# B6RUN: the 14B held fully on the card in B6 (6-bit + bf16 scale per 32), Triton reads it straight into the matvec. Greedy chat, tok/s.
import os, sys, json, time, math, torch, triton, triton.language as tl
SRC = os.environ.get("CALIPER_MODEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
_ST_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "I64": torch.int64,
          "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
_ST_HDR = {}
def st_header(path):  # safetensors: u64 header length, JSON header, then the raw tensor bytes
    if path not in _ST_HDR:
        with open(path, "rb") as fp:
            n = int.from_bytes(fp.read(8), "little"); _ST_HDR[path] = (8 + n, json.loads(fp.read(n)))
    return _ST_HDR[path]
def st_tensor(path, name):
    """One tensor, read with a plain file read into memory this process owns. No memory map: mapped file pages
    would sit in the working set beside the copy the engine makes of them, and on a 14B that is the checkpoint twice."""
    base, h = st_header(path); m = h[name]; a, b = m["data_offsets"]
    t = torch.empty(b - a, dtype=torch.uint8); mv = memoryview(t.numpy()); got = 0
    with open(path, "rb") as fp:
        fp.seek(base + a)
        while got < b - a:
            k = fp.readinto(mv[got:])
            if not k: raise IOError("short read: %s in %s" % (name, path))
            got += k
    return t.view(_ST_DT[m["dtype"]]).reshape(m["shape"])
@triton.jit
def gemv6(L, H, S, X, Y, R, C, sl, sh, ss, BR: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR); rm = r < R
    acc = tl.zeros((BR,), dtype=tl.float32)
    for k0 in range(0, C, BK):
        c = k0 + tl.arange(0, BK)
        lb = tl.load(L + r[:, None] * sl + (c // 2)[None, :], mask=rm[:, None], other=0).to(tl.int32)
        hb = tl.load(H + r[:, None] * sh + (c // 4)[None, :], mask=rm[:, None], other=0).to(tl.int32)
        q = (((hb >> ((c % 4) * 2)[None, :]) & 3) << 4) | ((lb >> ((c % 2) * 4)[None, :]) & 15)
        s = tl.load(S + r[:, None] * ss + (c // 32)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        x = tl.load(X + c).to(tl.float32)
        acc += tl.sum((q - 32).to(tl.float32) * s * x[None, :], axis=1)
    tl.store(Y + r, acc, mask=rm)
class B6:
    def __init__(s, w, bias=None):
        R, C = w.shape; Ls, Hs, Ss = [], [], []
        for r0 in range(0, R, 8192):
            wc = w[r0:r0 + 8192].cuda().float(); n = wc.shape[0]
            sc = (wc.view(n, C // 32, 32).abs().amax(-1) / 31).to(torch.bfloat16)
            sf = sc.float(); sf[sf == 0] = 1.0
            u = (torch.clamp(torch.round(wc.view(n, C // 32, 32) / sf[..., None]), -31, 31) + 32).to(torch.uint8).view(n, C)
            lo, hi = u & 15, u >> 4
            Ls.append(lo[:, 0::2] | (lo[:, 1::2] << 4)); Hs.append(hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)); Ss.append(sc)
            del wc, u, lo, hi, sf
        s.L = torch.cat(Ls).contiguous(); s.H = torch.cat(Hs).contiguous(); sc = torch.cat(Ss)
        torch.cuda.empty_cache()
        s.S = sc.contiguous(); s.R, s.C = R, C
        s.b = bias.cuda().float() if bias is not None else None
        s.y = torch.empty(R, device="cuda", dtype=torch.float32)
    def __call__(s, x):
        gemv6[(triton.cdiv(s.R, BR),)](s.L, s.H, s.S, x, s.y, s.R, s.C, s.L.stride(0), s.H.stride(0), s.S.stride(0), BR=BR, BK=BK)
        return s.y + s.b if s.b is not None else s.y.clone()
BR, BK = int(os.environ.get("B6_BR", 16)), int(os.environ.get("B6_BK", 256))
def main():
    t0 = time.time(); cfg = json.load(open(os.path.join(SRC, "config.json")))
    nl, nh, nkv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
    eps, theta = cfg["rms_norm_eps"], cfg.get("rope_theta", 1e6)
    _ix = os.path.join(SRC, "model.safetensors.index.json")
    if os.path.exists(_ix):
        idx = json.load(open(_ix))["weight_map"]                       # sharded
    else:                                                              # one file
        _one = "model.safetensors"
        idx = {k: _one for k in st_header(os.path.join(SRC, _one))[1] if k != "__metadata__"}
    def Tc(n):                       # cast whatever the model stores to the working dtype
        t = T(n); return t.to(WDT) if t.dtype != WDT else t

    def Topt(n):
        try: return T(n)
        except Exception: return None

    def T(n):
        return st_tensor(os.path.join(SRC, idx[n]), n)
    WDT = T("model.embed_tokens.weight").dtype                        # fp32, fp16 or bf16, as stored
    if WDT not in (torch.float32, torch.float16, torch.bfloat16):
        raise SystemExit("Caliper reads fp32, fp16 and bf16 safetensors; this model is " + str(WDT))
    print("weights are", str(WDT).split(".")[-1], flush=True)
    emb = T("model.embed_tokens.weight")  # stays on the host
    Ls = []
    for i in range(nl):
        p = "model.layers.%d." % i; a = p + "self_attn."; m = p + "mlp."
        Ls.append(dict(n1=T(p + "input_layernorm.weight").cuda().float(), n2=T(p + "post_attention_layernorm.weight").cuda().float(),
            q=B6(T(a + "q_proj.weight"), Topt(a + "q_proj.bias")), k=B6(T(a + "k_proj.weight"), Topt(a + "k_proj.bias")),
            v=B6(T(a + "v_proj.weight"), Topt(a + "v_proj.bias")), o=B6(T(a + "o_proj.weight")),
            g=B6(T(m + "gate_proj.weight")), u=B6(T(m + "up_proj.weight")), d=B6(T(m + "down_proj.weight"))))
    nf = T("model.norm.weight").cuda().float(); head = B6(Topt("lm_head.weight") if Topt("lm_head.weight") is not None else emb)   # tied embedding when absent
    torch.cuda.synchronize(); load_s = time.time() - t0
    used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
    print("B6RUN loaded in %.0fs, card used %.2f GB" % (load_s, used), flush=True)
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    MAXT = 4096; K = torch.zeros(nl, nkv, MAXT, hd, device="cuda"); V = torch.zeros_like(K)
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean() + eps) * w
    def rope(x, pos):
        f = pos * inv; c, s = torch.cos(f).repeat(2), torch.sin(f).repeat(2)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * c + torch.cat([-x2, x1], -1) * s
    def step(tok, pos):
        x = emb[tok].cuda().float()
        for i, l in enumerate(Ls):
            h = rms(x, l["n1"])
            q = rope(l["q"](h).view(nh, hd), pos); k = rope(l["k"](h).view(nkv, hd), pos); v = l["v"](h).view(nkv, hd)
            K[i, :, pos] = k; V[i, :, pos] = v
            kk = K[i, :, : pos + 1].repeat_interleave(nh // nkv, 0); vv = V[i, :, : pos + 1].repeat_interleave(nh // nkv, 0)
            att = torch.softmax((kk @ q[:, :, None]).squeeze(-1) / math.sqrt(hd), -1)
            x = x + l["o"]((att[:, None, :] @ vv).reshape(-1).contiguous())
            h = rms(x, l["n2"]); x = x + l["d"]((torch.nn.functional.silu(l["g"](h)) * l["u"](h)).contiguous())
        return head(rms(x, nf).contiguous())
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(SRC); stops = (tk.eos_token_id, tk.convert_tokens_to_ids("<|im_end|>"))
    def gen(msgs, maxnew=512):
        ids = tk.encode(tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))[-(MAXT - maxnew - 2):]
        with torch.no_grad():
            for p, t in enumerate(ids[:-1]): step(t, p)
            pos = len(ids) - 1; tok = ids[-1]; out = []
            for _ in range(min(maxnew, MAXT - pos - 1)):
                tok = int(torch.argmax(step(tok, pos))); pos += 1
                if tok in stops: break
                out.append(tok); yield tk.decode(out)
    if "--serve" in sys.argv:
        serve(gen); return
    msg = sys.argv[1] if len(sys.argv) > 1 else "Write a 120 word story about a lighthouse keeper."
    t0 = time.time(); t1 = None; text = ""; n = 0
    for text in gen([{"role": "user", "content": msg}], 96):
        if t1 is None: t1 = time.time()
        n += 1
    now = time.time()
    print("caliper %.2f tok/s generating (%.2f including the prompt pass, %d tokens)"
          % ((n - 1) / (now - t1) if n > 1 else 0.0, n / (now - t0), n))
    print(text)
def serve(gen):
    import http.server, threading, webbrowser
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.html"), "rb").read().replace(b"__MODES__", b"resident"); lock = threading.Lock()
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(page)
        def do_POST(self):
            msgs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"]
            self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.end_headers()
            with lock:
                prev = ""; n = 0; t0 = None
                for text in gen(msgs):
                    if t0 is None: t0 = time.time()
                    d = text[len(prev):]; prev = text; n += 1
                    frame = dict(d=d, tok_s=(n - 1) / max(1e-9, time.time() - t0))
                    self.wfile.write((json.dumps(frame) + "\n").encode()); self.wfile.flush()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8811), H)
    webbrowser.open("http://127.0.0.1:8811/"); srv.serve_forever()
if __name__ == "__main__": main()
