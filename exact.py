# EXACT: lossless 14B output on a 16 GB card. The 6-bit copy on the card guesses G tokens; the lossless bf16 weights,
# pinned in host RAM, stream through the card once per check and keep the longest run of guesses the lossless model
# itself would pick (greedy), plus its own next token. Every emitted token is the lossless model's choice.
import os, sys, json, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as b6m
from safetensors import safe_open
SRC = b6m.SRC; G = int(os.environ.get("CALIPER_GUESS", 10)); MAXT = 2048
def load():
    t0 = time.time(); cfg = json.load(open(os.path.join(SRC, "config.json")))
    nl, nh, nkv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]; hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
    eps, theta = cfg["rms_norm_eps"], cfg.get("rope_theta", 1e6)
    fh = {}; _ix = os.path.join(SRC, "model.safetensors.index.json")
    if os.path.exists(_ix):
        idx = json.load(open(_ix))["weight_map"]                       # sharded
    else:                                                              # one file
        from safetensors import safe_open as _so
        _one = "model.safetensors"
        idx = {k: _one for k in _so(os.path.join(SRC, _one), framework="pt").keys()}
    def Topt(n):
        try: return T(n)
        except Exception: return None

    def T(n):
        f = idx[n]
        if f not in fh: fh[f] = safe_open(os.path.join(SRC, f), framework="pt")
        return fh[f].get_tensor(n)
    WDT = T("model.embed_tokens.weight").dtype
    if WDT not in (torch.float32, torch.float16, torch.bfloat16):
        raise SystemExit("Caliper reads fp32, fp16 and bf16 safetensors; this model is " + str(WDT))
    emb = T("model.embed_tokens.weight")
    names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    host, small, draft = [], [], []
    per = sum(math.prod(T("model.layers.0." + n).shape) for n in names)
    hs = tuple((Topt("lm_head.weight") if Topt("lm_head.weight") is not None else emb).shape)
    BIG = torch.empty(per * nl + hs[0] * hs[1], dtype=WDT)  # one host block, registered (pinned) once, exact size
    torch.cuda.cudart().cudaHostRegister(BIG.data_ptr(), BIG.numel() * BIG.element_size(), 0)
    for i in range(nl):
        p = "model.layers.%d." % i; ws = [Topt(p + n) for n in names]
        flat = BIG[i * per:(i + 1) * per]; o = 0; shp = []
        for w in ws: flat[o:o + w.numel()] = w.reshape(-1); shp.append((o, tuple(w.shape))); o += w.numel()
        host.append((flat, shp))
        sm = {k: (lambda t: t.cuda() if t is not None else None)(Topt(p + k)) for k in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias")}
        small.append(sm)
        draft.append(dict(n1=sm["input_layernorm.weight"].float(), n2=sm["post_attention_layernorm.weight"].float(),
            q=b6m.B6(ws[0], sm["self_attn.q_proj.bias"]), k=b6m.B6(ws[1], sm["self_attn.k_proj.bias"]), v=b6m.B6(ws[2], sm["self_attn.v_proj.bias"]),
            o=b6m.B6(ws[3]), g=b6m.B6(ws[4]), u=b6m.B6(ws[5]), d=b6m.B6(ws[6])))
    head = Topt("lm_head.weight")
    if head is None: head = emb                      # tied embedding
    headp = BIG[per * nl:].view(hs); headp.copy_(head); dhead = b6m.B6(head); nf = T("model.norm.weight").cuda()
    print("EXACT loaded in %.0fs, card used %.2f GB" % (time.time() - t0, (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9), flush=True)
    return locals()
class Engine:
    def __init__(s):
        m = load(); s.__dict__.update(m); s.nl, s.nh, s.nkv, s.hd = m["nl"], m["nh"], m["nkv"], m["hd"]
        hd = s.hd; s.inv = 1.0 / (s.theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
        s.DK = torch.zeros(s.nl, s.nkv, MAXT, hd, device="cuda"); s.DV = torch.zeros_like(s.DK)
        s.TK = torch.zeros(s.nl, s.nkv, MAXT, hd, device="cuda", dtype=s.WDT); s.TV = torch.zeros_like(s.TK)
        big = max(f.numel() for f, _ in s.host); s.buf = [torch.empty(big, dtype=s.WDT, device="cuda") for _ in range(2)]
        s.cs = torch.cuda.Stream(); s.HC = 24576
    def rope(s, x, pos):  # x [..., L, h, hd] with pos tensor [L]
        f = pos[:, None].float() * s.inv[None]; c, si = torch.cos(f).repeat(1, 2)[:, None], torch.sin(f).repeat(1, 2)[:, None]
        x1, x2 = x[..., : s.hd // 2], x[..., s.hd // 2:]; return x * c + torch.cat([-x2, x1], -1) * si
    def dlogits(s, tok, pos):  # draft, one token, 6-bit weights on card
        x = s.emb[tok].cuda().float(); P = torch.tensor([pos], device="cuda")
        rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean() + s.eps) * w
        for i, l in enumerate(s.draft):
            h = rms(x, l["n1"])
            q = s.rope(l["q"](h).view(1, s.nh, s.hd), P)[0]; k = s.rope(l["k"](h).view(1, s.nkv, s.hd), P)[0]; v = l["v"](h).view(s.nkv, s.hd)
            s.DK[i, :, pos] = k; s.DV[i, :, pos] = v
            kk = s.DK[i, :, : pos + 1].repeat_interleave(s.nh // s.nkv, 0); vv = s.DV[i, :, : pos + 1].repeat_interleave(s.nh // s.nkv, 0)
            a = torch.softmax((kk @ q[:, :, None]).squeeze(-1) / math.sqrt(s.hd), -1)
            x = x + l["o"]((a[:, None, :] @ vv).reshape(-1).contiguous())
            h = rms(x, l["n2"]); x = x + l["d"]((torch.nn.functional.silu(l["g"](h)) * l["u"](h)).contiguous())
        return s.dhead(rms(x, s.nf.float()).contiguous())

    def dstep(s, tok, pos):
        return int(torch.argmax(s.dlogits(tok, pos)))
    def dstep_margin(s, tok, pos):
        """One resident step that also reports the copy's own confidence: the gap, in logits,
        between its best and second-best choice. Near zero means a near-tie."""
        top = torch.topk(s.dlogits(tok, pos), 2)
        return int(top.indices[0]), float(top.values[0] - top.values[1])

    def fetch(s, i):
        with torch.cuda.stream(s.cs):
            f, _ = s.host[i]; s.buf[i % 2][: f.numel()].copy_(f, non_blocking=True); ev = torch.cuda.Event(); ev.record(s.cs)
        return ev
    def tstep(s, toks, pos0):  # lossless check over L tokens at pos0.., weights streamed from host; returns greedy picks [L]
        L = len(toks); P = torch.arange(pos0, pos0 + L, device="cuda"); T = pos0 + L
        x = s.emb[torch.tensor(toks)].cuda()
        rms = lambda x, w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps) * w.float()).to(s.WDT)
        mask = torch.full((L, T), float("-inf"), device="cuda"); mask[:, :T] = 0; mask[:, pos0:] = torch.full((L, L), float("-inf"), device="cuda").triu(1)
        ev = s.fetch(0)
        for i in range(s.nl):
            torch.cuda.current_stream().wait_event(ev); b = s.buf[i % 2]; _, shp = s.host[i]
            W = [b[o:o + math.prod(sh)].view(sh) for o, sh in shp]; sm = s.small[i]
            if i + 1 < s.nl:
                done = torch.cuda.Event(); done.record(); s.cs.wait_event(done); ev = s.fetch(i + 1)
            h = rms(x, sm["input_layernorm.weight"])
            bq, bk, bv = sm["self_attn.q_proj.bias"], sm["self_attn.k_proj.bias"], sm["self_attn.v_proj.bias"]
            q = (h @ W[0].T + bq if bq is not None else h @ W[0].T).view(L, s.nh, s.hd)
            k = (h @ W[1].T + bk if bk is not None else h @ W[1].T).view(L, s.nkv, s.hd)
            v = (h @ W[2].T + bv if bv is not None else h @ W[2].T).view(L, s.nkv, s.hd)
            q, k = s.rope(q.float(), P), s.rope(k.float(), P)
            s.TK[i, :, pos0:T] = k.transpose(0, 1).to(s.WDT); s.TV[i, :, pos0:T] = v.transpose(0, 1)
            kk = s.TK[i, :, :T].float().repeat_interleave(s.nh // s.nkv, 0); vv = s.TV[i, :, :T].float().repeat_interleave(s.nh // s.nkv, 0)
            sc = torch.einsum("lhd,htd->hlt", q, kk) / math.sqrt(s.hd) + mask
            o = torch.einsum("hlt,htd->lhd", torch.softmax(sc, -1), vv).reshape(L, -1).to(s.WDT)
            x = x + o @ W[3].T
            h = rms(x, sm["post_attention_layernorm.weight"])
            mlp_in = torch.nn.functional.silu(h @ W[4].T) * (h @ W[5].T); y = mlp_in @ W[6].T
            x = x + y
        h = rms(x, s.nf); best_v = torch.full((L,), float("-inf"), device="cuda"); best_i = torch.zeros(L, dtype=torch.long, device="cuda")
        for r0 in range(0, s.headp.shape[0], s.HC):  # lm_head streamed in chunks through buffer 0
            hc = s.headp[r0:r0 + s.HC]; n = hc.numel(); torch.cuda.synchronize(); s.buf[0][:n].copy_(hc.reshape(-1)); lg = (h @ s.buf[0][:n].view(hc.shape).T).float()
            v, ix = lg.max(-1); upd = v > best_v; best_v = torch.where(upd, v, best_v); best_i = torch.where(upd, ix + r0, best_i)
        return best_i.tolist()
    def gen_fast(s, ids, maxnew=512, stops=()):
        tok = None
        for p, t in enumerate(ids): tok = s.dstep(t, p)
        pos = len(ids); out = []; t0 = None
        while len(out) < maxnew and tok not in stops:
            out.append(tok)
            el = (time.time() - t0) if t0 else 0.0
            yield out, dict(tok_s=(len(out) - 1) / el if el > 0 else 0.0)
            if t0 is None: t0 = time.time()
            tok = s.dstep(tok, pos); pos += 1

    def gen_audit(s, ids, maxnew=512, stops=()):
        out = []
        for out, st in s.gen_fast(ids, maxnew, stops):
            yield out, dict(st, phase="fast")
        if not out: return
        t0 = time.time()
        s.TK.zero_(); s.TV.zero_()
        seq = list(ids) + list(out)
        picks = s.tstep(seq[:-1], 0)                     # original's greedy pick at every position, one pass
        want = picks[len(ids) - 1:]                      # what the original would put where each reply token sits
        miss = [i for i, (a, b) in enumerate(zip(out, want)) if a != b]
        first = miss[0] if miss else len(out)
        yield out, dict(phase="audit", audit_s=round(time.time() - t0, 2), exact_prefix=first,
                        mismatch=miss, want=[int(want[i]) for i in miss], total=len(out))

    def verify(s, ids, toks):
        s.TK.zero_(); s.TV.zero_()
        seq = list(ids) + list(toks)
        picks = s.tstep(seq[:-1], 0); want = picks[len(ids) - 1:]
        miss = [i for i, (a, b) in enumerate(zip(toks, want)) if a != b]
        return dict(exact_prefix=(miss[0] if miss else len(toks)), mismatch=miss, want=[int(want[i]) for i in miss], total=len(toks))

    def gen(s, ids, maxnew=512, stops=()):
        picks = s.tstep(ids, 0); tok = picks[-1]; pos = len(ids)
        for p, t in enumerate(ids): s.dstep(t, p)  # fill draft cache
        out = [tok]; checks = drafted = accepted = 0; t0 = time.time()
        while len(out) < maxnew and tok not in stops and pos + G + 1 < MAXT:
            d = []; tt = tok
            for j in range(G): tt = s.dstep(tt, pos + j); d.append(tt)
            v = s.tstep([tok] + d, pos); checks += 1; drafted += G
            n = 0
            while n < G and d[n] == v[n]: n += 1
            accepted += n; new = d[:n] + [v[n]]; pos += n + 1; tok = new[-1]
            for t in new:
                if t in stops: break
                out.append(t)
            yield out, dict(checks=checks, accept=accepted / max(1, drafted), tok_s=len(out) / (time.time() - t0))
            if any(t in stops for t in new): break
def main():
    e = Engine()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(SRC); stops = (tk.eos_token_id, tk.convert_tokens_to_ids("<|im_end|>"))
    def run(msgs, maxnew=512, mode="exact"):
        ids = tk.encode(tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))[-(MAXT - maxnew - G - 4):]
        loop = e.gen if mode == "exact" else e.gen_audit
        with torch.no_grad():
            for out, st in loop(ids, maxnew, stops):
                if st.get("phase") == "audit":
                    # whole words on both sides. The original's first token is exact; the rest of its word is
                    # completed by the resident copy (highest position first so earlier cache stays intact).
                    T = tk.decode(out); spans = []
                    for i in sorted(st["mismatch"], reverse=True):
                        pos = len(ids) + i; w0 = st["want"][st["mismatch"].index(i)]
                        toks = [w0]; tok = e.dstep(w0, pos)
                        while len(toks) < 4 and tok not in stops:
                            piece = tk.decode([tok])
                            if not piece or piece[0] in " \n" or not piece[0].isalnum(): break
                            toks.append(tok); tok = e.dstep(tok, pos + len(toks) - 1)
                        a = len(tk.decode(out[:i])); b = len(tk.decode(out[:i + 1]))
                        while b < len(T) and T[b].isalnum(): b += 1          # resident's whole word
                        spans.append([a, b, tk.decode(toks), len(tk.decode([w0]))])
                    spans.sort()
                    st = dict(st, spans=spans, exact_chars=len(tk.decode(out[:st["exact_prefix"]])))
                    st.pop("want", None); st.pop("mismatch", None)
                yield tk.decode([t for t in out if t not in stops]), st
    if "--serve" in sys.argv:
        import http.server, threading, webbrowser
        page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.html"), "rb").read().replace(b"__MODES__", b"both"); lock = threading.Lock()
        def rid(): return "chatcmpl-%d" % int(time.time() * 1000)
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def hdr(self, ctype):
                self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            def do_OPTIONS(self):
                self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"); self.end_headers()
            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    self.hdr("application/json"); self.wfile.write(json.dumps({"object": "list", "data": [
                        {"id": "caliper-resident", "object": "model", "owned_by": "caliper"}, {"id": "caliper-exact", "object": "model", "owned_by": "caliper"}]}).encode()); return
                self.hdr("text/html; charset=utf-8"); self.wfile.write(page)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path.startswith("/verify"):
                    with lock, torch.no_grad():
                        ids = tk.encode(tk.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=False))
                        toks = tk.encode(body["text"]); r = e.verify(ids, toks)
                        r["exact_chars"] = len(tk.decode(toks[:r["exact_prefix"]])); r["want"] = [tk.decode([w]) for w in r["want"]]
                    self.hdr("application/json"); self.wfile.write(json.dumps(r).encode()); return
                if self.path.startswith("/v1/chat/completions"):
                    model = body.get("model", ""); mode = body.get("mode") or ("exact" if "exact" in model else "resident")
                    maxnew = int(body.get("max_tokens") or 512); stream = bool(body.get("stream"))
                    if stream:
                        self.hdr("text/event-stream")
                        with lock:
                            prev = ""; st = {}
                            for text, st in run(body["messages"], maxnew, mode):
                                d = text[len(prev):]; prev = text
                                if d: self.wfile.write(("data: " + json.dumps({"id": rid(), "object": "chat.completion.chunk", "model": "caliper-" + mode,
                                    "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}]}) + "\n\n").encode()); self.wfile.flush()
                            self.wfile.write(("data: " + json.dumps({"id": rid(), "object": "chat.completion.chunk", "model": "caliper-" + mode,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "caliper": {k: v for k, v in st.items() if k != "d"}}) + "\n\ndata: [DONE]\n\n").encode())
                        return
                    with lock:
                        text = ""; st = {}
                        for text, st in run(body["messages"], maxnew, mode): pass
                    self.hdr("application/json")
                    self.wfile.write(json.dumps({"id": rid(), "object": "chat.completion", "model": "caliper-" + mode,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": st.get("total", len(tk.encode(text)))}, "caliper": {k: v for k, v in st.items() if k != "d"}}).encode()); return
                msgs, mode = body["messages"], body.get("mode", "exact")
                self.hdr("text/plain; charset=utf-8")
                with lock:
                    prev = ""
                    for text, st in run(msgs, mode=mode):
                        frame = dict(d=text[len(prev):], **st); prev = text
                        self.wfile.write((json.dumps(frame) + "\n").encode()); self.wfile.flush()
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8811), H); webbrowser.open("http://127.0.0.1:8811/"); srv.serve_forever()
    msg = sys.argv[1] if len(sys.argv) > 1 else "Write a 120 word story about a lighthouse keeper."
    text, st = "", {}
    for text, st in run([{"role": "user", "content": msg}], 120): pass
    st["text"] = text; json.dump(st, open(os.path.join(os.path.dirname(b6m.OUT + "\\"), "EXACT.json"), "w"), indent=1)
    print("EXACT", json.dumps({k: v for k, v in st.items() if k != "text"})); print(text); print("EXACT DONE")
if __name__ == "__main__": main()
