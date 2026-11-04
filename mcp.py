# Caliper as an MCP server over stdio. Point an MCP client at:  py mcp.py
# Needs the engine running (py exact.py --serve). No dependencies beyond the standard library.
import os, sys, json, urllib.request
URL = os.environ.get("CALIPER_URL", "http://127.0.0.1:8811")
TOOLS = [
    {"name": "caliper_generate",
     "description": "Generate a reply with the local model. mode 'exact' returns tokens bit-identical to the full-precision reference. "
                    "mode 'resident' is fast and reports, after the reply, where the reference would have chosen differently.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"}, "mode": {"type": "string", "enum": ["resident", "exact"], "default": "resident"},
         "max_tokens": {"type": "integer", "default": 512}}, "required": ["prompt"]}},
    {"name": "caliper_verify",
     "description": "Check a piece of text against the full-precision reference: how far it is exactly what the reference "
                    "would have written after the prompt, and where it first differs.",
     "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "text": {"type": "string"}}, "required": ["prompt", "text"]}},
]
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600).read().decode()
def generate(a):
    msgs = [{"role": "user", "content": a["prompt"]}]
    r = json.loads(post("/v1/chat/completions", {"messages": msgs, "mode": a.get("mode", "resident"), "max_tokens": a.get("max_tokens", 512)}))
    text = r["choices"][0]["message"]["content"]; c = r.get("caliper", {})
    note = ""
    if "exact_prefix" in c:
        note = "\n\n[caliper] exact up to token %d of %d; %d place(s) where the reference would differ" % (c["exact_prefix"], c["total"], len(c.get("spans", [])))
    elif "accept" in c:
        note = "\n\n[caliper] exact mode: every token verified against the reference (%d checks)" % c.get("checks", 0)
    return text + note
def verify(a):
    r = json.loads(post("/verify", {"messages": [{"role": "user", "content": a["prompt"]}], "text": a["text"]}))
    return json.dumps(r, indent=1)
def reply(i, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": i}
    if error: m["error"] = {"code": -32000, "message": str(error)}
    else: m["result"] = result
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line)
        except Exception: continue
        i, method, p = req.get("id"), req.get("method", ""), req.get("params") or {}
        try:
            if method == "initialize":
                reply(i, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "caliper", "version": "0.1"}})
            elif method == "tools/list": reply(i, {"tools": TOOLS})
            elif method == "tools/call":
                name, args = p.get("name"), p.get("arguments") or {}
                out = generate(args) if name == "caliper_generate" else verify(args) if name == "caliper_verify" else None
                if out is None: reply(i, error="unknown tool " + str(name))
                else: reply(i, {"content": [{"type": "text", "text": out}]})
            elif method == "ping": reply(i, {})
            elif i is not None: reply(i, error="unknown method " + method)
        except Exception as ex:
            if i is not None: reply(i, error=ex)
if __name__ == "__main__": main()
