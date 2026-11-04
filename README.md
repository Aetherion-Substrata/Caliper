# Caliper

A local inference engine for one dense model on one consumer card, with an
**exact mode**: every token bit-identical to what the original bfloat16 weights
would have produced.

An engine by Aetherion Substrata Ltd.

## The two modes

| mode | what you get | tok/s |
| --- | --- | --- |
| `engine.py` | the model held entirely on the card in a 6-bit form | 12.4 |
| `exact.py` | every token bit-identical to the bfloat16 reference | 1.9 |

Generation rate, 14B dense, 16 GB card, greedy, single stream, desktop open.
Loading takes about 15 seconds; the prompt pass is counted separately.

### Resident mode

Each weight becomes a 6-bit signed step count; every 32 weights share one
bfloat16 scale. 6.5 bits a weight. A 14B model is 12.9 GB in this form and sits
on the card with room for the cache, so nothing streams during generation.

Measured cost in quality, two ways, on a frozen text and a 40-question
checkable quiz:

| | perplexity | quiz |
| --- | --- | --- |
| reference | 51.928 | 26/40 |
| 6-bit | 51.901 | 26/40 |

No measurable loss.

### Exact mode

The resident 6-bit model proposes a run of tokens. The original bfloat16
weights, held in pinned system RAM, stream across the bus once and check the
whole run at full precision. Tokens are kept as far as the check agrees, then
the check's own next token is taken. Every token emitted is the one the
full-precision model would have chosen.

Measured:

- the proposer agrees with the full-precision model on **95.8%** of tokens
- about 4.5 tokens of a 10-token run survive in live generation
- a run repeats token for token at a **fixed guess length**. Change it and a
  near-tie can land the other way: over 65 positions, one picked differently
  when the check covered 40 at a time instead of 8. The weights are the same
  either way - a wide reduction and a narrow one disagree on the last bits, as
  they do in any batched GPU arithmetic. Hold the guess length and the run is
  reproducible; quote it with any figure you publish

Speed is set by the bus: a check moves 29.5 GB of weights, about 1.9 s at the
measured 15.2 GB/s, which is nearly the whole cost of a token. That cost is paid
per check and not per token, so the figure that sets the speed is how many tokens
survive each check, and a longer guess helps only as far as the proposer keeps
agreeing.

## What it is for

Determinism against a reference. Evaluation runs that have to repeat exactly,
regression tests across quantisation changes, anything where "close enough" is
not a result.

## Running it

    pip install -r requirements.txt
    set CALIPER_MODEL=X:\path\to\your\model
    py exact.py --serve

Opens the harness locally with both modes and a switch in the header. The
resident copy and the original weights are loaded once and shared, so flipping
the switch costs nothing and you can put the same prompt through both.

    py engine.py --serve

Resident mode only. Loads in about half the time and needs no extra system RAM,
for when you do not want the exact mode at all.

Without `--serve`, either one answers a single prompt given as an argument and
prints the rate.

## The page

Copy on every turn, regenerate on replies, attach text files (they fold into
the prompt as fenced blocks), stop mid-reply, new chat, export the
conversation as markdown. Resident mode runs the check in the background after
the reply lands: words the reference would have chosen differently turn amber,
and clicking one swaps in the reference's word. Up to the cyan mark the reply
is exactly the reference's. Compare, on any reply, runs the same prompt through exact mode and shows a
true word diff: struck amber for what you got, cyan for what the reference
wrote, with the share of identical words. The compare icon in the header makes
that automatic, firing the moment a resident reply lands. It is not run
alongside the reply on purpose: both modes want the same card, and the check
moves 29.5 GB over the bus, so running them together would slow the reply
badly. Sounds are synthesised in the page; the speaker icon mutes them.

## Endpoints

Everything speaks JSON over the one local address.

    POST /v1/chat/completions   OpenAI-compatible, streaming or not.
                                model "caliper-resident" or "caliper-exact".
                                The response carries a "caliper" object with
                                the verification result.
    GET  /v1/models
    POST /verify                {"messages": [...], "text": "..."}
                                how far the text is exactly what the reference
                                would have written, and where it first differs.

So any client that talks to an OpenAI-style endpoint can point at Caliper.

## MCP

    py mcp.py

Caliper as an MCP server over stdio, standard library only. Two tools:
`caliper_generate` (prompt, mode) and `caliper_verify` (prompt, text). Needs
the engine running.

## Your own front end

The page is one standalone HTML file and the server is a few lines. Replace
`harness.html` with anything you like, or point your own client at it:

    POST /   {"mode":"exact","messages":[{"role":"user","content":"..."}]}

Both modes stream one JSON object per line as the reply is produced:

    {"d": "Red", "checks": 1, "accept": 0.7, "tok_s": 1.9}

`d` is the text just settled. In exact mode `accept` is the share of the
proposed run the original weights kept, so a front end can draw the
verification as it happens. `mode` is `exact` or `resident`.

## Models

Point `CALIPER_MODEL` at any local folder of safetensors with a Qwen2 or
Llama-style dense architecture. Weights may be **bf16, fp16 or fp32**; the
engine reads whichever the model stores and works in it throughout, so exact
mode verifies against the model's own numbers, not a converted copy. One file
or sharded, tied embeddings or a separate head, with or without attention
biases, explicit `head_dim` or derived: all handled.

Run so far:

| model | resident rate |
| --- | --- |
| Qwen2.5-0.5B-Instruct | 27.1 tok/s |
| Qwen2.5-1.5B-Instruct | 21.5 tok/s |
| Qwen2.5-7B-Instruct | 23.8 tok/s |
| Qwen2.5-14B-Instruct | 12.4 tok/s |
| Mistral-Nemo-Instruct-2407 (12B) | runs |
| Qwen2.5-0.5B cast to fp16 | 27.1 tok/s |
| Qwen2.5-0.5B cast to fp32 | 26.9 tok/s |

## What it takes

- Dense models, Qwen2 or Llama-style, in bf16, fp16 or fp32 safetensors.
- Resident mode: about 14 GB free on the card for a 14B.
- Exact mode: the model's own weights in system RAM, about 30 GB for a 14B.
- Greedy decoding.
- Bring your own weights. Point `CALIPER_MODEL` at the folder.

Exact mode works against original weights. Formats that have already discarded
them (GGUF, GPTQ, AWQ) have nothing left to verify against.

## Next

Streaming only the bits the resident copy does not already hold, which about
doubles the exact rate. A tuned resident kernel. Sampling that preserves the
reference model's distribution.

## Licence

Apache 2.0. See LICENSE and NOTICE.
