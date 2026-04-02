"""Quick stochastic test: temp=0.7, top_p=0.95, 5 tokens."""

import sys, os, math, random

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODEL = os.environ["TURBO_TEST_MODEL"]


def sample(logits, temp=0.7, top_p=0.95, seed=42):
    random.seed(seed)
    scaled = [l / temp for l in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    probs = [e / total for e in exps]
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0.0
    cutoff = len(indexed)
    for j, (_, p) in enumerate(indexed):
        cumsum += p
        if cumsum >= top_p:
            cutoff = j + 1
            break
    candidates = indexed[:cutoff]
    total_c = sum(p for _, p in candidates)
    r = random.random()
    cumsum = 0.0
    for idx, p in candidates:
        cumsum += p / total_c
        if r <= cumsum:
            return idx
    return candidates[0][0]


prompt = "The history of artificial intelligence began"

ctx = TurboContext(
    MODEL, n_ctx=2048, n_batch=512, cache_type="turbo4", n_gpu_layers=999
)
tokens = ctx.tokenize(prompt)
del ctx


def run(label, env):
    for k, v in env.items():
        os.environ[k] = v
    for k in ["TURBO_2PASS_SPLIT", "GGML_CUDA_DISABLE_GRAPHS", "TURBO_RECENT_WINDOW"]:
        if k not in env:
            os.environ.pop(k, None)
    ctx = TurboContext(
        MODEL, n_ctx=2048, n_batch=512, cache_type="turbo4", n_gpu_layers=999
    )
    ctx.decode(tokens, 0)
    out = []
    for i in range(10):
        t = tokens[-1] if i == 0 else out[-1]
        logits = ctx.decode([t], len(tokens) + i)
        out.append(sample(logits, seed=42 + i))
    text = ctx.detokenize(out)
    del ctx
    return out, text


print("BASELINE...", flush=True)
base_out, base_text = run("base", {"GGML_CUDA_DISABLE_GRAPHS": "1"})
print(f"  tokens: {base_out}", flush=True)
print(f"  text:   {base_text[:50]!r}", flush=True)

print("2-PASS...", flush=True)
tp_out, tp_text = run(
    "2pass",
    {
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "TURBO_2PASS_SPLIT": "1",
        "TURBO_RECENT_WINDOW": "64",
    },
)
print(f"  tokens: {tp_out}", flush=True)
print(f"  text:   {tp_text[:50]!r}", flush=True)

diverge = None
for i in range(len(base_out)):
    if base_out[i] != tp_out[i]:
        diverge = i
        break

if diverge is None:
    print(f"RESULT: EXACT MATCH (all {len(base_out)} sampled tokens)", flush=True)
else:
    common = sum(1 for a, b in zip(base_out, tp_out) if a == b)
    print(
        f"RESULT: Diverge at token {diverge}, match {common}/{len(base_out)}",
        flush=True,
    )

print("DONE", flush=True)
