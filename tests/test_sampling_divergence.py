"""Stochastic sampling divergence test: temp=0.8, top_p=0.95, 100 tokens."""

import sys, os

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODEL = os.environ["TURBO_TEST_MODEL"]
prompt = "Write a detailed explanation of how neural networks learn from data. "


def gen_sampled(ctx, prompt_tokens, n_gen=50, temp=0.8, top_p=0.95, seed=42):
    """Sample decode with temperature and top_p."""
    import random

    random.seed(seed)
    ctx.decode(prompt_tokens, 0)
    out = []
    for i in range(n_gen):
        t = prompt_tokens[-1] if i == 0 else out[-1]
        logits = ctx.decode([t], len(prompt_tokens) + i)
        # Apply temperature
        scaled = [l / temp for l in logits]
        # Simple top-p sampling (sort, cumulative, cut)
        indexed = sorted(enumerate(scaled), key=lambda x: -x[1])
        cumsum = 0.0
        import math

        probs = [math.exp(s - max(scaled)) for _, s in indexed]
        total = sum(probs)
        probs = [p / total for p in probs]
        cutoff = 0
        for j, p in enumerate(probs):
            cumsum += p
            if cumsum >= top_p:
                cutoff = j + 1
                break
        candidates = indexed[:cutoff]
        probs_c = [probs[j] for j in range(cutoff)]
        total_c = sum(probs_c)
        probs_c = [p / total_c for p in probs_c]
        # Weighted random choice
        r = random.random()
        cumsum = 0.0
        chosen = candidates[0][0]
        for j, (idx, _) in enumerate(candidates):
            cumsum += probs_c[j]
            if r <= cumsum:
                chosen = idx
                break
        out.append(chosen)
    return out


# Baseline
os.environ.pop("TURBO_2PASS_SPLIT", None)
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
ctx = TurboContext(
    MODEL, n_ctx=4096, n_batch=512, cache_type="turbo4", n_gpu_layers=999
)
prompt_tokens = ctx.tokenize(prompt)
base_out = gen_sampled(ctx, prompt_tokens, 50, temp=0.8, top_p=0.95, seed=42)
base_text = ctx.detokenize(base_out[:20])
del ctx

# 2-PASS
os.environ["TURBO_2PASS_SPLIT"] = "1"
os.environ["TURBO_RECENT_WINDOW"] = "64"
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
ctx = TurboContext(
    MODEL, n_ctx=4096, n_batch=512, cache_type="turbo4", n_gpu_layers=999
)
prompt_tokens = ctx.tokenize(prompt)
tp_out = gen_sampled(ctx, prompt_tokens, 50, temp=0.8, top_p=0.95, seed=42)
tp_text = ctx.detokenize(tp_out[:20])
del ctx

# Analysis
diverge_at = None
for i in range(len(base_out)):
    if base_out[i] != tp_out[i]:
        diverge_at = i
        break

common = sum(1 for a, b in zip(base_out, tp_out) if a == b)
pct = 100.0 * common / len(base_out)

print(f"Sampling test (temp=0.8, top_p=0.95, {len(base_out)} tokens):", flush=True)
if diverge_at is None:
    print(f"  Result: EXACT MATCH (all {len(base_out)} tokens identical)", flush=True)
else:
    print(f"  First divergence at token {diverge_at}", flush=True)
    print(f"  Match rate: {common}/{len(base_out)} ({pct:.0f}%)", flush=True)
    print(f"  Base  : {base_text[:60]!r}", flush=True)
    print(f"  2Pass : {tp_text[:60]!r}", flush=True)
print("DONE", flush=True)
