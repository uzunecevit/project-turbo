"""
Long Context + Norm Drift Test
Compares baseline vs 2-PASS at different context lengths.
Measures output quality degradation and token divergence.
"""

import sys, os, json, time

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext, TurboBridge

MODEL = os.environ.get("TURBO_TEST_MODEL", "")


def gen_greedy(ctx, prompt_tokens, n_gen=5):
    """Greedy decode from pre-tokenized prompt. Returns generated tokens."""
    ctx.decode(prompt_tokens, 0)
    output = []
    for i in range(n_gen):
        tok = prompt_tokens[-1] if i == 0 else output[-1]
        logits = ctx.decode([tok], len(prompt_tokens) + i)
        best = max(range(len(logits)), key=lambda j: logits[j])
        output.append(best)
    return output


def run_test(label, env_overrides, prompt_tokens, n_gen=5):
    """Run test with given env overrides."""
    for k, v in env_overrides.items():
        os.environ[k] = v
    for k in ["TURBO_2PASS_SPLIT", "GGML_CUDA_DISABLE_GRAPHS", "TURBO_RECENT_WINDOW"]:
        if k not in env_overrides:
            os.environ.pop(k, None)

    ctx = TurboContext(MODEL, n_ctx=4096, cache_type="turbo4", n_gpu_layers=999)
    t0 = time.time()
    tokens = gen_greedy(ctx, prompt_tokens, n_gen)
    elapsed = time.time() - t0
    text = ctx.detokenize(tokens)
    del ctx
    return {"tokens": tokens, "text": text, "elapsed": elapsed}


# Build long prompts by repeating a base sentence
base = "The quick brown fox jumps over the lazy dog. "
context_lengths = [256, 512, 1024, 2048]

results = {}
for ctx_len in context_lengths:
    prompt = base * (ctx_len // len(base.split()) + 1)
    tmp_bridge = TurboBridge()
    tmp_bridge.load_model(MODEL, 999)
    prompt_tokens = tmp_bridge.tokenize(prompt)[:ctx_len]
    del tmp_bridge

    print(f"\n{'=' * 60}", flush=True)
    print(f"Context length: {ctx_len} tokens", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Baseline (no 2-PASS)
    print(f"  BASELINE...", flush=True)
    r_base = run_test("baseline", {"GGML_CUDA_DISABLE_GRAPHS": "1"}, prompt_tokens, 5)
    print(f"    tokens: {r_base['tokens']}", flush=True)
    print(f"    text:   {r_base['text']}", flush=True)
    print(f"    time:   {r_base['elapsed']:.2f}s", flush=True)

    # 2-PASS
    print(f"  2-PASS...", flush=True)
    r_2pass = run_test(
        "2pass",
        {
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "TURBO_2PASS_SPLIT": "1",
            "TURBO_RECENT_WINDOW": "64",
        },
        prompt_tokens,
        5,
    )
    print(f"    tokens: {r_2pass['tokens']}", flush=True)
    print(f"    text:   {r_2pass['text']}", flush=True)
    print(f"    time:   {r_2pass['elapsed']:.2f}s", flush=True)

    # Comparison
    match = r_base["tokens"] == r_2pass["tokens"]
    common = sum(1 for a, b in zip(r_base["tokens"], r_2pass["tokens"]) if a == b)
    seq_pct = 100.0 * common / max(len(r_base["tokens"]), len(r_2pass["tokens"]))
    speedup = r_base["elapsed"] / r_2pass["elapsed"] if r_2pass["elapsed"] > 0 else 0

    results[ctx_len] = {
        "match": match,
        "seq_pct": seq_pct,
        "baseline_tokens": r_base["tokens"],
        "twopass_tokens": r_2pass["tokens"],
        "baseline_time": r_base["elapsed"],
        "twopass_time": r_2pass["elapsed"],
    }

    print(
        f"  RESULT: {'EXACT MATCH' if match else f'{seq_pct:.0f}% similarity'}",
        flush=True,
    )
    print(
        f"  Time: baseline={r_base['elapsed']:.2f}s 2pass={r_2pass['elapsed']:.2f}s",
        flush=True,
    )

# Summary
print(f"\n{'=' * 60}", flush=True)
print("LONG CONTEXT SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
print(
    f"{'Ctx':>6} | {'Match':>8} | {'Seq%':>6} | {'Base t':>8} | {'2Pass t':>8}",
    flush=True,
)
print(f"{'-' * 6}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 8}", flush=True)
for ctx_len in context_lengths:
    r = results[ctx_len]
    match_str = "EXACT" if r["match"] else f"{r['seq_pct']:.0f}%"
    print(
        f"{ctx_len:>6} | {match_str:>8} | {r['seq_pct']:>5.1f}% | {r['baseline_time']:>7.2f}s | {r['twopass_time']:>7.2f}s",
        flush=True,
    )
print(f"{'=' * 60}", flush=True)
