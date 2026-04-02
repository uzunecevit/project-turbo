"""Pre-production validation: CUDA Graphs ON + Perf sanity."""

import sys, os, time

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODEL = os.environ["TURBO_TEST_MODEL"]
prompt = "The meaning of life is"


def run(label, env, n_gen=5):
    for k, v in env.items():
        os.environ[k] = v
    for k in ["TURBO_2PASS_SPLIT", "GGML_CUDA_DISABLE_GRAPHS", "TURBO_RECENT_WINDOW"]:
        if k not in env:
            os.environ.pop(k, None)
    ctx = TurboContext(
        MODEL, n_ctx=2048, n_batch=512, cache_type="turbo4", n_gpu_layers=999
    )
    tokens = ctx.tokenize(prompt)
    t0 = time.time()
    ctx.decode(tokens, 0)
    out = []
    for i in range(n_gen):
        t = tokens[-1] if i == 0 else out[-1]
        logits = ctx.decode([t], len(tokens) + i)
        out.append(max(range(len(logits)), key=lambda j: logits[j]))
    elapsed = time.time() - t0
    text = ctx.detokenize(out)
    del ctx
    return out, text, elapsed


print("=" * 60, flush=True)
print("PRE-PRODUCTION VALIDATION", flush=True)
print("=" * 60, flush=True)

# Test 1: Baseline (graphs ON, no 2-PASS)
print("\n[Test 1] Baseline — CUDA Graphs ON, no 2-PASS", flush=True)
b_out, b_text, b_time = run("base", {}, 5)
print(f"  tokens: {b_out}", flush=True)
print(f"  text:   {b_text[:40]!r}", flush=True)
print(f"  time:   {b_time:.2f}s", flush=True)

# Test 2: 2-PASS with graphs OFF (known working)
print("\n[Test 2] 2-PASS — CUDA Graphs OFF", flush=True)
tp_off_out, tp_off_text, tp_off_time = run(
    "2pass_off",
    {
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "TURBO_2PASS_SPLIT": "1",
        "TURBO_RECENT_WINDOW": "64",
    },
    5,
)
match_off = "EXACT" if b_out == tp_off_out else "DIFF"
print(f"  tokens: {tp_off_out}", flush=True)
print(f"  vs base: {match_off}", flush=True)
print(f"  time:   {tp_off_time:.2f}s", flush=True)

# Test 3: 2-PASS with graphs ON (CRITICAL)
print("\n[Test 3] 2-PASS — CUDA Graphs ON (CRITICAL)", flush=True)
tp_on_out, tp_on_text, tp_on_time = run(
    "2pass_on", {"TURBO_2PASS_SPLIT": "1", "TURBO_RECENT_WINDOW": "64"}, 5
)
match_on = "EXACT" if b_out == tp_on_out else "DIFF"
print(f"  tokens: {tp_on_out}", flush=True)
print(f"  vs base: {match_on}", flush=True)
print(f"  time:   {tp_on_time:.2f}s", flush=True)

# Summary
print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Graphs OFF vs base: {match_off}", flush=True)
print(f"  Graphs ON  vs base: {match_on}", flush=True)
if match_off == "EXACT" and match_on == "EXACT":
    print(f"  RESULT: ALL PASS — production ready", flush=True)
elif match_off == "EXACT":
    print(f"  RESULT: Graphs ON diverges — investigate", flush=True)
else:
    print(f"  RESULT: Both fail — regression", flush=True)

# Perf sanity
print(
    f"\n  Perf: base={b_time:.2f}s 2pass_off={tp_off_time:.2f}s 2pass_on={tp_on_time:.2f}s",
    flush=True,
)
if tp_off_time > 0:
    slowdown_off = tp_off_time / b_time if b_time > 0 else 0
    print(f"  Slowdown (graphs OFF): {slowdown_off:.2f}x", flush=True)
if tp_on_time > 0:
    slowdown_on = tp_on_time / b_time if b_time > 0 else 0
    print(f"  Slowdown (graphs ON):  {slowdown_on:.2f}x", flush=True)

print("DONE", flush=True)
