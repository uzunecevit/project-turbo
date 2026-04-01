"""
PROJECT-TURBO | Latency Breakdown Test Suite — C-Level Performance
C-level perf metrics: prefill, decode, per-token breakdown, KV state.

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_latency.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge, TurboPerfData, TurboKVState

RESULTS = {}


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Prefill latency breakdown
# ═══════════════════════════════════════════════════════════════════════


def test_prefill_latency():
    """C-level prefill timing at various prompt lengths."""
    print("\n[TEST 1] Prefill Latency (C-level)")
    print("-" * 60)

    bridge = TurboBridge()
    bridge.load_model(MODEL, -1)
    handle = bridge.ctx_init(4096, 8, 41, 1, 1)

    results = []
    for n_words in [50, 100, 200, 500]:
        prompt = " ".join(["word"] * n_words)
        tokens = bridge.ctx_tokenize(prompt)

        bridge.ctx_kv_cache_clear()
        bridge.perf_reset()

        t0 = time.time()
        bridge.ctx_decode(tokens, pos=0)
        wall_time = time.time() - t0

        perf = bridge.perf_get()

        c_tps = perf.n_p_eval / (perf.t_p_eval_ms / 1000) if perf.t_p_eval_ms > 0 else 0
        wall_tps = len(tokens) / wall_time if wall_time > 0 else 0

        results.append(
            {
                "tokens": len(tokens),
                "c_ms": perf.t_p_eval_ms,
                "wall_ms": wall_time * 1000,
                "c_tok_s": c_tps,
                "wall_tok_s": wall_tps,
            }
        )

        print(
            f"  {len(tokens):>4} tok: Wall={wall_time * 1000:.1f}ms ({wall_tps:.0f} tok/s)"
            f"{f' | C={perf.t_p_eval_ms:.1f}ms ({c_tps:.0f} tok/s)' if perf.t_p_eval_ms > 0 else ''}"
        )

    bridge.ctx_free()
    bridge.unload_model()

    RESULTS["prefill"] = results
    print(f"  [INFO] C-level prefill metrics above (C timing is authoritative)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Decode latency per token
# ═══════════════════════════════════════════════════════════════════════


def test_decode_per_token():
    """C-level decode timing: per-token latency over 100 tokens."""
    print("\n[TEST 2] Decode Per-Token Latency (C-level)")
    print("-" * 60)

    bridge = TurboBridge()
    bridge.load_model(MODEL, -1)
    handle = bridge.ctx_init(4096, 8, 41, 1, 1)

    prompt = "Write a story about a robot who learns to paint:"
    tokens = bridge.ctx_tokenize(prompt)

    # Prefill
    bridge.perf_reset()
    bridge.ctx_decode(tokens, pos=0)
    prefill_perf = bridge.perf_get()
    print(
        f"  Prefill: {prefill_perf.t_p_eval_ms:.1f}ms / {prefill_perf.n_p_eval} tokens"
    )

    # Decode 100 tokens
    pos = len(tokens)
    bridge.perf_reset()

    t0 = time.time()
    for i in range(100):
        logits = bridge.ctx_get_logits()
        nv = bridge.ctx_n_vocab()
        best = max(range(nv), key=lambda j: logits[j])
        bridge.ctx_decode([best], pos=pos)
        pos += 1
    wall_time = time.time() - t0

    decode_perf = bridge.perf_get()

    c_per_token = decode_perf.t_eval_ms / max(decode_perf.n_eval, 1)
    wall_per_token = (wall_time * 1000) / 100
    c_tps = (
        decode_perf.n_eval / (decode_perf.t_eval_ms / 1000)
        if decode_perf.t_eval_ms > 0
        else 0
    )

    print(f"  Decode: C={decode_perf.t_eval_ms:.1f}ms / {decode_perf.n_eval} tokens")
    print(f"  C per-token: {c_per_token:.2f}ms")
    print(f"  Wall per-token: {wall_per_token:.2f}ms")
    print(f"  C throughput: {c_tps:.1f} tok/s")
    print(f"  Wall throughput: {100 / wall_time:.1f} tok/s")

    # KV state at end
    kv = bridge.kv_state()
    print(
        f"  KV state: {kv.n_pos}/{kv.n_ctx} ({kv.utilization:.1%}), "
        f"state={kv.state_bytes / 1024:.1f}KB"
    )

    bridge.ctx_free()
    bridge.unload_model()

    RESULTS["decode"] = {
        "c_per_token_ms": c_per_token,
        "wall_per_token_ms": wall_per_token,
        "c_tok_s": c_tps,
        "kv_utilization": kv.utilization,
    }


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: First token latency
# ═══════════════════════════════════════════════════════════════════════


def test_first_token_latency():
    """Time from init to first decoded token (TTFT)."""
    print("\n[TEST 3] First Token Latency (TTFT)")
    print("-" * 60)

    for prompt_tokens in [10, 50, 100, 200]:
        bridge = TurboBridge()
        bridge.load_model(MODEL, -1)
        handle = bridge.ctx_init(4096, 8, 41, 1, 1)

        prompt = " ".join(["hello"] * prompt_tokens)
        tokens = bridge.ctx_tokenize(prompt)

        # Full prefill + first decode
        t0 = time.time()
        bridge.ctx_decode(tokens, pos=0)
        logits = bridge.ctx_get_logits()
        nv = bridge.ctx_n_vocab()
        best = max(range(nv), key=lambda j: logits[j])
        bridge.ctx_decode([best], pos=len(tokens))
        ttft = time.time() - t0

        perf = bridge.perf_get()
        print(
            f"  {len(tokens):>3} tok prompt: TTFT={ttft * 1000:.1f}ms "
            f"(C prefill={perf.t_p_eval_ms:.1f}ms)"
        )

        bridge.ctx_free()
        bridge.unload_model()

    RESULTS["ttft"] = {"note": "see output above"}


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: KV state progression during generation
# ═══════════════════════════════════════════════════════════════════════


def test_kv_progression():
    """Track KV utilization as tokens are generated."""
    print("\n[TEST 4] KV State Progression")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    prompt = "Once upon a time"
    tokens = ctx.tokenize(prompt)

    # Prefill
    logits = ctx.decode(tokens)
    pos = len(tokens)

    # Generate and track KV state every 50 tokens
    print(f"  {'Token':>6} | {'KV Pos':>6} | {'Util%':>6} | {'State KB':>8}")
    print(f"  {'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 8}")

    kv = ctx.bridge.kv_state()
    print(
        f"  {'prefill':>6} | {kv.n_pos:>6} | {kv.utilization:>5.1%} | {kv.state_bytes / 1024:>7.1f}"
    )

    for i in range(200):
        best = max(range(len(logits)), key=lambda j: logits[j])
        ret = ctx.bridge.ctx_decode([best], pos=pos)
        if ret != 0:
            break
        logits = ctx.bridge.ctx_get_logits()
        pos += 1

        if (i + 1) % 50 == 0:
            kv = ctx.bridge.kv_state()
            print(
                f"  {i + 1:>6} | {kv.n_pos:>6} | {kv.utilization:>5.1%} | {kv.state_bytes / 1024:>7.1f}"
            )

    ctx.bridge.free()

    RESULTS["kv_progression"] = {"note": "see output above"}


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: C vs Python timing comparison
# ═══════════════════════════════════════════════════════════════════════


def test_c_vs_python_timing():
    """Compare C-level perf with Python wall clock to verify accuracy."""
    print("\n[TEST 5] C vs Python Timing Comparison")
    print("-" * 60)

    bridge = TurboBridge()
    bridge.load_model(MODEL, -1)
    handle = bridge.ctx_init(4096, 8, 41, 1, 1)

    prompt = "The meaning of life is"
    tokens = bridge.ctx_tokenize(prompt)

    # Prefill timing comparison
    bridge.perf_reset()
    t0 = time.time()
    bridge.ctx_decode(tokens, pos=0)
    wall_prefill = (time.time() - t0) * 1000
    perf = bridge.perf_get()

    # Decode timing comparison
    pos = len(tokens)
    bridge.perf_reset()
    t0 = time.time()
    for i in range(50):
        logits = bridge.ctx_get_logits()
        nv = bridge.ctx_n_vocab()
        best = max(range(nv), key=lambda j: logits[j])
        bridge.ctx_decode([best], pos=pos)
        pos += 1
    wall_decode = (time.time() - t0) * 1000
    perf = bridge.perf_get()

    print(
        f"  Prefill: C={perf.t_p_eval_ms:.1f}ms vs Python={wall_prefill:.1f}ms "
        f"(ratio={perf.t_p_eval_ms / wall_prefill:.2f}x)"
    )
    print(
        f"  Decode:  C={perf.t_eval_ms:.1f}ms vs Python={wall_decode:.1f}ms "
        f"(ratio={perf.t_eval_ms / wall_decode:.2f}x)"
    )

    bridge.ctx_free()
    bridge.unload_model()

    RESULTS["c_vs_python"] = {
        "prefill_ratio": perf.t_p_eval_ms / wall_prefill if wall_prefill > 0 else 0,
        "decode_ratio": perf.t_eval_ms / wall_decode if wall_decode > 0 else 0,
    }
    print(f"  [INFO] C timing is the authoritative metric")


# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    print(f"LATENCY BREAKDOWN TEST SUITE — C-Level Performance")
    print(f"{'=' * 60}")

    test_prefill_latency()
    test_decode_per_token()
    test_first_token_latency()
    test_kv_progression()
    test_c_vs_python_timing()

    print(f"\n{'=' * 60}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 60}")

    if "decode" in RESULTS:
        d = RESULTS["decode"]
        print(f"  Decode per-token (C): {d['c_per_token_ms']:.2f}ms")
        print(f"  Decode throughput (C): {d['c_tok_s']:.1f} tok/s")
        print(f"  KV utilization: {d['kv_utilization']:.1%}")
    print(f"  [INFO] All latency metrics use C-level perf (authoritative)")
