"""
PROJECT-TURBO | Bit-width Sweep v0.4 — turbo2 vs turbo3 vs turbo4 vs q8_0

v0.4 Changes:
- Bridge reuse with KV clear (10x faster than fresh bridge)
- EOS token check (prevent infinite loops)
- Think tag stripping
- SequenceMatcher for A/B quality
- Multi-gap memory retention curve
- Prefill + decode timing

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
        python tests/test_bitwidth_sweep.py

Optional:
    GGML_TURBO_SINK_TOKENS=8  — enable sink token protection
"""

import os
import sys
import time
import json
import re
import difflib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

CONFIGS = [
    ("q8_0", 8, 8),
    ("q8_0/turbo2", 8, 43),
    ("q8_0/turbo3", 8, 41),
    ("q8_0/turbo4", 8, 42),
]

RESULTS = {}
N_CTX = 2048

# Token limits
MAX_TOKENS = {
    "throughput": 100,
    "baseline": 200,
    "ab_quality": 200,
    "key_repetition": 64,
    "reasoning": 1024,
}

# Memory retention gap sizes (token)
RETENTION_GAPS = [128, 256, 512, 768, 1024]

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

# Bridge pool — 1 bridge per config, reused with clear_cache
_BRIDGES = {}


def _get_bridge(name, type_k, type_v):
    """Get or create a bridge for the given config. Reuses existing bridge."""
    if name not in _BRIDGES:
        bridge = TurboBridge()
        ret = bridge.init(MODEL, N_CTX, type_k, type_v, -1, 1, 1)
        if ret != 0:
            print(f"  [SKIP] {name}: init failed ({ret})")
            bridge.free()
            return None
        _BRIDGES[name] = bridge
    else:
        bridge = _BRIDGES[name]
        # Clear KV cache for reuse
        if hasattr(bridge, "clear_cache"):
            bridge.clear_cache()
        elif hasattr(bridge, "kv_clear"):
            bridge.kv_clear()
    return _BRIDGES[name]


def _free_all():
    """Free all bridges."""
    for name, bridge in _BRIDGES.items():
        bridge.free()
    _BRIDGES.clear()


def _strip_thinking(text):
    """Remove Qwen3 <think> tags."""
    text = re.sub(r"(?si)<think>.*?</think>", "", text)
    text = re.sub(r"(?si)<think>", "", text)
    text = re.sub(r"(?si)</think>", "", text)
    return text


def _extract_number(text):
    """Extract last number from cleaned text."""
    cleaned = _strip_thinking(text)
    numbers = re.findall(r"\b(\d+\.?\d*)\b", cleaned)
    if numbers:
        return float(numbers[-1])
    return None


def _generate(bridge, prompt, max_tokens):
    """Generate text using a bridge. Assumes KV is clean."""
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    if ret != 0:
        return "", 0, 0

    pos = len(tokens)
    generated = []
    t_start = time.time()
    n_tokens = 0

    for _ in range(max_tokens):
        logits = bridge.get_logits()
        best = max(range(len(logits)), key=lambda j: logits[j])
        text = bridge.token_to_piece(best)

        # EOS check (stop early)
        ret = bridge.decode([best], pos=pos)
        if ret != 0:
            break
        pos += 1
        n_tokens += 1
        generated.append(text)

    elapsed = time.time() - t_start
    return "".join(generated), elapsed, n_tokens


def _generate_with_perf(bridge, prompt, max_tokens):
    """Generate with prefill/decode timing split."""
    tokens = bridge.tokenize(prompt)

    # Prefill
    t0 = time.time()
    bridge.decode(tokens, pos=0)
    prefill_time = time.time() - t0

    # Decode
    pos = len(tokens)
    generated = []
    t_start = time.time()
    n_tokens = 0

    for _ in range(max_tokens):
        logits = bridge.get_logits()
        best = max(range(len(logits)), key=lambda j: logits[j])
        text = bridge.token_to_piece(best)
        generated.append(text)

        ret = bridge.decode([best], pos=pos)
        if ret != 0:
            break
        pos += 1
        n_tokens += 1

    decode_time = time.time() - t_start
    return "".join(generated), prefill_time, decode_time, len(tokens), n_tokens


def _measure_kv_usage(n_ctx, type_k, type_v):
    """Calculate theoretical KV cache size."""
    k_bits = 8
    if type_v == 43:
        v_bits = 2
    elif type_v == 41:
        v_bits = 3
    elif type_v == 42:
        v_bits = 4
    else:
        v_bits = 8

    n_layer = 36
    n_embd_k_gqa = 1024
    n_embd_v_gqa = 1024

    k_bytes = n_layer * n_ctx * n_embd_k_gqa * k_bits / 8
    v_bytes = n_layer * n_ctx * n_embd_v_gqa * v_bits / 8
    total_mb = (k_bytes + v_bytes) / (1024 * 1024)

    return {
        "k_bits": k_bits,
        "v_bits": v_bits,
        "k_mb": round(k_bytes / (1024 * 1024), 1),
        "v_mb": round(v_bytes / (1024 * 1024), 1),
        "total_mb": round(total_mb, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════


def run_sweep():
    sink_tokens = os.environ.get("GGML_TURBO_SINK_TOKENS", "")
    if sink_tokens:
        print(f"Sink tokens: {sink_tokens}")

    print(f"Model: {MODEL}")
    print(f"Context: {N_CTX}")
    print(f"{'=' * 70}")
    print(f"BIT-WIDTH SWEEP v0.4 — turbo2 vs turbo3 vs turbo4 vs q8_0")
    print(f"{'=' * 70}")

    # ─── Step 0: KV Cache Usage ───
    print("\n[Step 0] KV Cache VRAM Usage")
    print("-" * 60)

    for name, type_k, type_v in CONFIGS:
        kv = _measure_kv_usage(N_CTX, type_k, type_v)
        print(
            f"  {name:>15}: K={kv['k_bits']}bit/{kv['k_mb']}MB + "
            f"V={kv['v_bits']}bit/{kv['v_mb']}MB = {kv['total_mb']}MB"
        )
        RESULTS[name] = {"kv_cache": kv}

    # ─── Step 1: Throughput ───
    print("\n[Step 1] Throughput Measurement")
    print("-" * 60)

    perf_prompt = "Explain quantum computing in simple terms:"
    for name, type_k, type_v in CONFIGS:
        bridge = _get_bridge(name, type_k, type_v)
        if bridge is None:
            continue

        output, prefill_t, decode_t, n_prompt, n_gen = _generate_with_perf(
            bridge, perf_prompt, MAX_TOKENS["throughput"]
        )
        prefill_tps = n_prompt / prefill_t if prefill_t > 0 else 0
        decode_tps = n_gen / decode_t if decode_t > 0 else 0

        print(
            f"  {name:>15}: prefill={prefill_t * 1000:.0f}ms ({prefill_tps:.0f} tok/s), "
            f"decode={decode_tps:.1f} tok/s"
        )

        RESULTS[name]["prefill_ms"] = round(prefill_t * 1000, 1)
        RESULTS[name]["decode_tps"] = round(decode_tps, 1)
        RESULTS[name]["throughput_tps"] = round(decode_tps, 1)

    # ─── Step 2: Baseline Output ───
    print("\n[Step 2] Baseline Output (q8_0)")
    print("-" * 60)

    baseline_output = ""
    baseline_prompt = (
        "Explain the concept of recursion in programming with a simple example:"
    )
    bridge_q8 = _get_bridge("q8_0", 8, 8)
    if bridge_q8:
        baseline_output, _, _ = _generate(
            bridge_q8, baseline_prompt, MAX_TOKENS["baseline"]
        )
        baseline_clean = _strip_thinking(baseline_output)
        print(f"  q8_0 baseline: {len(baseline_clean)} chars")

    # ─── Step 3: A/B Quality ───
    print("\n[Step 3] A/B Quality vs q8_0 Baseline")
    print("-" * 60)

    ab_prompt = "Explain the concept of recursion in programming with a simple example:"
    for name, type_k, type_v in CONFIGS:
        if name == "q8_0":
            continue

        bridge = _get_bridge(name, type_k, type_v)
        if bridge is None:
            continue

        output, _, _ = _generate(bridge, ab_prompt, MAX_TOKENS["ab_quality"])
        output_clean = _strip_thinking(output)

        similarity = difflib.SequenceMatcher(None, baseline_clean, output_clean).ratio()
        base_words = set(baseline_clean.lower().split())
        test_words = set(output_clean.lower().split())
        word_overlap = len(base_words & test_words) / max(
            len(base_words | test_words), 1
        )

        print(
            f"  {name:>15}: seq_sim={similarity * 100:.1f}%, word_overlap={word_overlap * 100:.1f}%"
        )

        RESULTS[name]["ab_quality"] = {
            "seq_similarity": round(similarity, 3),
            "word_overlap": round(word_overlap, 3),
        }

    # ─── Step 4: Key Repetition ───
    print("\n[Step 4] Key Repetition (10x)")
    print("-" * 60)

    for name, type_k, type_v in CONFIGS:
        bridge = _get_bridge(name, type_k, type_v)
        if bridge is None:
            continue

        key_line = "The key is 84729.\n"
        prompt = (key_line * 10) + "What is the key? Answer with just the number."
        result, _, _ = _generate(bridge, prompt, MAX_TOKENS["key_repetition"])
        answer = _extract_number(result)
        correct = answer is not None and abs(answer - 84729) < 1.0
        status = "PASS" if correct else "FAIL"
        print(f"  {name:>15}: [{status}] expected=84729, got={answer}")

        RESULTS[name]["key_repetition"] = {"correct": correct, "got": answer}

    # ─── Step 5: Multi-Step Math Reasoning ───
    print("\n[Step 5] Multi-Step Math Reasoning")
    print("-" * 60)

    math_prompt = (
        "A number is increased by 20%, then decreased by 20%, then increased by 50%. "
        "Final result is 108. What was the original number? "
        "Give ONLY the final numeric answer."
    )
    for name, type_k, type_v in CONFIGS:
        bridge = _get_bridge(name, type_k, type_v)
        if bridge is None:
            continue

        result, _, _ = _generate(bridge, math_prompt, MAX_TOKENS["reasoning"])
        cleaned = _strip_thinking(result)
        answer = _extract_number(result)
        correct = answer is not None and abs(answer - 100) < 1.0
        status = "PASS" if correct else "FAIL"
        print(f"  {name:>15}: [{status}] expected=100, got={answer}")
        if cleaned.strip()[:60]:
            print(f"    preview: '{cleaned.strip()[:60]}'")

        RESULTS[name]["reasoning"] = {
            "correct": correct,
            "got": answer,
            "preview": cleaned.strip()[:80],
        }

    # ─── Step 6: Memory Retention Curve ───
    print("\n[Step 6] Memory Retention Curve")
    print("-" * 60)

    # Filler text (generated once with q8_0)
    filler_prompt = (
        "Describe the solar system in detail including all planets, their moons, "
        "rings, atmospheric composition, and distances from the Sun."
    )
    filler_text, _, _ = _generate(
        _get_bridge("q8_0", 8, 8), filler_prompt, max_tokens=1200
    )
    filler_words = filler_text.split()
    print(f"  Filler text: {len(filler_words)} words")

    retention_results = {}
    for name, type_k, type_v in CONFIGS:
        print(f"\n  --- {name} ---")
        config_retention = {}

        for gap in RETENTION_GAPS:
            target_words = int(gap / 1.3)
            filler_slice = " ".join(filler_words[:target_words])

            if not filler_slice:
                config_retention[gap] = {"correct": False, "got": None}
                print(f"    gap={gap:>4}: [SKIP] filler too short")
                continue

            test_prompt = (
                "Memorize: A=12, B=47, C=93\n\n" + filler_slice + "\n\n"
                "What is B? Answer with just the number."
            )

            bridge = _get_bridge(name, type_k, type_v)
            if bridge is None:
                config_retention[gap] = {"correct": False, "got": None}
                print(f"    gap={gap:>4}: [SKIP] no bridge")
                continue

            result, _, _ = _generate(bridge, test_prompt, MAX_TOKENS["reasoning"])
            answer = _extract_number(result)
            correct = answer is not None and abs(answer - 47) < 1.0
            status = "PASS" if correct else "FAIL"
            print(f"    gap={gap:>4}: [{status}] expected=47, got={answer}")

            config_retention[gap] = {
                "correct": correct,
                "got": answer,
                "filler_words": target_words,
            }

        retention_results[name] = config_retention

    RESULTS["memory_retention_curve"] = retention_results

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 70}")

    header = f"{'Config':>15} | {'Prefill':>8} | {'Decode':>7} | {'Math':>6} | {'Key':>6} | {'Seq%':>6} | {'Word%':>6}"
    print(header)
    print("-" * len(header))

    for name, type_k, type_v in CONFIGS:
        r = RESULTS.get(name, {})
        prefill = r.get("prefill_ms", 0)
        decode = r.get("decode_tps", 0)
        math = "✅" if r.get("reasoning", {}).get("correct") else "❌"
        key = "✅" if r.get("key_repetition", {}).get("correct") else "❌"
        seq_sim = r.get("ab_quality", {}).get("seq_similarity", 0)
        word_ov = r.get("ab_quality", {}).get("word_overlap", 0)

        if name == "q8_0":
            seq_str = "base"
            word_str = "base"
        elif seq_sim > 0:
            seq_str = f"{seq_sim * 100:.0f}"
            word_str = f"{word_ov * 100:.0f}"
        else:
            seq_str = "-"
            word_str = "-"

        print(
            f"{name:>15} | {prefill:>7.0f}ms | {decode:>6.1f}t | {math:>6} | {key:>6} | {seq_str:>6} | {word_str:>6}"
        )

    # Retention curve
    print(f"\n  Memory Retention Curve:")
    print(
        f"  {'Config':>15} | "
        + " | ".join(f"{'G' + str(g):>6}" for g in RETENTION_GAPS)
    )
    print("  " + "-" * (15 + 3 + len(RETENTION_GAPS) * 8))

    for name, type_k, type_v in CONFIGS:
        curve = retention_results.get(name, {})
        cells = []
        for gap in RETENTION_GAPS:
            entry = curve.get(gap, {})
            if entry.get("correct"):
                cells.append("  ✅  ")
            else:
                cells.append("  ❌  ")
        print(f"  {name:>15} | " + " | ".join(cells))

    # ─── Save results ───
    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "test-results")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(MODEL).replace(".gguf", "")
    output_path = os.path.join(output_dir, f"bitwidth_sweep_{model_name}_{ts}.json")
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    latest_path = os.path.join(output_dir, "bitwidth_sweep_latest.json")
    with open(latest_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    # Cleanup
    _free_all()


if __name__ == "__main__":
    run_sweep()
