"""
PROJECT-TURBO | Bit-width Sweep v0.5 — turbo2 vs turbo3 vs turbo4 vs q8_0

v0.5: Fresh bridge per test (no reuse). 4 configs × 4 tests = 16 bridges.
Focus: get reliable results, skip slow retention curve.

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

CONFIGS = [
    ("q8_0", 8, 8),
    ("q8_0/turbo2", 8, 43),
    ("q8_0/turbo3", 8, 41),
    ("q8_0/turbo4", 8, 42),
]

RESULTS = {}
N_CTX = 2048

MAX_TOKENS = {
    "throughput": 100,
    "ab_quality": 150,
    "key_repetition": 64,
    "reasoning": 512,
}


def _strip_thinking(text):
    text = re.sub(r"(?si)<think>.*?</think>", "", text)
    text = re.sub(r"(?si)<think>", "", text)
    text = re.sub(r"(?si)</think>", "", text)
    return text


def _extract_number(text):
    cleaned = _strip_thinking(text)
    numbers = re.findall(r"\b(\d+\.?\d*)\b", cleaned)
    if numbers:
        return float(numbers[-1])
    return None


def _make_bridge(name, type_k, type_v):
    bridge = TurboBridge()
    ret = bridge.init(MODEL, N_CTX, type_k, type_v, -1, 1, 1)
    if ret != 0:
        print(f"  [SKIP] {name}: init failed ({ret})")
        bridge.free()
        return None
    return bridge


def _gen(bridge, prompt, max_tokens):
    """Generate with fresh bridge context."""
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    if ret != 0:
        return ""

    pos = len(tokens)
    out = []
    for _ in range(max_tokens):
        logits = bridge.get_logits()
        best = max(range(len(logits)), key=lambda j: logits[j])
        text = bridge.token_to_piece(best)
        out.append(text)
        ret = bridge.decode([best], pos=pos)
        if ret != 0:
            break
        pos += 1
    return "".join(out)


def run_sweep():
    sink = os.environ.get("GGML_TURBO_SINK_TOKENS", "")
    if sink:
        print(f"Sink tokens: {sink}")

    print(f"Model: {MODEL}")
    print(f"Context: {N_CTX}")
    print(f"{'=' * 70}")
    print(f"BIT-WIDTH SWEEP v0.5")
    print(f"{'=' * 70}")

    # ─── Step 1: Throughput ───
    print("\n[Step 1] Throughput")
    print("-" * 60)

    perf_prompt = "Explain quantum computing in simple terms:"
    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue

        tokens = bridge.tokenize(perf_prompt)
        t0 = time.time()
        bridge.decode(tokens, pos=0)
        prefill_t = time.time() - t0

        pos = len(tokens)
        n = 0
        t1 = time.time()
        for _ in range(MAX_TOKENS["throughput"]):
            logits = bridge.get_logits()
            best = max(range(len(logits)), key=lambda j: logits[j])
            ret = bridge.decode([best], pos=pos)
            if ret != 0:
                break
            pos += 1
            n += 1
        decode_t = time.time() - t1
        decode_tps = n / decode_t if decode_t > 0 else 0
        prefill_tps = len(tokens) / prefill_t if prefill_t > 0 else 0

        print(
            f"  {name:>15}: prefill={prefill_t * 1000:.0f}ms ({prefill_tps:.0f}t/s), decode={decode_tps:.1f}t/s"
        )
        RESULTS[name] = {
            "prefill_ms": round(prefill_t * 1000, 1),
            "prefill_tps": round(prefill_tps, 0),
            "decode_tps": round(decode_tps, 1),
        }
        bridge.free()

    # ─── Step 2: Key Repetition ───
    print("\n[Step 2] Key Repetition")
    print("-" * 60)

    key_prompt = (
        "The key is 84729.\n" * 10
    ) + "What is the key? Answer with just the number."
    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        result = _gen(bridge, key_prompt, MAX_TOKENS["key_repetition"])
        ans = _extract_number(result)
        ok = ans is not None and abs(ans - 84729) < 1.0
        st = "PASS" if ok else "FAIL"
        print(f"  {name:>15}: [{st}] 84729 → {ans}")
        RESULTS[name]["key_repetition"] = {"correct": ok, "got": ans}
        bridge.free()

    # ─── Step 3: Math Reasoning ───
    print("\n[Step 3] Math Reasoning")
    print("-" * 60)

    math_prompt = (
        "A number is increased by 20%, then decreased by 20%, then increased by 50%. "
        "Final result is 108. What was the original number? "
        "Give ONLY the final numeric answer."
    )
    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        result = _gen(bridge, math_prompt, MAX_TOKENS["reasoning"])
        ans = _extract_number(result)
        ok = ans is not None and abs(ans - 100) < 1.0
        st = "PASS" if ok else "FAIL"
        preview = _strip_thinking(result).strip()[:60]
        print(f"  {name:>15}: [{st}] 100 → {ans}")
        if preview:
            print(f"                    '{preview}'")
        RESULTS[name]["reasoning"] = {"correct": ok, "got": ans}
        bridge.free()

    # ─── Step 4: A/B Quality ───
    print("\n[Step 4] A/B Quality")
    print("-" * 60)

    ab_prompt = "Explain the concept of recursion in programming with a simple example:"

    # Baseline
    bridge_base = _make_bridge("q8_0", 8, 8)
    baseline = _gen(bridge_base, ab_prompt, MAX_TOKENS["ab_quality"])
    baseline_clean = _strip_thinking(baseline)
    print(f"  q8_0 baseline: {len(baseline_clean)} chars")
    bridge_base.free()

    for name, tk, tv in CONFIGS:
        if name == "q8_0":
            continue
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        output = _gen(bridge, ab_prompt, MAX_TOKENS["ab_quality"])
        output_clean = _strip_thinking(output)

        sim = difflib.SequenceMatcher(None, baseline_clean, output_clean).ratio()
        bw = set(baseline_clean.lower().split())
        tw = set(output_clean.lower().split())
        overlap = len(bw & tw) / max(len(bw | tw), 1)

        print(f"  {name:>15}: seq={sim * 100:.1f}% word={overlap * 100:.1f}%")
        RESULTS[name]["ab_quality"] = {
            "seq_similarity": round(sim, 3),
            "word_overlap": round(overlap, 3),
        }
        bridge.free()

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 70}")

    header = f"{'Config':>15} | {'Pre ms':>7} | {'Toks/s':>7} | {'Math':>5} | {'Key':>5} | {'Seq%':>6} | {'Wrd%':>6}"
    print(header)
    print("-" * len(header))

    for name, tk, tv in CONFIGS:
        r = RESULTS.get(name, {})
        pm = r.get("prefill_ms", 0)
        dt = r.get("decode_tps", 0)
        m = "✅" if r.get("reasoning", {}).get("correct") else "❌"
        k = "✅" if r.get("key_repetition", {}).get("correct") else "❌"
        ss = r.get("ab_quality", {}).get("seq_similarity", 0)
        wo = r.get("ab_quality", {}).get("word_overlap", 0)

        if name == "q8_0":
            sp, wp = "base", "base"
        elif ss > 0:
            sp, wp = f"{ss * 100:.0f}", f"{wo * 100:.0f}"
        else:
            sp, wp = "-", "-"

        print(
            f"{name:>15} | {pm:>6.0f}ms | {dt:>6.1f} | {m:>5} | {k:>5} | {sp:>6} | {wp:>6}"
        )

    # ─── Save ───
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "test-results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mname = os.path.basename(MODEL).replace(".gguf", "")

    path = os.path.join(out_dir, f"bitwidth_sweep_{mname}_{ts}.json")
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nSaved: {path}")

    latest = os.path.join(out_dir, "bitwidth_sweep_latest.json")
    with open(latest, "w") as f:
        json.dump(RESULTS, f, indent=2)


if __name__ == "__main__":
    run_sweep()
