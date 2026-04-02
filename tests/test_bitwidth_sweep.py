"""
PROJECT-TURBO | Bit-width Sweep v0.6 — turbo2 vs turbo3 vs turbo4 vs q8_0

v0.6: Chat template + proper sampling (non-greedy).
      Throughput = raw greedy speed. Quality tests = C sampler chain.
      System prompt included for instruct model correctness.

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
    ("q8_0/q8_0", 8, 8),  # Baseline
    ("q8_0/turbo4", 8, 42),  # V=4-bit (safe)
    ("q8_0/turbo3", 8, 41),  # V=3-bit (aggressive)
    ("turbo4/turbo4", 42, 42),  # Full 4-bit
    ("turbo3/turbo3", 41, 41),  # Full 3-bit (most aggressive)
]

RESULTS = {}
N_CTX = 2048

MAX_TOKENS = {
    "throughput": 100,
    "ab_quality": 150,
    "key_repetition": 64,
    "reasoning": 1024,
}

# Qwen3 sampling presets (official)
# NOTE: temp=0.6 for thinking (more stable than Qwen3.5's 1.0)
THINKING_SAMPLING = {
    "temp": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "penalty_last_n": -1,
    "penalty_repeat": 1.0,
    "penalty_freq": 0.0,
    "penalty_present": 1.5,
}
NON_THINKING_SAMPLING = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "penalty_last_n": -1,
    "penalty_repeat": 1.0,
    "penalty_freq": 0.0,
    "penalty_present": 1.5,
}

# Model loaded once, shared across configs
_GLOBAL_MODEL_LOADED = False


def _ensure_model_loaded():
    """Load model once (shared across all configs)."""
    global _GLOBAL_MODEL_LOADED
    if _GLOBAL_MODEL_LOADED:
        return
    bridge = TurboBridge()
    ret = bridge.lib.turbo_load_model(MODEL.encode(), -1)
    if ret != 0:
        raise RuntimeError(f"turbo_load_model failed: {ret}")
    _GLOBAL_MODEL_LOADED = True


def _make_bridge(name, type_k, type_v):
    """Create a fresh context with specific KV cache types. Model is shared."""
    _ensure_model_loaded()
    bridge = TurboBridge()
    # Use handle-based API: ctx_init creates context from already-loaded model
    handle = bridge.ctx_init(N_CTX, type_k, type_v, 1, 1)
    if not handle:
        print(f"  [SKIP] {name}: ctx_init failed")
        return None
    bridge._handle = handle
    return bridge


def _strip_thinking(text):
    # Strip ChatML special tokens (character sequences, not special token IDs)
    im_start = chr(60) + "|im_start|"
    im_end = chr(60) + "|im_end|"
    # Strip <think>...</think> blocks
    think_start = chr(60) + "think" + chr(62)
    think_end = chr(60) + "/think" + chr(62)
    text = re.sub(
        r"(?si)" + re.escape(think_start) + r".*?" + re.escape(think_end), "", text
    )
    text = re.sub(r"(?si)" + re.escape(think_start), "", text)
    text = re.sub(r"(?si)" + re.escape(think_end), "", text)
    # Strip ChatML tags and role markers
    text = re.sub(r"(?si)" + re.escape(im_start) + r"[^<]*?", "", text)
    text = re.sub(r"(?si)" + re.escape(im_end), "", text)
    text = re.sub(r"(?m)^(system|user|assistant)\s*\n?", "", text)
    return text.strip()


def _extract_key_number(text):
    """Extract key number from text. Looks for first standalone number."""
    # First: look for 5-digit number (typical key format)
    match = re.search(r"(?<!\d)(\d{4,6})(?!\d)", text)
    if match:
        return float(match.group(1))
    # Fallback: any standalone number
    numbers = re.findall(r"(?<!\d)(\d+\.?\d*)(?!\d)", text)
    if numbers:
        return float(numbers[0])
    return None


def _extract_number(text):
    """Extract the final answer number from model output.

    Handles both:
    - Answer outside think block (non-thinking or completed thinking)
    - Answer inside think block (model finished reasoning but answer is in thinking)
    """
    # Try stripping think blocks first
    stripped = _strip_thinking(text)
    answer = _find_answer_number(stripped)
    if answer is not None:
        return answer
    # If no answer found outside think blocks, search inside them too
    return _find_answer_number(text)


def _find_answer_number(text):
    """Find the answer number using explicit patterns. Last standalone number fallback."""
    patterns = [
        # Strong: explicit "final answer" or "the answer is"
        r"(?:final answer|the answer is)[:\s]*(?:is|=|:)?\s*(\d+\.?\d*)",
        # Strong: "original number is X"
        r"(?:original number|sonuç)[:\s]*(?:is|=|:)?\s*(\d+\.?\d*)",
        # Strong: "x = 75"
        r"(?:^|\n)\s*x\s*=\s*(\d+\.?\d*)",
        # Strong: bold **75**
        r"\*\*(\d+\.?\d*)\*\*",
        # Strong: \boxed{75}
        r"\\boxed\{(\d+\.?\d*)\}",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))
    # Fallback: LAST standalone number (math problems have intermediate numbers first)
    numbers = re.findall(r"(?<!\d)(\d+\.?\d*)(?!\d)", text)
    if numbers:
        return float(numbers[-1])
    return None


def _format_prompt(bridge, user_content):
    """Apply chat template for proper model formatting. NEVER fallback silently."""
    return bridge.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_content},
        ],
        add_ass=True,
    )


def _gen_greedy(bridge, prompt, max_tokens):
    """Generate with greedy decoding (for throughput measurement only)."""
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


def _gen_sampled(bridge, prompt, max_tokens, sampling=None, seed=42):
    """Generate with proper C sampler chain (non-greedy)."""
    if sampling is None:
        sampling = THINKING_SAMPLING

    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    if ret != 0:
        return ""

    sampler = bridge.sampler_init(
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
        min_p=sampling["min_p"],
        temp=sampling["temp"],
        seed=seed,
        penalty_last_n=sampling.get("penalty_last_n", 0),
        penalty_repeat=sampling.get("penalty_repeat", 1.0),
        penalty_freq=sampling.get("penalty_freq", 0.0),
        penalty_present=sampling.get("penalty_present", 0.0),
    )

    pos = len(tokens)
    out = []
    for _ in range(max_tokens):
        token = bridge.sampler_sample(sampler)
        text = bridge.token_to_piece(token)
        out.append(text)
        ret = bridge.decode([token], pos=pos)
        if ret != 0:
            break
        pos += 1

    bridge.sampler_free(sampler)
    return "".join(out)


def run_sweep():
    sink = os.environ.get("GGML_TURBO_SINK_TOKENS", "")
    if sink:
        print(f"Sink tokens: {sink}")

    print(f"Model: {MODEL}")
    print(f"Context: {N_CTX}")
    print(f"{'=' * 70}")
    print(f"BIT-WIDTH SWEEP v0.7 (sampler order fix + stable sampling)")
    print(f"{'=' * 70}")

    # ─── Step 0: Ground Truth (q8_0/q8_0 baseline, zero ambiguity) ───
    print("\n[Step 0] Ground Truth Runner (q8_0/q8_0, zero ambiguity)")
    print("-" * 60)

    gt_bridge = _make_bridge("ground_truth", 8, 8)
    if gt_bridge is not None:
        # Ground truth math
        gt_math_raw = (
            "Solve this step by step. Show your work, then give the final answer.\n\n"
            "A number is increased by 20%, then decreased by 20%, then increased by 50%.\n"
            "Final result is 108.\n\n"
            "What was the original number?"
        )
        gt_math_prompt = _format_prompt(gt_bridge, gt_math_raw)
        gt_math = _gen_sampled(
            gt_bridge,
            gt_math_prompt,
            MAX_TOKENS["reasoning"],
            sampling=THINKING_SAMPLING,
            seed=42,
        )
        gt_math_ans = _extract_number(gt_math)
        gt_math_ok = gt_math_ans is not None and abs(gt_math_ans - 75) < 1.0
        print(
            f"  Math:   expected=75, got={gt_math_ans} -> {'PASS' if gt_math_ok else 'FAIL'}"
        )
        print(f"  Output: '{_strip_thinking(gt_math).strip()[:100]}'")

        # Ground truth key
        gt_key_raw = (
            "The key is 84729.\n" * 10
        ) + "What is the key? Answer with just the number."
        gt_key_prompt = _format_prompt(gt_bridge, gt_key_raw)
        gt_key = _gen_sampled(
            gt_bridge,
            gt_key_prompt,
            MAX_TOKENS["key_repetition"],
            sampling=NON_THINKING_SAMPLING,
            seed=42,
        )
        gt_key_ans = _extract_key_number(_strip_thinking(gt_key))
        gt_key_ok = gt_key_ans is not None and abs(gt_key_ans - 84729) < 1.0
        print(
            f"  Key:    expected=84729, got={gt_key_ans} -> {'PASS' if gt_key_ok else 'FAIL'}"
        )

        RESULTS["ground_truth"] = {
            "math": {
                "correct": gt_math_ok,
                "got": gt_math_ans,
                "output": _strip_thinking(gt_math).strip()[:200],
            },
            "key": {"correct": gt_key_ok, "got": gt_key_ans},
        }
        gt_bridge.free()
    else:
        print("  [SKIP] ground truth bridge init failed")

    # ─── Step 1: Throughput (greedy + sampling, both measured) ───
    print("\n[Step 1] Throughput (greedy + sampling)")
    print("-" * 60)

    first_bridge = _make_bridge("ref", 8, 8)
    perf_prompt_raw = "Explain quantum computing in simple terms:"
    if first_bridge:
        perf_prompt = _format_prompt(first_bridge, perf_prompt_raw)
        first_bridge.free()
    else:
        perf_prompt = perf_prompt_raw

    # Step 1a: Greedy throughput (raw speed)
    print("\n  1a. Greedy (raw speed):")
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
            f"    {name:>15}: prefill={prefill_t * 1000:.0f}ms ({prefill_tps:.0f}t/s), decode={decode_tps:.1f}t/s"
        )
        RESULTS[name] = {
            "prefill_ms": round(prefill_t * 1000, 1),
            "prefill_tps": round(prefill_tps, 0),
            "decode_tps": round(decode_tps, 1),
        }
        bridge.free()

    # Step 1b: Sampling throughput (real-world speed)
    print("\n  1b. Sampling (real-world):")
    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue

        tokens = bridge.tokenize(perf_prompt)
        t0 = time.time()
        bridge.decode(tokens, pos=0)
        prefill_t = time.time() - t0

        sampler = bridge.sampler_init(
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            temp=0.6,
            seed=42,
            penalty_last_n=-1,
            penalty_repeat=1.0,
            penalty_freq=0.0,
            penalty_present=1.5,
        )
        pos = len(tokens)
        n = 0
        t1 = time.time()
        for _ in range(MAX_TOKENS["throughput"]):
            token = bridge.sampler_sample(sampler)
            if token < 0:
                break
            ret = bridge.decode([token], pos=pos)
            if ret != 0:
                break
            pos += 1
            n += 1
        decode_t = time.time() - t1
        decode_tps = n / decode_t if decode_t > 0 else 0

        bridge.sampler_free(sampler)
        print(
            f"    {name:>15}: prefill={prefill_t * 1000:.0f}ms, decode={decode_tps:.1f}t/s (sampling)"
        )
        RESULTS[name]["decode_tps_sampling"] = round(decode_tps, 1)
        bridge.free()

    # ─── Step 2: Key Repetition (non-thinking mode) ───
    print("\n[Step 2] Key Repetition (non-thinking sampling)")
    print("-" * 60)

    key_raw = (
        "The key is 84729.\n" * 10
    ) + "What is the key? Answer with just the number."

    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        key_prompt = _format_prompt(bridge, key_raw)
        result = _gen_sampled(
            bridge,
            key_prompt,
            MAX_TOKENS["key_repetition"],
            sampling=NON_THINKING_SAMPLING,
        )
        ans = _extract_key_number(_strip_thinking(result))
        ok = ans is not None and abs(ans - 84729) < 1.0
        st = "PASS" if ok else "FAIL"
        print(f"  {name:>15}: [{st}] 84729 -> {ans}")
        RESULTS[name]["key_repetition"] = {"correct": ok, "got": ans}
        bridge.free()

    # ─── Step 3: Math Reasoning (thinking mode) ───
    print("\n[Step 3] Math Reasoning (thinking sampling)")
    print("-" * 60)

    math_raw = (
        "Solve this step by step. Show your work, then give the final answer.\n\n"
        "A number is increased by 20%, then decreased by 20%, then increased by 50%.\n"
        "Final result is 108.\n\n"
        "What was the original number?"
    )

    for name, tk, tv in CONFIGS:
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        math_prompt = _format_prompt(bridge, math_raw)
        result = _gen_sampled(
            bridge,
            math_prompt,
            MAX_TOKENS["reasoning"],
            sampling=THINKING_SAMPLING,
        )
        ans = _extract_number(result)
        ok = ans is not None and abs(ans - 75) < 1.0
        st = "PASS" if ok else "FAIL"
        preview = _strip_thinking(result).strip()[:60]
        print(f"  {name:>15}: [{st}] 75 -> {ans}")
        if preview:
            print(f"                    '{preview}'")
        RESULTS[name]["reasoning"] = {"correct": ok, "got": ans}
        bridge.free()

    # ─── Step 4: A/B Quality (non-thinking, fixed seed) ───
    print("\n[Step 4] A/B Quality (non-thinking, fixed seed)")
    print("-" * 60)

    ab_raw = "Explain the concept of recursion in programming with a simple example:"

    # Baseline
    bridge_base = _make_bridge("q8_0", 8, 8)
    ab_prompt_base = _format_prompt(bridge_base, ab_raw)
    baseline = _gen_sampled(
        bridge_base,
        ab_prompt_base,
        MAX_TOKENS["ab_quality"],
        sampling=NON_THINKING_SAMPLING,
        seed=42,
    )
    baseline_clean = _strip_thinking(baseline)
    print(f"  q8_0 baseline: {len(baseline_clean)} chars")
    bridge_base.free()

    for name, tk, tv in CONFIGS:
        if name == "q8_0/q8_0":
            continue
        bridge = _make_bridge(name, tk, tv)
        if bridge is None:
            continue
        ab_prompt = _format_prompt(bridge, ab_raw)
        output = _gen_sampled(
            bridge,
            ab_prompt,
            MAX_TOKENS["ab_quality"],
            sampling=NON_THINKING_SAMPLING,
            seed=42,
        )
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
    print(f"\n{'=' * 80}")
    print("SONUC OZETI")
    print(f"{'=' * 80}")

    header = (
        f"{'Config':>15} | {'Pre ms':>7} | {'Greedy':>7} | {'Sample':>7} |"
        f" {'Math':>5} | {'Key':>5} | {'Seq%':>6} | {'Wrd%':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, tk, tv in CONFIGS:
        r = RESULTS.get(name, {})
        pm = r.get("prefill_ms", 0)
        dt = r.get("decode_tps", 0)
        ds = r.get("decode_tps_sampling", 0)
        m_ok = r.get("reasoning", {}).get("correct")
        k_ok = r.get("key_repetition", {}).get("correct")
        m = "OK" if m_ok else "FAIL"
        k = "OK" if k_ok else "FAIL"
        ss = r.get("ab_quality", {}).get("seq_similarity", 0)
        wo = r.get("ab_quality", {}).get("word_overlap", 0)

        if name == "q8_0/q8_0":
            sp, wp = "base", "base"
        elif ss > 0:
            sp, wp = f"{ss * 100:.0f}", f"{wo * 100:.0f}"
        else:
            sp, wp = "-", "-"

        print(
            f"{name:>15} | {pm:>6.0f}ms | {dt:>6.1f} | {ds:>6.1f} | {m:>5} | {k:>5} | {sp:>6} | {wp:>6}"
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
