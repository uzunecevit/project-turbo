"""
PROJECT-TURBO | Reasoning Chain Test Suite — Error Amplification
Küçük KV hatalarını büyüten testler: multi-step math, memory retention,
contradiction traps.

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_reasoning.py
"""

import re
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge

RESULTS = {}


def _normalize_output(text):
    # Remove Think tags
    text = re.sub(r"(?si)<think>(.*?)</think>", r"\1", text)
    text = re.sub(r"(?si)<think>", "", text)
    text = re.sub(r"(?si)</think>", "", text)
    # Remove chat artifacts
    text = re.sub(r"(?im)^(user|assistant|system)\s*$", "", text)
    text = re.sub(r"(?i)(user|assistant)\s*$", "", text)
    # Strip LaTeX
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\[()]", "", text)
    # Remove output tags
    text = re.sub(r"(?si)<output>(.*?)</output>", r"\1", text)
    text = re.sub(r"(?si)<output>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_number(text):
    cleaned = _normalize_output(text)
    numbers = re.findall(r"\b\d+\.?\d*\b", cleaned)
    if numbers:
        return float(numbers[-1])
    glued = re.findall(r"(\d+\.?\d*)[a-zA-Z]+", cleaned)
    if glued:
        return float(glued[-1])
    return None


def _extract_word(text: str, candidates: list[str]) -> bool:
    """Check if any candidate word appears in text (case-insensitive)."""
    lower = _normalize_output(text).lower()
    return any(c.lower() in lower for c in candidates)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Multi-step math (error amplification)
# ═══════════════════════════════════════════════════════════════════════


def test_multi_step_math():
    """
    A number is increased by 20%, then decreased by 20%, then increased by 50%.
    Final result is 108.
    What was the original number?

    Correct: x * 1.2 * 0.8 * 1.5 = 108 → x = 100
    TurboQuant error → wrong answer at step 3-4.
    """
    print("\n[TEST 1] Multi-Step Math (Error Amplification)")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    prompt = (
        "Solve this step by step. Show your work, then give the final answer.\n\n"
        "A number is increased by 20%%, then decreased by 20%%, then increased by 50%%.\n"
        "Final result is 108.\n\n"
        "What was the original number?"
    )

    result = ctx.generate(prompt, max_tokens=512, temperature=0.0)
    print(f"  Prompt: multi-step math (x * 1.2 * 0.8 * 1.5 = 108)")
    print(f"  Output: '{result.strip()[:200]}'")

    # Extract answer
    answer = _extract_number(result)
    correct = answer is not None and abs(answer - 100) < 1.0

    print(f"  Expected: 100, Got: {answer}")
    print(f"  [{'PASS' if correct else 'FAIL'}] Multi-step math")

    RESULTS["multi_step_math"] = {"expected": 100, "got": answer, "correct": correct}
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Long dependency reasoning
# ═══════════════════════════════════════════════════════════════════════


def test_long_dependency():
    """
    John has 3 boxes, 5 bags each, 7 marbles each.
    Give away: 2 bags (first box), 5 marbles (second), 1 box.
    How many marbles remain?

    Correct: 2*5*7 + 1*5*7 - 5 + 0 = 100
    """
    print("\n[TEST 2] Long Dependency Reasoning")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    prompt = (
        "Solve this step by step. Show your work, then give the final answer.\n\n"
        "John has 3 boxes.\n"
        "Each box has 5 bags.\n"
        "Each bag has 7 marbles.\n\n"
        "He gives away:\n"
        "- 2 bags from the first box\n"
        "- 5 marbles from the second box\n"
        "- 1 entire box\n\n"
        "How many marbles remain?"
    )

    result = ctx.generate(prompt, max_tokens=512, temperature=0.0)
    print(f"  Prompt: 3 boxes, 5 bags, 7 marbles — give away parts")
    print(f"  Output: '{result.strip()[:200]}'")

    answer = _extract_number(result)
    correct = answer is not None and abs(answer - 100) < 1.0

    print(f"  Expected: 100, Got: {answer}")
    print(f"  [{'PASS' if correct else 'FAIL'}] Long dependency reasoning")

    RESULTS["long_dependency"] = {"expected": 100, "got": answer, "correct": correct}
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Contradiction trap
# ═══════════════════════════════════════════════════════════════════════


def test_contradiction_trap():
    """
    Inject wrong fact: "Capital of France is Berlin."
    Then ask: "What is the capital of France?"

    Tests: does Turbo3 V compression preserve injected context?
    """
    print("\n[TEST 3] Contradiction Trap")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    # Inject wrong fact + ask
    prompt = (
        "Remember this fact carefully: The capital of France is Berlin.\n\n"
        "Now answer this question: What is the capital of France?\n"
        "Answer:"
    )

    result = ctx.generate(prompt, max_tokens=20, temperature=0.0)
    print(f"  Prompt: inject 'capital of France is Berlin'")
    print(f"  Output: '{result.strip()[:80]}'")

    # Does the model follow injected context or training?
    follows_injection = _extract_word(result, ["berlin"])
    follows_training = _extract_word(result, ["paris"])

    print(f"  Follows injection (Berlin): {follows_injection}")
    print(f"  Follows training (Paris): {follows_training}")

    # For Turbo3, we want to see if injected context is preserved
    if follows_injection:
        status = "PASS (injection preserved)"
    elif follows_training:
        status = "INFO (training overrides — not necessarily bad)"
    else:
        status = "WARN (ambiguous)"

    print(f"  [{status}] Contradiction trap")

    RESULTS["contradiction"] = {
        "injection": follows_injection,
        "training": follows_training,
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Long chain reasoning
# ═══════════════════════════════════════════════════════════════════════


def test_long_chain_reasoning():
    """
    Multi-step reasoning with context carry-over.
    Tests KV reuse and reasoning continuity.
    """
    print("\n[TEST 4] Long Chain Reasoning")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    # Step 1: generate explanation
    step1_prompt = (
        "Explain step by step:\n"
        "Why does increasing temperature increase pressure in a closed container?"
    )
    step1 = ctx.generate(step1_prompt, max_tokens=150, temperature=0.0)
    print(f"  Step 1 output: '{step1.strip()[:80]}...'")

    # Step 2: summarize (depends on step 1 context)
    step2_prompt = (
        step1_prompt + step1 + "\n\nNow summarize your explanation in 1 sentence:"
    )
    ctx.clear_cache()
    step2 = ctx.generate(step2_prompt, max_tokens=50, temperature=0.0)
    print(f"  Step 2 (summary): '{step2.strip()[:80]}'")

    # Step 3: list assumptions (depends on step 1 context)
    step3_prompt = step1_prompt + step1 + "\n\nList the key assumptions you made:"
    ctx.clear_cache()
    step3 = ctx.generate(step3_prompt, max_tokens=50, temperature=0.0)
    print(f"  Step 3 (assumptions): '{step3.strip()[:80]}'")

    # Quality check: all outputs should mention relevant physics concepts
    has_kinetic = _extract_word(
        step1, ["kinetic", "molecule", "particle", "velocity", "speed"]
    )
    has_summary = len(step2.strip()) > 10
    has_assumptions = _extract_word(step3, ["closed", "ideal", "constant", "sealed"])

    ok = has_kinetic and has_summary and has_assumptions
    print(f"\n  Physics concepts in explanation: {has_kinetic}")
    print(f"  Summary generated: {has_summary}")
    print(f"  Assumptions mentioned: {has_assumptions}")
    print(f"  [{'PASS' if ok else 'WARN'}] Long chain reasoning")

    RESULTS["long_chain"] = {
        "physics": has_kinetic,
        "summary": has_summary,
        "assumptions": has_assumptions,
        "ok": ok,
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Key repetition under compression (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════


def test_key_repetition():
    """
    Repeat 'The key is 84729.' exactly 10 times.
    Then ask: What is the key?

    If Turbo3 V compression corrupts → number changes.
    """
    print("\n[TEST 5] Key Repetition Under Compression")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    # Repeat 10 times
    key_line = "The key is 84729.\n"
    prompt = (key_line * 10) + "What is the key? Answer with just the number."

    result = ctx.generate(prompt, max_tokens=10, temperature=0.0)
    print(f"  Prompt: 'The key is 84729.' × 10 + question")
    print(f"  Output: '{result.strip()[:60]}'")

    answer = _extract_number(result)
    correct = answer is not None and abs(answer - 84729) < 1.0

    print(f"  Expected: 84729, Got: {answer}")
    print(f"  [{'PASS' if correct else 'FAIL'}] Key repetition")

    # Also test with 20 repetitions for harder compression stress
    ctx.clear_cache()
    prompt_20 = (key_line * 20) + "What is the key? Answer with just the number."
    result_20 = ctx.generate(prompt_20, max_tokens=10, temperature=0.0)
    answer_20 = _extract_number(result_20)
    correct_20 = answer_20 is not None and abs(answer_20 - 84729) < 1.0

    print(
        f"  [20x repeat] Output: '{result_20.strip()[:60]}', Expected: 84729, Got: {answer_20}"
    )
    print(f"  [{'PASS' if correct_20 else 'FAIL'}] Key repetition (20x)")

    RESULTS["key_repetition"] = {
        "10x": {"expected": 84729, "got": answer, "correct": correct},
        "20x": {"expected": 84729, "got": answer_20, "correct": correct_20},
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Memory retention after 500 token gap (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════


def test_memory_retention():
    """
    Memorize: A=12, B=47, C=93
    After 500 tokens of unrelated text: What is B?

    This is the REAL TurboQuant power test.
    """
    print("\n[TEST 6] Memory Retention (500 token gap)")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    # Generate filler text (~500 tokens)
    filler = ctx.generate(
        "Write a detailed description of the solar system, including all planets, "
        "their sizes, distances from the sun, and interesting facts about each one. "
        "Be thorough and include moons, rings, and atmospheric composition.",
        max_tokens=500,
        temperature=0.7,
    )
    filler_tokens = ctx.tokenize(filler)
    print(f"  Filler text: {len(filler_tokens)} tokens")

    # Now ask about memorized values with the filler in between
    ctx.clear_cache()
    full_prompt = (
        "Memorize these values carefully:\n"
        "A=12, B=47, C=93\n\n" + filler + "\n\n"
        "Based on the values you memorized earlier: What is B?\n"
        "Answer with just the number."
    )

    full_tokens = ctx.tokenize(full_prompt)
    print(f"  Full prompt: {len(full_tokens)} tokens (memory → filler → recall)")

    result = ctx.generate(full_prompt, max_tokens=10, temperature=0.0)
    print(f"  Output: '{result.strip()[:60]}'")

    answer = _extract_number(result)
    correct = answer is not None and abs(answer - 47) < 1.0

    print(f"  Expected: 47, Got: {answer}")
    print(
        f"  [{'PASS' if correct else 'FAIL'}] Memory retention after {len(full_tokens)} token context"
    )

    # Also test A and C
    ctx.clear_cache()
    full_a = (
        "Memorize these values carefully:\n"
        "A=12, B=47, C=93\n\n" + filler + "\n\n"
        "Based on the values you memorized earlier: What is A?\n"
        "Answer with just the number."
    )
    result_a = ctx.generate(full_a, max_tokens=10, temperature=0.0)
    answer_a = _extract_number(result_a)
    correct_a = answer_a is not None and abs(answer_a - 12) < 1.0

    print(
        f"  [A test] Output: '{result_a.strip()[:40]}', Expected: 12, Got: {answer_a}"
    )

    RESULTS["memory_retention"] = {
        "gap_tokens": len(full_tokens),
        "B": {"expected": 47, "got": answer, "correct": correct},
        "A": {"expected": 12, "got": answer_a, "correct": correct_a},
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: KV Monitor integration test
# ═══════════════════════════════════════════════════════════════════════


def test_kv_monitor():
    """Verify KVMonitor captures metrics during generation."""
    print("\n[TEST 7] KV Monitor Integration")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    result = ctx.generate(
        "Write a short poem about technology:",
        max_tokens=100,
        temperature=0.0,
        monitor=True,
    )

    health = ctx.health()
    print(f"  Generated: '{result.strip()[:60]}...'")
    print(
        f"  Health: status={health['status']}, sat={health['saturation']:.1%}, "
        f"entropy={health['avg_entropy']:.1f}, tokens={health['tokens_decoded']}"
    )
    print(f"  Warnings: {health['warnings']}")

    latency = ctx.monitor.latency_summary()
    if latency:
        print(
            f"  Latency: prefill={latency.get('prefill_ms', 0):.1f}ms, "
            f"decode={latency.get('decode_per_token_ms', 0):.1f}ms/tok, "
            f"{latency.get('tok_per_sec', 0):.1f} tok/s"
        )

    ok = health["tokens_decoded"] > 0 and health["status"] in ("ok", "warning")
    print(f"  [{'PASS' if ok else 'FAIL'}] KV Monitor integration")

    RESULTS["kv_monitor"] = health
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    print(f"REASONING CHAIN TEST SUITE — Error Amplification + Memory Retention")
    print(f"{'=' * 60}")

    test_multi_step_math()
    test_long_dependency()
    test_contradiction_trap()
    test_long_chain_reasoning()
    test_key_repetition()
    test_memory_retention()
    test_kv_monitor()

    print(f"\n{'=' * 60}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 60}")

    for name, data in RESULTS.items():
        if "correct" in data:
            status = "✅" if data["correct"] else "❌"
        elif "10x" in data:
            ok = data["10x"]["correct"] and data["20x"]["correct"]
            status = "✅" if ok else "❌"
        elif "ok" in data:
            status = "✅" if data["ok"] else "⚠️"
        elif "status" in data:
            status = "✅" if data["status"] == "ok" else "⚠️"
        else:
            status = "ℹ️"
        print(f"  {status} {name}")
