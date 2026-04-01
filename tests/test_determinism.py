"""
PROJECT-TURBO | Determinism Test Suite — Cross-Session Consistency
KV clear sonrası determinism, cross-context consistency,
injected vs original knowledge.

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_determinism.py
"""

import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge

RESULTS = {}


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Cross-reset determinism
# ═══════════════════════════════════════════════════════════════════════


def test_cross_reset_determinism():
    """
    Same prompt, same context, clear cache between runs.
    Greedy (temperature=0) → output should be identical.
    """
    print("\n[TEST 1] Cross-Reset Determinism")
    print("-" * 60)

    prompt = "What is 2 + 2? Answer with just the number."
    outputs = []

    for run in range(3):
        ctx = TurboContext(
            model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
        )
        result = ctx.generate(prompt, max_tokens=10, temperature=0.0)
        outputs.append(result.strip())
        ctx.bridge.free()
        del ctx
        gc.collect()  # ensure __del__ cleanup completes before next context

    print(f"  Run 1: '{outputs[0]}'")
    print(f"  Run 2: '{outputs[1]}'")
    print(f"  Run 3: '{outputs[2]}'")

    all_same = len(set(outputs)) == 1
    print(
        f"  [{'PASS' if all_same else 'FAIL'}] Cross-reset determinism: {'all identical' if all_same else 'VARIATION DETECTED'}"
    )

    RESULTS["cross_reset"] = {"outputs": outputs, "deterministic": all_same}


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Same-context determinism (no clear)
# ═══════════════════════════════════════════════════════════════════════


def test_same_context_determinism():
    """
    Same context, generate, then generate again WITHOUT clear.
    Tests if KV state affects subsequent generations deterministically.
    """
    print("\n[TEST 2] Same-Context Determinism")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    prompt = "What is 3 + 3? Answer with just the number."

    r1 = ctx.generate(prompt, max_tokens=10, temperature=0.0)
    ctx.clear_cache()
    r2 = ctx.generate(prompt, max_tokens=10, temperature=0.0)

    print(f"  Generation 1: '{r1.strip()}'")
    print(f"  After clear, Generation 2: '{r2.strip()}'")

    same = r1.strip() == r2.strip()
    print(f"  [{'PASS' if same else 'FAIL'}] Same-context determinism")

    RESULTS["same_context"] = {"g1": r1.strip(), "g2": r2.strip(), "same": same}
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Injected vs original knowledge
# ═══════════════════════════════════════════════════════════════════════


def test_injected_vs_original():
    """
    Inject 'Capital of France is Berlin.'
    Clear cache.
    Ask: 'Capital of France is?'
    Check: Berlin (KV leak) or Paris (correct).
    """
    print("\n[TEST 3] Injected vs Original Knowledge (KV Leak Test)")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    # Inject wrong fact
    ctx.generate(
        "Remember this fact: The capital of France is Berlin.",
        max_tokens=5,
        temperature=0.0,
    )

    # Clear cache — KV should be wiped
    ctx.clear_cache()

    # Ask the question (no injection this time)
    result = ctx.generate(
        "What is the capital of France? Answer with just the city name.",
        max_tokens=10,
        temperature=0.0,
    )

    print(f"  Injected: 'capital of France is Berlin'")
    print(f"  After clear, asked: 'capital of France is?'")
    print(f"  Output: '{result.strip()}'")

    has_berlin = "berlin" in result.lower()
    has_paris = "paris" in result.lower()

    if has_berlin:
        status = "FAIL (KV leak — Berlin persists after clear)"
    elif has_paris:
        status = "PASS (correct — Paris, no KV leak)"
    else:
        status = f"WARN (ambiguous: '{result.strip()[:40]}')"

    print(f"  [{status}]")

    RESULTS["kv_leak"] = {
        "output": result.strip(),
        "berlin": has_berlin,
        "paris": has_paris,
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Multi-turn without clear (context continuity)
# ═══════════════════════════════════════════════════════════════════════


def test_multi_turn_continuity():
    """
    Turn 1: establish fact ('My name is Alice')
    Turn 2: ask about it (clear + generate)
    Turn 3: clear again + ask (should NOT remember)
    """
    print("\n[TEST 4] Multi-Turn Continuity")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    # Turn 1: inject fact
    t1 = ctx.generate(
        "My name is Alice and I like cats.",
        max_tokens=5,
        temperature=0.0,
    )
    print(f"  Turn 1: 'My name is Alice and I like cats.' → '{t1.strip()[:40]}'")

    # Turn 2: clear cache, inject fact again, then ask
    ctx.clear_cache()
    full_prompt = "My name is Alice and I like cats.\n\nWhat is my name?"
    t2 = ctx.generate(full_prompt, max_tokens=10, temperature=0.0)
    print(f"  Turn 2 (with context): 'What is my name?' → '{t2.strip()[:40]}'")

    has_alice_with_context = "alice" in t2.lower()

    # Turn 3: clear + ask WITHOUT the fact
    ctx.clear_cache()
    t3 = ctx.generate(
        "What is my name?",
        max_tokens=10,
        temperature=0.0,
    )
    print(f"  Turn 3 (no context): 'What is my name?' → '{t3.strip()[:40]}'")

    has_alice_without_context = "alice" in t3.lower()

    print(f"\n  Alice found (with context): {has_alice_with_context}")
    print(f"  Alice found (no context): {has_alice_without_context}")

    if has_alice_with_context and not has_alice_without_context:
        status = "PASS (context works, clear works)"
    elif has_alice_with_context and has_alice_without_context:
        status = "WARN (KV leak after clear)"
    elif not has_alice_with_context:
        status = "INFO (model didn't recall — depends on prompt)"
    else:
        status = "INFO (ambiguous)"

    print(f"  [{status}]")

    RESULTS["multi_turn"] = {
        "with_context": has_alice_with_context,
        "no_context": has_alice_without_context,
    }
    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    print(f"DETERMINISM TEST SUITE — Cross-Session Consistency")
    print(f"{'=' * 60}")

    test_cross_reset_determinism()
    test_same_context_determinism()
    test_injected_vs_original()
    test_multi_turn_continuity()

    print(f"\n{'=' * 60}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 60}")

    for name, data in RESULTS.items():
        if "deterministic" in data:
            status = "✅" if data["deterministic"] else "❌"
        elif "same" in data:
            status = "✅" if data["same"] else "❌"
        elif "paris" in data and not data.get("berlin"):
            status = "✅"
        elif "berlin" in data and data.get("berlin"):
            status = "❌ (KV leak)"
        elif "with_context" in data:
            if data["with_context"] and not data["no_context"]:
                status = "✅"
            elif data["with_context"] and data["no_context"]:
                status = "⚠️ (KV leak)"
            else:
                status = "ℹ️"
        else:
            status = "ℹ️"
        print(f"  {status} {name}")
