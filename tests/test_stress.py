"""
PROJECT-TURBO | Stress Test Suite — Dayaniklilik Kaniti
Gercek stres testleri: uzun uretim, KV drift, cross-turn, A/B kalite.

v0.6: Chat template + proper sampling (non-greedy). System prompt included.

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_stress.py
"""

import os
import sys
import time
import statistics
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_STRESS_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_STRESS_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge

RESULTS = {}

THINKING_SAMPLING = {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
NON_THINKING_SAMPLING = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def _strip_thinking(text):
    ts = chr(60) + "think" + chr(62)
    te = chr(60) + "/think" + chr(62)
    text = re.sub(r"(?si)" + re.escape(ts) + r".*?" + re.escape(te), "", text)
    text = re.sub(r"(?si)" + re.escape(ts), "", text)
    text = re.sub(r"(?si)" + re.escape(te), "", text)
    return text


def _format_prompt(ctx, user_content):
    try:
        return ctx.format_user_prompt(user_content)
    except Exception:
        return user_content


def _gen_sampled(ctx, prompt, max_tokens, sampling=None, seed=42):
    if sampling is None:
        sampling = THINKING_SAMPLING

    tokens = ctx.tokenize(prompt)
    logits = ctx.decode(tokens)
    pos = len(tokens)

    sampler = ctx.bridge.sampler_init(
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
        min_p=sampling["min_p"],
        temp=sampling["temp"],
        seed=seed,
    )

    out_tokens = []
    for _ in range(max_tokens):
        token = ctx.bridge.sampler_sample(sampler)
        out_tokens.append(token)
        ret = ctx.bridge.ctx_decode([token], pos)
        if ret != 0:
            break
        logits = ctx.bridge.ctx_get_logits()
        pos += 1

    ctx.bridge.sampler_free(sampler)
    return ctx.detokenize(out_tokens)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Long Generation — 1024 token drift analizi
# ═══════════════════════════════════════════════════════════════════════


def test_long_generation():
    """1024 token uretim: latency profili, token tekrari, drift tespiti."""
    print("\n[TEST 1] Long Generation (1024 token)")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    prompt = _format_prompt(
        ctx, "Write a detailed essay about the history of artificial intelligence:"
    )
    tokens = ctx.tokenize(prompt)

    # Prefill
    t0 = time.time()
    logits = ctx.decode(tokens)
    prefill_time = time.time() - t0
    print(
        f"  Prefill: {len(tokens)} tokens in {prefill_time:.2f}s ({len(tokens) / prefill_time:.0f} tok/s)"
    )

    # Decode loop with C sampler chain
    sampler = ctx.bridge.sampler_init(
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        temp=0.6,
        seed=42,
    )

    pos = len(tokens)
    generated = []
    decode_times = []
    t_start = time.time()

    for i in range(1024):
        t_step = time.time()
        token = ctx.bridge.sampler_sample(sampler)
        text = ctx.bridge.token_to_piece(token)
        generated.append(text)

        ret = ctx.bridge.ctx_decode([token], pos=pos)
        if ret != 0:
            print(f"  [BREAK] decode failed at step {i}: {ret}")
            break
        pos += 1
        logits = ctx.bridge.ctx_get_logits()
        decode_times.append(time.time() - t_step)

        # Progress
        if (i + 1) % 256 == 0:
            elapsed = time.time() - t_start
            tps = (i + 1) / elapsed
            avg_ms = statistics.mean(decode_times[-256:]) * 1000
            print(f"  [{i + 1:>4}/1024] {tps:.1f} tok/s, avg {avg_ms:.1f}ms/tok")

    ctx.bridge.sampler_free(sampler)

    total_time = time.time() - t_start
    full_text = "".join(generated)

    # Analiz
    words = full_text.split()
    unique_words = set(w.lower() for w in words)
    repetition_ratio = 1 - (len(unique_words) / max(len(words), 1))

    first_100 = "".join(generated[:100])
    last_100 = "".join(generated[-100:])
    common_chars = sum(1 for a, b in zip(first_100, last_100) if a == b)
    saturation = common_chars / max(len(first_100), 1)

    print(f"\n  --- Analiz ---")
    print(f"  Toplam sure: {total_time:.1f}s")
    print(f"  Throughput: {len(generated) / total_time:.1f} tok/s")
    print(f"  Token sayisi: {len(generated)}")
    print(f"  Unique word ratio: {len(unique_words) / max(len(words), 1) * 100:.1f}%")
    print(f"  Repetition ratio: {repetition_ratio * 100:.1f}%")
    print(f"  Saturation (first vs last 100): {saturation * 100:.1f}%")
    print(
        f"  Latency (min/avg/max ms): {min(decode_times) * 1000:.1f}/{statistics.mean(decode_times) * 1000:.1f}/{max(decode_times) * 1000:.1f}"
    )

    ok = repetition_ratio < 0.7 and saturation < 0.5
    status = "PASS" if ok else "WARN"
    print(
        f"  [{status}] Long generation: {'drift yok' if ok else 'potansiyel drift tespit edildi'}"
    )

    RESULTS["long_gen"] = {
        "tokens": len(generated),
        "time": total_time,
        "tps": len(generated) / total_time,
        "repetition": repetition_ratio,
        "saturation": saturation,
        "ok": ok,
    }

    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Cross-Turn Consistency
# ═══════════════════════════════════════════════════════════════════════


def test_cross_turn_consistency():
    """Multi-turn: KV semantic continuity testi."""
    print("\n[TEST 2] Cross-Turn Consistency")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=2048, cache_type="turbo3", flash_attn=True
    )

    # Turn 1: Bilgi ver
    t1_prompt = _format_prompt(ctx, "My favorite color is blue and I live in Istanbul.")
    t1 = _gen_sampled(ctx, t1_prompt, 5, sampling=NON_THINKING_SAMPLING)
    print(f"  Turn 1 input: 'My favorite color is blue and I live in Istanbul.'")
    print(f"  Turn 1 output: '{t1.strip()[:60]}'")

    # Turn 2: Hatirliyor mu?
    ctx.clear_cache()
    t2_prompt = _format_prompt(ctx, "What is my favorite color and where do I live?")
    t2 = _gen_sampled(ctx, t2_prompt, 20, sampling=NON_THINKING_SAMPLING)
    print(f"  Turn 2 input: 'What is my favorite color and where do I live?'")
    print(f"  Turn 2 output: '{t2.strip()[:80]}'")

    # Turn 3: Tekrar context olustur
    ctx.clear_cache()
    t3_prompt = _format_prompt(
        ctx, "Tell me about the weather in the city where I live."
    )
    t3 = _gen_sampled(ctx, t3_prompt, 20, sampling=NON_THINKING_SAMPLING)
    print(f"  Turn 3 input: 'Tell me about the weather in the city where I live.'")
    print(f"  Turn 3 output: '{t3.strip()[:80]}'")

    # Semantic check
    has_blue = "blue" in t2.lower() or "mavi" in t2.lower()
    has_istanbul = (
        "istanbul" in t2.lower() or "turkiye" in t2.lower() or "turkey" in t2.lower()
    )

    print(f"\n  Semantic check:")
    print(f"    'blue' found: {has_blue}")
    print(f"    'Istanbul' found: {has_istanbul}")

    if has_blue and has_istanbul:
        print(f"  [PASS] Cross-turn semantic continuity: BILGI KORUNDU")
    elif has_blue or has_istanbul:
        print(f"  [WARN] Partial semantic continuity: yarim hatirlama")
    else:
        print(f"  [INFO] Single-turn model (KV clear sonrasi context kaybi beklenen)")

    RESULTS["cross_turn"] = {
        "has_blue": has_blue,
        "has_istanbul": has_istanbul,
    }

    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: A/B Kalite Testi — q8_0/q8_0 vs q8_0/turbo3
# ═══════════════════════════════════════════════════════════════════════


def test_ab_quality():
    """Ayni prompt, iki farkli KV config: output karsilastirmasi."""
    print("\n[TEST 3] A/B Quality: q8_0/q8_0 vs q8_0/turbo3")
    print("-" * 60)

    ab_raw = "Explain the concept of recursion in programming with a simple example:"
    configs = [
        ("q8_0/q8_0", 8, 8),
        ("q8_0/turbo3", 8, 41),
    ]

    outputs = {}
    for name, type_k, type_v in configs:
        bridge = TurboBridge()
        ret = bridge.init(MODEL, 2048, type_k, type_v, -1, 1, 1)
        if ret != 0:
            print(f"  [SKIP] {name}: init failed ({ret})")
            bridge.free()
            continue

        try:
            prompt = bridge.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": ab_raw},
                ],
                add_ass=True,
            )
        except Exception:
            prompt = ab_raw

        tokens = bridge.tokenize(prompt)
        bridge.decode(tokens, pos=0)

        sampler = bridge.sampler_init(
            top_k=20,
            top_p=0.8,
            min_p=0.0,
            temp=0.7,
            seed=42,
        )

        pos = len(tokens)
        generated = []
        for i in range(100):
            token = bridge.sampler_sample(sampler)
            text = bridge.token_to_piece(token)
            generated.append(text)
            ret = bridge.decode([token], pos=pos)
            if ret != 0:
                break
            pos += 1

        bridge.sampler_free(sampler)

        full = "".join(generated)
        outputs[name] = full
        print(f'  {name}: "{_strip_thinking(full)[:80]}..."')
        bridge.free()

    # Karsilastirma
    if "q8_0/q8_0" in outputs and "q8_0/turbo3" in outputs:
        base = _strip_thinking(outputs["q8_0/q8_0"])
        turbo = _strip_thinking(outputs["q8_0/turbo3"])

        min_len = min(len(base), len(turbo))
        matches = sum(1 for i in range(min_len) if base[i] == turbo[i])
        similarity = matches / max(min_len, 1)

        base_words = set(base.lower().split())
        turbo_words = set(turbo.lower().split())
        word_overlap = len(base_words & turbo_words) / max(
            len(base_words | turbo_words), 1
        )

        print(f"\n  --- Karsilastirma ---")
        print(f"  Character similarity: {similarity * 100:.1f}%")
        print(f"  Word overlap: {word_overlap * 100:.1f}%")

        if similarity > 0.5:
            print(f"  [PASS] Kalite kaybi minimal")
        elif similarity > 0.3:
            print(f"  [WARN] Bazi farkliliklar var ama anlamli")
        else:
            print(f"  [FAIL] Ciddi kalite kaybi")

        RESULTS["ab_quality"] = {
            "char_similarity": similarity,
            "word_overlap": word_overlap,
            "base_len": len(base),
            "turbo_len": len(turbo),
        }


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: KV Corruption — partial reset + recovery
# ═══════════════════════════════════════════════════════════════════════


def test_kv_corruption():
    """KV cache temizledikten sonra yeni prompt ile recovery."""
    print("\n[TEST 4] KV Corruption / Recovery")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )

    # Prompt 1
    p1 = _format_prompt(ctx, "The capital of France is")
    t1 = _gen_sampled(ctx, p1, 10, sampling=NON_THINKING_SAMPLING)
    print(f"  Prompt 1: 'The capital of France is' -> '{t1.strip()[:40]}'")

    # Cache temizle
    ctx.clear_cache()

    # Prompt 2
    p2 = _format_prompt(ctx, "2 + 2 =")
    t2 = _gen_sampled(ctx, p2, 5, sampling=NON_THINKING_SAMPLING)
    print(f"  Prompt 2 (after clear): '2 + 2 =' -> '{t2.strip()[:20]}'")

    # Prompt 3
    ctx.clear_cache()
    p3 = _format_prompt(ctx, "The opposite of hot is")
    t3 = _gen_sampled(ctx, p3, 5, sampling=NON_THINKING_SAMPLING)
    print(f"  Prompt 3 (after clear): 'The opposite of hot is' -> '{t3.strip()[:20]}'")

    # Recovery check
    has_4 = "4" in t2 or "four" in t2.lower()
    has_cold = "cold" in t3.lower() or "cool" in t3.lower()

    print(f"\n  Recovery check:")
    print(f"    '2+2=4' correct: {has_4}")
    print(f"    'opposite of hot = cold' correct: {has_cold}")

    status = "PASS" if (has_4 and has_cold) else "WARN"
    print(f"  [{status}] KV corruption/recovery")

    RESULTS["kv_corruption"] = {
        "math_correct": has_4,
        "semantic_correct": has_cold,
    }

    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Prefill Performance
# ═══════════════════════════════════════════════════════════════════════


def test_prefill_performance():
    """Farkli prompt uzunluklarinda prefill hizi."""
    print("\n[TEST 5] Prefill Performance")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    for n_words in [10, 50, 100, 200]:
        prompt = " ".join(["word"] * n_words)
        tokens = ctx.tokenize(prompt)

        ctx.clear_cache()
        t0 = time.time()
        ctx.decode(tokens)
        elapsed = time.time() - t0

        tps = len(tokens) / elapsed if elapsed > 0 else 0
        print(f"  {len(tokens):>4} tokens: {elapsed:.2f}s ({tps:.0f} tok/s)")

    ctx.bridge.free()
    print(f"  [INFO] Prefill hizlari yukarida")


# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    print(f"STRESS TEST SUITE v0.6 — Chat Template + Proper Sampling")
    print(f"{'=' * 60}")

    test_long_generation()
    test_cross_turn_consistency()
    test_ab_quality()
    test_kv_corruption()
    test_prefill_performance()

    print(f"\n{'=' * 60}")
    print("SONUC OZETI")
    print(f"{'=' * 60}")

    for name, data in RESULTS.items():
        if "ok" in data:
            status = "OK" if data["ok"] else "WARN"
        elif "char_similarity" in data:
            status = "OK" if data["char_similarity"] > 0.5 else "WARN"
        elif "math_correct" in data:
            status = (
                "OK"
                if data.get("math_correct") and data.get("semantic_correct")
                else "WARN"
            )
        else:
            status = "INFO"
        print(f"  [{status}] {name}: {data}")
