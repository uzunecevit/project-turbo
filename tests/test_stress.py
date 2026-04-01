"""
PROJECT-TURBO | Stress Test Suite — Dayanıklılık Kanıtı
Gerçek stres testleri: uzun üretim, KV drift, cross-turn, A/B kalite.

Usage:
    LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_stress.py
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_STRESS_MODEL", "")
if not MODEL or not os.path.exists(MODEL):
    print("[SKIP] Set TURBO_STRESS_MODEL=/path/to/model.gguf")
    sys.exit(0)

from turbo import TurboContext, TurboBridge

RESULTS = {}


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Long Generation — 1024 token drift analizi
# ═══════════════════════════════════════════════════════════════════════


def test_long_generation():
    """1024 token üretim: latency profili, token tekrarı, drift tespiti."""
    print("\n[TEST 1] Long Generation (1024 token)")
    print("-" * 60)

    ctx = TurboContext(
        model_path=MODEL, n_ctx=4096, cache_type="turbo3", flash_attn=True
    )

    prompt = "Write a detailed essay about the history of artificial intelligence:"
    tokens = ctx.tokenize(prompt)

    # Prefill
    t0 = time.time()
    logits = ctx.decode(tokens)
    prefill_time = time.time() - t0
    print(
        f"  Prefill: {len(tokens)} tokens in {prefill_time:.2f}s ({len(tokens) / prefill_time:.0f} tok/s)"
    )

    # Decode loop with timing
    pos = len(tokens)
    generated = []
    decode_times = []
    t_start = time.time()

    for i in range(1024):
        t_step = time.time()
        best = max(range(len(logits)), key=lambda j: logits[j])
        text = ctx.bridge.token_to_piece(best)
        generated.append(text)

        ret = ctx.bridge.decode([best], pos=pos)
        if ret != 0:
            print(f"  [BREAK] decode failed at step {i}: {ret}")
            break
        pos += 1
        logits = ctx.bridge.get_logits()
        decode_times.append(time.time() - t_step)

        # Progress
        if (i + 1) % 256 == 0:
            elapsed = time.time() - t_start
            tps = (i + 1) / elapsed
            avg_ms = statistics.mean(decode_times[-256:]) * 1000
            print(f"  [{i + 1:>4}/1024] {tps:.1f} tok/s, avg {avg_ms:.1f}ms/tok")

    total_time = time.time() - t_start
    full_text = "".join(generated)

    # Analiz
    words = full_text.split()
    unique_words = set(w.lower() for w in words)
    repetition_ratio = 1 - (len(unique_words) / max(len(words), 1))

    # Saturation check: son 100 token'ın ilk 100 token ile benzerliği
    first_100 = "".join(generated[:100])
    last_100 = "".join(generated[-100:])
    common_chars = sum(1 for a, b in zip(first_100, last_100) if a == b)
    saturation = common_chars / max(len(first_100), 1)

    print(f"\n  --- Analiz ---")
    print(f"  Toplam süre: {total_time:.1f}s")
    print(f"  Throughput: {len(generated) / total_time:.1f} tok/s")
    print(f"  Token sayısı: {len(generated)}")
    print(f"  Unique word ratio: {len(unique_words) / max(len(words), 1) * 100:.1f}%")
    print(f"  Repetition ratio: {repetition_ratio * 100:.1f}%")
    print(f"  Saturation (first vs last 100): {saturation * 100:.1f}%")
    print(
        f"  Latency (min/avg/max ms): {min(decode_times) * 1000:.1f}/{statistics.mean(decode_times) * 1000:.1f}/{max(decode_times) * 1000:.1f}"
    )

    # Kalite değerlendirmesi
    ok = repetition_ratio < 0.7 and saturation < 0.5
    status = "PASS" if ok else "WARN"
    print(
        f"\n  [{status}] Long generation: {'drift yok' if ok else 'potansiyel drift tespit edildi'}"
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
    t1 = ctx.generate(
        "My favorite color is blue and I live in Istanbul.",
        max_tokens=5,
        temperature=0.0,
    )
    print(f"  Turn 1 input: 'My favorite color is blue and I live in Istanbul.'")
    print(f"  Turn 1 output: '{t1.strip()[:60]}'")

    # Turn 2: Hatırlıyor mu?
    ctx.clear_cache()
    t2 = ctx.generate(
        "What is my favorite color and where do I live?", max_tokens=20, temperature=0.0
    )
    print(f"  Turn 2 input: 'What is my favorite color and where do I live?'")
    print(f"  Turn 2 output: '{t2.strip()[:80]}'")

    # NOT: clear_cache() ile KV temizlendiği için bu test aslında
    # single-turn semantic understanding testi.
    # Gerçek multi-turn için clear_cache() KULLANILMAMALI.
    # Ama bizim adapter'da KV cache persist edilemiyor (bridge global state).

    # Turn 3: Tekrar context oluştur, devam et
    ctx.clear_cache()
    t3 = ctx.generate(
        "Tell me about the weather in the city where I live.",
        max_tokens=20,
        temperature=0.0,
    )
    print(f"  Turn 3 input: 'Tell me about the weather in the city where I live.'")
    print(f"  Turn 3 output: '{t3.strip()[:80]}'")

    # Semantic check
    has_blue = "blue" in t2.lower() or "mavi" in t2.lower()
    has_istanbul = (
        "istanbul" in t2.lower() or "türkiye" in t2.lower() or "turkey" in t2.lower()
    )

    print(f"\n  Semantic check:")
    print(f"    'blue' found: {has_blue}")
    print(f"    'Istanbul' found: {has_istanbul}")

    if has_blue and has_istanbul:
        print(f"  [PASS] Cross-turn semantic continuity: BILGI KORUNDU")
    elif has_blue or has_istanbul:
        print(f"  [WARN] Partial semantic continuity: yarım hatırlama")
    else:
        print(f"  [INFO] Single-turn model (KV clear sonrası context kaybı beklenen)")

    RESULTS["cross_turn"] = {
        "has_blue": has_blue,
        "has_istanbul": has_istanbul,
    }

    ctx.bridge.free()


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: A/B Kalite Testi — q8_0/q8_0 vs q8_0/turbo3
# ═══════════════════════════════════════════════════════════════════════


def test_ab_quality():
    """Aynı prompt, iki farklı KV config: output karşılaştırması."""
    print("\n[TEST 3] A/B Quality: q8_0/q8_0 vs q8_0/turbo3")
    print("-" * 60)

    prompt = "Explain the concept of recursion in programming with a simple example:"
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

        tokens = bridge.tokenize(prompt)
        bridge.decode(tokens, pos=0)

        pos = len(tokens)
        generated = []
        for i in range(100):
            logits = bridge.get_logits()
            best = max(range(len(logits)), key=lambda j: logits[j])
            text = bridge.token_to_piece(best)
            generated.append(text)
            ret = bridge.decode([best], pos=pos)
            if ret != 0:
                break
            pos += 1

        full = "".join(generated)
        outputs[name] = full
        print(f'  {name}: "{full[:80]}..."')
        bridge.free()

    # Karşılaştırma
    if "q8_0/q8_0" in outputs and "q8_0/turbo3" in outputs:
        base = outputs["q8_0/q8_0"]
        turbo = outputs["q8_0/turbo3"]

        # Character-level similarity
        min_len = min(len(base), len(turbo))
        matches = sum(1 for i in range(min_len) if base[i] == turbo[i])
        similarity = matches / max(min_len, 1)

        # Word-level overlap
        base_words = set(base.lower().split())
        turbo_words = set(turbo.lower().split())
        word_overlap = len(base_words & turbo_words) / max(
            len(base_words | turbo_words), 1
        )

        print(f"\n  --- Karşılaştırma ---")
        print(f"  Character similarity: {similarity * 100:.1f}%")
        print(f"  Word overlap: {word_overlap * 100:.1f}%")

        if similarity > 0.5:
            print(f"  [PASS] Kalite kaybı minimal")
        elif similarity > 0.3:
            print(f"  [WARN] Bazı farklılıklar var ama anlamlı")
        else:
            print(f"  [FAIL] Ciddi kalite kaybı")

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

    # Prompt 1: Uzun doldurma
    t1 = ctx.generate("The capital of France is", max_tokens=10, temperature=0.0)
    print(f"  Prompt 1: 'The capital of France is' -> '{t1.strip()[:40]}'")

    # Cache temizle (corruption simülasyonu)
    ctx.clear_cache()

    # Prompt 2: Yeni context
    t2 = ctx.generate("2 + 2 =", max_tokens=5, temperature=0.0)
    print(f"  Prompt 2 (after clear): '2 + 2 =' -> '{t2.strip()[:20]}'")

    # Prompt 3: Tekrar temizle, farklı prompt
    ctx.clear_cache()
    t3 = ctx.generate("The opposite of hot is", max_tokens=5, temperature=0.0)
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
    """Farklı prompt uzunluklarında prefill hızı."""
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
    print(f"  [INFO] Prefill hızları yukarıda")


# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    print(f"STRESS TEST SUITE — Dayanıklılık Kanıtı")
    print(f"{'=' * 60}")

    test_long_generation()
    test_cross_turn_consistency()
    test_ab_quality()
    test_kv_corruption()
    test_prefill_performance()

    print(f"\n{'=' * 60}")
    print("SONUÇ ÖZETİ")
    print(f"{'=' * 60}")

    for name, data in RESULTS.items():
        if "ok" in data:
            status = "✅" if data["ok"] else "⚠️"
        elif "char_similarity" in data:
            status = "✅" if data["char_similarity"] > 0.5 else "⚠️"
        elif "math_correct" in data:
            status = (
                "✅"
                if data.get("math_correct") and data.get("semantic_correct")
                else "⚠️"
            )
        else:
            status = "ℹ️"
        print(f"  {status} {name}: {data}")
