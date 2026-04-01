"""
PROJECT-TURBO | Inference tests — requires a GGUF model
Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_inference.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")


def skip_if_no_model():
    if not MODEL or not os.path.exists(MODEL):
        print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
        sys.exit(0)


# ── Bridge Tests ──────────────────────────────────────────────────────


def test_bridge_init():
    """Bridge ile model yükleme ve vocab erişimi."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)  # K=q8_0, V=turbo3
    assert ret == 0, f"turbo_init failed: {ret}"

    nv = bridge.n_vocab()
    assert nv > 1000, f"unexpected vocab size: {nv}"
    print(f"[PASS] bridge_init: vocab={nv}")

    bridge.free()


def test_bridge_tokenize():
    """Tokenize → detokenize round-trip."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    bridge.init(MODEL, 512, 8, 41, -1, 0, 1)

    tokens = bridge.tokenize("Hello world, how are you?")
    assert len(tokens) > 0, "tokenize returned empty"

    text = "".join(bridge.token_to_piece(t) for t in tokens)
    assert "Hello" in text or "hello" in text.lower(), f"round-trip failed: '{text}'"
    print(f"[PASS] tokenize_roundtrip: {len(tokens)} tokens -> '{text[:40]}'")

    bridge.free()


def test_bridge_decode():
    """Tek token decode."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    bridge.init(MODEL, 512, 8, 41, -1, 0, 1)

    tokens = bridge.tokenize("Hi")
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0, f"decode failed: {ret}"

    logits = bridge.get_logits()
    assert len(logits) > 0, "logits empty"
    assert any(l != 0.0 for l in logits), "all logits are zero"
    print(f"[PASS] bridge_decode: {len(logits)} logits")

    bridge.free()


# ── Asymmetric K/V Tests ─────────────────────────────────────────────


def test_asymmetric_kv_q80_turbo3():
    """K=q8_0, V=turbo3 — asimetrik strateji."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)  # K=q8_0, V=turbo3
    assert ret == 0

    tokens = bridge.tokenize("The meaning of life is")
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    logits = bridge.get_logits()
    best = max(range(len(logits)), key=lambda i: logits[i])
    text = bridge.token_to_piece(best)
    print(f"[PASS] asymmetric_kv: next='{text}'")

    bridge.free()


def test_asymmetric_kv_q80_turbo4():
    """K=q8_0, V=turbo4 — 4-bit V compression."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 42, -1, 0, 1)  # K=q8_0, V=turbo4
    assert ret == 0, f"turbo4 init failed: {ret}"

    tokens = bridge.tokenize("Hello")
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    logits = bridge.get_logits()
    best = max(range(len(logits)), key=lambda i: logits[i])
    text = bridge.token_to_piece(best)
    print(f"[PASS] turbo4_kv: next='{text}'")

    bridge.free()


def test_asymmetric_kv_q80_turbo2():
    """K=q8_0, V=turbo2 — 2-bit V compression."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 43, -1, 0, 1)  # K=q8_0, V=turbo2
    assert ret == 0, f"turbo2 init failed: {ret}"

    tokens = bridge.tokenize("Hello")
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    logits = bridge.get_logits()
    best = max(range(len(logits)), key=lambda i: logits[i])
    text = bridge.token_to_piece(best)
    print(f"[PASS] turbo2_kv: next='{text}'")

    bridge.free()


def test_baseline_q80_q80():
    """K=q8_0, V=q8_0 — baseline karşılaştırma."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 8, -1, 1, 1)  # K=q8_0, V=q8_0, flash_attn=1
    if ret != 0:
        print(
            f"[SKIP] baseline_q8_0: spiritbuun fork K=q8_0/V=q8_0 desteklemiyor (ret={ret})"
        )
        return

    tokens = bridge.tokenize("The meaning of life is")
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    logits = bridge.get_logits()
    best = max(range(len(logits)), key=lambda i: logits[i])
    text = bridge.token_to_piece(best)
    print(f"[PASS] baseline_q8_0: next='{text}'")

    bridge.free()


# ── Generation Tests ─────────────────────────────────────────────────


def test_greedy_generation():
    """Greedy (temperature=0) üretim."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )
    text = ctx.generate("Hello", max_tokens=10, temperature=0.0)
    assert len(text) > 0, "generation empty"
    print(f"[PASS] greedy_gen: '{text.strip()[:50]}'")
    ctx.bridge.free()


def test_sampling_generation():
    """Temperature sampling üretimi."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )
    text = ctx.generate("Once upon a time", max_tokens=15, temperature=0.7, top_p=0.9)
    assert len(text) > 0, "sampling generation empty"
    print(f"[PASS] sampling_gen: '{text.strip()[:50]}'")
    ctx.bridge.free()


def test_multi_prompt():
    """Ardışık prompt'larla KV cache temizleme."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )

    prompts = ["Hello", "Merhaba", "Python"]
    for p in prompts:
        ctx.clear_cache()
        out = ctx.generate(p, max_tokens=5, temperature=0.0)
        assert len(out) > 0, f"empty output for '{p}'"

    print(f"[PASS] multi_prompt: {len(prompts)} prompts OK")
    ctx.bridge.free()


def test_incremental_decode():
    """Her adımda tek token decode — temel generation loop."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    bridge.init(MODEL, 1024, 8, 41, -1, 1, 1)

    tokens = bridge.tokenize("Hello world")
    bridge.decode(tokens, pos=0)

    pos = len(tokens)
    generated = []
    for i in range(5):
        logits = bridge.get_logits()
        best = max(range(len(logits)), key=lambda i: logits[i])
        text = bridge.token_to_piece(best)
        generated.append(text)

        ret = bridge.decode([best], pos=pos)
        assert ret == 0, f"incremental decode failed at step {i}"
        pos += 1

    result = "".join(generated)
    assert len(result) > 0
    print(f"[PASS] incremental_decode: '{result[:40]}'")
    bridge.free()


# ── Flash Attention Tests ────────────────────────────────────────────


def test_flash_attn_on():
    """Flash Attention açık."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=512, cache_type="turbo3", flash_attn=True
    )
    text = ctx.generate("Hi", max_tokens=5, temperature=0.0)
    assert len(text) > 0
    print(f"[PASS] flash_attn_on: '{text.strip()[:30]}'")
    ctx.bridge.free()


def test_flash_attn_off():
    """Flash Attention kapalı."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=512, cache_type="turbo3", flash_attn=False
    )
    text = ctx.generate("Hi", max_tokens=5, temperature=0.0)
    assert len(text) > 0
    print(f"[PASS] flash_attn_off: '{text.strip()[:30]}'")
    ctx.bridge.free()


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    skip_if_no_model()
    print(f"Model: {MODEL}\n")

    tests = [
        ("bridge_init", test_bridge_init),
        ("bridge_tokenize", test_bridge_tokenize),
        ("bridge_decode", test_bridge_decode),
        ("asymmetric_kv_turbo3", test_asymmetric_kv_q80_turbo3),
        ("asymmetric_kv_turbo4", test_asymmetric_kv_q80_turbo4),
        ("asymmetric_kv_turbo2", test_asymmetric_kv_q80_turbo2),
        ("baseline_q8_0", test_baseline_q80_q80),
        ("greedy_generation", test_greedy_generation),
        ("sampling_generation", test_sampling_generation),
        ("multi_prompt", test_multi_prompt),
        ("incremental_decode", test_incremental_decode),
        ("flash_attn_on", test_flash_attn_on),
        ("flash_attn_off", test_flash_attn_off),
    ]

    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except SystemExit:
            skipped += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(
        f"Results: {passed} passed, {failed} failed, {skipped} skipped, {passed + failed + skipped} total"
    )
    print(f"{'=' * 50}")
    sys.exit(1 if failed > 0 else 0)
