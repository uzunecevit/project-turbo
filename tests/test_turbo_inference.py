"""
PROJECT-TURBO | Inference tests — requires a GGUF model

v0.6: Chat template + proper sampling. System prompt included.

Usage:
    TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_inference.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("TURBO_TEST_MODEL", "")

NON_THINKING_SAMPLING = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def skip_if_no_model():
    if not MODEL or not os.path.exists(MODEL):
        print("[SKIP] Set TURBO_TEST_MODEL=/path/to/model.gguf")
        sys.exit(0)


def _format_prompt(bridge, user_content):
    try:
        return bridge.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_content},
            ],
            add_ass=True,
        )
    except Exception:
        return user_content


# ── Bridge Tests ──────────────────────────────────────────────────────


def test_bridge_init():
    """Bridge ile model yukleme ve vocab erisimi."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)  # K=q8_0, V=turbo3
    assert ret == 0, f"turbo_init failed: {ret}"

    nv = bridge.n_vocab()
    assert nv > 1000, f"unexpected vocab size: {nv}"
    print(f"[PASS] bridge_init: vocab={nv}")

    bridge.free()


def test_bridge_tokenize():
    """Tokenize -> detokenize round-trip."""
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


# ── Chat Template Tests ──────────────────────────────────────────────


def test_chat_template_get():
    """GGUF'dan chat template okuma."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)
    assert ret == 0

    tmpl = bridge.get_chat_template()
    if tmpl:
        print(f"[PASS] chat_template_get: '{tmpl[:50]}...'")
    else:
        print(f"[PASS] chat_template_get: NULL (will use fallback)")

    bridge.free()


def test_chat_template_apply():
    """Chat template uygulama."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)
    assert ret == 0

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    try:
        result = bridge.apply_chat_template(messages, add_ass=True)
        assert len(result) > 0, "empty template result"
        assert "2+2" in result or "2+2" in result.lower(), (
            f"content missing: '{result[:80]}'"
        )
        print(f"[PASS] chat_template_apply: '{result[:80]}...'")
    except Exception as e:
        print(f"[FAIL] chat_template_apply: {e}")
        raise

    bridge.free()


def test_sampler_chain():
    """C sampler chain olusturma ve token ornekleme."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)
    assert ret == 0

    prompt = _format_prompt(bridge, "Hello")
    tokens = bridge.tokenize(prompt)
    bridge.decode(tokens, pos=0)

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.6, seed=42)
    assert sampler != 0, "sampler_init failed"

    token = bridge.sampler_sample(sampler)
    assert token >= 0, f"invalid token: {token}"

    text = bridge.token_to_piece(token)
    print(f"[PASS] sampler_chain: token={token}, text='{text}'")

    bridge.sampler_free(sampler)
    bridge.free()


# ── Asymmetric K/V Tests ─────────────────────────────────────────────


def test_asymmetric_kv_q80_turbo3():
    """K=q8_0, V=turbo3 — asimetrik strateji."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 41, -1, 0, 1)  # K=q8_0, V=turbo3
    assert ret == 0

    prompt = _format_prompt(bridge, "The meaning of life is")
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.7, seed=42)
    token = bridge.sampler_sample(sampler)
    text = bridge.token_to_piece(token)
    print(f"[PASS] asymmetric_kv: next='{text}'")

    bridge.sampler_free(sampler)
    bridge.free()


def test_asymmetric_kv_q80_turbo4():
    """K=q8_0, V=turbo4 — 4-bit V compression."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 42, -1, 0, 1)  # K=q8_0, V=turbo4
    assert ret == 0, f"turbo4 init failed: {ret}"

    prompt = _format_prompt(bridge, "Hello")
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.7, seed=42)
    token = bridge.sampler_sample(sampler)
    text = bridge.token_to_piece(token)
    print(f"[PASS] turbo4_kv: next='{text}'")

    bridge.sampler_free(sampler)
    bridge.free()


def test_asymmetric_kv_q80_turbo2():
    """K=q8_0, V=turbo2 — 2-bit V compression."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 43, -1, 0, 1)  # K=q8_0, V=turbo2
    assert ret == 0, f"turbo2 init failed: {ret}"

    prompt = _format_prompt(bridge, "Hello")
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.7, seed=42)
    token = bridge.sampler_sample(sampler)
    text = bridge.token_to_piece(token)
    print(f"[PASS] turbo2_kv: next='{text}'")

    bridge.sampler_free(sampler)
    bridge.free()


def test_baseline_q80_q80():
    """K=q8_0, V=q8_0 — baseline karsilastirma."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    ret = bridge.init(MODEL, 512, 8, 8, -1, 1, 1)  # K=q8_0, V=q8_0, flash_attn=1
    if ret != 0:
        print(
            f"[SKIP] baseline_q8_0: spiritbuun fork K=q8_0/V=q8_0 desteklemiyor (ret={ret})"
        )
        return

    prompt = _format_prompt(bridge, "The meaning of life is")
    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    assert ret == 0

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.7, seed=42)
    token = bridge.sampler_sample(sampler)
    text = bridge.token_to_piece(token)
    print(f"[PASS] baseline_q8_0: next='{text}'")

    bridge.sampler_free(sampler)
    bridge.free()


# ── Generation Tests ─────────────────────────────────────────────────


def test_low_temp_generation():
    """Dusuk temperature ile uretim (greedy degil, Qwen3 uyumlu)."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )
    prompt = ctx.format_user_prompt("Hello")
    text = ctx.generate(prompt, max_tokens=10, temperature=0.1, top_p=0.95, top_k=20)
    assert len(text) > 0, "generation empty"
    print(f"[PASS] low_temp_gen: '{text.strip()[:50]}'")
    ctx.bridge.free()


def test_sampling_generation():
    """Temperature sampling uretimi."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )
    prompt = ctx.format_user_prompt("Once upon a time")
    text = ctx.generate(
        prompt,
        max_tokens=15,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    )
    assert len(text) > 0, "sampling generation empty"
    print(f"[PASS] sampling_gen: '{text.strip()[:50]}'")
    ctx.bridge.free()


def test_chat_template_generation():
    """Chat template + sampling ile tam uretim."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )
    prompt = ctx.format_user_prompt("What is the capital of France?")
    text = ctx.generate(
        prompt,
        max_tokens=30,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    )
    assert len(text) > 0, "chat template generation empty"
    print(f"[PASS] chat_template_gen: '{text.strip()[:80]}'")
    ctx.bridge.free()


def test_multi_prompt():
    """Ardisik prompt'larla KV cache temizleme."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=1024, cache_type="turbo3", flash_attn=True
    )

    prompts = ["Hello", "Merhaba", "Python"]
    for p in prompts:
        ctx.clear_cache()
        prompt = ctx.format_user_prompt(p)
        out = ctx.generate(prompt, max_tokens=5, temperature=0.1, top_p=0.95, top_k=20)
        assert len(out) > 0, f"empty output for '{p}'"

    print(f"[PASS] multi_prompt: {len(prompts)} prompts OK")
    ctx.bridge.free()


def test_incremental_decode():
    """Her adimda tek token decode — temel generation loop."""
    from turbo import TurboBridge

    bridge = TurboBridge()
    bridge.init(MODEL, 1024, 8, 41, -1, 1, 1)

    prompt = _format_prompt(bridge, "Hello world")
    tokens = bridge.tokenize(prompt)
    bridge.decode(tokens, pos=0)

    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.7, seed=42)

    pos = len(tokens)
    generated = []
    for i in range(5):
        token = bridge.sampler_sample(sampler)
        text = bridge.token_to_piece(token)
        generated.append(text)

        ret = bridge.decode([token], pos=pos)
        assert ret == 0, f"incremental decode failed at step {i}"
        pos += 1

    bridge.sampler_free(sampler)

    result = "".join(generated)
    assert len(result) > 0
    print(f"[PASS] incremental_decode: '{result[:40]}'")
    bridge.free()


# ── Flash Attention Tests ────────────────────────────────────────────


def test_flash_attn_on():
    """Flash Attention acik."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=512, cache_type="turbo3", flash_attn=True
    )
    prompt = ctx.format_user_prompt("Hi")
    text = ctx.generate(prompt, max_tokens=5, temperature=0.1, top_p=0.95, top_k=20)
    assert len(text) > 0
    print(f"[PASS] flash_attn_on: '{text.strip()[:30]}'")
    ctx.bridge.free()


def test_flash_attn_off():
    """Flash Attention kapali."""
    from turbo import TurboContext

    ctx = TurboContext(
        model_path=MODEL, n_ctx=512, cache_type="turbo3", flash_attn=False
    )
    prompt = ctx.format_user_prompt("Hi")
    text = ctx.generate(prompt, max_tokens=5, temperature=0.1, top_p=0.95, top_k=20)
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
        ("chat_template_get", test_chat_template_get),
        ("chat_template_apply", test_chat_template_apply),
        ("sampler_chain", test_sampler_chain),
        ("asymmetric_kv_turbo3", test_asymmetric_kv_q80_turbo3),
        ("asymmetric_kv_turbo4", test_asymmetric_kv_q80_turbo4),
        ("asymmetric_kv_turbo2", test_asymmetric_kv_q80_turbo2),
        ("baseline_q8_0", test_baseline_q80_q80),
        ("low_temp_generation", test_low_temp_generation),
        ("sampling_generation", test_sampling_generation),
        ("chat_template_generation", test_chat_template_generation),
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
