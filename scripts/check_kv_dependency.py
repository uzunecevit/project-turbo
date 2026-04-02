"""
PROJECT-TURBO | KV Dependency Analyzer v0.8

Modelin KV cache'e ne kadar bağımlı olduğunu ölçer.
TurboQuant uygulanabilirliğini otomatik belirler.

Usage:
    LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python scripts/check_kv_dependency.py /path/to/model.gguf

Optional:
    --ctx 2048          Context length (default: 2048)
    --n-repeats 5       Retention test repeat count (default: 5)
"""

import os
import sys
import argparse
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turbo import TurboBridge


def get_model_info(bridge):
    """Read model metadata."""
    info = bridge.model_info()
    return info


def measure_kv_memory(bridge, n_ctx=2048, type_k=8, type_v=8):
    """Measure KV cache memory for given config. Decode enough tokens to fill cache."""
    handle = bridge.ctx_init(n_ctx, type_k, type_v, 1, 1)
    if not handle:
        return 0

    bridge._handle = handle
    # Decode a longer prompt to populate KV cache
    prompt = bridge.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "Write a detailed essay about artificial intelligence.",
            },
        ],
        add_ass=True,
    )
    tokens = bridge.tokenize(prompt)
    bridge.decode(tokens, pos=0)

    # Decode additional tokens to fill cache
    sampler = bridge.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.6, seed=42)
    pos = len(tokens)
    for _ in range(min(256, n_ctx - pos - 10)):
        token = bridge.sampler_sample(sampler)
        if token < 0:
            break
        ret = bridge.decode([token], pos=pos)
        if ret != 0:
            break
        pos += 1
    bridge.sampler_free(sampler)

    kv = bridge.kv_state()
    memory = kv.state_bytes
    bridge.free()
    return memory


def measure_kv_memory_per_layer(bridge, n_ctx=2048, type_k=8, type_v=8):
    """Estimate per-layer KV memory by dividing total by KV layer count."""
    total_memory = measure_kv_memory(bridge, n_ctx, type_k, type_v)
    n_layer = bridge.model_n_layer()
    n_head_kv = bridge.model_n_head_kv()
    return total_memory, n_layer, n_head_kv


def quick_retention_test(bridge, n_repeats=5):
    """Quick key retention test. Returns 0.0-1.0."""
    key_value = 99123
    key_line = f"The key is {key_value}.\n"
    prompt_raw = (
        key_line * n_repeats
    ) + f"What is the key? Answer with just the number."

    try:
        prompt = bridge.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_raw},
            ],
            add_ass=True,
        )
    except Exception:
        prompt = prompt_raw

    tokens = bridge.tokenize(prompt)
    ret = bridge.decode(tokens, pos=0)
    if ret != 0:
        return 0.0

    sampler = bridge.sampler_init(
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        temp=0.7,
        seed=42,
        penalty_last_n=-1,
        penalty_repeat=1.0,
        penalty_freq=0.0,
        penalty_present=1.5,
    )

    pos = len(tokens)
    out = []
    for _ in range(20):
        token = bridge.sampler_sample(sampler)
        if token < 0:
            break
        text = bridge.token_to_piece(token)
        out.append(text)
        ret = bridge.decode([token], pos=pos)
        if ret != 0:
            break
        pos += 1

    bridge.sampler_free(sampler)

    result = "".join(out)
    # Strip ChatML tags
    im_start = chr(60) + "|im_start|"
    im_end = chr(60) + "|im_end|"
    cleaned = re.sub(r"(?si)" + re.escape(im_start) + r"[^<]*?", "", result)
    cleaned = re.sub(r"(?si)" + re.escape(im_end), "", cleaned)
    cleaned = re.sub(r"(?m)^(system|user|assistant)\s*\n?", "", cleaned)

    # Look for the key
    match = re.search(rf"(?<!\d)({key_value})(?!\d)", cleaned)
    return 1.0 if match else 0.0


def score_to_recommendation(score):
    """Convert 0-100 score to recommendation string."""
    if score >= 80:
        return "TURBO_FULL", "turbo3/turbo4 uygulanabilir"
    elif score >= 50:
        return "TURBO_SAFE", "Sadece turbo4, dikkatli uygulanabilir"
    elif score >= 30:
        return "TURBO_MINIMAL", "Sadece throughput testi, kalite testi atlanabilir"
    else:
        return "TURBO_SKIP", "TurboQuant uygulama, sonuç anlamsız olur"


def analyze(model_path, n_ctx=2048, n_repeats=5):
    """Full KV dependency analysis."""
    print(f"Model: {model_path}")
    print(f"Context: {n_ctx}")
    print(f"{'=' * 60}")

    bridge = TurboBridge()
    bridge.lib.turbo_load_model(model_path.encode(), -1)

    # Step 1: Model Info
    print("\n[1] Model Metadata")
    print("-" * 40)
    info = get_model_info(bridge)
    print(f"  Architecture:    {info['arch']}")
    print(f"  Description:     {info['desc']}")
    print(f"  Layers:          {info['n_layer']}")
    print(f"  Attention Heads: {info['n_head']}")
    print(f"  KV Heads:        {info['n_head_kv']}")
    print(f"  Embedding Dim:   {info['n_embd']}")
    print(f"  Train Context:   {info['n_ctx_train']}")

    # Step 2: KV Memory Measurement
    print("\n[2] KV Cache Memory")
    print("-" * 40)

    configs = [
        ("q8_0/q8_0", 8, 8),
        ("q8_0/turbo4", 8, 42),
        ("q8_0/turbo3", 8, 41),
        ("turbo4/turbo4", 42, 42),
        ("turbo3/turbo3", 41, 41),
    ]

    kv_memories = {}
    for name, type_k, type_v in configs:
        mem = measure_kv_memory(bridge, n_ctx, type_k, type_v)
        kv_memories[name] = mem
        mem_mb = mem / (1024 * 1024)
        print(f"  {name:>15}: {mem_mb:>8.1f} MB")

    # Savings calculation
    baseline = kv_memories.get("q8_0/q8_0", 1)
    for name, _, _ in configs[1:]:
        if baseline > 0:
            savings = (1 - kv_memories[name] / baseline) * 100
            print(f"  {'savings':>15}: {savings:>7.1f}% vs q8_0/q8_0")

    # Step 3: KV Layer Ratio
    print("\n[3] KV Layer Analysis")
    print("-" * 40)
    n_head = info["n_head"]
    n_head_kv = info["n_head_kv"]
    if n_head > 0:
        kv_ratio = n_head_kv / n_head
        print(f"  KV Head Ratio:   {n_head_kv}/{n_head} = {kv_ratio:.2%}")
    else:
        kv_ratio = 0
        print(f"  KV Head Ratio:   N/A (no attention heads)")

    n_layer = info["n_layer"]
    # Detect architecture type from metadata
    arch = info["arch"].lower()
    # Check for GDN/hybrid: KV memory is much smaller than expected for dense model
    baseline_mem = kv_memories.get("q8_0/q8_0", 0)
    if n_head_kv > 0 and n_layer > 0 and baseline_mem > 0:
        d_head = info["n_embd"] // n_head if n_head > 0 else 128
        expected_dense = (
            n_layer * n_head_kv * d_head * 2 * n_ctx * 1
        )  # q8_0, all layers
        kv_layer_ratio = baseline_mem / expected_dense if expected_dense > 0 else 0
        kv_layers = max(1, int(kv_layer_ratio * n_layer))
        if kv_layer_ratio < 0.5:
            print(f"  KV Layers:       {kv_layers}/{n_layer} (HYBRID/GDN detected)")
            print(f"  Memory ratio:    {kv_layer_ratio:.1%} of expected dense model")
        else:
            kv_layers = n_layer
            print(f"  KV Layers:       {kv_layers}/{n_layer} (dense)")
    else:
        kv_layers = n_layer
        print(f"  KV Layers:       {kv_layers}/{n_layer} (unknown, assuming dense)")

    # Step 4: Quick Retention Test
    print("\n[4] Quick Retention Test (q8_0/q8_0)")
    print("-" * 40)
    retention = quick_retention_test(bridge, n_repeats)
    print(f"  Retention Rate: {retention:.0%}")

    # Step 5: KV Dependency Score
    print("\n[5] KV Dependency Score")
    print("-" * 40)

    # Layer ratio component (0-100)
    layer_score = (kv_layers / n_layer) * 100 if n_layer > 0 else 0

    # Retention component (0-100)
    retention_score = retention * 100

    # Memory ratio component (0-100)
    # Dense model at q8_0/q8_0 for 2048 ctx ≈ n_layer * n_head_kv * d_head * 2 * n_ctx * 1 byte
    # If actual memory is close to expected, model is KV-dominant
    if n_head_kv > 0 and n_layer > 0:
        d_head = info["n_embd"] // n_head if n_head > 0 else 128
        expected_per_layer = n_head_kv * d_head * 2 * n_ctx * 1  # q8_0 = 1 byte
        expected_total = kv_layers * expected_per_layer
        if expected_total > 0:
            memory_ratio = (
                min(kv_memories.get("q8_0/q8_0", 0) / expected_total, 1.0) * 100
            )
        else:
            memory_ratio = 0
    else:
        memory_ratio = 0

    # Weighted score
    score = 0.4 * layer_score + 0.3 * retention_score + 0.3 * memory_ratio
    score = int(score)
    rec_code, rec_text = score_to_recommendation(score)

    print(
        f"  Layer Component:  {layer_score:.0f}/100 ({kv_layers}/{n_layer} KV layers)"
    )
    print(f"  Retention:        {retention_score:.0f}/100")
    print(f"  Memory Ratio:     {memory_ratio:.0f}/100")
    print(f"  ─────────────────────────")
    print(f"  TOTAL SCORE:      {score}/100")
    print(f"  Recommendation:   {rec_code}")
    print(f"                    {rec_text}")

    # Step 6: Optimal Config Estimation
    print("\n[6] Estimated Optimal Configs")
    print("-" * 40)

    if score >= 80:
        print("  Aggressive:  turbo3/turbo3 (riskli ama max tasarruf)")
        print("  Balanced:    turbo4/turbo4 (önerilen)")
        print("  Safe:        q8_0/turbo4")
    elif score >= 50:
        print("  Balanced:    q8_0/turbo4 (önerilen)")
        print("  Safe:        q8_0/q8_0")
    elif score >= 30:
        print("  Throughput:  turbo4/turbo4 (sadece hız testi)")
        print("  Quality:     q8_0/q8_0 (kalite testi)")
    else:
        print("  ❌ Bu modelde TurboQuant anlamsız")
        print("  Önerilen: Dense transformer modelleri kullan (LLaMA, Mistral)")

    print(f"\n{'=' * 60}")
    print(f"KV DEPENDENCY SCORE: {score}/100 → {rec_code}")
    print(f"{'=' * 60}")

    return {
        "model": os.path.basename(model_path),
        "arch": arch,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "kv_layers": kv_layers,
        "kv_ratio": kv_ratio,
        "retention": retention,
        "score": score,
        "recommendation": rec_code,
        "kv_memories": kv_memories,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Dependency Analyzer")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument(
        "--n-repeats", type=int, default=5, help="Retention test repeats"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    analyze(args.model, args.ctx, args.n_repeats)
