"""
PROJECT-TURBO | 128K context inference with TurboQuant KV cache
Usage:
    LD_LIBRARY_PATH=./build/bin python examples/128k_inference.py --model model.gguf
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from turbo import TurboLlama, setup


def main():
    parser = argparse.ArgumentParser(description="PROJECT-TURBO 128K Context Inference")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument(
        "--n-ctx", type=int, default=131072, help="Context size (default: 128K)"
    )
    parser.add_argument(
        "--cache-type", choices=["turbo2", "turbo3", "turbo4"], default="turbo3"
    )
    parser.add_argument(
        "--prompt",
        default="Explain quantum entanglement in detail.",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max generation tokens"
    )
    parser.add_argument("--bridge", default=None, help="Path to turbo_bridge.so")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    # Verify bridge loads
    setup(custom_lib_path=args.bridge)

    print(f"\n[TURBO] Model: {args.model}")
    print(f"[TURBO] Context: {args.n_ctx:,} tokens")
    print(f"[TURBO] Cache: {args.cache_type}")
    print(f"[TURBO] Prompt: {args.prompt[:80]}...")
    print("-" * 72)

    t0 = time.time()
    llm = TurboLlama(
        model_path=args.model,
        cache_type=args.cache_type,
        n_ctx=args.n_ctx,
        n_gpu_layers=-1,
        flash_attn=True,
        bridge_path=args.bridge,
    )

    output = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        stop=["</s>", "\n\n\n"],
    )

    elapsed = time.time() - t0
    text = output["choices"][0]["text"]

    print("\n" + "=" * 72)
    print(f"[TURBO] Generated in {elapsed:.2f}s")
    print(f"[TURBO] Output:\n{text[:500]}{'...' if len(text) > 500 else ''}")
    print("=" * 72)


if __name__ == "__main__":
    main()
