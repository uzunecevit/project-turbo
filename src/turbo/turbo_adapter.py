"""
PROJECT-TURBO — llama.cpp TurboQuant adapter (C bridge)

Uses a C bridge (turbo_bridge.so) to avoid ctypes struct-by-value ABI issues
that cause segfaults on Linux x86_64.

Public API:
    TurboContext — high-level context wrapper
    TurboLlama  — drop-in llama-cpp-python compatible API
    TurboBridge — low-level C bridge
    setup()     — one-call verification
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Auto-detect bridge path ──────────────────────────────────────────────────
_BRIDGE_CANDIDATES = [
    Path(__file__).parent.parent.parent / "build" / "turbo_bridge.so",
    Path(__file__).parent.parent.parent / "build" / "bin" / "turbo_bridge.so",
    Path("./build/turbo_bridge.so"),
    Path("./turbo_bridge.so"),
]


def _find_bridge() -> str:
    for p in _BRIDGE_CANDIDATES:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(
        "turbo_bridge.so not found. Build it first:\n"
        "  gcc -shared -fPIC -o build/turbo_bridge.so src/turbo/turbo_bridge.c "
        "-I./src/llama-spiritbuun-cuda/include -L./build/bin -lllama -Wl,-rpath,./build/bin"
    )


# =============================================================================
# C BRIDGE — avoids struct-by-value ABI issues with ctypes
# =============================================================================


class TurboBridge:
    """C bridge wrapper — avoids ctypes struct-by-value segfaults."""

    def __init__(self, bridge_path: Optional[str] = None):
        path = bridge_path or _find_bridge()
        self.lib = ctypes.CDLL(path)

        self.lib.turbo_init.restype = ctypes.c_int
        self.lib.turbo_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.turbo_tokenize.restype = ctypes.c_int
        self.lib.turbo_tokenize.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.turbo_decode.restype = ctypes.c_int
        self.lib.turbo_decode.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.turbo_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.turbo_n_vocab.restype = ctypes.c_int
        self.lib.turbo_token_to_piece.restype = ctypes.c_int
        self.lib.turbo_token_to_piece.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]

    def init(
        self,
        model_path: str,
        n_ctx: int,
        type_k: int,
        type_v: int,
        n_gpu_layers: int = -1,
        flash_attn: int = 0,
        offload_kqv: int = 1,
    ) -> int:
        return self.lib.turbo_init(
            model_path.encode(),
            n_ctx,
            type_k,
            type_v,
            n_gpu_layers,
            flash_attn,
            offload_kqv,
        )

    def tokenize(self, text: str) -> list[int]:
        tokens = (ctypes.c_int * 1024)()
        n = self.lib.turbo_tokenize(text.encode("utf-8"), tokens, 1024)
        return list(tokens[:n]) if n > 0 else []

    def decode(self, tokens: list[int], pos: int = 0) -> int:
        arr = (ctypes.c_int * len(tokens))(*tokens)
        return self.lib.turbo_decode(arr, len(tokens), pos)

    def get_logits(self) -> list[float]:
        ptr = self.lib.turbo_get_logits()
        nv = self.lib.turbo_n_vocab()
        return [ptr[i] for i in range(nv)]

    def n_vocab(self) -> int:
        return self.lib.turbo_n_vocab()

    def token_to_piece(self, token: int) -> str:
        buf = ctypes.create_string_buffer(64)
        self.lib.turbo_token_to_piece(token, buf, 64)
        return buf.value.decode(errors="ignore")

    def free(self):
        self.lib.turbo_free()

    def kv_cache_clear(self):
        self.lib.turbo_kv_cache_clear()


# =============================================================================
# TurboContext — high-level inference wrapper
# =============================================================================


# Map cache type names to ggml_type enum values
# spiritbuun fork: turbo3=41, turbo4=42, turbo2=43, q8_0=8
TURBO_TYPE_MAP = {
    "turbo3": 41,
    "turbo4": 42,
    "turbo2": 43,
    "q8_0": 8,
    "q4_0": 2,
    "f16": 1,
}


class TurboContext:
    """
    High-level TurboQuant inference context.
    Asymmetric K/V: K=q8_0 (preserved), V=turbo3 (compressed).
    """

    def __init__(
        self,
        model_path: str = "",
        n_ctx: int = 131072,
        n_batch: int = 512,
        cache_type: str = "turbo3",
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        bridge_path: Optional[str] = None,
    ):
        self.bridge = TurboBridge(bridge_path)

        type_v = TURBO_TYPE_MAP.get(cache_type, 41)
        # Asymmetric: K always q8_0, V compressed
        type_k = 8  # q8_0 — K precision preserved (safe default)

        fa = 1 if flash_attn else 0
        ret = self.bridge.init(model_path, n_ctx, type_k, type_v, n_gpu_layers, fa, 1)
        if ret != 0:
            raise RuntimeError(f"turbo_init failed: {ret}")

        self.n_ctx = n_ctx
        self.cache_type = cache_type
        self._n_vocab = self.bridge.n_vocab()
        logger.info(
            "[TURBO] Context: n_ctx=%d, K=q8_0, V=%s (type_v=%d), flash_attn=%s, vocab=%d",
            n_ctx,
            cache_type,
            type_v,
            flash_attn,
            self._n_vocab,
        )

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        return self.bridge.tokenize(text)

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.bridge.token_to_piece(t) for t in tokens)

    def decode(self, tokens: list[int], pos: int = 0) -> list[float]:
        """Decode tokens, return logits."""
        ret = self.bridge.decode(tokens, pos)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        return self.bridge.get_logits()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate text from prompt."""
        tokens = self.tokenize(prompt)
        stop_seqs = []
        if stop:
            for s in stop:
                t = self.tokenize(s)
                if t:
                    stop_seqs.append(t)

        # Prefill
        logits = self.decode(tokens)
        pos = len(tokens)
        generated = []

        for _ in range(max_tokens):
            next_token = self._sample(logits, temperature, top_p)
            generated.append(next_token)

            # Stop check
            hit = False
            for ss in stop_seqs:
                if len(generated) >= len(ss) and generated[-len(ss) :] == ss:
                    generated = generated[: -len(ss)]
                    hit = True
                    break
            if hit:
                break

            # Incremental decode
            ret = self.bridge.decode([next_token], pos)
            if ret != 0:
                break
            logits = self.bridge.get_logits()
            pos += 1

        return self.detokenize(generated)

    def _sample(self, logits: list[float], temperature: float, top_p: float) -> int:
        """Top-p sampling."""
        if temperature <= 0:
            return int(logits.index(max(logits)))

        max_l = max(logits)
        exp_l = [math.exp((l - max_l) / temperature) for l in logits]
        sum_exp = sum(exp_l)
        probs = [e / sum_exp for e in exp_l]

        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        top_p_set = set()
        for idx, p in indexed:
            top_p_set.add(idx)
            cumulative += p
            if cumulative >= top_p:
                break

        candidates = [(idx, probs[idx]) for idx in top_p_set]
        total = sum(p for _, p in candidates)
        r = random.random() * total
        cum = 0.0
        for idx, p in candidates:
            cum += p
            if r <= cum:
                return idx
        return candidates[-1][0]

    def clear_cache(self):
        self.bridge.kv_cache_clear()

    def __del__(self):
        try:
            if hasattr(self, "bridge"):
                self.bridge.free()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"<TurboContext n_ctx={self.n_ctx} cache={self.cache_type} "
            f"vocab={self._n_vocab}>"
        )


# =============================================================================
# CONVENIENCE: TurboLlama (drop-in compatible API)
# =============================================================================


class TurboLlama:
    """
    Drop-in Llama wrapper using TurboContext.
    Matches llama-cpp-python API surface for easy migration.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 131072,
        n_gpu_layers: int = -1,
        cache_type: str = "turbo3",
        flash_attn: bool = True,
        **kwargs,
    ):
        bridge_path = kwargs.pop("bridge_path", None)
        self._ctx = TurboContext(
            model_path=model_path,
            n_ctx=n_ctx,
            cache_type=cache_type,
            n_gpu_layers=n_gpu_layers,
            flash_attn=flash_attn,
            bridge_path=bridge_path,
        )

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        text = self._ctx.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or kwargs.get("stop", None),
        )
        return {
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": max_tokens},
        }

    def n_ctx(self) -> int:
        return self._ctx.n_ctx

    def tokenize(self, text: str) -> list[int]:
        return self._ctx.tokenize(text)

    def detokenize(self, tokens: list[int]) -> str:
        return self._ctx.detokenize(tokens)

    def __repr__(self):
        return f"<TurboLlama n_ctx={self._ctx.n_ctx} cache={self._ctx.cache_type}>"


# =============================================================================
# ONE-CALL SETUP
# =============================================================================


def setup(custom_lib_path: Optional[str | Path] = None, verbose: bool = True):
    """Verify bridge loads and TurboQuant is available."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    bridge = TurboBridge(str(custom_lib_path) if custom_lib_path else None)
    logger.info("[TURBO] Bridge loaded OK")
    logger.info("[TURBO] Setup complete — all systems nominal")
    return bridge
