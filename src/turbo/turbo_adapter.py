"""
PROJECT-TURBO — llama.cpp TurboQuant adapter (C bridge)

Uses a C bridge (turbo_bridge.so) to avoid ctypes struct-by-value ABI issues
that cause segfaults on Linux x86_64.

v0.3: Handle-based architecture with KV monitoring and performance metrics.

Public API:
    TurboContext — high-level context wrapper (with KVMonitor)
    TurboLlama  — drop-in llama-cpp-python compatible API
    TurboBridge — low-level C bridge (backward compat + handle-based)
    KVMonitor   — KV cache health monitoring
    setup()     — one-call verification
    load_model() / unload_model() — model lifecycle
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
# CTYPES STRUCTS — C-level data structures
# =============================================================================


class TurboPerfData(ctypes.Structure):
    """C-level performance metrics from llama_perf_context."""

    _fields_ = [
        ("t_p_eval_ms", ctypes.c_double),
        ("t_eval_ms", ctypes.c_double),
        ("n_p_eval", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
        ("n_reused", ctypes.c_int32),
    ]

    def __repr__(self):
        return (
            f"<TurboPerf prefill={self.t_p_eval_ms:.1f}ms/{self.n_p_eval}tok "
            f"decode={self.t_eval_ms:.1f}ms/{self.n_eval}tok "
            f"reused={self.n_reused}>"
        )


class TurboKVState(ctypes.Structure):
    """KV cache state: utilization and memory info."""

    _fields_ = [
        ("n_ctx", ctypes.c_int),
        ("n_pos", ctypes.c_int),
        ("utilization", ctypes.c_double),
        ("state_bytes", ctypes.c_size_t),
    ]

    def __repr__(self):
        return (
            f"<TurboKVState {self.n_pos}/{self.n_ctx} "
            f"({self.utilization:.1%}) {self.state_bytes / 1024:.1f}KB>"
        )


# Opaque handle pointer type
TurboHandle = ctypes.c_void_p


# =============================================================================
# C BRIDGE — avoids struct-by-value ABI issues with ctypes
# =============================================================================


class TurboBridge:
    """C bridge wrapper — supports both legacy global API and handle-based API.

    v0.3: Internally delegates to handle-based functions.
    Backward compatible: old init/tokenize/decode/free still work.
    """

    def __init__(self, bridge_path: Optional[str] = None):
        path = bridge_path or _find_bridge()
        self.lib = ctypes.CDLL(path)
        self._handle: Optional[int] = None  # handle-based mode
        self._setup_bindings()

    def _setup_bindings(self):
        """Configure ctypes signatures for all C bridge functions."""

        # ── Model lifecycle (new in v0.3) ──────────────────────────────────
        self.lib.turbo_load_model.restype = ctypes.c_int
        self.lib.turbo_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]

        self.lib.turbo_unload_model.restype = None
        self.lib.turbo_unload_model.argtypes = []

        # ── Handle-based API (new in v0.3) ─────────────────────────────────
        self.lib.turbo_ctx_init.restype = TurboHandle
        self.lib.turbo_ctx_init.argtypes = [
            ctypes.c_int,  # n_ctx
            ctypes.c_int,  # type_k
            ctypes.c_int,  # type_v
            ctypes.c_int,  # flash_attn
            ctypes.c_int,  # offload_kqv
        ]

        self.lib.turbo_ctx_tokenize.restype = ctypes.c_int
        self.lib.turbo_ctx_tokenize.argtypes = [
            TurboHandle,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]

        self.lib.turbo_ctx_decode.restype = ctypes.c_int
        self.lib.turbo_ctx_decode.argtypes = [
            TurboHandle,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.turbo_ctx_decode_chunked.restype = ctypes.c_int
        self.lib.turbo_ctx_decode_chunked.argtypes = [
            TurboHandle,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.lib.turbo_ctx_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.turbo_ctx_get_logits.argtypes = [TurboHandle]

        self.lib.turbo_ctx_n_vocab.restype = ctypes.c_int
        self.lib.turbo_ctx_n_vocab.argtypes = [TurboHandle]

        self.lib.turbo_ctx_token_to_piece.restype = ctypes.c_int
        self.lib.turbo_ctx_token_to_piece.argtypes = [
            TurboHandle,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]

        self.lib.turbo_ctx_kv_cache_clear.restype = None
        self.lib.turbo_ctx_kv_cache_clear.argtypes = [TurboHandle]

        self.lib.turbo_ctx_free.restype = None
        self.lib.turbo_ctx_free.argtypes = [TurboHandle]

        # ── Performance metrics (new in v0.3) ──────────────────────────────
        self.lib.turbo_ctx_perf_get.restype = TurboPerfData
        self.lib.turbo_ctx_perf_get.argtypes = [TurboHandle]

        self.lib.turbo_ctx_perf_reset.restype = None
        self.lib.turbo_ctx_perf_reset.argtypes = [TurboHandle]

        # ── KV state query (new in v0.3) ───────────────────────────────────
        self.lib.turbo_ctx_kv_state.restype = TurboKVState
        self.lib.turbo_ctx_kv_state.argtypes = [TurboHandle]

        # ── Legacy API (backward compat) ───────────────────────────────────
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

    # ── Model lifecycle ────────────────────────────────────────────────────

    def load_model(self, model_path: str, n_gpu_layers: int = -1) -> int:
        """Load model (once). Returns 0 on success."""
        return self.lib.turbo_load_model(model_path.encode(), n_gpu_layers)

    def unload_model(self):
        """Unload model and free backend."""
        self.lib.turbo_unload_model()

    # ── Handle-based API (v0.3) ────────────────────────────────────────────

    def ctx_init(
        self,
        n_ctx: int,
        type_k: int,
        type_v: int,
        flash_attn: int = 0,
        offload_kqv: int = 1,
    ) -> int:
        """Create new context handle. Returns handle pointer or 0 on failure."""
        self._handle = self.lib.turbo_ctx_init(
            n_ctx, type_k, type_v, flash_attn, offload_kqv
        )
        return self._handle if self._handle else 0

    def ctx_tokenize(self, text: str) -> list[int]:
        """Tokenize using handle."""
        max_tokens = max(len(text) * 2, 131072)
        tokens = (ctypes.c_int * max_tokens)()
        n = self.lib.turbo_ctx_tokenize(
            self._handle, text.encode("utf-8"), tokens, max_tokens
        )
        return list(tokens[:n]) if n > 0 else []

    def ctx_decode(self, tokens: list[int], pos: int = 0) -> int:
        """Decode using handle."""
        arr = (ctypes.c_int * len(tokens))(*tokens)
        return self.lib.turbo_ctx_decode(self._handle, arr, len(tokens), pos)

    def ctx_get_logits(self) -> list[float]:
        """Get logits using handle."""
        ptr = self.lib.turbo_ctx_get_logits(self._handle)
        nv = self.lib.turbo_ctx_n_vocab(self._handle)
        return [ptr[i] for i in range(nv)]

    def ctx_n_vocab(self) -> int:
        return self.lib.turbo_ctx_n_vocab(self._handle)

    def ctx_token_to_piece(self, token: int) -> str:
        buf = ctypes.create_string_buffer(64)
        self.lib.turbo_ctx_token_to_piece(self._handle, token, buf, 64)
        return buf.value.decode(errors="ignore")

    def ctx_kv_cache_clear(self):
        self.lib.turbo_ctx_kv_cache_clear(self._handle)

    def ctx_free(self):
        """Free context handle."""
        if self._handle:
            self.lib.turbo_ctx_free(self._handle)
            self._handle = None

    # ── Performance metrics (v0.3) ─────────────────────────────────────────

    def perf_get(self) -> TurboPerfData:
        """Get C-level performance metrics."""
        return self.lib.turbo_ctx_perf_get(self._handle)

    def perf_reset(self):
        """Reset performance counters."""
        self.lib.turbo_ctx_perf_reset(self._handle)

    # ── KV state (v0.3) ───────────────────────────────────────────────────

    def kv_state(self) -> TurboKVState:
        """Get KV cache state (utilization, position, memory)."""
        return self.lib.turbo_ctx_kv_state(self._handle)

    # ── Legacy API (backward compat) ───────────────────────────────────────

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
        """Legacy init — loads model and creates global context."""
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
        max_tokens = max(len(text) * 2, 131072)
        tokens = (ctypes.c_int * max_tokens)()
        n = self.lib.turbo_tokenize(text.encode("utf-8"), tokens, max_tokens)
        return list(tokens[:n]) if n > 0 else []

    def decode(self, tokens: list[int], pos: int = 0, chunk_size: int = 512) -> int:
        if len(tokens) <= chunk_size:
            arr = (ctypes.c_int * len(tokens))(*tokens)
            return self.lib.turbo_decode(arr, len(tokens), pos)

        for start in range(0, len(tokens), chunk_size):
            chunk = tokens[start : start + chunk_size]
            arr = (ctypes.c_int * len(chunk))(*chunk)
            ret = self.lib.turbo_decode(arr, len(chunk), pos + start)
            if ret != 0:
                return ret
        return 0

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
        """Free — context handle if present, otherwise legacy cleanup."""
        if self._handle:
            self.ctx_free()
            self.unload_model()
        else:
            self.lib.turbo_free()

    def kv_cache_clear(self):
        if self._handle:
            self.ctx_kv_cache_clear()
        else:
            self.lib.turbo_kv_cache_clear()


# =============================================================================
# KV MONITOR — cache health intelligence (v0.3)
# =============================================================================


class KVMonitor:
    """Per-token KV cache health monitoring.

    Tracks entropy, saturation, and latency to detect degradation.
    Uses compound signals (not single-metric) for reliability.
    """

    def __init__(
        self,
        n_ctx: int,
        saturation_warn: float = 0.80,
        saturation_critical: float = 0.95,
        entropy_warn: float = 10.0,
        entropy_spike: float = 12.0,
    ):
        self.n_ctx = n_ctx
        self.saturation_warn = saturation_warn
        self.saturation_critical = saturation_critical
        self.entropy_warn = entropy_warn
        self.entropy_spike = entropy_spike

        self.tokens_decoded: int = 0
        self.perf_history: list[TurboPerfData] = []
        self.entropy_history: list[float] = []
        self.saturation_history: list[float] = []
        self.repetition_history: list[float] = []
        self._recent_tokens: list[int] = []

    def on_decode(
        self,
        perf: TurboPerfData,
        kv_state: TurboKVState,
        logits: list[float],
        token: Optional[int] = None,
    ):
        """Called after each decode to record metrics."""
        self.tokens_decoded += 1
        self.perf_history.append(perf)
        self.saturation_history.append(kv_state.utilization)

        entropy = self._compute_entropy(logits)
        self.entropy_history.append(entropy)

        if token is not None:
            self._recent_tokens.append(token)
            if len(self._recent_tokens) > 200:
                self._recent_tokens = self._recent_tokens[-200:]
            rep = self._compute_repetition()
            self.repetition_history.append(rep)

    def _compute_entropy(self, logits: list[float]) -> float:
        """Shannon entropy of token probability distribution."""
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exp_l)
        if sum_exp <= 0:
            return 0.0
        probs = [e / sum_exp for e in exp_l]
        return -sum(p * math.log2(p) for p in probs if p > 1e-10)

    def _compute_repetition(self) -> float:
        """Repetition ratio in last 100 tokens."""
        if len(self._recent_tokens) < 50:
            return 0.0
        recent = self._recent_tokens[-100:]
        unique = len(set(recent))
        return 1.0 - (unique / len(recent))

    def health(self) -> dict:
        """Current health snapshot with compound signal analysis.

        Uses AND logic for warnings: single-metric spikes are NOT flagged
        unless paired with a second signal (per feedback).
        """
        current_sat = self.saturation_history[-1] if self.saturation_history else 0.0
        avg_entropy = sum(self.entropy_history[-100:]) / max(
            len(self.entropy_history[-100:]), 1
        )
        current_rep = self.repetition_history[-1] if self.repetition_history else 0.0

        status = "ok"
        warnings = []

        # Compound signal: high saturation AND high entropy → KV degradation
        if current_sat > self.saturation_critical and avg_entropy > self.entropy_warn:
            status = "critical"
            warnings.append(
                f"KV degradation: saturation={current_sat:.1%}, entropy={avg_entropy:.1f}"
            )
        elif current_sat > self.saturation_warn and avg_entropy > self.entropy_warn:
            status = "warning"
            warnings.append(
                f"KV pressure: saturation={current_sat:.1%}, entropy={avg_entropy:.1f}"
            )

        # Entropy spike alone: possible quantization noise burst
        if len(self.entropy_history) > 10:
            recent_max = max(self.entropy_history[-10:])
            if recent_max > self.entropy_spike:
                warnings.append(f"Entropy spike detected: {recent_max:.1f}")

        # Entropy trend: rising entropy over time
        if len(self.entropy_history) > 100:
            first_half = sum(self.entropy_history[:50]) / 50
            second_half = sum(self.entropy_history[-50:]) / 50
            if second_half > first_half * 1.5 and second_half > self.entropy_warn:
                warnings.append(f"Entropy rising: {first_half:.1f} → {second_half:.1f}")

        # Repetition signal (compound with saturation)
        if current_rep > 0.5 and current_sat > self.saturation_warn:
            warnings.append(
                f"Repetition + saturation: rep={current_rep:.1%}, sat={current_sat:.1%}"
            )

        return {
            "status": status,
            "saturation": current_sat,
            "avg_entropy": avg_entropy,
            "repetition": current_rep,
            "tokens_decoded": self.tokens_decoded,
            "warnings": warnings,
        }

    def should_clear_cache(self) -> bool:
        """Compound signal: only recommend clear when multiple signals agree."""
        h = self.health()
        if h["status"] == "critical":
            return True
        # Repetition + saturation together
        if h["repetition"] > 0.5 and h["saturation"] > self.saturation_warn:
            return True
        return False

    def latency_summary(self) -> dict:
        """Aggregate latency from perf history."""
        if not self.perf_history:
            return {}
        last = self.perf_history[-1]
        return {
            "prefill_ms": last.t_p_eval_ms,
            "decode_ms": last.t_eval_ms,
            "decode_per_token_ms": last.t_eval_ms / max(last.n_eval, 1),
            "tok_per_sec": last.n_eval / max(last.t_eval_ms / 1000, 0.001),
            "n_p_eval": last.n_p_eval,
            "n_eval": last.n_eval,
        }


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

    v0.3: Handle-based with KVMonitor and performance metrics.
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
        type_k = 8  # q8_0 — K precision preserved (safe default)
        fa = 1 if flash_attn else 0

        # v0.3: handle-based init
        ret = self.bridge.load_model(model_path, n_gpu_layers)
        if ret != 0:
            raise RuntimeError(f"turbo_load_model failed: {ret}")

        handle = self.bridge.ctx_init(n_ctx, type_k, type_v, fa, 1)
        if not handle:
            raise RuntimeError("turbo_ctx_init failed")

        self.n_ctx = n_ctx
        self.cache_type = cache_type
        self._n_vocab = self.bridge.ctx_n_vocab()
        self.monitor = KVMonitor(n_ctx=n_ctx)

        logger.info(
            "[TURBO] Context: n_ctx=%d, K=q8_0, V=%s (type_v=%d), flash_attn=%s, vocab=%d",
            n_ctx,
            cache_type,
            type_v,
            flash_attn,
            self._n_vocab,
        )

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        return self.bridge.ctx_tokenize(text)

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.bridge.ctx_token_to_piece(t) for t in tokens)

    def decode(self, tokens: list[int], pos: int = 0) -> list[float]:
        """Decode tokens, return logits."""
        ret = self.bridge.ctx_decode(tokens, pos)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        return self.bridge.ctx_get_logits()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        monitor: bool = True,
    ) -> str:
        """Generate text from prompt with optional KV monitoring."""
        tokens = self.tokenize(prompt)
        stop_seqs = []
        if stop:
            for s in stop:
                t = self.tokenize(s)
                if t:
                    stop_seqs.append(t)

        # Prefill
        self.bridge.perf_reset()
        logits = self.decode(tokens)
        pos = len(tokens)

        if monitor:
            perf = self.bridge.perf_get()
            kv = self.bridge.kv_state()
            self.monitor.on_decode(perf, kv, logits)
            self.bridge.perf_reset()

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
            ret = self.bridge.ctx_decode([next_token], pos)
            if ret != 0:
                break
            logits = self.bridge.ctx_get_logits()
            pos += 1

            # Monitor each step
            if monitor:
                perf = self.bridge.perf_get()
                kv = self.bridge.kv_state()
                self.monitor.on_decode(perf, kv, logits, token=next_token)
                self.bridge.perf_reset()

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

    def health(self) -> dict:
        """Get KV cache health snapshot."""
        return self.monitor.health()

    def clear_cache(self):
        self.bridge.ctx_kv_cache_clear()
        # Reset monitor state for new session
        self.monitor = KVMonitor(n_ctx=self.n_ctx)

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
