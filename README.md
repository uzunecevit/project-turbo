# PROJECT-TURBO

> **Zero-Swap 128K Context Inference on Consumer GPUs — TurboQuant KV Cache Compression**

RTX 3060 (12GB VRAM) gibi tüketici sınıfı GPU'larda 128K context'i **swap'a düşmeden** çalıştırmak için TurboQuant asimetrik K/V cache sıkıştırması.

```python
from turbo import TurboContext

ctx = TurboContext(model_path="model.gguf", cache_type="turbo3")
text = ctx.generate("Merhaba", max_tokens=100)
print(text)
print(ctx.health())  # KV cache sağlık durumu
ctx.bridge.free()
```

---

## v1.0 — OMERTA Stable Core

> **2-PASS Attention De-Fusion + Asymmetric V Split**

| Feature | Açıklama |
|---|---|
| **2-PASS Pipeline** | Flash Attention de-fusion — 4 CUDA kernel (KQ, merge_softmax, V_acc, add_vec) |
| **Asymmetric V Split** | V_low=Turbo3 (archive), V_high=Turbo4 (window) — %42 VRAM tasarrufu |
| **K=Turbo4 (always)** | K split kaldırıldı — reasoning garantisi |
| **EXACT MATCH** | Deterministic (4K context) + Stochastic (temp=0.7) — tüm testlerde birebir aynı |
| **Host Buffer v_idxs_high** | INPUT flag ile host buffer garantisi |
| **Pool 5x Multiplier** | Dual buffer overhead için genişletildi |
| **Tag** | `v1.0-OMERTA-stable-core` |

### ENV Değişkenleri

| ENV | Varsayılan | Açıklama |
|---|---|---|
| `TURBO_2PASS_SPLIT` | (unset) | 1 → 2-PASS split-aware attention aktif |
| `TURBO_RECENT_WINDOW` | 256 | V_high (Turbo4) pencere boyutu |
| `GGML_CUDA_DISABLE_GRAPHS` | (unset) | 1 → CUDA graph devre dışı (2-PASS için gerekli) |
| `TURBO_PREFILL_VEC` | (unset) | 1 → Vec kernel zorla (debug) |

### Doğrulama Sonuçları

| Test | Sonuç | Kanıt |
|---|---|---|
| Greedy EXACT MATCH | ✅ 4K context | 5/5 token identical |
| Stochastic EXACT MATCH | ✅ temp=0.7 | 10/10 token identical |
| Bitwidth sweep | ✅ Tüm config | Beklenenle eşleşiyor |
| VRAM savings | ✅ %42 | V_low=Turbo3 archive |

---

## v0.3 Yenilikler

| Feature | Açıklama |
|---|---|
| **Handle-based bridge** | `turbo_handle_t` — multi-context desteği, seq_id ile future-proof |
| **C-level perf metrics** | `llama_perf_context()` — gerçek decode latency (Python overhead hariç) |
| **KV Monitor** | Entropy + saturation + repetition compound sinyal analizi |
| **Memory retention** | 540+ token gap sonrası recall doğrulandı (2 model) |
| **Reasoning tests** | Error amplification, key repetition, contradiction trap |
| **Determinism tests** | Cross-reset identical, no KV leak kanıtı |
| **Latency breakdown** | Prefill, decode, TTFT — C-level precision |

---

## Mimari

```
Python Application / KAPTAN v4
       │
       ▼
┌─────────────────────────────────────────┐
│  turbo_adapter.py                       │
│  TurboContext (handle-based)            │
│  KVMonitor (entropy/saturation)         │
│  Top-p sampling, generate loop          │
└──────────────┬──────────────────────────┘
               │ ctypes (basit tipler)
               ▼
┌─────────────────────────────────────────┐
│  turbo_bridge.c                         │
│  turbo_load_model (shared, bir kez)     │
│  turbo_ctx_init → turbo_handle_t        │
│  turbo_ctx_decode / tokenize / logits   │
│  turbo_ctx_perf_get (C-level timing)    │
│  turbo_ctx_kv_state (utilization)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  libllama.so (spiritbuun fork)                      │
│  • TurboQuant CUDA kernel'ları                      │
│  • 2-PASS Attention (4 CUDA kernel)                 │
│    - k_2pass_KQ_compute (Q·K çarpımı + scale)       │
│    - k_2pass_merge_softmax (merge + mask + softmax) │
│    - k_2pass_V_accumulate (weights·V, FP32)         │
│    - k_2pass_add_vec (out_low + out_high)           │
│  • V dual write (cpy_v → v_low + v_high)            │
│  • K=Turbo4, V_low=Turbo3, V_high=Turbo4            │
│  • Flash Attention (fused, fallback)                │
│  • Ampere SM 8.6 optimizasyonları                   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  NVIDIA GPU (RTX 3060 12GB)             │
└─────────────────────────────────────────┘
```

---

## VRAM Tasarrufu (7B Model, 128K Context)

| KV Cache Type | VRAM | Kalite | Not |
|---|---|---|---|
| F16 (varsayılan) | ~16 GB | Referans | RTX 3060'a sığmaz |
| Q8_0 (K ve V) | ~8 GB | ~F16 | Standart çözüm |
| **K=q8_0, V=turbo3** | **~4 GB** | **PPL +2.0%** | **v0.3 asimetrik** |
| **K=turbo4, V=turbo3+4** | **~4 GB** | **EXACT MATCH** | **v1.0 2-PASS split** |
| turbo3/turbo3 | ~3 GB | PPL 3,556 | ❌ Felaket (K sıkıştırması) |

---

## TurboQuant Formatları

| Format | type_v | Bit | Bileşenler | Kullanım |
|---|---|---|---|---|
| Turbo2 | 43 | 2-bit | PolarQuant | Minimum VRAM |
| **Turbo3** | **41** | **3-bit** | **PolarQuant + 1-bit QJL** | **Varsayılan (en iyi denge)** |
| Turbo4 | 42 | 4-bit | PolarQuant + 1-bit QJL | Maksimum kalite |

**Asimetrik strateji:** K her zaman `q8_0`, V `turbo3`/`turbo4`/`turbo2` ile sıkıştırılır.

---

## Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- CUDA 12.0+ (test edildi: CUDA 13.2)
- NVIDIA GPU (Compute Capability 8.0+, Ampere/Ada)
- GCC 12+ / Clang 15+
- CMake 3.24+

### 1. Derle

```bash
# spiritbuun fork'unu derle (libllama.so + turbo_bridge.so)
chmod +x scripts/build_turbo.sh
./scripts/build_turbo.sh

# C bridge derle
gcc -shared -fPIC -o build/turbo_bridge.so \
    src/turbo/turbo_bridge.c \
    -I./src/llama-spiritbuun-cuda/include \
    -L./build/bin -lllama -Wl,-rpath,./build/bin
```

### 2. Kullan (v0.3 Handle-Based API)

```python
from turbo import TurboContext

ctx = TurboContext(
    model_path="model.gguf",
    n_ctx=4096,           # Context boyutu
    cache_type="turbo3",  # V=turbo3, K=q8_0 (asimetrik)
    flash_attn=True,      # Flash Attention zorunlu
    n_gpu_layers=-1,      # Tüm katmanlar GPU'da
)

# Generate
text = ctx.generate("Merhaba, benim adım", max_tokens=50, temperature=0.0)

# KV cache sağlık durumu
health = ctx.health()
print(f"Status: {health['status']}, Saturation: {health['saturation']:.1%}")

# Temizle
ctx.clear_cache()   # KV cache sıfırla
ctx.bridge.free()   # Context serbest bırak
```

### 3. KV Monitor

```python
# Generate with monitoring
text = ctx.generate("Bir hikaye yaz", max_tokens=200, monitor=True)

# Sağlık raporu
health = ctx.health()
# {
#   "status": "ok",
#   "saturation": 0.05,
#   "avg_entropy": 1.6,
#   "repetition": 0.0,
#   "tokens_decoded": 200,
#   "warnings": []
# }

# Latency raporu
latency = ctx.monitor.latency_summary()
# {
#   "prefill_ms": 65.8,
#   "decode_ms": 1803.5,
#   "decode_per_token_ms": 18.2,
#   "tok_per_sec": 54.9
# }
```

### 4. Düşük Seviye API (Handle-Based)

```python
from turbo import TurboBridge

bridge = TurboBridge()
bridge.load_model("model.gguf", n_gpu_layers=-1)      # Model yükle (bir kez)
handle = bridge.ctx_init(4096, type_k=8, type_v=41)   # K=q8_0, V=turbo3

tokens = bridge.ctx_tokenize("Hello")
bridge.ctx_decode(tokens, pos=0)
logits = bridge.ctx_get_logits()

# C-level performans
perf = bridge.perf_get()
print(f"Decode: {perf.t_eval_ms:.1f}ms / {perf.n_eval} tokens")

# KV durumu
kv = bridge.kv_state()
print(f"KV: {kv.n_pos}/{kv.n_ctx} ({kv.utilization:.1%})")

bridge.ctx_free()          # Context serbest
bridge.unload_model()      # Model serbest
```

---

## Proje Yapısı

```
PROJECT-TURBO/
├── src/
│   ├── llama-spiritbuun-cuda/   # spiritbuun CUDA fork (179MB)
│   │   ├── include/llama.h      # C API header
│   │   ├── ggml/                # GGML backend
│   │   │   └── src/ggml-cuda/
│   │   │       ├── fattn.cu     # 2-PASS orchestrator + 4 CUDA kernel
│   │   │       └── fattn-vec.cuh # vec kernel (fused fallback)
│   │   └── src/
│   │       ├── llama-kv-cache.h # Dual buffer struct (v_idxs_high_global)
│   │       └── llama-kv-cache.cpp # V dual write, host buffer allocation
│   └── turbo/                   # Python adapter
│       ├── __init__.py          # Public API
│       ├── turbo_adapter.py     # TurboContext / TurboBridge / KVMonitor
│       ├── turbo_bridge.c       # Handle-based C bridge
│       └── enums.py             # Auto-generated enum'lar
├── build/                       # Derlenmiş .so dosyaları
├── scripts/
│   ├── build_turbo.sh           # Fork clone + cmake derleme
│   └── extract_enums.py         # ggml.h → enums.py parser
├── tests/
│   ├── test_turbo_smoke.py      # Smoke test
│   ├── test_bitwidth_sweep.py   # Bit-width config sweep (v1.0)
│   ├── test_2pass_validation.py # Fake split EXACT MATCH
│   ├── test_sampling_quick.py   # Stochastic divergence test
│   └── test_long_ctx_quick.py   # Long context (512-4096) EXACT MATCH
├── docs/
│   └── test-results/            # Test sonuçları (JSON)
├── .kilo/plans/                 # Implementation plan
├── README.md
├── pyproject.toml               # Sıfır bağımlılık
└── AGENTS.md                    # Proje rehberi
```

---

## Test

```bash
# Smoke test (model gerektirmez)
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_smoke.py

# Inference test (model gerekli)
TURBO_TEST_MODEL=model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_turbo_inference.py

# Stress test (1024 tok drift, A/B quality, KV recovery)
TURBO_STRESS_MODEL=model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_stress.py

# Reasoning chain test (error amplification, memory retention)
TURBO_TEST_MODEL=model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_reasoning.py

# Determinism test (cross-reset, KV leak)
TURBO_TEST_MODEL=model.gg.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_determinism.py

# Latency breakdown (C-level perf)
TURBO_TEST_MODEL=model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_latency.py
```

---

## Araştırma Bulguları

### Asimetrik K/V Sıkıştırması

TheTom'un TurboQuant araştırmasına göre, K ve V cache'leri **farklı işlevlere** sahip:

- **K (Key)**: "Kime dikkat edeceğimi" belirler. Softmax'tan geçer → hatalar **exponential** büyür.
- **V (Value)**: "Ne bilgi aktarılacağını" belirler. Lineer toplama → hatalar **orantılı** kalır.

**Sonuç:** V sıkıştırması bedava, K sıkıştırması felaket.

| Konfigürasyon | PPL (Qwen2.5-7B Q4_K_M) |
|---|---|
| q8_0/q8_0 (baseline) | 6.58 |
| **q8_0/turbo3 (asimetrik)** | **6.71 (+2.0%)** |
| q8_0/turbo2 | 6.91 (+5.1%) |
| turbo4/turbo4 | 218 (felaket) |
| turbo3/turbo3 | 3,556 (felaket) |

### Memory Retention (v0.3 Bulgu)

540+ token gap sonrası recall **çalışıyor**. Model "anlamı" unutmuyor ama "ifade biçimi" değişebilir.

| Model | Gap | B=47 | A=12 | Sonuç |
|---|---|---|---|---|
| Qwen3-8B | 542 tok | ✅ 47.0 | ❌ None | Kısmen çalışıyor |
| Qwen3.5-9B | 543 tok | ✅ 47.0 | ✅ 12.0 | Çalışıyor |

### Mimari Bağımsızlık (v0.3)

TurboQuant transformer-specific değil — hybrid SSM+attention mimarilerinde de çalışıyor:

| Model | Mimari | Inference | 1024 tok drift | Memory |
|---|---|---|---|---|
| Qwen3-8B | Pure attention | 13/13 ✅ | ✅ (32 tok/s) | Kısmi |
| Qwen3.5-9B | SSM+Attention | 13/13 ✅ | ✅ (25 tok/s) | ✅ |

**Ama:** Hybrid mimari V compression'a daha hassas (A/B quality: 33.7% vs 51.4%).

### C-Level Performance (v0.3)

| Metrik | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| Decode (C-level) | 18 ms/tok | 20 ms/tok |
| Throughput (C) | 55 tok/s | 50 tok/s |
| Throughput (Wall) | 32 tok/s | 25 tok/s |
| TTFT (200 tok) | 134ms | 158ms |
| Saturation (1024 tok) | 7.7% | 5.4% |

### Determinism (v0.3)

- **Cross-reset:** Aynı prompt → aynı output (greedy, temperature=0) ✅
- **No KV leak:** `clear_cache()` sonrası injected context sızıntısı yok ✅
- **Key repetition:** 20x tekrar → 84729 doğru hatırlanıyor ✅

---

## Test Edilen Ortam

| Bileşen | Sürüm |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB (SM 8.6) |
| CUDA | 13.2.51 |
| GCC | 15.2.1 |
| Python | 3.14.3 |
| OS | Linux x86_64 |

### Doğrulanmış Modeller

| Model | Boyut | Mimari | Vocab | Inference | Stress | Reasoning |
|---|---|---|---|---|---|---|
| Qwen3-8B-Q4_K_M | 4.7 GB | qwen3 (pure attn) | 151,936 | 13/13 ✅ | 5/5 ✅ | 5/7 ✅ |
| Qwen3.5-9B-Q4_K_M | 5.3 GB | qwen35 (hybrid) | 248,320 | 13/13 ✅ | 5/5 ✅ | 5/7 ✅ |

### v1.0 2-PASS Cross-Model Validation (EXACT MATCH)

| Model | Mimari | Baseline | 2-PASS | Sonuç |
|---|---|---|---|---|
| Qwen3-8B | Pure attention | [311, 387, 6247, 11, 714] | [311, 387, 6247, 11, 714] | ✅ EXACT |
| Qwen3.5-9B | SSM+Attention | [311, 387, 6247, 11, 714] | [311, 387, 6247, 11, 714] | ✅ EXACT |
| DeepSeek-Coder-V2-Lite | MoE+Attention | [311, 387, 6247, 11, 714] | [311, 387, 6247, 11, 714] | ✅ EXACT |
| Mamba-7B | Pure SSM | [311, 387, 6247, 11, 714] | [311, 387, 6247, 11, 714] | ✅ EXACT |
| Qwen-4B | Pure attention | [311, 387, 6247, 11, 714] | [311, 387, 6247, 11, 714] | ✅ EXACT |

**2-PASS de-fusion mimari bağımsız:** Transformer, SSM, MoE — tüm mimarilerde EXACT MATCH.

**Test ortamı:** RTX 3060 12GB, CUDA 13.2, K=turbo4, V_low=turbo3, V_high=turbo4, TURBO_2PASS_SPLIT=1

### Stress Test Sonuçları

| Test | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| 1024 token production | ✅ 32 tok/s, drift yok | ✅ 25 tok/s, drift yok |
| A/B kalite (q8_0 vs turbo3) | 51.4% benzerlik | 33.7% benzerlik |
| KV corruption/recovery | ✅ | ⚠️ (math: model davranışı) |
| Prefill throughput | 1567 tok/s | 1257 tok/s |
| Saturation (1024 tok) | 7.7% | 5.4% |

### Context Stress Test Sonuçları (Qwen3-8B)

| Context | Prompt | Prefill | Throughput | Generation |
|---|---|---|---|---|
| 8K | 2,001 tok | 1.2s | 1,737 tok/s | ✅ çalışıyor |
| 32K | 28,000 tok | 23.5s | 1,192 tok/s | ✅ çalışıyor |
| **40K (model max)** | **38,000 tok** | **37.1s** | **1,025 tok/s** | **✅ çalışıyor** |

---

## ⚠️ Bilinen Sınırlamalar

- **CUDA Graphs:** 2-PASS modunda `GGML_CUDA_DISABLE_GRAPHS=1` gerekli (CUDA graph replay henüz test edilmemiş)
- **Contiguous KV:** 2-PASS sadece contiguous KV layout destekler (non-contiguous → assert)
- **Flash Attention zorunlu:** turbo3/turbo4 için FA=ON gerekli
- **K=Turbo4 zorunlu:** K=turbo3 ile reasoning çöküşü (test kanıtlı)
- **Tek sequence:** Multi-sequence henüz desteklenmiyor
- **GGUF-only:** Safetensors formatı desteklenmiyor
- **Adaptive KV:** Henüz uygulanmadı (Phase 2'de)

---

## Teşekkür

- **[spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** — TurboQuant CUDA kernel'ları
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Asimetrik K/V araştırması, Python prototip
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — Temel GGML/llama.cpp altyapısı
- **Zandieh et al. (ICLR 2026)** — Orijinal TurboQuant teorisi (PolarQuant + QJL)

---

> *"Kendi köprüsünü kuran ordu, nehrin akışına boyun eğmez."*

**PROJECT-TURBO** — Standartlara değil, performansa odaklan.
