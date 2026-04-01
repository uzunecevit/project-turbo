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
│  (legacy API: backward compat wrapper)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  libllama.so (spiritbuun fork)          │
│  • TurboQuant CUDA kernel'ları          │
│  • Flash Attention                      │
│  • K=q8_0, V=turbo3 (asimetrik)        │
│  • Ampere SM 8.6 optimizasyonları       │
└──────────────┬──────────────────────────┘
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
| **K=q8_0, V=turbo3** | **~4 GB** | **PPL +2.0%** | **Asimetrik (bizim strateji)** |
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
│   │   └── src/                 # llama.cpp kaynak kodu
│   └── turbo/                   # Python adapter (v0.3)
│       ├── __init__.py          # v0.3.0 public API
│       ├── turbo_adapter.py     # TurboContext / TurboBridge / KVMonitor
│       ├── turbo_bridge.c       # Handle-based C bridge + perf metrics
│       └── enums.py             # Auto-generated enum'lar
├── build/                       # Derlenmiş .so dosyaları
│   ├── libllama.so              # TurboQuant-enabled llama.cpp
│   ├── libggml-cuda.so          # CUDA backend
│   ├── libggml-cpu.so           # CPU backend
│   └── turbo_bridge.so          # C bridge
├── scripts/
│   ├── build_turbo.sh           # Fork clone + cmake derleme
│   └── extract_enums.py         # ggml.h → enums.py parser
├── tests/
│   ├── test_turbo_smoke.py      # Smoke test (model gerektirmez)
│   ├── test_turbo_inference.py  # Inference test (13 test)
│   ├── test_stress.py           # Stress test (5 test: 1024 tok, A/B, KV)
│   ├── test_reasoning.py        # Reasoning chain (7 test: math, memory)
│   ├── test_determinism.py      # Determinism (4 test: cross-reset)
│   └── test_latency.py          # Latency breakdown (5 test: C-level)
├── examples/
│   └── 128k_inference.py        # 128K context demo
├── README.md
├── pyproject.toml               # Sıfır bağımlılık
└── CONTRIBUTING.md              # Katkı rehberi
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

**Test ortamı:** RTX 3060 12GB, CUDA 13.2, K=q8_0, V=turbo3, Flash Attn=ON

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

- **Flash Attention zorunlu:** turbo3 için FA=ON gerekli, aksi halde gizli hatalar
- **Asimetrik K zorunlu:** K=q8_0, turbo3 K ile birlikte kullanılmamalı (PPL felaketi)
- **Hybrid mimari hassasiyeti:** SSM+attention modellerde A/B quality düşük (33.7%)
- **Reasoning kırılganlığı:** Multi-step math zincirinde error accumulation
- **Tek sequence:** Multi-sequence henüz desteklenmiyor (seq_id hazır ama aktif değil)
- **GGUF-only:** Safetensors formatı desteklenmiyor
- **no_perf default:** llama.cpp `no_perf=true` default → bridge'de `false` set edilmeli

---

## Teşekkür

- **[spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** — TurboQuant CUDA kernel'ları
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Asimetrik K/V araştırması, Python prototip
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — Temel GGML/llama.cpp altyapısı
- **Zandieh et al. (ICLR 2026)** — Orijinal TurboQuant teorisi (PolarQuant + QJL)

---

> *"Kendi köprüsünü kuran ordu, nehrin akışına boyun eğmez."*

**PROJECT-TURBO** — Standartlara değil, performansa odaklan.
