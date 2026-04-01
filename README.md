# PROJECT-TURBO

> **Zero-Swap 128K Context Inference on Consumer GPUs — TurboQuant KV Cache Compression**

RTX 3060 (12GB VRAM) gibi tüketici sınıfı GPU'larda 128K context'i **swap'a düşmeden** çalıştırmak için TurboQuant asimetrik K/V cache sıkıştırması.

```
from turbo import TurboContext
ctx = TurboContext(model_path="model.gguf", cache_type="turbo3")
print(ctx.generate("Merhaba", max_tokens=100))
```

---

## Ne Yaptık?

PROJECT-TURBO, llama.cpp'nin deneysel **TurboQuant** quantizasyon formatlarını (2/3/4-bit KV cache) Python'a bağlayan bir **C bridge + ctypes adapter** katmanıdır.

**Temel bulgular:**

| Keşif | Detay |
|---|---|
| **Asimetrik K/V** | K=q8_0 (korunmuş), V=turbo3 (sıkıştırılmış) → PPL +2.0% |
| K sıkıştırması felaket | turbo3/turbo3 → PPL 3,556 (Qwen Q4_K_M) |
| V sıkıştırması bedava | q8_0/turbo3 → PPL +1-5% (7 model doğrulandı) |
| C bridge zorunlu | Python ctypes struct-by-value ABI sorunu → `turbo_bridge.c` |
| Flash Attn gerekli | turbo3 için FA=ON zorunlu, aksi halde gizli hatalar |

---

## Mimari

```
Python Application / KAPTAN v4
       │
       ▼
┌─────────────────────────────────────┐
│  turbo_adapter.py                   │
│  TurboContext / TurboLlama          │
│  Top-p sampling, generate loop      │
└──────────────┬──────────────────────┘
               │ ctypes (basit tipler)
               ▼
┌─────────────────────────────────────┐
│  turbo_bridge.c                     │
│  turbo_init / turbo_decode /        │
│  turbo_tokenize / turbo_get_logits  │
│  (struct-by-value C tarafında)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  libllama.so (spiritbuun fork)      │
│  • TurboQuant CUDA kernel'ları      │
│  • Flash Attention                  │
│  • K=q8_0, V=turbo3 (asimetrik)    │
│  • Ampere SM 8.6 optimizasyonları   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  NVIDIA GPU (RTX 3060 12GB)         │
└─────────────────────────────────────┘
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

### 2. Kullan

```python
from turbo import TurboContext

ctx = TurboContext(
    model_path="model.gguf",
    n_ctx=4096,           # Context boyutu
    cache_type="turbo3",  # V=turbo3, K=q8_0 (asimetrik)
    flash_attn=True,      # Flash Attention zorunlu
    n_gpu_layers=-1,      # Tüm katmanlar GPU'da
)

# Greedy generation
text = ctx.generate("Merhaba, benim adım", max_tokens=50, temperature=0.0)
print(text)

# Sampling generation
text = ctx.generate("Yapay zeka nedir?", max_tokens=100, temperature=0.7, top_p=0.9)
print(text)

# Temizle
ctx.clear_cache()  # KV cache temizle (yeni prompt için)
ctx.bridge.free()
```

### 3. Düşük Seviye API

```python
from turbo import TurboBridge

bridge = TurboBridge()
bridge.init("model.gguf", 4096, type_k=8, type_v=41)  # K=q8_0, V=turbo3

tokens = bridge.tokenize("Hello")
bridge.decode(tokens, pos=0)
logits = bridge.get_logits()

# Token seçimi
best = logits.index(max(logits))
print(bridge.token_to_piece(best))
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
│   └── turbo/                   # Python adapter (48KB)
│       ├── __init__.py          # v0.2.0 public API
│       ├── turbo_adapter.py     # TurboContext / TurboBridge
│       ├── turbo_bridge.c       # C wrapper (struct ABI fix)
│       └── enums.py             # Auto-generated enum'lar
├── build/                       # Derlenmiş .so dosyaları (454MB)
│   ├── libllama.so              # TurboQuant-enabled llama.cpp
│   ├── libggml-cuda.so          # CUDA backend
│   ├── libggml-cpu.so           # CPU backend
│   └── turbo_bridge.so          # C bridge
├── scripts/
│   ├── build_turbo.sh           # Fork clone + cmake derleme
│   └── extract_enums.py         # ggml.h → enums.py parser
├── tests/
│   └── test_turbo_smoke.py      # Smoke test suite
├── examples/
│   └── 128k_inference.py        # 128K context demo
├── config/                      # Gelecek konfigürasyonlar
├── README.md
├── pyproject.toml               # Sıfır bağımlılık
└── Proje_İlerleme_Raporu.md     # Detaylı teknik rapor
```

---

## Test

```bash
# Smoke test (model gerektirmez)
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_smoke.py

# Inference test (model gerekli)
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python -c "
from turbo import TurboContext
ctx = TurboContext(model_path='model.gguf', cache_type='turbo3')
print(ctx.generate('Hello', max_tokens=20))
"

# 128K context demo
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python examples/128k_inference.py \
    --model model.gguf \
    --n-ctx 131072 \
    --cache-type turbo3
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

### Neden C Bridge?

Python ctypes, Linux x86_64'da büyük struct'ları by-value passing'te **stack alignment** sorunu yaşıyor. `llama_init_from_model(model, params)` çağrısı C'de çalışırken Python ctypes'te segfault veriyor.

**Çözüm:** `turbo_bridge.c` — struct passing'i C tarafında yapıp Python'a sadece basit tipler (`int`, `float*`) döndürüyoruz.

### spiritbuun Fork

`spiritbuun/llama-cpp-turboquant-cuda` — en aktif TurboQuant CUDA fork:
- 295 ⭐, 21 fork
- TURBO3_0/TURBO4_0/TURBO2_0 enum'ları destekliyor
- Flash Attention + CUDA F16 + FA_ALL_QUANTS
- Norm correction (PPL iyileştirmesi)

---

## Test Edilen Ortam

| Bileşen | Sürüm |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB (SM 8.6) |
| CUDA | 13.2.51 |
| GCC | 15.2.1 |
| Python | 3.14.3 |
| OS | Linux x86_64 |
| Model | Qwen3 8B Q4_K_M |

---

## ⚠️ Bilinen Sınırlamalar

- **Flash Attention zorunlu:** turbo3 için FA=ON gerekli, aksi halde gizli hatalar
- **Asimetrik K zorunlu:** K=q8_0, turbo3 K ile birlikte kullanılmamalı (PPL felaketi)
- **Tek sequence:** Multi-sequence henüz desteklenmiyor
- **GGUF-only:** Safetensors formatı desteklenmiyor

---

## Teşekkür

- **[spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)** — TurboQuant CUDA kernel'ları
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Asimetrik K/V araştırması, Python prototip
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — Temel GGML/llama.cpp altyapısı
- **Zandieh et al. (ICLR 2026)** — Orijinal TurboQuant teorisi (PolarQuant + QJL)

---

> *"Kendi köprüsünü kuran ordu, nehrin akışına boyun eğmez."*

**PROJECT-TURBO** — Standartlara değil, performansa odaklan.
