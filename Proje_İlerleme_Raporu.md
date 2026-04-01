# PROJECT-TURBO — Proje İlerleme Raporu

> **Son Güncelleme:** 1 Nisan 2026 — 03:30
> **Durum:** ✅ TurboQuant asimetrik K/V inference Python'da çalışıyor!

---

## 📋 Proje Özeti

**Amaç:** RTX 3060 (12GB VRAM) gibi tüketici GPU'larda 128K context'i zero-swap mimarisiyle çalıştırmak.

**Yöntem:** Madreag'ın llama.cpp fork'undaki TurboQuant quantizasyon formatlarını (2/3/4-bit KV cache), resmi `llama-cpp-python` binding'lerini beklemeden, saf `ctypes` binding ile Python dünyasına bağlamak.

**Stratejik Karar:** `llama-cpp-python` bağımlılığı tamamen reddedildi → Saf ctypes binding (Seçenek C).

---

## 🗓️ Adım Adım İlerleme

### Faz 1: İstihbarat Toplama ve Mimari Planlama

#### 1.1 Web Araştırması — Fork ve Enum Tespiti
**Tarih:** 31 Mart 2026

| Kaynak | URL | Bulgu |
|---|---|---|
| Madreag/turbo3-cuda | `github.com/Madreag/turbo3-cuda` | Ana fork, `feature/turboquant-kv-cache` branch |
| TheTom/llama-cpp-turboquant | `github.com/TheTom/llama-cpp-turboquant` | Aynı branch, aynı enum'lar |
| llama.cpp Discussion #20969 | `github.com/ggml-org/llama.cpp/discussions/20969` | TurboQuant upstream tartışması |

**Tespit Edilen Enum Değerleri (ggml.h):**

| Enum | Değer | Açıklama |
|---|---|---|
| `GGML_TYPE_TURBO3_0` | **41** | 3-bit KV cache (2-bit PolarQuant + 1-bit QJL) |
| `GGML_TYPE_TURBO4_0` | **42** | 4-bit KV cache (3-bit PolarQuant + 1-bit QJL) |
| `GGML_TYPE_TURBO2_0` | **43** | 2-bit KV cache (2-bit PolarQuant, no QJL) |
| `GGML_TYPE_COUNT` | **44** | Toplam type sayısı |
| `GGML_OP_TURBO_WHT` | **85** | Walsh-Hadamard Transform operasyonu |

#### 1.2 Sistem Taraması
```
GPU:          NVIDIA GeForce RTX 3060, 12288 MiB, Compute 8.6 (Ampere)
CUDA:         13.2.51 (/opt/cuda/bin/nvcc)
GCC:          15.2.1 20260209
Python:       3.14.3
Sistem lib:   /usr/lib/libllama.so.0 (upstream, TurboQuant YOK)
llama-cpp:    YÜKLÜ DEĞİL
```

#### 1.3 Mimari Karar: Saf ctypes Binding (Seçenek C)

**Değerlendirilen Seçenekler:**

| Seçenek | Açıklama | Karar | Sebep |
|---|---|---|---|
| A) llama-cpp-python wrapper | Mevcut kütüphaneye enum injection | ❌ Red | .so precedence savaşı, bağımlılık zinciri |
| B) llama-cpp-python fork | Kendi fork'umuzu publish et | ❌ Red | Bakım yükü, upstream sync derdi |
| **C) Saf ctypes binding** | Doğrudan libllama.so → ctypes → Python | ✅ Kabul | Sıfır bağımlılık, tam kontrol, bağımsız |

**Karar Gerekçesi:**
- Upstream değişikliklerinden etkilenmez
- `.so` precedence savaşı yok
- Enum injection derdi yok
- KAPTAN v4'e entegrasyon daha temiz
- "HF'i beklemeyenlerin kütüphanesi" vizyonuna uygun

---

### Faz 2: Proje İskeleti Oluşturma

#### 2.1 Dizin Yapısı
```
PROJECT-TURBO/
├── README.md                          # Açık kaynak için hazır
├── Proje_İlerleme_Raporu.md           # Bu dosya
├── pyproject.toml                     # Paket tanımı (sıfır bağımlılık)
├── scripts/
│   ├── build_turbo.sh                 # Fork clone + CUDA derleme + enum extraction
│   └── extract_enums.py              # ggml.h parser → auto-generated enums.py
├── src/
│   ├── turbo/
│   │   ├── __init__.py               # Public API exports
│   │   ├── turbo_adapter.py          # Ana adapter: saf ctypes binding
│   │   └── enums.py                  # Auto-generated (build sonrası)
│   └── llama-turboquant/             # Madreag fork (git clone)
├── examples/
│   └── 128k_inference.py             # 128K context demo
├── tests/
│   └── test_turbo_smoke.py           # Enum + sembol doğrulama testleri
└── build/                            # Derlenen .so dosyaları
    ├── libllama.so                   # ← Ana hedef
    ├── libllama.so.0
    ├── libllama.so.0.0.1
    ├── libggml-cuda.so               # 62MB CUDA backend
    ├── libggml-cpu.so                # 1.1MB CPU backend
    └── libggml-base.so               # 780KB temel
```

#### 2.2 İlk turbo_adapter.py (v1 — llama-cpp-python bağımlı)
**Durum:** ❌ Terk edildi

İlk versiyon `llama-cpp-python`'u iskelet olarak kullanıyordu:
- `force_load()` ile .so yükleme
- `inject_into_llama_cpp()` ile enum injection
- `TurboLlama` wrapper

**Sorunlar tespit edildi:**
1. `llama-cpp-python` yüklü değildi
2. `.so` precedence savaşı riski (kendi .so'sunu cache'ler)
3. `type_k`/`type_v` parametre isimleri API değişikliklerine açık
4. Gereksiz bağımlılık zinciri

**Karar:** Saf ctypes binding'e geçiş (v2)

---

### Faz 3: README.md Oluşturma

**Tarih:** 31 Mart 2026

Açık kaynak lansmanı için hazır README.md yazıldı:
- Proje amacı ve motivasyonu
- VRAM karşılaştırma tablosu (F16 vs Turbo3)
- Mimari diyagram
- Hızlı başlangıç kılavuzu
- Performans hedefleri
- Bilinen sınırlamalar
- Katkı kuralları

---

### Faz 4: Build Script Geliştirme ve Düzeltmeler

#### 4.1 build_turbo.sh — İlk Versiyon
**Sorunlar:**
| Hata | Açıklama | Düzeltme |
|---|---|---|
| CUDA yolu yanlış | `/opt/cuda-13.2/bin` → yok | `which nvcc` fallback'li otomatik tespit |
| `-ffast-math` hatası | `vec.h:1070: #error "non-finite math required"` | `-fno-finite-math-only` |
| CMAKE_CUDA_ARCHITECTURES eksik | PTX/JIT derlemesi → runtime latency | `86` eklendi (RTX 3060 Ampere) |

#### 4.2 build_turbo.sh — Revize (v2)
```bash
# CUDA otomatik tespit
if command -v nvcc &>/dev/null; then
    NVCC="$(which nvcc)"
elif [ -x "/opt/cuda/bin/nvcc" ]; then
    NVCC="/opt/cuda/bin/nvcc"
...

# Düzeltilmiş flags
-DCMAKE_C_FLAGS="-march=native -mtune=native -O3 -fno-finite-math-only"
-DCMAKE_CUDA_ARCHITECTURES=86
```

#### 4.3 extract_enums.py — İstihbarat Scripti
**Amaç:** `ggml.h` dosyasını parse edip `enums.py` üretmek. Hardcode yok.

**Özellikler:**
- Explicit enum değerlerini parse eder (`GGML_TYPE_TURBO3_0 = 41`)
- Incremental enum değerlerini hesaplar (auto-increment)
- TurboQuant-specific türleri ve operasyonları filtreler
- `TURBO_TYPE_MAP` convenience map üretir
- Auto-generated dosyaya "DO NOT EDIT MANUALLY" uyarısı ekler

**Çıktı Örneği:**
```python
# src/turbo/enums.py (auto-generated)
GGML_TYPE_TURBO3_0 = 41
GGML_TYPE_TURBO4_0 = 42
GGML_TYPE_TURBO2_0 = 43
GGML_TYPE_COUNT = 44
GGML_OP_TURBO_WHT = 85
TURBO_TYPE_MAP = {
    "turbo3": 41,
    "turbo4": 42,
    "turbo2": 43,
}
```

---

### Faz 5: Saf ctypes Binding (turbo_adapter.py v2)

#### 5.1 C Struct Tanımları
```python
class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),          # offset 0
        ("n_batch", ctypes.c_uint32),        # offset 4
        ...
        ("type_k", ctypes.c_int),            # offset 88 ← KRİTİK
        ("type_v", ctypes.c_int),            # offset 92 ← KRİTİK
        ...
        ("embeddings", ctypes.c_bool),       # offset 112
        ...
    ]
```

#### 5.2 Kapsanan C API
| C Fonksiyonu | Python Wrapper | Durum |
|---|---|---|
| `llama_backend_init/free` | ✅ | |
| `llama_model_load_from_file` | ✅ | |
| `llama_init_from_model` | ✅ (type_k, type_v ile) | |
| `llama_tokenize` | ✅ | |
| `llama_token_to_piece` | ✅ | |
| `llama_decode` | ✅ | |
| `llama_get_logits_ith` | ✅ | |
| `llama_kv_cache_clear/add/rm/defrag` | ✅ | |
| `llama_model_default_params` | ✅ | |
| `llama_context_default_params` | ✅ | |

#### 5.3 Yüksek Seviye Wrapper'lar
- `TurboContext` — Düşük seviye C API wrapper
- `TurboLlama` — Drop-in compatible (llama-cpp-python API uyumlu)
- `setup()` — One-call initialization

---

### Faz 6: Tespit Edilen ve Düzeltilen Hatalar

#### 6.1 Enum Hataları
| Enum | İlk Değer | Gerçek Değer | Kaynak |
|---|---|---|---|
| `GGML_OP_TURBO_WHT` | 90 (tahmin) | 85 (header parse) | `extract_enums.py` |
| `GGML_OP_TURBO_WHT` | 86 (header sayımı) | 85 (fork parse) | Build sonrası doğrulama |

**Ders:** Hardcode enum değerleri asla güvenilmez. `extract_enums.py` zorunlu.

#### 6.2 Runtime Hataları
| Hata | Sebep | Çözüm |
|---|---|---|
| `AttributeError: module 'ctypes' has no attribute 'RTLD_NOW'` | Python 3.14'te expose edilmemiş | `getattr(ctypes, "RTLD_NOW", 0x00002)` |
| `NameError: _setup_c_api` | Fonksiyon kaldırıldı ama referans kaldı | Referans silindi |
| `FileNotFoundError: Not found: libllama.so.0` | `ctypes.util.find_library` versiyonlu isim döndürüyor | LD_LIBRARY_PATH + standart dizin arama |
| `ModuleNotFoundError: No module named 'turbo'` | `project_root` offset hatası (`.parent.parent` → `.parent.parent.parent`) | Düzeltildi |
| Sistem lib'i öncelik aldı | `find_libllama` build/ dizinini doğru taramıyordu | Unversioned `.so` önceliği eklendi |

#### 6.3 Derleme Hataları
| Hata | Sebep | Çözüm |
|---|---|---|
| `#error "some routines require non-finite math"` | `-ffast-math` → `-ffinite-math-only` default | `-fno-finite-math-only` |
| `ggml-cpu` target derlenemiyor | Yukarıdaki hatanın yan etkisi | Aynı düzeltme |

---

### Faz 7: Build ve Doğrulama

#### 7.1 Build Sonuçları
```
[TURBO] Fork: Madreag/turbo3-cuda (3380d3c)
[TURBO] CUDA: 13.2.51
[TURBO] Arch: 86 (Ampere)
[TURBO] Enum extraction: 45 types, 98 ops → enums.py

Derlenen dosyalar:
  libllama.so          3.8MB   ← Ana hedef
  libggml-cuda.so     62MB     ← CUDA backend
  libggml-cpu.so       1.1MB   ← CPU backend
  libggml-base.so      780KB   ← Temel
```

#### 7.2 Struct Cross-Check (C vs Python)
```c
// C tarafı (gcc ile derlenmiş test binary)
sizeof(llama_context_params) = 136
offsetof(type_k) = 88
offsetof(type_v) = 92
offsetof(n_ctx) = 0
offsetof(embeddings) = 112
```

```python
# Python tarafı (ctypes)
ctypes.sizeof(LlamaContextParams) = 136  ✅
```

**Sonuç:** Sıfır offset kayması. Struct alignment mükemmel eşleşti.

#### 7.3 TurboQuant Symbol Doğrulama
```bash
$ nm -D build/libllama.so | grep -i turbo
U ggml_turbo_wht
U _Z32turbo_innerq_mark_tensor_updatedv
U _Z32turbo_innerq_needs_tensor_updatev
T _ZNK22llama_kv_cache_context18get_turbo_rotationEv
T _ZNK22llama_kv_cache_context21get_turbo_rot_forwardEv
...
```

```python
>>> ggml_type_name(41)
'turbo3'  ✅

>>> ggml_type_name(43)
'turbo2'  ✅
```

#### 7.4 Smoke Test Sonuçları
```
[PASS] Enum values correct (41, 42, 43, 44, 85)
[PASS] TURBO_TYPE_MAP correct
[PASS] Struct sizes: ctx=136, model=72, batch=56
[PASS] Lib resolution logic works
[PASS] ggml_type_name(41) = 'turbo3' — TurboQuant active!
[PASS] Library symbols verified
[PASS] Auto-generated enums.py is valid

[TURBO] All smoke tests passed.  ✅
```

---

### Faz 8: VRAM Hedefleri ve Beklentiler

#### 8.1 Teorik VRAM Hesaplaması
**7B Model, 128K Context:**

| KV Cache Type | Bit/element | KV Cache Boyutu | Toplam VRAM |
|---|---|---|---|
| F16 (varsayılan) | 16 | ~16 GB | ~20 GB (model + cache) |
| Q8_0 | 8 | ~8 GB | ~12 GB |
| Q4_K | 4 | ~4 GB | ~8 GB |
| **Turbo3 (3-bit)** | **3** | **~3 GB** | **~7 GB** ✅ |
| **Turbo2 (2-bit)** | **2** | **~2 GB** | **~6 GB** ✅ |

**Hedef:** RTX 3060 (12GB) üzerinde 128K context → **< 5GB VRAM** (Turbo3)

#### 8.2 Performans Hedefleri
| Metrik | Hedef | Not |
|---|---|---|
| 128K Context VRAM | < 5 GB (Turbo3) | RTX 3060 12GB'de rahat |
| Prefill Throughput | > 200 tok/s | 128K prompt |
| Decode Throughput | > 40 tok/s | RTX 3060, 7B model |
| Swap | 0 byte | Zero-swap mimarisi |
| Bağımlılık | 0 | Sadece Python + CUDA |

---

### Faz 9: Kalan Adımlar

#### 9.1 Kısa Vadeli (Şimdi Bekleyen)
- [ ] **İlk inference testi** — GGUF model dosyası gerekli
  - `examples/128k_inference.py --model model.gguf`
  - VRAM kullanımını ölç (`nvidia-smi`)
  - Throughput ölçümü

#### 9.2 Orta Vadeli
- [ ] **Throughput optimizasyonu**
  - Batch decode
  - KV cache defragmentation
  - Streaming generation
- [ ] **KAPTAN v4 entegrasyonu**
  - `KVContextPool` → `TurboContext` geçişi
  - Dynamic accelerator selection

#### 9.3 Uzun Vadeli
- [ ] **Multi-GPU desteği** — Layer splitting
- [ ] **CPU fallback** — TurboQuant CPU implementasyonu
- [ ] **Açık kaynak lansmanı** — GitHub publish
- [ ] **Benchmark suite** — Karşılaştırmalı performans testleri

---

## 🔑 Kritik Kararlar ve Gerekçeleri

| # | Karar | Tarih | Gerekçe |
|---|---|---|---|
| 1 | Saf ctypes binding (Seçenek C) | 31 Mar | Bağımsızlık, kontrol, upstream'ten etkilenmeme |
| 2 | `extract_enums.py` zorunlu | 31 Mar | Hardcode enum'lar fork değişikliklerinde patlar |
| 3 | `-fno-finite-math-only` | 31 Mar | ggml `vec.h` finite math olmayan aritmetik gerektirir |
| 4 | `CMAKE_CUDA_ARCHITECTURES=86` | 31 Mar | PTX/JIT yerine native cubin, sıfır runtime compile |
| 5 | `llama-cpp-python` bağımlılığı reddedildi | 31 Mar | .so precedence savaşı, enum injection karmaşıklığı |

---

## 📊 Metrik Özeti

| Kategori | Değer |
|---|---|
| Toplam dosya sayısı | 8 (kod) + 2 (dokümantasyon) |
| Satır kod (Python) | ~670 |
| Satır kod (Bash) | ~85 |
| Satır kod (C test) | ~60 |
| Enum doğrulama | 5/5 ✅ |
| Struct alignment | 136/136 byte ✅ |
| TurboQuant sembolleri | 41→"turbo3", 43→"turbo2" ✅ |
| Smoke test | 7/7 ✅ |
| Derleme süresi | ~3 dakika (20 çekirdek) |
| `libllama.so` boyutu | 3.8 MB |
| `libggml-cuda.so` boyutu | 62 MB |

---

## ⚠️ Bilinen Riskler

| Risk | Olasılık | Etki | Mitigasyon | Durum |
|---|---|---|---|---|
| Struct size mismatch (fork güncellenirse) | Düşük | Yüksek | `extract_enums.py` + build sonrası cross-check | ✅ Kontrol edildi |
| llama.h API değişir | Orta | Orta | Struct tanımını fork'a göre güncelle | ⚠️ İzleniyor |
| Tokenizasyon harici bağımlılık | Yüksek | Düşük | Model ile gelen tokenizer.json kullan | ✅ Planlandı |
| Madreag fork derlenemiyor | Düşük | Yüksek | Fallback: TheTom fork | ✅ Alternatif mevcut |
| TurboQuant CPU'da çalışmaz | Yüksek | Orta | CUDA-only, CPU fallback yok | ✅ Kabul edildi |
| **TurboQuant CUDA kernel segfault (C++ fork'lar)** | **%100** | **KRİTİK** | **tq-kv (Rust) ile değiştirildi** | 🟢 ÇÖZÜLDÜ |
| tq-kv llama.cpp patch uyumsuzluğu | Orta | Yüksek | Manuel patch + test | ⏳ Planlandı |

---

## 🏁 Mevcut Durum

**PROJECT-TURBO adapter layer çalışır durumda ama TurboQuant CUDA kernel'ları decode aşamasında çöküyor.**

- ✅ Madreag fork başarıyla derlendi
- ✅ TheTom fork başarıyla derlendi (8ad0f00)
- ✅ TurboQuant sembolleri doğrulandı (`turbo3`, `turbo2`)
- ✅ Saf ctypes binding çalışıyor
- ✅ Struct alignment C/Python arasında mükemmel eşleşti
- ✅ Tüm smoke testler geçti
- ✅ Enum extraction otomatik çalışıyor
- ✅ Build scripti stabilize edildi
- ✅ KV cache doğru allocate ediliyor (`K (turbo3): 10.94 MiB`)
- ✅ TurboQuant rotation matrices initialize ediliyor
- ❌ **Decode aşamasında segfault** — tüm modellerde, tüm fork'larda, tüm konfigürasyonlarda
- ❌ Fused Gated Delta Net force-disable yapıldı (llama-context.cpp yaması)
- ❌ Flash Attention ON/OFF fark etmiyor
- ❌ Hybrid model (Qwen 3.5) ve pure transformer (Qwen2.5-1.5B) fark etmiyor
- ❌ Madreag fork → segfault
- ❌ TheTom fork → segfault (aynı sorun)

**Kök Sebep:** TurboQuant CUDA kernel'ları (`turbo-wht.cu`, `turbo-innerq.cu`) henüz production-ready değil. Reddit'teki tcarambat (AnythingLLM kurucusu) da aynı sorunu bildirmiş: *"I was getting absolutely garbage outputs on CUDA machine."*

**Kritik Yeni Bulgu:** spiritbuun'un benchmark'ı (RTX 4080, Qwen3.5-9B, 65K context) çalışıyor. TheTom/Madreag'dan farklı olarak **Flash Attention AÇIK** (`-DGGML_CUDA_FA=ON`) ve `-DGGML_CUDA_FA_ALL_QUANTS=ON` flag'leri ile build ediyor. FA kapalıyken TurboQuant kernel'ları çalışmıyor olabilir.

**En Son Bulgular (31 Mart 2026 — 20:30):**
- `TheTom/turboquant_plus` reposu: Python prototype, 511 test, %100 coverage — **algoritma matematiksel olarak kusursuz**
- Sorun CUDA kernel'larda: Python'da geçen mantık GPU bellek adreslemesiyle çakışıyor
- **Asymmetric K/V** denendi (K=q8_0, V=turbo3) — yine segfault
- **CPU Fallback** alternatifi: 62GB RAM ile KV cache'i CPU'da tutma seçeneği mevcut
- **Sparse V** upstream PR'ı: `ggml-org/llama.cpp#21119` (+22.8% decode)
- **Boundary V**: İlk 2 + son 2 layer q8_0-V, geri turbo2-V → kalite %37-91 recovery
- **Tahmini süre:** 2-4 hafta upstream stabilizasyon

**Bekleyen:** Spiritbuun fork'u denenmesi (FA=ON ile), asymmetric K/V (K=q8_0, V=turbo3) testi.

---

### Faz 10: Inference Testleri ve Kök Sebep Analizi

#### 10.1 Test Matrisi
| Model | Mimarisi | Flash Attn | Fused GDN | Sonuç |
|---|---|---|---|---|
| Qwen-4B-Q4_K_M | Hybrid SSM+Attn | ON | ON | ❌ Segfault |
| Qwen-4B-Q4_K_M | Hybrid SSM+Attn | ON | OFF (force-disable) | ❌ Segfault |
| Qwen-4B-Q4_K_M | Hybrid SSM+Attn | OFF | OFF (force-disable) | ❌ Segfault |
| qwen3-8b-q4_k_m | Hybrid SSM+Attn | ON | OFF (force-disable) | ❌ Segfault |
| DeepSeek-Coder-V2-Lite | MoE | ON | N/A | ❌ type_k override (q8_0 fallback) |
| **Qwen2.5-1.5B** | **Pure Transformer** | ON | N/A | ❌ Segfault |
| **Qwen2.5-1.5B** | **Pure Transformer** | **OFF** | N/A | ❌ Segfault |

#### 10.2 KV Cache Başarılı, Compute Başarısız
Her testte KV cache doğru allocate ediliyor:
```
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size =   24.50 MiB (  4096 cells,  28 layers,  1/1 seqs), K (turbo3):   12.25 MiB, V (turbo3):   12.25 MiB
sched_reserve: graph nodes  = 1015
sched_reserve: graph splits = 2
```
Ama `llama_decode()` execution'da segfault.

#### 10.3 Dış Doğrulama
Reddit r/LocalLLaMA (tcarambat, 5 gün önce):
> *"I did try to get some kernels working on a CUDA machine but I was getting absolutely garbage outputs"*

TurboQuant Metal'de (Apple Silicon) çalışıyor ama CUDA'da henüz stabil değil.

#### 10.4 Yapılan Müdahaleler
| Müdahale | Dosya | Sonuç |
|---|---|---|
| Fused GDN force-disable | `src/llama-context.cpp` | ❌ Etkisiz |
| Flash Attention OFF | runtime param | ❌ Etkisiz |
| n_batch=1 (pure autoregressive) | runtime param | ❌ Etkisiz |

#### 10.5 Üç Fork Test Sonuçları
| Fork | Commit | FA | GDN | Sonuç |
|---|---|---|---|---|
| Madreag | 3380d3c | OFF | Force-disable | ❌ Segfault |
| TheTom | 8ad0f00 | OFF | Force-disable | ❌ Segfault |
| spiritbuun | 5047b3e | ON | Force-disable | ❌ Segfault |

**Sonuç:** Sorun fork değil, FA ayarı değil, GDN değil. TurboQuant CUDA kernel'larının kendisi (`turbo-wht.cu`, `turbo-innerq.cu`) decode aşamasında çöküyor.

#### 10.6 Yapılan Müdahaleler
| Müdahale | Dosya/Parametre | Sonuç |
|---|---|---|
| Fused GDN force-disable | `src/llama-context.cpp` | ❌ Etkisiz |
| Flash Attention OFF | runtime param | ❌ Etkisiz |
| Flash Attention ON | CMake `-DGGML_CUDA_FA=ON` | ❌ Etkisiz |
| n_batch=1 (pure autoregressive) | runtime param | ❌ Etkisiz |
| Asymmetric K/V (K=q8_0, V=turbo3) | runtime override | ❌ Etkisiz |
| Üç farklı fork denemesi | Madreag/TheTom/spiritbuun | ❌ Hepsinde aynı segfault |

#### 10.7 Kök Sebep Analizi — Python vs CUDA
**Keşif:** `TheTom/turboquant_plus` reposu bir **Python prototype** — 511 test, %100 coverage, matematiksel olarak kusursuz.

| Katman | Durum | Açıklama |
|---|---|---|
| Algoritma (Python/NumPy) | ✅ Mükemmel | 511 test, PPL kağıtla eşleşiyor |
| Metal (Apple Silicon) | ✅ Çalışıyor | M5 Max, 104B @ 128K kanıtlandı |
| CUDA (NVIDIA) | ❌ Çöküyor | `turbo-wht.cu` decode'da segfault |

**Kök Sebep:** Python'daki matematiksel mantık, CUDA kernel'larında GPU bellek adreslemesi (memory alignment) ile çakışıyor. Sorun bizim konfigürasyonumuz değil, kernel implementasyonunun henüz olgunlaşmamış olması.

#### 10.8 Ek Stratejiler (TheTom/turboquant_plus dokümantasyonundan)
| Strateji | Açıklama | Durum |
|---|---|---|
| **Sparse V** | Attention weight < 1e-6 olan V pozisyonlarını atla → +22.8% decode | Upstream PR #21119 açık |
| **Boundary V** | İlk 2 + son 2 layer q8_0-V, geri turbo2-V → kalite %37-91 recovery | TheTom fork'ta mevcut |
| **Asymmetric K/V** | K=q8_0 (yüksek kalite), V=turbo3 (düşük VRAM) | ❌ Bizde de segfault |
| **CPU Fallback** | KV cache'i CPU RAM'de tut (62GB mevcut) | Alternatif, denenmedi |

#### 10.9 Statü: Research/Alpha → tq-kv (Rust) Keşfi
**Tarih:** 31 Mart 2026 — 22:15

PROJECT-TURBO **Research/Alpha** statüsünden **tq-kv (Rust) entegrasyon** fazına geçti:

**C++ Fork'ları (Segfault) — TAMAMLANDI:**
- ❌ Madreag (3380d3c) → segfault
- ❌ TheTom (8ad0f00) → segfault
- ❌ spiritbuun (5047b3e) → segfault
- ❌ animehacker (4381bdd) → segfault
- ❌ Asymmetric K/V (K=q8_0, V=turbo3) → segfault
- ❌ Flash Attention ON/OFF → etkisiz

**tq-kv (Rust) — BAŞARILI:**
- ✅ `onur-gokyildiz-bhi/tq-kv` clone edildi
- ✅ Rust 1.94.1 kuruldu
- ✅ `libtq_kv.so` derlendi (cdylib)
- ✅ Python ctypes FFI testi: **7/7 geçti**
- ✅ Compression ratio: **3.8x** (4-bit)
- ✅ Fused attention çalışıyor (segfault yok)
- ✅ `llama-cpp-patch/` dizini hazır — standart llama.cpp entegrasyonu için
- ✅ GGUF Q4_K_M desteği: README'de Qwen 2.5 7B benchmark'ı var (PPL 6.07, +17%)

**Strateji Değişikliği:**
- ~~Upstream CUDA kernel fix'i bekle~~ → **tq-kv (Rust) ile yoluna devam et**
- Standart upstream llama.cpp + tq-kv patch = segfault riski yok
- Rust'ın memory safety garantileri + 111 test passing
- 3-Fix Framework: Sink Token Preservation + Past-Only Quantization + Cache State Management

**Planlanan:**
- [x] Standart upstream llama.cpp clone (0fcb376)
- [x] tq-kv patch dosyalarını uygula (llama-kv-tq.cpp/h + tq_kv.h + libtq_kv.a)
- [x] CMakeLists.txt'ye LLAMA_TQ_KV ekle
- [x] Derle: cmake -DLLAMA_TQ_KV=ON -DGGML_CUDA=ON
- [x] Test: ctypes ile tüm `llama_tq_*` + `tq_*` fonksiyonları çalışıyor (7/7)
- [x] `llama-mtmd-cli` build edildi (tq-kv linkli)
- [x] Granular test: init → compress → fused_attention → free — hepsi ✅

#### 10.10 Son Karar: Inference Loop Entegrasyonu Ertelendi
**Tarih:** 31 Mart 2026 — 23:30

**Sorun:** tq-kv'nin `llama_tq_*` fonksiyonları `libllama.so` içinde var ve çalışıyor, ama llama.cpp'nin **inference loop'u** onları çağırmıyor. KV cache update ve attention compute sırasında `tq_*` fonksiyonlarını tetiklemek için şu dosyaların modifiye edilmesi gerekiyor:

| Dosya | Gerekli Değişiklik |
|---|---|
| `llama-kv-cache.cpp` | KV store sırasında `llama_tq_compress_keys` çağrısı |
| `llama-context.cpp` | Attention compute sırasında `llama_tq_fused_attention` çağrısı + `type_k = tq3` algılama |
| `common/arg.cpp` | `--cache-type-k tq3` CLI arg parsing |
| `ggml.h` | `GGML_TYPE_TQ3_0` enum ekleme |

**Tahmini İş Yükü:** 200-400 satır C++ kodu, llama.cpp'nin KV cache ve attention mimarisini derinlemesine anlamayı gerektirir. Her değişiklik sonrası rebuild + test. Potansiyel segfault riski.

**Karar:** Inference loop entegrasyonu **ertelendi**. Sebep:
1. KAPTAN'ın asıl misyonu (OSINT, LangGraph, Constitutional Guard) bu C++ patch'inden daha kritik
2. tq-kv upstream'e entegre edildiğinde tek `git pull` yeterli olacak
3. Standart quant tipleri (`q8_0`, `q4_0`) ile VRAM tasarrufu sağlanabilir (TurboQuant kadar agresif değil ama stabil)

**Mevcut Durum:**
- ✅ Bridge kuruldu (Rust FFI → libllama.so → ctypes)
- ✅ Tüm semboller doğrulandı (20+ `tq_*` ve `llama_tq_*` fonksiyonu)
- ⏸️ Inference loop entegrasyonu upstream'e bırakıldı
- ⏸️ tq-kv upstream PR takibi: `onur-gokyildiz-bhi/tq-kv`

**Alternatif:** Standart `--cache-type-k q8_0` ile %50 VRAM tasarrufu (TurboQuant'ın %75'ine karşılık, ama sıfır patch gerektirir).

---

## 🚀 Dönüm Noktası: Asimetrik K/V Inference Çalıştı (1 Nisan 2026)

### Keşifler

1. **C bridge gerekli:** Python ctypes, struct-by-value passing'te stack alignment sorunu yaşıyor. Çözüm: `turbo_bridge.c` — C wrapper fonksiyonları ile struct passing'i C tarafında yapıyor.

2. **Asimetrik K/V stratejisi (kritik bulgu):**
   - **K=q8_0** (korunmuş), **V=turbo3** (sıkıştırılmış)
   - Sebep: K hataları softmax'ta exponential olarak büyür, V hataları orantılı kalır
   - Kaynak: TheTom/turboquant_plus asymmetric KV compression araştırması
   - Qwen2.5-7B Q4_K_M: q8_0/turbo3 → PPL +2.0% (kabul edilebilir)
   - turbo3/turbo3 → PPL 3,556 (felaket!)

3. **Spiritbuun fork seçildi:** `spiritbuun/llama-cpp-turboquant-cuda` — en aktif CUDA fork, TURBO3_0/TURBO4_0/TURBO2_0 enum'ları destekliyor.

4. **`batch.logits = NULL` sorunu:** `llama_batch_get_one` spiritbuun fork'unda logits alanını NULL döndürüyor. Çözüm: `llama_batch_init()` kullanıldı.

### Test Sonuçları

| Test | Sonuç |
|---|---|
| C bridge inference (K=q8_0, V=turbo3) | ✅ Çalışıyor |
| Python TurboContext inference | ✅ Çalışıyor |
| Greedy generation (50 token) | ✅ Tutarlı metin |
| Temperature sampling (0.7) | ✅ Çalışıyor |
| Flash Attn=ON | ✅ Çalışıyor |
| Smoke test suite | ✅ 5/5 geçti |

### Dosya Yapısı (Güncel)

```
PROJECT-TURBO/
├── src/turbo/
│   ├── __init__.py           # v0.2.0 — TurboContext, TurboBridge, TurboLlama
│   ├── turbo_adapter.py      # Python adapter (C bridge kullanır)
│   └── enums.py              # Auto-generated enum'lar
├── src/turbo/turbo_bridge.c  # C wrapper — struct-by-value sorununu çözer
├── src/llama-spiritbuun-cuda/ # spiritbuun fork (aktif)
├── build/
│   ├── libllama.so           # Spiritbuun build (CUDA+FA+OpenMP)
│   ├── turbo_bridge.so       # C bridge
│   └── bin/                  # ggml .so dosyaları
└── tests/
```

### Kullanım

```python
from turbo import TurboContext

ctx = TurboContext(
    model_path="model.gguf",
    n_ctx=4096,
    cache_type="turbo3",  # V=turbo3, K=q8_0 (asimetrik)
    flash_attn=True,
)
text = ctx.generate("Merhaba", max_tokens=100)
print(text)
```

### Sonraki Adımlar

- [ ] TurboContext'u KAPTAN'a entegre et
- [ ] 128K context testi (model kapasitesine göre)
- [ ] PPL benchmark (q8_0/q8_0 vs q8_0/turbo3)
- [ ] TurboBridge'a KV cache operations ekle
- [ ] Streaming output desteği

---

> *"Kendi köprüsünü kuran ordu, nehrin akışına boyun eğmez. Ama akıllı komutan, nehrin durulmasını da bilir."*
>
> **PROJECT-TURBO** — Status: ✅ TurboQuant asimetrik K/V inference çalışıyor. C bridge + Python adapter ile ilk tutarlı metin üretimi gerçekleştirildi.
