# PROJECT-TURBO — Master Roadmap v1.0

> **Son Güncelleme:** 2026-04-01
> **Durum:** Aktif geliştirme — v0.3 stabilize edildi, v0.4 pipeline refactor hedefi

---

## 1. Mevcut Durum (Snapshot)

### 1.1 Bit-width Sweep Sonuçları

**Qwen3-8B (Pure Attention)**
| Config | Tok/s | Math | Key | Char% | Word% | KV (MB) |
|---|---|---|---|---|---|---|
| q8_0 | 34.5 | ❌ (9.0) | ✅ | base | base | 153 |
| turbo2 | 33.8 | ❌ (75.0) | ✅ | 10% | 20% | 99 |
| turbo3 | 32.9 | ✅ (100.0) | ✅ | 51% | 35% | 108 |
| turbo4 | 33.1 | ❌ (0.0) | ✅ | 86% | 88% | 114 |

**Qwen3.5-9B (Hybrid SSM+Attention)**
| Config | Tok/s | Math | Key | Char% | Word% | KV (MB) |
|---|---|---|---|---|---|---|
| q8_0 | 24.5 | ❌ (75) | ✅ | base | base | 153 |
| turbo2 | 24.4 | ❌ (0) | ❌ (8) | 7% | 25% | 99 |
| turbo3 | 24.5 | ❌ (20) | ❌ (8) | 23% | 26% | 108 |
| turbo4 | 25.0 | ❌ (75) | ✅ | 5% | 28% | 114 |

### 1.2 Kritik Bulgular

1. **turbo3 = en iyi trade-off** (pure attention'da)
2. **Hybrid mimari turbo quantization'a uygun değil** (Qwen3.5-9B'de ciddi kalite kaybı)
3. **turbo2 + Boundary V** denemesi yapılmadı — potansiyel olarak turbo3'ten daha iyi olabilir
4. **turbo4** baseline'a en yakın ama math reasoning'de bile hatalı

### 1.3 Bilinen Sorunlar

| Sorun | Etki | Root Cause |
|---|---|---|
| "54" hatası (quantization bias) | Math reasoning'de sabit sayı | RoPE sonrası quantization |
| Memory retention FAIL | 500+ token gap'te bilgi kaybı | V compression + sink token eksik |
| PPL spike (3556) | GGUF + Turbo3 double quant | Softmax'ta exponential hata |
| A/B quality düşük | Hybrid mimarilerde | SSM+Attention etkileşimi |
| Test altyapısı yavaş | 10+ dakika/sweep | Bridge reuse yok |

---

## 2. Pipeline Sorunu (En Kritik)

### 2.1 Mevcut Pipeline (YANLIŞ)

```
Kcur (F32) → RoPE → FWHT → Quantize (turbo3) → KV Cache
```

**Problem:** RoPE sonrası quantization Gaussian varsayımını bozor. PPL +%17.
**Kaynak:** TurboQuant makalesi + geliştirici notu

### 2.2 Hedef Pipeline (DOĞRU)

**Option A (Önerilen):**
```
Kcur (F32) → Quantize (turbo3) → Store compressed
→ Decode: Dequant → RoPE → Attention
```

**Option B (Hardcore):**
```
Kcur (F32) → Quantize → RoPE inside quant domain → Store
```

### 2.3 Pipeline'ı Kim Yaptı?

**Dosyalar:**
- `src/llama-spiritbuun-cuda/src/models/llama.cpp:68-78` — RoPE application
- `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/set-rows.cu:367-414` — Quant kernel
- `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/turbo-quant-cuda.cuh:336-411` — TURBO3 pipeline
- `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/fattn.cu` — Flash Attention dequant

---

## 3. Yol Haritası (Phase Sıralaması)

### PHASE 0: Hızlı Sinyaller (1-2 saat)

#### 0A: Sink Token Testi
- `GGML_TURBO_SINK_TOKENS=8` env var ekle
- Memory retention (200/500 gap) ölç
- **Beklenti:** Yüzey düzeltme, root cause çözmez
- **Risk:** Yok (kernel değişikliği yok)

#### 0B: Boundary V Kontrolü
- spiritbuun'da `turbo-sink.cu` zaten var ✅
- Boundary V (Layer-Aware) mevcut mu kontrol et
- `turbo2 + Boundary V` config test et

#### 0C: Test Altyapısı Hızlandırma
- Bridge reuse pattern: `_generate_fresh` → `_generate_reuse` + `clear_cache()`
- Sweep süresi: 10+ dk → 2-3 dk

### PHASE 1: RoPE Pipeline Refactor (2-3 saat — CERRAHİ)

#### 1A: RoPE Öncesi Quantization (K Vektörleri)
- `llama.cpp:68-78`'de K için RoPE'i kaldır
- Quantization pipeline'ını RoPE'den ÖNCE taşı
- Dequant + RoPE adedimını attention'da ekle

#### 1B: Flash Attention'da Dequant + RoPE
- `fattn.cu`'da K dequantize edildikten sonra RoPE uygula
- Q pre-rotation mevcut (FWHT)
- K inverse dequant + RoPE ekle

#### 1C: PPL Testi
- WikiText-2'de PPL ölç
- Hedef: +%17'den +%3.7'ye düşürmek
- Boundary V + RoPE öncesi quant birlikte test et

### PHASE 2: Kalite Validasyonu (1-2 saat)

#### 2A: Entropy Drift Tracking
- KVMonitor ile entropy ölç
- Her 100 token'da drift raporla

#### 2B: Attention Dağılımı Analizi
- İlk tokenlara bias ölç
- Sink token + RoPE öncesi quant kombinasyonu

#### 2C: Multi-Model Test
- Qwen3-8B: turbo3 (mevcut)
- Qwen3.5-9B: turbo4 + Boundary V
- Mamba-7B: deneme (SSM mimari)

### PHASE 3: Documentation & README (30 dk)

#### 3A: Test Results Kaydetme
- Timestamped JSON: `docs/test-results/`
- README'de tablo olarak kullan

#### 3B: README Güncelleme
- Bit-width sweep sonuçları
- Pipeline refactor bulguları
- Boundary V / Sparse V durumu

---

## 4. Beklentiler ve Metrikler

### Phase 0 Beklentileri
| Test | Sink Token | Boundary V | Karar |
|---|---|---|---|
| Memory retention (200) | Hafif iyileşir | İyileşir | Kontrol et |
| Memory retention (500) | ❌ FAIL | ❌ FAIL | Pipeline lazım |
| Math reasoning | Değişmez | Hafif iyileşir | Pipeline lazım |
| Key repetition | Değişmez | Değişmez | Zaten iyi |
| PPL | Azalır | Azalır | Kontrol et |

### Phase 1 Beklentileri
| Metric | Before | After (Target) |
|---|---|---|
| PPL gap vs q8_0 | +%17 | +%3.7 |
| Memory retention (500) | FAIL | PASS |
| Math reasoning | %0-51 | %80+ |
| A/B quality (turbo3) | 33-51% | 70%+ |

### Phase 2 Beklentileri
| Metric | Before | After (Target) |
|---|---|---|
| Qwen3.5-9B key repetition | FAIL (turbo2/3) | PASS |
| Qwen3.5-9B A/B quality | 5-23% | 50%+ |
| Hybrid mimari support | Kötü | Kabuledilebilir |

---

## 5. Acil Öncelik Listesi

```
ÖNCELİK 1: Boundary V kontrol (spiritbuun'da var mı?)
ÖNCELİK 2: Test altyapısı hızlandırma
ÖNCELİK 3: Sink token test (hızlı sinyal)
ÖNCELİK 4: RoPE pipeline refactor plan
```

## 6. Risk Matrisi

| Aksiyon | Risk | Etki | Geri Dönüş Zorluk |
|---|---|---|---|
| Sink token test | Sıfır | Düşük | Yok |
| Boundary V test | Sıfır | Orta | Yok |
| Test speedup | Düşük | Orta | Kolay |
| RoPE refactor | **Yüksek** | **Yüksek** | Zor |

---

## 7. Dosya Referansları

| Dosya | Amaç |
|---|---|
| `src/llama-spiritbuun-cuda/src/models/llama.cpp` | RoPE application |
| `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/set-rows.cu` | Quantization kernel |
| `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/turbo-quant-cuda.cuh` | TURBO3 pipeline |
| `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/turbo-sink.cu` | Sink token support |
| `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/fattn.cu` | Flash Attention |
| `src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/turbo-wht.cu` | FWHT kernel |
| `tests/test_bitwidth_sweep.py` | Bit-width test suite |
| `docs/test-results/` | Test sonuçları |

---

## 8. Kaynaklar

| Kaynak | URL | Not |
|---|---|---|
| TurboQuant makalesi | arxiv.org/abs/2504.19874 | Random rotation + QJL |
| turboquant_plus | github.com/TheTom/turboquant_plus | MoE + Boundary V |
| spiritbuun fork | submodule | TurboQuant CUDA kernels |
| PROJECT-TURBO | mevcut repo | v0.3 stabilize |
