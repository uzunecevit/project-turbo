# PROJECT-TURBO — Test Planı

> **Amaç:** Mevcut test suite'i genişleterek TurboQuant implementasyonunun kalitesini, performansını ve teorik iddialarını deneysel olarak doğrulamak.

---

## Mevcut Test Durumu

| Dosya | Kapsam | Test Sayısı |
|---|---|---|
| `tests/test_turbo_smoke.py` | Bridge loading, context init, type map | 5 |
| `tests/test_turbo_inference.py` | Temel inference | 13 |
| `tests/test_stress.py` | Long gen, cross-turn, A/B, KV recovery, prefill | 5 |
| `tests/test_reasoning.py` | Math, memory, contradiction, key repetition, KV monitor | 7 |
| `tests/test_determinism.py` | Cross-reset, KV leak | 4 |
| `tests/test_latency.py` | Prefill, decode, TTFT (C-level) | 5 |
| **TOPLAM** | | **39** |

---

## Öncelikli Eksik Testler

### T1: Perplexity (PPL) Otomatik Ölçüm

**Öncelik:** KRİTİK | **Zorluk:** Orta | **Süre:** 1-2 gün

**Neden:** Makale PPL raporluyor, bizde yok. En kritik kalite metriği.

**Ne yapılacak:**
- WikiText-2 veya PTB üzerinde PPL hesapla
- Baseline (q8_0/q8_0) vs turbo2 vs turbo3 vs turbo4 karşılaştır
- Farklı context uzunluklarında (512, 2048, 4096) PPL ölç
- Model bazında (Qwen3-8B, Qwen3.5-9B) karşılaştır

**Başarı kriteri:**
- turbo3 PPL, q8_0/q8_0 baseline'dan %5'ten fazla sapmamalı
- turbo2 PPL, baseline'dan %10'dan fazla sapmamalı

**Çıktı:** `tests/test_perplexity.py`

**Komut:**
```bash
TURBO_TEST_MODEL=model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src \
    python tests/test_perplexity.py
```

---

### T2: Bit-width Sweep (turbo2 vs turbo3 vs turbo4)

**Öncelik:** YÜKSEK | **Zorluk:** Düşük | **Süre:** 0.5 gün

**Neden:** "turbo3 default" iddiasını her workload için kanıtla.

**Ne yapılacak:**
- Tüm 3 formatı aynı prompt setinde çalıştır
- Metrikler: throughput, memory retention accuracy, reasoning accuracy, A/B quality
- VRAM kullanımını ölç (nvidia-smi veya C-level KV state)

**Test matrisi:**
| Format | Bit | PPL | Tok/s | Memory Ret. | Reasoning | A/B Quality |
|---|---|---|---|---|---|---|
| turbo2 | 2-bit | TBD | TBD | TBD | TBD | TBD |
| turbo3 | 3-bit | TBD | TBD | TBD | TBD | TBD |
| turbo4 | 4-bit | TBD | TBD | TBD | TBD | TBD |
| q8_0 | 8-bit | baseline | baseline | baseline | baseline | - |

**Çıktı:** `tests/test_bitwidth_sweep.py`

---

### T3: Distortion Rate Ölçümü

**Öncelik:** YÜKSEK | **Zorluk:** Orta | **Süre:** 1-2 gün

**Neden:** Makalenin "near-optimal distortion" iddiasını deneysel olarak doğrula. Shannon lower bound ~2.7x faktör.

**Ne yapılacak:**
- Quantize edilmiş KV cache ile orijinal (f16) arasındaki MSE'yi hesapla
- Farklı bit-width'lerde distortion rate ölç
- Makalenin teorik bound'u ile karşılaştır
- Inner product distortion (D_prod) ayrıca ölç

**Formüller:**
```
D_mse  = E[||x - Q^{-1}(Q(x))||²]
D_prod = E[|<y,x> - <y,Q^{-1}(Q(x))>|²]
```

**Başarı kriteri:**
- turbo3 distortion, Shannon lower bound'un 3x faktör içinde olmalı
- Makalenin deneysel sonuçlarıyla tutarlı olmalı

**Çıktı:** `tests/test_distortion.py`

---

### T4: Flash Attention Dependency Analizi

**Öncelik:** YÜKSEK | **Zorluk:** Yüksek | **Süre:** 2-3 gün

**Neden:** turbo3 FA=OFF ile sessiz hatalar veriyor. Root cause bilinmiyor.

**Ne yapılacak:**
- turbo3'ü FA=OFF ile çalıştır → PPL patlamasını ölç
- turbo3'ü FA=ON ile çalıştır → referans PPL
- Farklı attention path'lerde (FA vs non-FA) KV cache içeriğini karşılaştır
- Hangi computation step'te sapma başladığını bul (attention logits, softmax output, output projection)
- turbo4 ve turbo2'de de aynı testi yap

**Hipotezler:**
1. PolarQuant hesaplamaları FA kernel'ına gömülü, non-FA path desteklemiyor
2. FA kernel'ı angle/radius hesaplarını optimize ediyor, non-FA'da precision kaybı var
3. Softmax computation FA'da farklı, non-FA'da overflow/underflow

**Çıktı:** `tests/test_fa_dependency.py` + teknik not

---

### T5: PolarQuant vs Random Rotation Karşılaştırması

**Öncelik:** ORTA | **Zorluk:** Yüksek | **Süre:** 3-4 gün

**Neden:** spiritbuun'un tercihini bilimsel olarak doğrula. Hangisi daha iyi distortion/runtime trade-off sunuyor?

**Ne yapılacak:**
- Aynı input vektör setini her iki yöntemle quantize et
- MSE distortion'ı karşılaştır
- Runtime hızını ölç (GPU kernel level, ns/vector)
- Farklı dimension'larda (64, 128, 256, 512) test et
- Farklı bit-width'lerde (2, 3, 4) test et

**Karşılaştırma matrisi:**
| Metrik | PolarQuant | Random Rotation | Fark |
|---|---|---|---|
| MSE distortion (3-bit) | TBD | TBD | TBD |
| Runtime (ns/vector) | TBD | TBD | TBD |
| Memory overhead | TBD | TBD | TBD |
| FA dependency | Evet | Hayır | - |

**Not:** Random rotation implementasyonu Google makalesindeki algoritmaya göre yazılmalı (Algorithm 1).

**Çıktı:** `tests/test_polar_vs_rotation.py` + teknik not

---

## İkincil Ama Faydalı Testler

### T6: Cross-Architecture Validation

**Öncelik:** ORTA | **Zorluk:** Düşük | **Süre:** 1 gün

**Ne yapılacak:**
- Llama-3-8B, Mistral-7B, Gemma-2B modellerinde test et
- Makale Gemma + Mistral'da test etmiş, biz sadece Qwen'de
- Her modelde: inference, stress, reasoning testlerini çalıştır

**Çıktı:** `tests/test_cross_arch.py`

---

### T7: Temperature Sensitivity

**Öncelik:** DÜŞÜK | **Zorluk:** Düşük | **Süre:** 0.5 gün

**Ne yapılacak:**
- temperature=0.0, 0.3, 0.7, 1.0, 1.5'te kalite/çeşitlilik ölç
- Her T'de: repetition ratio, unique word ratio, semantic coherence
- turbo3 vs q8_0 karşılaştırması her T'de

**Çıktı:** Mevcut `test_stress.py`'ye eklenecek

---

### T8: Long Context Scaling

**Öncelik:** ORTA | **Zorluk:** Düşük | **Süre:** 0.5 gün

**Ne yapılacak:**
- 8K, 16K, 32K, 64K context'lerde test et
- Saturation, throughput, memory usage ölç
- 128K iddiası var ama 40K'da test edilmiş

**Test matrisi:**
| Context | Prefill (ms) | Decode (ms/tok) | Saturation | VRAM (GB) |
|---|---|---|---|---|
| 8K | TBD | TBD | TBD | TBD |
| 16K | TBD | TBD | TBD | TBD |
| 32K | TBD | TBD | TBD | TBD |
| 64K | TBD | TBD | TBD | TBD |

**Çıktı:** Mevcut `test_stress.py`'ye eklenecek veya `tests/test_context_scaling.py`

---

### T9: Hybrid Architecture Deep Dive

**Öncelik:** ORTA | **Zorluk:** Orta | **Süre:** 1-2 gün

**Neden:** Qwen3.5-9B'de A/B quality %33.7 — neden bu kadar düşük?

**Ne yapılacak:**
- Qwen3.5-9B'de token-level error localization
- Hangi token'larda sapma oluyor? (SSM katmanları vs attention katmanları)
- SSM katmanlarındaki KV cache vs attention katmanlarındaki KV cache'i ayrı ayrı analiz et
- Error amplification pattern'lerini bul

**Çıktı:** `tests/test_hybrid_deep_dive.py` + teknik not

---

### T10: Multi-Turn Persistent KV Cache

**Öncelik:** YÜKSEK | **Zorluk:** Orta | **Süre:** 1 gün

**Neden:** Mevcut cross-turn testi aslında single-turn (clear_cache() kullanıyor). Gerçek multi-turn test yok.

**Ne yapılacak:**
- `clear_cache()` KULLANMADAN 5+ turn conversation simüle et
- Her turn'da önceki context'in korunup korunmadığını test et
- KV cache doldukça performans ve kalite değişimini ölç
- Context window limit'e yaklaştığında behavior'ı test et

**Çıktı:** `tests/test_multi_turn.py`

---

## Test Öncelik ve Zaman Çizelgesi

```
Hafta 1:
├── T1: PPL Ölçüm          (KRİTİK)
├── T2: Bit-width Sweep    (YÜKSEK)
└── T3: Distortion Rate    (YÜKSEK)

Hafta 2:
├── T4: FA Dependency      (YÜKSEK)
├── T10: Multi-Turn        (YÜKSEK)
└── T8: Context Scaling    (ORTA)

Hafta 3:
├── T5: PolarQuant vs RR   (ORTA)
├── T6: Cross-Arch         (ORTA)
└── T9: Hybrid Deep Dive   (ORTA)

Hafta 4:
├── T7: Temperature        (DÜŞÜK)
├── Dokümantasyon güncelleme
└── README güncelleme
```

---

## Test Altyapısı Gereksinimleri

### Modeller
| Model | Boyut | Amaç |
|---|---|---|
| Qwen3-8B-Q4_K_M | 4.7 GB | Primary (mevcut) |
| Qwen3.5-9B-Q4_K_M | 5.3 GB | Hybrid arch (mevcut) |
| Llama-3-8B-Q4_K_M | ~4.5 GB | Cross-arch (T6) |
| Mistral-7B-Q4_K_M | ~4.1 GB | Cross-arch (T6) |
| Gemma-2B-Q4_K_M | ~1.5 GB | Cross-arch, hızlı test (T6) |

### Datasetler
| Dataset | Amaç |
|---|---|
| WikiText-2 | PPL ölçüm (T1) |
| PTB | PPL doğrulama (T1) |
| LongBench | Long context benchmark (T8) |
| NeedleInAHaystack | Memory retention (T10) |

### Ortam
- GPU: NVIDIA RTX 3060 12GB (mevcut)
- CUDA: 13.2+
- Python: 3.11+
- LD_LIBRARY_PATH=./build/bin
- PYTHONPATH=src

---

## Başarı Metrikleri

| Metrik | Hedef |
|---|---|
| turbo3 PPL (Qwen3-8B) | < 7.0 (baseline: 6.58) |
| turbo3 distortion bound | < 3x Shannon lower bound |
| turbo3 memory retention (500+ gap) | > 80% accuracy |
| turbo3 reasoning accuracy | > 70% (7 test) |
| turbo3 A/B quality (pure attn) | > 50% |
| turbo3 A/B quality (hybrid) | > 40% (şu an %33.7) |
| Multi-turn (5+ turn) | Context loss yok |
| Long context (64K) | Saturation < 50% |

---

## Notlar

- Tüm testler model gerektirir (smoke test hariç)
- `TURBO_TEST_MODEL` veya `TURBO_STRESS_MODEL` env var kullanılmalı
- Her test `LD_LIBRARY_PATH=./build/bin` ile çalışmalı
- Test sonuçları `docs/test-results/` dizinine kaydedilmeli
- Benchmark sonuçları JSON formatında olmalı (otomatik raporlama için)
