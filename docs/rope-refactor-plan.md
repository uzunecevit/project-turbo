# RoPE Pipeline Refactor — Cerrahi Plan

> **Amaç:** Kuantizasyonu RoPE'den ÖNCE taşıyarak PPL'i +%17'den +%3.7'ye düşürmek
> **Risk:** Yüksek — CUDA kernel pipeline değişiyor
> **Süre:** 2-3 saat (C-level debugging gerektirir)

---

## 1. Mevcut Pipeline (YANLIŞ)

```
Kcur (F32) → RoPE → FWHT → Quantize (turbo3) → KV Cache
                         ↑
                         Bu noktada Gaussian varsayımı bozuluyor
```

**Problem:** RoPE rotary embedding, vektörleri pozisyona göre döndürür. Döndürülmüş vektörlerin istatistiksel dağılımı artık Gaussian değil — bu yüzden Lloyd-Max codebook centroid'leri yanlışı temsil eder.

## 2. Hedef Pipeline (DOĞRU)

```
Kcur (F32) → Quantize (turbo3) → Store compressed
→ Decode: Dequant → RoPE → Attention
```

**Avantaj:** Quantize edilmiş K vektörleri Gaussian dağılımına uygun kalır. RoPE inference zamanında uygulanır.

## 3. Etkilenen Dosyalar

| Dosya | Değişiklik | Satır |
|---|---|---|
| `src/llama.cpp` | K için RoPE'i kaldır | 68-78 |
| `set-rows.cu` | Quantization pipeline (değişmez) | 367-414 |
| `fattn.cu` | Dequant → RoPE ekle | 375-405 |
| `fattn-vec.cuh` | K dequant sonrası RoPE ekle | 125-370 |

## 4. Adım Adım Uygulama

### Adım 1: `llama.cpp` — K RoPE'yi Kaldır

**Mevcut kod (line 68-78):**
```cpp
Kcur = ggml_rope_ext(
    ctx0, Kcur, inp_pos, rope_factors,
    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
    ext_factor, attn_factor, beta_fast, beta_slow);
```

**Yeni kod:**
```cpp
// K RoPE kaldırıldı — quantization sonrası inference zamanında uygulanacak
// Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, ...);  // SATIRI YORUM SATIRI YAP
```

**Dikkat:** Bu sadece K için geçerli. Q için RoPE KALACAK.

### Adım 2: `fattn.cu` — Dequant + RoPE Ekle

**Mevcut dequant kernel (line 215-230):**
```cpp
// k_turbo2_dequant_f16: sadece dequantize eder
```

**Yeni dequant + RoPE kernel:**
```cpp
__global__ void k_turbo2_dequant_rope_f16(
    const void * src, half * dst, const int64_t ne0, const int64_t ne1,
    const int * positions,  // YENİ: pozisyon bilgisi
    const float * rope_factors, float freq_base, int n_rot, int rope_type)
{
    // ... dequantize mevcut kod ...

    // ROPE EKLEME: her element için rotary embedding uygula
    const int pos = positions[j];  // token pozisyonu
    const float theta = freq_base * powf(1.0f, -2.0f * (j % n_rot) / n_rot);
    const float cos_val = cosf(pos * theta);
    const float sin_val = sinf(pos * theta);

    // rotary embedding: [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
    if (j % 2 == 0) {
        dst[j] = val * cos_val - dst[j+1] * sin_val;
    } else {
        dst[j] = val * sin_val + dst[j-1] * cos_val;
    }
}
```

### Adım 3: Attention'da RoPE Sonrası K Kullanımı

**Mevcut:** K zaten RoPE'li → dequant → attention
**Yeni:** K RoPE'siz → dequant → RoPE → attention

### Adım 4: Q RoPE Eşlemesi

**Kritik:** Q hala RoPE'li. Ama K artık RoPE'siz. Attention hesaplamasında:
- Q: RoPE'li (mevcut)
- K: dequant + RoPE (yeni)
- Q · K = RoPE'li · RoPE'li = doğru inner product ✓

## 5. Test Doğrulama

### PPL Testi
```
# Baseline: q8_0/q8_0
python tests/test_perplexity.py --model qwen3-8b --config q8_0/q8_0

# Hedef: RoPE öncesi quant
python tests/test_perplexity.py --model qwen3-8b --config q8_0/turbo3

# Beklenti: +%17 → +%3.7
```

### Memory Retention Testi
```
# Sink token + RoPE öncesi quant
GGML_TURBO_SINK_TOKENS=8 python tests/test_bitwidth_sweep.py

# Beklenti: 500 token gap'te PASS
```

## 6. Rollback Planı

Her adımda commit:
```
git add -A && git commit -m "rope: step N description"
```

Hata durumunda:
```
git reset --hard <son_iyi_commit>
```

## 7. Risk Değerlendirmesi

| Risk | Olasılık | Etki | Çözüm |
|---|---|---|---|
| RoPE uygulama hatası | Orta | Yüksek | Dequant kernel'da hata ayıklama |
| Attention pattern bozulması | Düşük | Kritik | Q·K inner product doğrulama |
| PPL artması | Orta | Orta | Adım adım rollback |
| Compile hatası | Düşük | Düşük | CMake rebuild |

## 8. Başarı Kriterleri

| Metric | Before | After (Target) | Test |
|---|---|---|---|
| PPL @ q8_0/turbo3 | +%17 | <+%5 | WikiText-2 |
| Memory retention (500) | FAIL | PASS | test_bitwidth_sweep |
| Math reasoning | %0-51 | %80+ | test_bitwidth_sweep |
| Key repetition | ✅ | ✅ | test_bitwidth_sweep |
| A/B quality | 33-51% | 70%+ | test_bitwidth_sweep |
