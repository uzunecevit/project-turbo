# TurboQuant (Google) vs spiritbuun Implementasyonu — Teknik Karşılaştırma

## Makale Özeti

**TurboQuant** (Zandieh & Mirrokni, Google Research, ICLR 2026) — yüksek boyutlu vektörler için near-optimal distortion rate'e sahip online vektör quantization algoritması.

**İki aşamalı tasarım:**
1. **Random rotation** → koordinatlar Beta dağılımına dönüşür → Lloyd-Max scalar quantizer (b-1 bit)
2. **QJL residual** → 1-bit inner product bias düzeltmesi

**Teorik bound:** Shannon lower bound'un ~2.7x faktör içinde.

## spiritbuun Implementasyonu

`spiritbuun/llama-cpp-turboquant-cuda` — llama.cpp fork'unda TurboQuant CUDA kernel'ları.

**Formatlar:**
- Turbo2 (type_v=43): 2-bit PolarQuant
- Turbo3 (type_v=41): 3-bit PolarQuant + QJL
- Turbo4 (type_v=42): 4-bit PolarQuant + QJL

## Temel Farklar

### Algoritma

| Konu | Google Makalesi | spiritbuun (bizim) |
|---|---|---|
| Dönüşüm | Random rotation + Beta dağılımı | PolarQuant (polar koordinat) |
| Quantizer | Lloyd-Max scalar quantizer | PolarQuant scalar quantizer |
| Residual | QJL (1-bit) | QJL (1-bit) — aynı |
| Entropy encoding | Teorik %5 tasarruf, uygulanmamış | Yok |

### Neden PolarQuant?

Random rotation yerine polar koordinat dönüşümü seçilmiş. Olası nedenler:

1. **Runtime hızı:** Polar dönüşüm (açı + yarıçap hesaplama) GPU'da matris çarpımından daha hızlı
2. **FA entegrasyonu:** PolarQuant hesaplamaları Flash Attention kernel'ının içine gömülmüş olabilir
3. **Cache locality:** Polar koordinatlar angle/radius olarak saklanır, attention hesabında tek erişim yeterli

### KV Stratejisi

| Konu | Makale | PROJECT-TURBO |
|---|---|---|
| K quantization | Belirtilmemiş (simetrik varsayılır) | q8_0 (8-bit, düşük bias) |
| V quantization | TurboQuant (3.5 bit önerilen) | turbo3 (3-bit) |
| Strateji | Simetrik | Asimetrik |

**Neden asimetrik?** Makalenin kendi teorisi destekliyor:
- MSE-optimize quantizer'lar inner product'ta **biased**
- Attention'da softmax(Q·K) hesaplanır → K'daki bias **exponential** büyür
- V'deki bias ise **lineer** toplama → orantılı kalır
- Sonuç: K'yi düşük bias tut (q8_0), V'yi sıkıştır (turbo3)

### Flash Attention Bağımlılığı

**spiritbuun:** turbo3 Flash Attention'sız çalışmaz (sessiz hatalar, PPL patlar)

**Makale:** Random rotation tabanlı orijinal TurboQuant FA bağımlılığına sahip değil

**Neden:** PolarQuant hesaplamaları FA kernel'ının içine gömülü — angle/radius hesapları FA'nın attention computation loop'unda yapılıyor. FA olmayan path bu hesaplamaları desteklemiyor.

## Doğrulanan Uyumluluklar

1. **V için 3-bit makul:** Makale 2.5-3.5 bit aralığını doğruluyor. turbo3 (3-bit) bu aralıkta.
2. **Asimetrik K/V haklı:** Makale MSE-optimize quantizer'ların inner product bias'ını kanıtlıyor. K=q8_0 stratejisi bu teoriyle uyumlu.
3. **QJL residual aynı:** Her iki yaklaşım da QJL'yi 1-bit bias düzeltmesi olarak kullanıyor.
4. **Needle-in-haystack:** Makalede perfect retrieval, bizde de memory retention (540+ token gap) çalışıyor.

## Bilinen Farklar ve Limitler

1. **PolarQuant vs Random Rotation:** Hangisinin distortion'ı daha düşük karşılaştırılmamış. spiritbuun'un tercihi hız odaklı.
2. **"turbo3" ≠ orijinal TurboQuant:** Format isimleri aynı ama algoritmik temel farklı.
3. **Multi-step reasoning:** Makale test etmemiş. Biz systematic quantization bias bulduk (54 hatası).
4. **Teorik bound:** Makale Shannon lower bound kanıtı sunuyor. Bizim implementasyonda teorik bound yok — sadece deneysel validasyon var.
5. **Entropy encoding:** Makale %5 tasarruf potansiyeli belirtiyor ama uygulamamış. 3-bit → 2.8 bit teorik olarak mümkün ama decode latency artabilir.

## KAPTAN Entegrasyonu İçin Notlar

- turbo3 güvenilir: memory retention, determinism, no KV leak kanıtlandı
- multi-step math'de precision kaybı var → agentic approach (external calculator) ile çözülür
- A/B quality: pure transformer'da 51.4%, hybrid'de 33.7% → mimariye göre model seçimi önemli
- C-level perf metrics (no_perf=false) enable edilmeli, aksi halde timing 0 döner
