# Contributing to PROJECT-TURBO

Katkıda bulunmak isteyen herkese açık! İşte yapmanız gerekenler:

## Nasıl Katkıda Bulunulur?

1. **Fork** edin
2. **Feature branch** oluşturun: `git checkout -b feature/your-feature`
3. **Değişikliklerinizi** yapın
4. **Test edin**: `LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_smoke.py`
5. **Commit** atın: `git commit -m 'feat: your feature description'`
6. **Push** edin: `git push origin feature/your-feature`
7. **Pull Request** açın

## Katkı Kuralları

- **Performans > Her şey** — Her değişiklik performansı korumalı veya iyileştirmeli
- **Bağımlılık ekleme** — Proje sıfır Python bağımlılığı ile çalışmalı
- **Enum'ları hardcode etme** — `scripts/extract_enums.py` kullanarak otomatik üret
- **Test yaz** — Değişikliklerinizi `tests/` altına test ile destekleyin
- **C bridge** — Struct-by-value sorunları için `turbo_bridge.c` kullanın, doğrudan ctypes ile struct passing yapmayın

## Mimarik Notlar

### C Bridge Neden Var?

Python ctypes, Linux x86_64'da büyük struct'ları by-value passing'te stack alignment sorunu yaşıyor. `llama_init_from_model(model, params)` C'de çalışırken Python ctypes'te segfault verebilir.

**Çözüm:** `turbo_bridge.c` — tüm struct passing'i C tarafında yapılır, Python'a sadece basit tipler döndürülür.

### Asimetrik K/V Stratejisi

K her zaman `q8_0` (korunmuş), V `turbo3`/`turbo4`/`turbo2` ile sıkıştırılır. **K'yı turbo3 ile sıkıştırmayın** — softmax'ta exponential hata büyümesi nedeniyle PPL felakete uğrar.

### Fork Seçimi

`spiritbuun/llama-cpp-turboquant-cuda` kullanıyoruz çünkü:
- En aktif CUDA fork
- TURBO3_0/TURBO4_0/TURBO2_0 enum'ları destekliyor
- Flash Attention + CUDA F16 + FA_ALL_QUANTS
- Norm correction ile PPL iyileştirmesi

## Sorun Bildirme

GitHub Issues üzerinden bildirin. Lütfen şunları ekleyin:
- GPU modeli ve VRAM
- CUDA sürümü
- Python sürümü
- Model adı ve quant tipi
- Hata mesajı ve tam traceback

## Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.
