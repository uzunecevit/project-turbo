#!/bin/bash
# Rollback script: 2-pass kernel modification geri alma
# Kullanım: bash scripts/rollback_2pass.sh
#
# Ne yapar:
# 1. Kernel dosyalarını backup'dan geri yükler
# 2. C bridge'i yeniden derler
# 3. Test çalıştırır

set -e

PROJECT="/home/ayandon/PROJECT-TURBO"
BACKUP="$PROJECT/backup/pre-2pass-kernel"

echo "=== GERİ DÖNÜŞ: 2-Pass Kernel Modifikasyonu ==="
echo ""

# 1. Kernel dosyalarını geri yükle
echo "[1/4] Kernel dosyalarını geri yüklüyor..."
cp "$BACKUP/llama-kv-cache.h" "$PROJECT/src/llama-spiritbuun-cuda/src/llama-kv-cache.h"
cp "$BACKUP/llama-kv-cache.cpp" "$PROJECT/src/llama-spiritbuun-cuda/src/llama-kv-cache.cpp"
cp "$BACKUP/fattn-vec.cuh" "$PROJECT/src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/fattn-vec.cuh"
cp "$BACKUP/fattn.cu" "$PROJECT/src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/fattn.cu"
cp "$BACKUP/fattn-common.cuh" "$PROJECT/src/llama-spiritbuun-cuda/ggml/src/ggml-cuda/fattn-common.cuh"
echo "  OK"

# 2. C bridge yeniden derle
echo "[2/4] C bridge yeniden derliyor..."
cd "$PROJECT"
gcc -shared -fPIC -o build/turbo_bridge.so src/turbo/turbo_bridge.c \
    -I./src/llama-spiritbuun-cuda/include \
    -L./build/bin -lllama -Wl,-rpath,./build/bin 2>&1
echo "  OK"

# 3. Smoke test
echo "[3/4] Smoke test çalıştırıyor..."
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python -c "
from turbo import TurboBridge
b = TurboBridge()
b.lib.turbo_load_model('/home/ayandon/KAPTAN/modeller/qwen3-8b-q4_k_m.gguf'.encode(), -1)
h = b.ctx_init(512, 42, 42, 1, 1)
b._handle = h
tokens = b.tokenize('Hello')
b.decode(tokens, pos=0)
sampler = b.sampler_init(top_k=20, top_p=0.95, min_p=0.0, temp=0.6, seed=42)
t = b.sampler_sample(sampler)
text = b.token_to_piece(t)
print(f'Sample OK: token={t} text={repr(text)}')
b.sampler_free(sampler)
b.free()
" 2>/dev/null
echo "  OK"

# 4. Git geri alma (opsiyonel)
echo "[4/4] Git tag kontrol..."
echo "  Geri dönmek için: git checkout pre-2pass-kernel"
echo "  Veya: git reset --hard pre-2pass-kernel"

echo ""
echo "=== GERİ DÖNÜŞ TAMAMLANDI ==="
echo "Sistem pre-2pass-kernel durumuna döndü."
