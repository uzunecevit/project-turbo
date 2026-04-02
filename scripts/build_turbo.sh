#!/usr/bin/env bash
set -euo pipefail

# PROJECT-TURBO | Surgical .so Factory
# Compiles TurboQuant-enabled libllama.so from Madreag fork
# Target: RTX 3060 (Ampere 8.6), CUDA 13.2, zero-swap 128K context

TURBO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${TURBO_ROOT}/build"
SRC_DIR="${TURBO_ROOT}/src/llama-spiritbuun-cuda"
ENUMS_OUT="${TURBO_ROOT}/src/turbo/enums.py"

FORK_URL="${FORK_URL:-https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git}"
FORK_BRANCH="${FORK_BRANCH:-feature/turboquant-kv-cache}"

# ── CUDA Detection ──────────────────────────────────────────────────────────
if command -v nvcc &>/dev/null; then
    NVCC="$(which nvcc)"
elif [ -x "/opt/cuda/bin/nvcc" ]; then
    NVCC="/opt/cuda/bin/nvcc"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
    NVCC="/usr/local/cuda/bin/nvcc"
else
    echo "[TURBO] ERROR: nvcc not found in PATH or standard locations"
    echo "[TURBO] Install CUDA toolkit or set PATH accordingly"
    exit 1
fi

CUDA_VER="$("$NVCC" --version | grep 'release' | awk '{print $5}' | tr -d ',')"
echo "[TURBO] CUDA: $CUDA_VER ($NVCC)"
echo "[TURBO] Arch: native ($(nproc) cores)"

# ── Clone Fork ───────────────────────────────────────────────────────────────
if [ ! -d "$SRC_DIR" ] || [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    echo "[TURBO] Cloning $FORK_URL ($FORK_BRANCH)..."
    mkdir -p "$(dirname "$SRC_DIR")"
    git clone --depth 1 --branch "$FORK_BRANCH" "$FORK_URL" "$SRC_DIR"
else
    echo "[TURBO] Source tree present at $SRC_DIR"
    # Update if it's a worktree or regular clone
    if [ -d "$SRC_DIR/.git" ] || [ -f "$SRC_DIR/.git" ]; then
        echo "[TURBO] Fetching latest from $FORK_URL..."
        git -C "$SRC_DIR" fetch origin --depth=1 2>/dev/null || true
        git -C "$SRC_DIR" reset --hard "origin/$FORK_BRANCH" 2>/dev/null || true
    fi
fi

# ── Apply OMERTA Patches ────────────────────────────────────────────────────
PATCH_DIR="${TURBO_ROOT}/patches"
if [ -d "$PATCH_DIR" ]; then
    echo "[TURBO] Applying OMERTA patches..."
    for patch in "$PATCH_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            echo "[TURBO]   $(basename "$patch")"
            git -C "$SRC_DIR" apply --check "$patch" 2>/dev/null && \
                git -C "$SRC_DIR" apply "$patch" || \
                echo "[TURBO]   SKIP (already applied or conflict)"
        fi
    done
fi

# ── CMake Configure ─────────────────────────────────────────────────────────
echo "[TURBO] Configuring CMake..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_C_FLAGS="-march=native -mtune=native -O3 -fno-finite-math-only" \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3 -fno-finite-math-only" \
    -DCMAKE_CUDA_FLAGS="-Xcompiler=-march=native,-O3,-fno-finite-math-only" \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"

# ── Build ────────────────────────────────────────────────────────────────────
echo "[TURBO] Building libllama.so ($(nproc) threads)..."
cmake --build "$BUILD_DIR" --target llama -j"$(nproc)"

# ── Collect .so files ───────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
find "$BUILD_DIR" \( -name "libllama.so" -o -name "libllama.so.*" -o -name "libggml*.so*" \) \
    -exec cp -v {} "$BUILD_DIR/" \;

echo ""
echo "[TURBO] ═══════════════════════════════════════════════════"
echo "[TURBO] Build complete."
echo "[TURBO] Artifacts in: $BUILD_DIR/"
ls -lh "$BUILD_DIR"/lib*.so* 2>/dev/null || echo "[TURBO] WARNING: No .so files found"
echo "[TURBO] ═══════════════════════════════════════════════════"
