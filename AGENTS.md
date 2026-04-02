# AGENTS.md — PROJECT-TURBO

> TurboQuant asymmetric K/V cache compression — zero-swap 128K context on consumer GPUs.

## Project Structure

```
src/turbo/                     # Python package (public API)
  __init__.py                  # Exports: TurboContext, TurboBridge, TurboLlama, KVMonitor, setup
  turbo_adapter.py             # High-level adapter + ctypes C bridge bindings
  turbo_bridge.c               # C bridge (avoids ctypes struct-by-value ABI issues)
  enums.py                     # Auto-generated from C headers — DO NOT EDIT MANUALLY
src/llama-spiritbuun-cuda/     # llama.cpp fork (git submodule, spiritbuun/llama-cpp-turboquant-cuda)
scripts/
  build_turbo.sh               # Full build script (CMake + CUDA + C bridge compile)
  extract_enums.py             # Generates src/turbo/enums.py from ggml.h
tests/                         # All tests (no pytest fixtures, run as scripts or via pytest)
examples/                      # Usage examples
build/                         # Build artifacts (libllama.so, turbo_bridge.so, etc.)
```

## Build Commands

### Full build (libllama.so + CUDA kernels)
```bash
bash scripts/build_turbo.sh
```
This clones the spiritbuun fork, runs CMake with CUDA, and copies `.so` artifacts to `build/`.

### Compile C bridge only
```bash
gcc -shared -fPIC -o build/turbo_bridge.so src/turbo/turbo_bridge.c \
    -I./src/llama-spiritbuun-cuda/include \
    -L./build/bin -lllama -Wl,-rpath,./build/bin
```

### Regenerate enums from C headers
```bash
python scripts/extract_enums.py src/llama-spiritbuun-cuda/include/ggml.h src/turbo/enums.py
```

## Test Commands

All tests require these environment variables:
- `LD_LIBRARY_PATH=./build/bin` — shared libraries location
- `PYTHONPATH=src` — Python package path
- `TURBO_TEST_MODEL=/path/to/model.gguf` — model path (tests skip if unset)

### Run all tests via pytest
```bash
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src pytest tests/ -v
```

### Run a single test file
```bash
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src pytest tests/test_turbo_smoke.py -v
```

### Run a single test function
```bash
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src pytest tests/test_turbo_smoke.py::test_bridge_loads -v
```

### Run tests directly as scripts (smoke tests don't need a model)
```bash
LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_turbo_smoke.py
```

### Stress / reasoning / bitwidth sweep tests
```bash
TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_stress.py
TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_reasoning.py
TURBO_TEST_MODEL=/path/to/model.gguf LD_LIBRARY_PATH=./build/bin PYTHONPATH=src python tests/test_bitwidth_sweep.py
```

## Code Style

### Python (3.11+)
- **No external dependencies** — the `turbo` package has zero PyPI dependencies. Only `ctypes`, `logging`, `math`, `pathlib` from stdlib.
- Use `from __future__ import annotations` at top of modules.
- Type hints on all public method signatures. Use `Optional[X]` from `typing`, `list[int]` / `dict[str, ...]` (lowercase generics, 3.11+).
- Docstrings: triple-quoted, concise. One-line for simple methods, multi-line for complex classes.
- Logging via `logging.getLogger(__name__)`, never `print()` in library code (print is fine in tests).
- Private methods/attributes prefixed with `_` (e.g. `_find_bridge`, `_handle`, `_setup_bindings`).

### Formatting
- 4-space indentation. No trailing whitespace.
- Line length: soft limit ~100 chars, hard limit ~120 for readability.
- Section separators: `# ══════════...` banner comments for major sections within files.
- Imports grouped: stdlib first, then local. Alphabetical within groups.

### C Bridge (`turbo_bridge.c`)
- C99 style. `/* */` block comments.
- All public functions prefixed `turbo_` (e.g. `turbo_ctx_init`, `turbo_ctx_decode`).
- Handle-based API is primary; legacy global API exists for backward compatibility.
- Error returns: `0` for success, `-1` / `-2` / `NULL` for failure. Never raise from C.

### Naming Conventions
- Python classes: `PascalCase` (`TurboBridge`, `TurboContext`, `KVMonitor`).
- Python functions/methods: `snake_case` (`ctx_tokenize`, `kv_state`, `perf_get`).
- C structs: `snake_case` with `_t` suffix (`turbo_handle_t`, `turbo_perf_t`).
- Constants / enum maps: `UPPER_SNAKE_CASE` (`TURBO_TYPE_MAP`, `_BRIDGE_CANDIDATES`).
- Test functions: `test_` prefix, descriptive names (`test_asymmetric_kv_q80_turbo3`).

### Error Handling
- Python: raise `RuntimeError` with descriptive message for C bridge failures. Check return codes explicitly (`if ret != 0: raise RuntimeError(...)`).
- C: return error codes (`-1`, `NULL`), validate handles and pointers before use.
- `__del__` methods must catch exceptions silently (cleanup should never crash).
- Tests: use `assert` with informative messages (`assert ret == 0, f"init failed: {ret}"`).

### Key Constraints
- **Never hardcode enum values** — use `scripts/extract_enums.py` to regenerate `enums.py`.
- **Never use ctypes struct-by-value passing** for large structs — use `turbo_bridge.c` instead (ABI alignment issues on Linux x86_64).
- **K must always be `q8_0`** — compressing K with turbo causes PPL degradation due to softmax exponential error growth.
- **Zero Python dependencies** — do not add `import numpy`, `import torch`, etc. to the `turbo` package.
