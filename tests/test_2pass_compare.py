import ctypes, os, sys

# Test: Compare baseline vs 2-PASS output for first decode token
# Uses the same bridge functions as bitwidth_sweep

bl = ctypes.CDLL("./build/turbo_bridge.so")
bl.turbo_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
bl.turbo_load_model.restype = ctypes.c_int
bl.turbo_ctx_init.argtypes = [ctypes.c_int] * 5
bl.turbo_ctx_init.restype = ctypes.c_void_p
bl.turbo_ctx_tokenize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
bl.turbo_ctx_tokenize.restype = ctypes.c_int
bl.turbo_ctx_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
bl.turbo_ctx_decode.restype = ctypes.c_int
bl.turbo_ctx_get_logits.restype = ctypes.POINTER(ctypes.c_float)
bl.turbo_ctx_n_vocab.restype = ctypes.c_int
bl.turbo_ctx_token_to_piece.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
]

mp = os.environ["TURBO_TEST_MODEL"]


def run_test(label):
    """Run decode and return first decoded token."""
    bl.turbo_load_model(mp.encode(), 999)
    h = bl.turbo_ctx_init(2048, 8, 42, 1, 1)
    toks = (ctypes.c_int * 8)()
    nt = bl.turbo_ctx_tokenize(h, b"Hello world", toks, 8, 1)

    # Prefill
    bl.turbo_ctx_decode(h, toks, nt, 0)

    # Decode 3 tokens (no logits access)
    for i in range(3):
        ret = bl.turbo_ctx_decode(h, toks, 1, nt + i)
        if ret != 0:
            print(f"{label}: decode[{i}] FAILED ret={ret}", flush=True)
            return
    print(f"{label}: 3 decode steps OK", flush=True)


# Test without 2-PASS (baseline)
os.environ.pop("TURBO_2PASS_SPLIT", None)
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
print("=== BASELINE (no 2-PASS) ===", flush=True)
run_test("BASELINE")

# Test with 2-PASS
os.environ["TURBO_2PASS_SPLIT"] = "1"
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
os.environ["TURBO_RECENT_WINDOW"] = "64"
print("=== 2-PASS ===", flush=True)
run_test("2PASS")

print("BOTH TESTS COMPLETED", flush=True)
