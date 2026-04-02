import ctypes, os, sys

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

mp = os.environ["TURBO_TEST_MODEL"]
bl.turbo_load_model(mp.encode(), 999)
h = bl.turbo_ctx_init(2048, 8, 42, 1, 1)

toks = (ctypes.c_int * 8)()
nt = bl.turbo_ctx_tokenize(h, b"Hello", toks, 8, 1)
bl.turbo_ctx_decode(h, toks, nt, 0)
print("prefill ok", flush=True)

# Try WITHOUT GGML_CUDA_DISABLE_GRAPHS
# With CUDA graphs, decode goes through graph replay — no 2-PASS
ret = bl.turbo_ctx_decode(h, toks, 1, nt)
print(f"decode ret={ret}", flush=True)

nv = bl.turbo_ctx_n_vocab(h)
print(f"n_vocab={nv}", flush=True)

lp = bl.turbo_ctx_get_logits(h)
print(f"logits_ptr={lp}", flush=True)

if lp and nv > 0:
    v = lp[0]
    print(f"logits[0]={v}", flush=True)

print("DONE", flush=True)
