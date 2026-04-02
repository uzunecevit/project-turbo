import ctypes, os, sys

bl = ctypes.CDLL("./build/turbo_bridge.so")
bl.turbo_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
bl.turbo_load_model.restype = ctypes.c_int
bl.turbo_ctx_init.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
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
print(f"nt={nt}", flush=True)

bl.turbo_ctx_decode(h, toks, nt, 0)
print("prefill ok", flush=True)

ret = bl.turbo_ctx_decode(h, toks, 1, nt)
print(f"decode ret={ret}", flush=True)

nv = bl.turbo_ctx_n_vocab(h)
print(f"n_vocab={nv}", flush=True)

lp = bl.turbo_ctx_get_logits(h)
print(f"logits_ptr={lp}", flush=True)

if lp:
    vals = [lp[i] for i in range(10)]
    print(f"first10={vals}", flush=True)
    best = max(range(min(nv, 1000)), key=lambda j: lp[j])
    print(f"BEST_TOKEN={best}", flush=True)
else:
    print("logits_ptr is NULL", flush=True)

print("DONE", flush=True)
