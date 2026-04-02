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

ret = bl.turbo_ctx_decode(h, toks, 1, nt)
print(f"decode ret={ret}", flush=True)

# Try n_vocab first (should be simple)
nv = bl.turbo_ctx_n_vocab(h)
print(f"n_vocab={nv}", flush=True)

# Try get_logits with NULL check
lp = bl.turbo_ctx_get_logits(h)
print(f"logits_ptr={lp}", flush=True)

if lp and nv > 0:
    # Just access first element
    v = lp[0]
    print(f"logits[0]={v}", flush=True)
    # Access first 5
    for i in range(5):
        print(f"logits[{i}]={lp[i]}", flush=True)

print("DONE", flush=True)
