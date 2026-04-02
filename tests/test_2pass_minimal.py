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

mp = os.environ["TURBO_TEST_MODEL"]
bl.turbo_load_model(mp.encode(), 999)
h = bl.turbo_ctx_init(2048, 8, 42, 1, 1)

toks = (ctypes.c_int * 8)()
nt = bl.turbo_ctx_tokenize(h, b"Hello", toks, 8, 1)
print(f"nt={nt}", flush=True)

bl.turbo_ctx_decode(h, toks, nt, 0)
print("prefill ok", flush=True)

for i in range(3):
    ret = bl.turbo_ctx_decode(h, toks, 1, nt + i)
    print(f"decode[{i}] ret={ret}", flush=True)

print("ALL DONE", flush=True)
