"""Quick long context test: 512-2048 tokens, baseline vs 2-PASS."""

import sys, os

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODEL = os.environ["TURBO_TEST_MODEL"]
base = "The quick brown fox jumps over the lazy dog. "

os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
ctx0 = TurboContext(
    MODEL, n_ctx=8192, n_batch=512, cache_type="turbo4", n_gpu_layers=999
)
all_tokens = ctx0.tokenize(base * 500)  # ~4500 tokens
del ctx0


def chunked_decode(ctx, tokens, chunk_size=512):
    for start in range(0, len(tokens), chunk_size):
        ctx.decode(tokens[start : start + chunk_size], start)


def gen_greedy(ctx, prompt_tokens, n_gen=3):
    chunked_decode(ctx, prompt_tokens)
    out = []
    for i in range(n_gen):
        t = prompt_tokens[-1] if i == 0 else out[-1]
        logits = ctx.decode([t], len(prompt_tokens) + i)
        out.append(max(range(len(logits)), key=lambda j: logits[j]))
    return out


for ctx_len in [512, 1024, 2048, 4096]:
    prompt = all_tokens[:ctx_len]

    os.environ.pop("TURBO_2PASS_SPLIT", None)
    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    ctx = TurboContext(
        MODEL, n_ctx=8192, n_batch=512, cache_type="turbo4", n_gpu_layers=999
    )
    base_out = gen_greedy(ctx, prompt)
    base_text = ctx.detokenize(base_out)
    del ctx

    os.environ["TURBO_2PASS_SPLIT"] = "1"
    os.environ["TURBO_RECENT_WINDOW"] = "64"
    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    ctx = TurboContext(
        MODEL, n_ctx=8192, n_batch=512, cache_type="turbo4", n_gpu_layers=999
    )
    tp_out = gen_greedy(ctx, prompt)
    tp_text = ctx.detokenize(tp_out)
    del ctx

    match = (
        "EXACT"
        if base_out == tp_out
        else f"{sum(1 for a, b in zip(base_out, tp_out) if a == b)}/{len(base_out)}"
    )
    print(
        f"Ctx {ctx_len}: match={match} base={base_text[:30]!r} 2pass={tp_text[:30]!r}",
        flush=True,
    )

print("DONE", flush=True)
