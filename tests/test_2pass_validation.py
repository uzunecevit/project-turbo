import sys, os

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODEL = os.environ.get("TURBO_TEST_MODEL", "")


def gen_greedy(ctx, prompt, n_tokens=5):
    tokens = ctx.tokenize(prompt)
    logits = ctx.decode(tokens, 0)
    output_tokens = []
    for i in range(n_tokens):
        tok = tokens[-1] if i == 0 else output_tokens[-1]
        logits = ctx.decode([tok], len(tokens) + i)
        best = max(range(len(logits)), key=lambda j: logits[j])
        output_tokens.append(best)
    text = ctx.detokenize(output_tokens)
    return output_tokens, text


os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
os.environ.pop("TURBO_2PASS_SPLIT", None)

print("=== BASELINE ===", flush=True)
b1 = TurboContext(MODEL, n_ctx=2048, cache_type="turbo4", n_gpu_layers=999)
tokens_b, text_b = gen_greedy(b1, "The meaning of life is", 5)
print(f"  tokens: {tokens_b}", flush=True)
print(f"  text:   {text_b}", flush=True)
del b1

os.environ["TURBO_2PASS_SPLIT"] = "1"
os.environ["TURBO_RECENT_WINDOW"] = "64"
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"

print("=== 2-PASS ===", flush=True)
b2 = TurboContext(MODEL, n_ctx=2048, cache_type="turbo4", n_gpu_layers=999)
tokens_2, text_2 = gen_greedy(b2, "The meaning of life is", 5)
print(f"  tokens: {tokens_2}", flush=True)
print(f"  text:   {text_2}", flush=True)
del b2

print("=" * 60, flush=True)
match = tokens_b == tokens_2
print(f"Token match: {match}", flush=True)
if match:
    print("RESULT: EXACT MATCH", flush=True)
else:
    for i in range(min(len(tokens_b), len(tokens_2))):
        if tokens_b[i] != tokens_2[i]:
            print(
                f"First divergence at step {i}: {tokens_b[i]} vs {tokens_2[i]}",
                flush=True,
            )
            break
    common = sum(1 for a, b in zip(tokens_b, tokens_2) if a == b)
    seq_pct = 100.0 * common / max(len(tokens_b), len(tokens_2))
    print(f"Sequence similarity: {seq_pct:.1f}%", flush=True)
print("=" * 60, flush=True)
