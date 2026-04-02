"""Cross-model 2-PASS validation: EXACT MATCH across all architectures."""

import sys, os

sys.path.insert(0, "src")
from turbo.turbo_adapter import TurboContext

MODELS = [
    ("Qwen3-8B", "/home/ayandon/KAPTAN/modeller/qwen3-8b-q4_k_m.gguf"),
    ("Qwen3.5-9B", "/home/ayandon/KAPTAN/modeller/Qwen3.5-9B-Q4_K_M.gguf"),
    (
        "DeepSeek-Coder-V2-Lite",
        "/home/ayandon/KAPTAN/modeller/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
    ),
    ("Mamba-7B", "/home/ayandon/KAPTAN/modeller/Mamba-7B-Q4_K_M.gguf"),
    ("Qwen-4B", "/home/ayandon/KAPTAN/modeller/Qwen-4B-Q4_K_M.gguf"),
]


def run(label, model_path, env, n_gen=5):
    for k, v in env.items():
        os.environ[k] = v
    for k in ["TURBO_2PASS_SPLIT", "GGML_CUDA_DISABLE_GRAPHS", "TURBO_RECENT_WINDOW"]:
        if k not in env:
            os.environ.pop(k, None)
    try:
        ctx = TurboContext(
            model_path, n_ctx=2048, n_batch=512, cache_type="turbo4", n_gpu_layers=999
        )
        tokens = ctx.tokenize("The meaning of life is")
        ctx.decode(tokens, 0)
        out = []
        for i in range(n_gen):
            t = tokens[-1] if i == 0 else out[-1]
            logits = ctx.decode([t], len(tokens) + i)
            out.append(max(range(len(logits)), key=lambda j: logits[j]))
        text = ctx.detokenize(out)
        del ctx
        return out, text, None
    except Exception as e:
        return None, None, str(e)


print("=" * 70, flush=True)
print("CROSS-MODEL 2-PASS VALIDATION", flush=True)
print("=" * 70, flush=True)

results = []
for name, path in MODELS:
    print(f"\n{name}: ", end="", flush=True)

    b_out, b_text, b_err = run("base", path, {"GGML_CUDA_DISABLE_GRAPHS": "1"}, 5)
    if b_err:
        print(f"BASELINE FAILED: {b_err}", flush=True)
        results.append((name, "FAIL", "baseline crash", None))
        continue

    tp_out, tp_text, tp_err = run(
        "2pass",
        path,
        {
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "TURBO_2PASS_SPLIT": "1",
            "TURBO_RECENT_WINDOW": "64",
        },
        5,
    )
    if tp_err:
        print(f"2-PASS FAILED: {tp_err}", flush=True)
        results.append((name, "FAIL", "2pass crash", None))
        continue

    if b_out == tp_out:
        print(f"EXACT MATCH  [{b_out}]", flush=True)
        results.append((name, "EXACT", None, b_out))
    else:
        common = sum(1 for a, b in zip(b_out, tp_out) if a == b)
        print(f"DIV {common}/{len(b_out)}  base={b_out} 2pass={tp_out}", flush=True)
        results.append((name, f"{common}/{len(b_out)}", None, b_out))

print(f"\n{'=' * 70}", flush=True)
print("SUMMARY", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"{'Model':<28} | {'Result':<12} | {'Tokens'}", flush=True)
print(f"{'-' * 28}-+-{'-' * 12}-+-{'-' * 20}", flush=True)
for name, result, err, tokens in results:
    tok_str = str(tokens) if tokens else "N/A"
    if err:
        print(f"{name:<28} | {result:<12} | {err}", flush=True)
    else:
        print(f"{name:<28} | {result:<12} | {tok_str}", flush=True)
print(f"{'=' * 70}", flush=True)
print("DONE", flush=True)
