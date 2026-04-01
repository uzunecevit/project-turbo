/* turbo_bridge.c — C wrapper for struct-by-value calls */
#include <stdlib.h>
#include <string.h>
#include "llama.h"

/* Global state */
static struct llama_model *g_model = NULL;
static struct llama_context *g_ctx = NULL;
static const struct llama_vocab *g_vocab = NULL;

int turbo_init(const char *model_path, int n_ctx, int type_k, int type_v,
               int n_gpu_layers, int flash_attn, int offload_kqv) {
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mp);
    if (!g_model) return -1;

    g_vocab = llama_model_get_vocab(g_model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.type_k = type_k;
    cp.type_v = type_v;
    cp.flash_attn_type = flash_attn;
    cp.offload_kqv = offload_kqv;

    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) return -2;

    return 0;
}

int turbo_tokenize(const char *text, int *tokens, int max_tokens) {
    if (!g_vocab) return -1;
    return llama_tokenize(g_vocab, text, strlen(text), tokens, max_tokens, true, false);
}

int turbo_decode(int *tokens, int n_tokens, int pos_offset) {
    if (!g_ctx) return -1;
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_offset + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;
    int ret = llama_decode(g_ctx, batch);
    llama_batch_free(batch);
    return ret;
}

int turbo_decode_chunked(int *tokens, int n_tokens, int pos_offset, int chunk_size) {
    if (!g_ctx) return -1;
    if (chunk_size <= 0) chunk_size = 512;
    for (int start = 0; start < n_tokens; start += chunk_size) {
        int count = (start + chunk_size < n_tokens) ? chunk_size : (n_tokens - start);
        int ret = turbo_decode(tokens + start, count, pos_offset + start);
        if (ret != 0) return ret;
    }
    return 0;
}

const float *turbo_get_logits(void) {
    if (!g_ctx) return NULL;
    return llama_get_logits(g_ctx);
}

int turbo_n_vocab(void) {
    if (!g_vocab) return 0;
    return llama_vocab_n_tokens(g_vocab);
}

int turbo_token_to_piece(int token, char *buf, int buf_size) {
    if (!g_vocab) return -1;
    return llama_token_to_piece(g_vocab, token, buf, buf_size, 0, false);
}

void turbo_free(void) {
    if (g_ctx) { llama_free(g_ctx); g_ctx = NULL; }
    if (g_model) { llama_model_free(g_model); g_model = NULL; }
    llama_backend_free();
}

void turbo_kv_cache_clear(void) {
    if (g_ctx) {
        llama_memory_t mem = llama_get_memory(g_ctx);
        if (mem) llama_memory_clear(mem, true);
    }
}
