/* turbo_bridge.c — Handle-based C bridge for TurboQuant inference
 *
 * v0.3: Handle-based architecture with backward compatibility.
 * - Model shared across contexts (loaded once)
 * - Each context gets its own handle (turbo_handle_t)
 * - Old global API preserved as wrappers
 * - Performance metrics via llama_perf_context
 * - KV state queries via llama_memory_seq_pos_*
 *
 * Compile:
 *   gcc -shared -fPIC -o build/turbo_bridge.so src/turbo/turbo_bridge.c \
 *       -I./src/llama-spiritbuun-cuda/include \
 *       -L./build/bin -lllama -Wl,-rpath,./build/bin
 */
#include <stdlib.h>
#include <string.h>
#include "llama.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * Handle type — each context gets its own handle
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    struct llama_context       *ctx;
    const struct llama_vocab   *vocab;
    int32_t                     id;
    int32_t                     seq_id;   /* default sequence id for multi-seq support */
} turbo_handle_t;

/* ═══════════════════════════════════════════════════════════════════════════════
 * Performance / KV state structs
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double  t_p_eval_ms;    /* prompt eval time (ms) */
    double  t_eval_ms;      /* token generation time (ms) */
    int32_t n_p_eval;       /* prompt tokens processed */
    int32_t n_eval;         /* tokens generated */
    int32_t n_reused;       /* graph reuses */
} turbo_perf_t;

typedef struct {
    int      n_ctx;         /* total context capacity */
    int      n_pos;         /* current position (last decoded token) */
    double   utilization;   /* n_pos / n_ctx */
    size_t   state_bytes;   /* serialized state size */
} turbo_kv_state_t;

/* ═══════════════════════════════════════════════════════════════════════════════
 * Model lifecycle — shared across all handles
 * ═══════════════════════════════════════════════════════════════════════════════ */

static struct llama_model *g_model = NULL;

int turbo_load_model(const char *model_path, int n_gpu_layers) {
    if (g_model) return 0;  /* already loaded */

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mp);
    if (!g_model) return -1;

    return 0;
}

void turbo_unload_model(void) {
    if (g_model) {
        llama_model_free(g_model);
        g_model = NULL;
    }
    llama_backend_free();
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Handle-based API — primary interface
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int g_next_id = 0;

turbo_handle_t *turbo_ctx_init(int n_ctx, int type_k, int type_v,
                                int flash_attn, int offload_kqv) {
    if (!g_model) return NULL;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx          = n_ctx;
    cp.type_k         = type_k;
    cp.type_v         = type_v;
    cp.flash_attn_type = flash_attn;
    cp.offload_kqv    = offload_kqv;
    cp.no_perf        = false;  /* enable C-level performance tracking */

    struct llama_context *ctx = llama_init_from_model(g_model, cp);
    if (!ctx) return NULL;

    turbo_handle_t *h = (turbo_handle_t *)calloc(1, sizeof(turbo_handle_t));
    if (!h) {
        llama_free(ctx);
        return NULL;
    }

    h->ctx     = ctx;
    h->vocab   = llama_model_get_vocab(g_model);
    h->id      = g_next_id++;
    h->seq_id  = 0;

    return h;
}

int turbo_ctx_tokenize(const turbo_handle_t *h, const char *text,
                        int *tokens, int max_tokens) {
    if (!h || !h->vocab) return -1;
    return llama_tokenize(h->vocab, text, strlen(text), tokens, max_tokens, true, false);
}

int turbo_ctx_decode(turbo_handle_t *h, int *tokens, int n_tokens, int pos_offset) {
    if (!h || !h->ctx) return -1;

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = pos_offset + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = h->seq_id;
        batch.logits[i]     = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;
    int ret = llama_decode(h->ctx, batch);
    llama_batch_free(batch);
    return ret;
}

int turbo_ctx_decode_chunked(turbo_handle_t *h, int *tokens, int n_tokens,
                              int pos_offset, int chunk_size) {
    if (!h || !h->ctx) return -1;
    if (chunk_size <= 0) chunk_size = 512;
    for (int start = 0; start < n_tokens; start += chunk_size) {
        int count = (start + chunk_size < n_tokens) ? chunk_size : (n_tokens - start);
        int ret = turbo_ctx_decode(h, tokens + start, count, pos_offset + start);
        if (ret != 0) return ret;
    }
    return 0;
}

const float *turbo_ctx_get_logits(const turbo_handle_t *h) {
    if (!h || !h->ctx) return NULL;
    return llama_get_logits(h->ctx);
}

int turbo_ctx_n_vocab(const turbo_handle_t *h) {
    if (!h || !h->vocab) return 0;
    return llama_vocab_n_tokens(h->vocab);
}

int turbo_ctx_token_to_piece(const turbo_handle_t *h, int token, char *buf, int buf_size) {
    if (!h || !h->vocab) return -1;
    return llama_token_to_piece(h->vocab, token, buf, buf_size, 0, false);
}

void turbo_ctx_kv_cache_clear(turbo_handle_t *h) {
    if (h && h->ctx) {
        llama_memory_t mem = llama_get_memory(h->ctx);
        if (mem) llama_memory_clear(mem, true);
    }
}

void turbo_ctx_free(turbo_handle_t *h) {
    if (!h) return;
    if (h->ctx) {
        llama_free(h->ctx);
        h->ctx = NULL;
    }
    free(h);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Performance metrics — C-level timing
 * ═══════════════════════════════════════════════════════════════════════════════ */

turbo_perf_t turbo_ctx_perf_get(const turbo_handle_t *h) {
    turbo_perf_t perf = {0, 0, 0, 0, 0};
    if (!h || !h->ctx) return perf;

    struct llama_perf_context_data data = llama_perf_context(h->ctx);
    perf.t_p_eval_ms = data.t_p_eval_ms;
    perf.t_eval_ms   = data.t_eval_ms;
    perf.n_p_eval    = data.n_p_eval;
    perf.n_eval      = data.n_eval;
    perf.n_reused    = data.n_reused;
    return perf;
}

void turbo_ctx_perf_reset(turbo_handle_t *h) {
    if (h && h->ctx) {
        llama_perf_context_reset(h->ctx);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KV state query — cache utilization
 * ═══════════════════════════════════════════════════════════════════════════════ */

turbo_kv_state_t turbo_ctx_kv_state(const turbo_handle_t *h) {
    turbo_kv_state_t state = {0, 0, 0.0, 0};
    if (!h || !h->ctx) return state;

    state.n_ctx = (int)llama_n_ctx(h->ctx);

    llama_memory_t mem = llama_get_memory(h->ctx);
    if (mem) {
        llama_pos pos_max = llama_memory_seq_pos_max(mem, h->seq_id);
        state.n_pos = (pos_max >= 0) ? (int)(pos_max + 1) : 0;
    }

    state.utilization = (state.n_ctx > 0) ? (double)state.n_pos / state.n_ctx : 0.0;
    state.state_bytes = llama_state_get_size(h->ctx);

    return state;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Backward-compatible API — wraps handle-based functions via global state
 * These functions are preserved for existing tests and TurboBridge class.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static turbo_handle_t *g_handle = NULL;

int turbo_init(const char *model_path, int n_ctx, int type_k, int type_v,
               int n_gpu_layers, int flash_attn, int offload_kqv) {
    if (g_handle) return 0;  /* already initialized */

    int ret = turbo_load_model(model_path, n_gpu_layers);
    if (ret != 0) return ret;

    g_handle = turbo_ctx_init(n_ctx, type_k, type_v, flash_attn, offload_kqv);
    if (!g_handle) return -2;

    return 0;
}

int turbo_tokenize(const char *text, int *tokens, int max_tokens) {
    return turbo_ctx_tokenize(g_handle, text, tokens, max_tokens);
}

int turbo_decode(int *tokens, int n_tokens, int pos_offset) {
    return turbo_ctx_decode(g_handle, tokens, n_tokens, pos_offset);
}

int turbo_decode_chunked(int *tokens, int n_tokens, int pos_offset, int chunk_size) {
    return turbo_ctx_decode_chunked(g_handle, tokens, n_tokens, pos_offset, chunk_size);
}

const float *turbo_get_logits(void) {
    return turbo_ctx_get_logits(g_handle);
}

int turbo_n_vocab(void) {
    return turbo_ctx_n_vocab(g_handle);
}

int turbo_token_to_piece(int token, char *buf, int buf_size) {
    return turbo_ctx_token_to_piece(g_handle, token, buf, buf_size);
}

void turbo_free(void) {
    if (g_handle) {
        turbo_ctx_free(g_handle);
        g_handle = NULL;
    }
    turbo_unload_model();
}

void turbo_kv_cache_clear(void) {
    turbo_ctx_kv_cache_clear(g_handle);
}
