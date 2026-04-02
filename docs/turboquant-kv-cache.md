# Dual-Context Architecture — KAPTAN

> **Amaç:** Aynı model ağırlıkları üzerinde iki ayrı KV cache (Commander + Scout)
> **Fayda:** Zero-swap latency, model sharing, parallel inference

---

## Mimari

```
┌─────────────────────────────────────────────────┐
│  Model Weights (qwen3.5-9b-q4_k_m.gguf)        │
│  ┌─────────────────────────────────────────┐    │
│  │  Commander KV Cache (turbo4)            │    │
│  │  Full context, primary inference        │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │  Scout KV Cache (turbo3)                │    │
│  │  Context scaling, draft generation      │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │  Arbitrator (Qwen-4B, q8_0)             │    │
│  │  CRITICAL decisions only                │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Avantajlar

### 1. Zero-Swap Latency
- Same model weights (pointer sharing)
- No RAM dump → 0ms context switching
- Parallel KV cache operations

### 2. Anchor + Tail Sync
- System prompt (Anchor) → Scout'a enjekte
- Last 512 tokens (Tail) → Scout'a enjekte
- Karakter senkronizasyonu garanti

### 3. Arbitrator Pattern
- 4B model: CRITICAL + disagreement resolution
- %5-10 usage rate (VRAM efficient)
- 9B bias loop koruması

## Savaş Düzeni

| Birim | Model | Config | VRAM | Amaç |
|---|---|---|---|---|
| Commander | Qwen3.5-9B | turbo4 | 6.0 GB | Primary inference |
| Scout | Qwen3.5-9B | turbo3 | 4.2 GB | Draft + scaling |
| Arbitrator | Qwen-4B | q8_0 | 2.8 GB | Critical decisions |

## VRAM Watchdog

- Hard limit: 10.5 GB
- Commander + Scout: ~10.2 GB
- Arbitrator: 2.8 GB (lazy loaded)

## Python API

```python
from turbo import TurboDualContext

# Create dual-context
dc = TurboDualContext(
    model_path="/path/to/model.gguf",
    commander_config={"type_k": 8, "type_v": 42},  # turbo4
    scout_config={"type_k": 8, "type_v": 41},      # turbo3
)

# Commander inference
response = dc.commander.generate("Hello, world!")

# Scout draft
draft = dc.scout.generate("Draft response")

# Arbitrator (if needed)
if dc.needs_arbitration(commander_resp, scout_resp):
    verdict = dc.arbitrator.evaluate(commander_resp, scout_resp)

dc.free()
```

## Implementation Status

- [x] C bridge: dual-context support (turbo_ctx_init)
- [x] Python: TurboBridge handle-based
- [ ] Python: TurboDualContext wrapper
- [ ] Anchor + Tail sync
- [ ] Arbitrator integration
- [ ] VRAM watchdog
