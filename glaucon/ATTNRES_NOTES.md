# Attention Residuals — Experiment Notes

## Background

Attention Residuals (AttnRes) replaces standard residual connections with learned
inter-layer routing. Instead of `x_t = x_{t-1} + f_t(x_{t-1})`, each layer output
`y_t = f_t(h_t)` is stored, and the input to each subsequent layer is a weighted sum
of all previous outputs: `h_t = sum_s a_{t,s} * y_s`.

Reference: Su et al., "Attention Residuals" (arXiv:2603.15031)

## Current implementation: Paper-faithful Block AttnRes

We implement the paper's **Block AttnRes** variant: sublayer-level routing over
block-compressed sources. Each sublayer (attention or MLP) has its own learned
query vector that routes over completed block representations plus the current
intra-block partial sum.

### Configuration

Single env var: `ATTN_RES_BLOCK_SIZE` (default: 4)

Block size is measured in sublayers (each transformer layer has 2: attn + MLP).
So `ATTN_RES_BLOCK_SIZE=4` means 2 transformer layers per compression block.

With L=11 transformer layers (22 sublayers):
- `BLOCK_SIZE=4`: 5 full blocks + 1 partial (1 layer) + embedding = ~7 sources max
- `BLOCK_SIZE=6`: 3 full blocks + 1 partial (2 layers) + embedding = ~5 sources max
- `BLOCK_SIZE=8`: 2 full blocks + 1 partial (3 layers) + embedding = ~4 sources max
- `BLOCK_SIZE=22`: 1 partial block (all layers) + embedding = 2 sources

The paper uses ~8 blocks for a 48-layer MoE model. Smaller block sizes give finer
routing granularity but more sources to attend over.

### Architecture

Parameters: `attn_res_queries` of shape `(2L+1, model_dim)` — one query per sublayer
consumer (attn + MLP per layer) plus a final output query. Stored in fp32 (via
`CONTROL_TENSOR_NAME_PATTERNS`), cast to bf16 in the forward pass.

Forward loop (pseudocode):
```python
blocks = []           # completed block representations
partial = embedding   # running intra-block sum

for i in range(num_layers):
    # --- Attention sublayer ---
    h = block_attn_res(blocks, partial, queries[2*i])
    if i % layers_per_block == 0:       # block boundary
        blocks.append(partial)
        partial = None
    attn_out = attn_i(norm(h))
    partial = (partial + attn_out) if partial is not None else attn_out

    # --- MLP sublayer ---
    h = block_attn_res(blocks, partial, queries[2*i + 1])
    mlp_out = mlp_i(norm(h))
    partial = partial + mlp_out

out = block_attn_res(blocks, partial, queries[2*L])
return norm(out)
```

Key properties:
- **Embedding is isolated**: boundary fires at layer 0, so embedding is always its
  own block. The paper notes the model assigns substantial attention to embedding.
- **h is freshly routed each sublayer**: no residual accumulates on h.
- **Sublayer outputs accumulate on partial**: traditional residual within blocks.
- **partial is an attnres source**: the MLP can attend to the current partial (which
  includes the attn output), preserving the attn→MLP information path without a
  hardcoded skip connection. This is key — see "Internal residual" below.
- **Uneven last block is fine**: just stays as a smaller partial block.

### Routing function (`_block_attn_res`)

```python
sources = blocks + [partial]             # list of [B, T, D]
V = torch.stack(sources)                 # [N, B, T, D]
K = RMSNorm(V)                           # normalize keys
q = query.to(bf16)                       # fp32 → bf16
logits = (q * K).sum(dim=-1)             # [N, B, T]
w = softmax(logits, dim=0).unsqueeze(-1) # [N, B, T, 1]
return (w * V).sum(dim=0)                # [B, T, D]
```

This is the paper's formulation: `a_{c,s} ∝ exp(w_c · RMSNorm(y_s))`.

## What we removed

AttnRes subsumes several existing mechanisms:
- U-Net skip connections (`skip_weights`, encoder/decoder split)
- `resid_mix` (learned x/x0 mixing per block)
- `attn_scale` / `mlp_scale` (per-dim output gating)
- `ln_scale` (1/sqrt(layer) normalization scaling)
- `Block` class entirely — `GPT` now holds `attns` (ModuleList of CausalSelfAttention)
- Static routing mode (`attn_res_logits`, causal mask) — superseded by learned queries
- Sublayer granularity mode — superseded by paper-faithful block compression

## Implementation details

### torch.compile performance

Benchmarked on H100. Key finding from earlier work: **explicit scalar*tensor
accumulation loops compile 2x faster than stack+einsum** (2.87ms vs 6.37ms for
routing-only overhead). The compiler fuses scalar broadcast multiply + add chains
into a single kernel.

```python
# Fast (compiles well):
x_in = weights[row, 0] * ys[0]
for s in range(1, row + 1):
    x_in = x_in + weights[row, s] * ys[s]

# Slow (materializes intermediate tensor):
stk = torch.stack(ys, dim=0)
x_in = torch.einsum('s,sbtd->btd', weights[row, :len(ys)], stk)
```

**TODO**: The current `_block_attn_res` uses the stack+sum pattern for clarity.
This needs benchmarking on H100 — if step time regresses, rewrite to use the
Python-loop accumulation pattern (compute logits in a loop, then accumulate
weighted values in a loop). The number of sources is small (≤~7 with default
config) so either approach may be fine.

### FP32 query storage

Queries are stored in fp32 via `CONTROL_TENSOR_NAME_PATTERNS` and cast to bf16
in `_block_attn_res`. This helps optimizer stability without contaminating the
forward pass with fp32 compute (the earlier fp32 contamination bug that cascaded
through all weighted sums).

### Internal residual via partial block

In the previous block-level implementation, we needed an explicit internal residual
(`h + attn_out`) — without it, the model barely learned (stuck at ~4.9 loss). This
was because the block-level routing gave the MLP no direct access to the attn output.

In the paper-faithful implementation, this problem doesn't arise: the MLP sublayer
routes over `blocks + [partial]`, and `partial` already contains the attn output
(via `partial = partial + attn_out`). The MLP's query can attend to this partial
to recover the attn→MLP information path. No hardcoded skip needed.

### Zero-init for standard residual equivalence

Zero queries → `w · RMSNorm(y) = 0` for all sources → uniform softmax → equal-weight
sum. At initialization, this is equivalent to standard residual connections.

## Historical proxy run results (200 steps, 1xH100)

These used the old static/block implementation. Kept for reference.

Baseline: standard residual with skip connections, resid_mix, attn/mlp scale, ln_scale.

| Config                     | Step 100 | Step 200 | val_bpb@200 | step_avg | peak mem  |
|----------------------------|----------|----------|-------------|----------|-----------|
| **Baseline**               | 3.43     | 2.85     | (1.497@500) | 637ms    | 21718 MiB |
| Block+intres static LR=0.1 | 3.34     | 2.98     | 1.7896      | 620ms    | 19806 MiB |
| Block+intres static LR=0.5 | 3.29     | 2.94     | 1.7662      | 621ms    | 19806 MiB |
| Block+intres static LR=1.0 | 3.26     | 2.93     | 1.7559      | 621ms    | 19806 MiB |
| Sublayer static LR=0.5     | 3.26     | 2.92     | 1.7563      | 865ms    | 23173 MiB |

## Next steps

### Proxy sweep on 1xH100 (200–500 steps)

Primary goal: verify step time is competitive with baseline (637ms target).
Secondary goal: compare loss trajectories across block sizes.

Proposed runs:
1. `ATTN_RES_BLOCK_SIZE=4  ATTN_RES_LR=0.1`  — default (2 layers/block, ~7 sources)
2. `ATTN_RES_BLOCK_SIZE=8  ATTN_RES_LR=0.1`  — coarser (4 layers/block, ~4 sources)
3. `ATTN_RES_BLOCK_SIZE=22 ATTN_RES_LR=0.1`  — single block (just embedding + partial)

If step time regresses significantly, rewrite `_block_attn_res` to use the
Python-loop accumulation pattern that we know compiles well.

Also sweep `ATTN_RES_LR` — prior results suggest higher LR (0.5–1.0) helps routing
converge faster, but that was with static mode. Full mode may behave differently.

### Full-scale 8xH100

Take the best 1–2 configs from the proxy sweep to full training.
