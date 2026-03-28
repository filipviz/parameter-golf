# Attention Residuals — Experiment Notes

## Background

Attention Residuals (AttnRes) replaces standard residual connections with learned
inter-layer routing. Instead of `x_t = x_{t-1} + f_t(x_{t-1})`, each layer output
`y_t = f_t(h_t)` is stored, and the input to each subsequent layer is a weighted sum
of all previous outputs: `h_t = sum_s a_{t,s} * y_s`.

Reference: Su et al., "Attention Residuals" (arXiv:2603.15031)

## What we implemented

Two modes behind `ATTN_RES_MODE`:

- **`static`**: Data-independent causal routing matrix. An `(N, N)` lower-triangular
  logit matrix, softmaxed row-wise. Each row gives the weights a consumer uses to
  combine all preceding y's. This is a simpler ablation — not from the paper itself.

- **`full`**: The paper's AttnRes mechanism. Learned query vectors `w_c` per consumer,
  with `a_{c,s} = exp(w_c . RMSNorm(y_s))`, softmaxed over sources. Per-token routing.

Two granularity levels behind `ATTN_RES_GRANULARITY`:

- **`block`** (recommended): One residual output per block. The block function has an
  internal attn→MLP residual: `y = mlp(norm(h + attn(norm(h))))`. Matrix is `(L+1, L+1)`.

- **`sublayer`**: Each attn and MLP sublayer produces a separate residual output.
  Matrix is `(2L+1, 2L+1)`. Mathematically closer to the paper but ~40% slower.

## What we removed

AttnRes subsumes several existing mechanisms:
- U-Net skip connections (`skip_weights`, encoder/decoder split)
- `resid_mix` (learned x/x0 mixing per block)
- `attn_scale` / `mlp_scale` (per-dim output gating)
- `ln_scale` (1/sqrt(layer) normalization scaling)
- `Block` class entirely — `GPT` now holds `attns` (ModuleList of CausalSelfAttention)

## Implementation details

### torch.compile performance

Benchmarked extensively on H100. Key finding: **explicit scalar*tensor accumulation
loops compile 2x faster than stack+einsum** (2.87ms vs 6.37ms for the routing-only
overhead). The compiler fuses the chain of scalar broadcast multiply + add ops into
a single kernel.

```python
# Fast (compiles well):
x_in = weights[row, 0] * ys[0]
for s in range(1, row + 1):
    x_in = x_in + weights[row, s] * ys[s]

# Slow (materializes intermediate tensor):
stk = torch.stack(ys, dim=0)
x_in = torch.einsum('s,sbtd->btd', weights[row, :len(ys)], stk)
```

Other findings:
- `softmax(dim=-1)` vs `softmax(dim=0)` via transpose: no meaningful difference
- Reconstructing the causal mask each forward vs `register_buffer`: within noise;
  we use `register_buffer` for cleaner code
- `per-row softmax` (variable-length slices) triggers "online softmax disabled"
  warning and is slightly slower

### FP32 contamination bug (fixed)

`CONTROL_TENSOR_NAME_PATTERNS` originally included `attn_res_logits` and
`attn_res_queries`, causing `restore_low_dim_params_to_fp32` to keep them in fp32.
This made the softmax weights fp32, which then upcasted every subsequent weighted
sum operation to fp32 — cascading through the entire forward pass. Fixed by removing
AttnRes params from the control tensor patterns.

### Internal residual is essential

Without the internal `h + attn_out` residual inside the block function, the model
barely learns (train_loss stuck at ~4.9 after 200 steps vs ~2.9 with it). The attn
bottleneck is too narrow for all information to flow through.

### Zero-init for standard residual equivalence

Both modes zero-init their routing parameters:
- `static`: zero logits → uniform softmax → equal-weight sum = standard residual
- `full`: zero queries → `w . RMSNorm(y) = 0` → uniform softmax = standard residual

## Proxy run results (200 steps, 1xH100)

Baseline: standard residual with skip connections, resid_mix, attn/mlp scale, ln_scale.

| Config                     | Step 100 | Step 200 | val_bpb@200 | step_avg | peak mem  |
|----------------------------|----------|----------|-------------|----------|-----------|
| **Baseline**               | 3.43     | 2.85     | (1.497@500) | 637ms    | 21718 MiB |
| Block+intres static LR=0.1 | 3.34     | 2.98     | 1.7896      | 620ms    | 19806 MiB |
| Block+intres static LR=0.5 | 3.29     | 2.94     | 1.7662      | 621ms    | 19806 MiB |
| Block+intres static LR=1.0 | 3.26     | 2.93     | 1.7559      | 621ms    | 19806 MiB |
| Sublayer static LR=0.5     | 3.26     | 2.92     | 1.7563      | 865ms    | 23173 MiB |

**Block-level with internal residual at LR=1.0 is the sweet spot:**
- 2.5% faster per step than baseline (621ms vs 637ms)
- 1.9GB less peak memory (19.8 vs 21.7 GiB)
- Loss slightly behind at 200 steps but routing is still converging
- Sublayer gives identical loss but 40% slower — not worth it

## Proposed full-scale experiments

### Run 1: `attnres_block_static_lr100`
```bash
SEED=3 RUN_ID="attnres_block_static_lr100" ITERATIONS=500 WARMDOWN_ITERS=3500 \
    ATTN_RES_MODE=static ATTN_RES_LR=1.0 ATTN_RES_GRANULARITY=block
```

### Run 2: `attnres_block_static_lr050`
```bash
SEED=3 RUN_ID="attnres_block_static_lr050" ITERATIONS=500 WARMDOWN_ITERS=3500 \
    ATTN_RES_MODE=static ATTN_RES_LR=0.5 ATTN_RES_GRANULARITY=block
```

Compare final val_bpb against baseline's 1.4966.

### Future directions (if promising)
- Test `full` mode (learned queries) at block granularity — adds ~0.8ms overhead
  but enables data-dependent routing
- Try even higher routing LR (2.0, 5.0)
- Experiment with different block internal structure (e.g., no internal residual
  but with a learned gate)
