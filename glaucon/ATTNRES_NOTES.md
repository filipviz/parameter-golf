# Attention Residuals — Experiment Notes

## Background

Attention Residuals (AttnRes) replaces standard residual connections with learned
inter-layer routing. Instead of `x_t = x_{t-1} + f_t(x_{t-1})`, each layer output
`y_t = f_t(h_t)` is stored, and the input to each subsequent layer is a weighted sum
of all previous outputs: `h_t = sum_s a_{t,s} * y_s`.

Reference: Su et al., "Attention Residuals" (arXiv:2603.15031)

## Current implementation: Two-phase block-runner

We implement the paper's **Block AttnRes** with a **two-phase block-runner** that
mirrors Algorithm 1 from the paper. This replaced an earlier naive per-sublayer
implementation that had severe backward pass performance issues.

### Configuration

- `ATTN_RES_BLOCK_SIZE` (default: 4): block size in sublayers (attn+mlp = 2 per layer)
- `ATTN_RES_LR` (default: 0.10): learning rate for pseudo-query parameters

With L=11 transformer layers (22 sublayers):
- `BLOCK_SIZE=2`: 11 full blocks + embedding = 12 sources max (finest granularity)
- `BLOCK_SIZE=4`: 5 full blocks + 1 partial + embedding = ~7 sources max
- `BLOCK_SIZE=8`: 2 full blocks + 1 partial + embedding = ~4 sources max

### Architecture

Parameters: `attn_res_queries` of shape `(2L+1, model_dim)` — one query per sublayer
consumer (attn + MLP per layer) plus a final output query. Stored in fp32 (via
`CONTROL_TENSOR_NAME_PATTERNS`), cast to bf16 in the forward pass.

### Two-phase forward loop

**Phase 1 (`_inter_block_phase`)**: At each block boundary, batch ALL sublayer
queries for the upcoming block and compute scores against ALL completed blocks in
one matmul pass. Returns online softmax statistics (unnormalized numerator,
denominator, max logit) for each query.

**Phase 2 (`_merge_with_partial`)**: For each sublayer within the block, merge the
pre-computed inter-block result with the evolving partial block using online softmax.
This is O(B×T×D) elementwise — very cheap.

```python
for each block b:
    # Phase 1: one batched pass over all completed blocks
    completed_blocks.append(partial)
    prev_blocks = torch.stack(completed_blocks)
    inter_num, inter_den, inter_m = inter_block_phase(prev_blocks, q_block)

    for each sublayer j in block b:
        # Phase 2: cheap merge with evolving partial
        if first sublayer (no partial):
            h = inter_num[j] / inter_den[j]
        else:
            h = merge_with_partial(inter_num[j], inter_den[j], inter_m[j], partial, q[j])

        sublayer_out = attn_or_mlp(norm(h), ...)
        partial = partial + sublayer_out if partial is not None else sublayer_out
```

This reduces the naive O(L×N) source-scan pattern to O(N_blocks) batched passes
plus O(L) cheap elementwise merges.

### Key design properties

- **Embedding is isolated**: boundary fires at layer 0, so embedding is always its
  own block.
- **h is freshly routed each sublayer**: no residual accumulates on h.
- **Sublayer outputs accumulate on partial**: traditional residual within blocks.
- **partial is an attnres source**: the MLP can attend to the current partial (which
  includes the attn output), preserving the attn→MLP information path without a
  hardcoded skip connection.
- **Zero-init**: zero queries → uniform softmax → equal-weight sum at initialization.

## What we removed

AttnRes subsumes several existing mechanisms:
- U-Net skip connections (`skip_weights`, encoder/decoder split)
- `resid_mix` (learned x/x0 mixing per block)
- `attn_scale` / `mlp_scale` (per-dim output gating)
- `ln_scale` (1/sqrt(layer) normalization scaling)
- `Block` class entirely — `GPT` now holds `attns` (ModuleList of CausalSelfAttention)

## Implementation details

### torch.autocast dtype promotions

Under `torch.autocast(dtype=bf16)`, several ops silently promote to fp32:
- `torch.exp()` — always promotes
- `.sum()` — promotes reductions
- `F.softmax()` — promotes
- `torch.log()`, `.pow()`, `.prod()`, `.norm()` — promote

This is **CUDA-specific** — CPU autocast does not promote these ops.

This caused two bugs during development:
1. fp32 output from routing reaching the fused triton MLP kernel (which requires
   matching dtypes) → compilation error
2. Unnecessary fp32 intermediate storage bloating the backward graph

Solution: let the softmax computation (exp, sum) run in fp32 for numerical stability,
but cast routing weights back to bf16 before weighted sums. In `_inter_block_phase`,
this means casting `w` and `den` to source dtype after exp/sum. In
`_merge_with_partial`, casting `a` and `b` after exp.

### Inter-block phase implementation

Benchmarked several strategies for the batched inter-block computation:

| Strategy | N=3,S=8 | N=6,S=4 | N=11,S=2 |
|---|---|---|---|
| permute + bmm | 2.02ms | 2.53ms | 5.03ms |
| einsum | 2.36ms | 2.50ms | 3.79ms |
| broadcast (no reshape) | 2.25ms | 2.06ms | 4.63ms |
| **matmul scores + loop wsum** | **0.76ms** | **1.05ms** | **1.60ms** |

The winner uses `matmul` for scores (avoids autocast `.sum()` promotion, single
fused kernel) and Python-loop weighted sum over N (the compiler fuses
`scalar*tensor` accumulation chains — same pattern that worked well in the original
per-call routing benchmarks).

### Backward pass: why naive per-sublayer routing was slow

The original implementation called `_block_attn_res` independently for each of the
23 sublayer routing points. Each call created ~9N autograd nodes (N norms, N
element-wise multiplies, N sums, softmax ops, N weighted multiply-adds). With
BS=2 (up to 12 sources), this meant ~2000+ routing nodes in the backward graph.

The backward was the dominant bottleneck:

| Config | Forward | Backward | Total | Est step (×8) | vs baseline |
|---|---|---|---|---|---|
| Baseline | — | — | ~81ms | 650ms | — |
| BS=8 naive | 26ms | 67ms | 93ms | 745ms | +15% |
| BS=4 naive | 27ms | 103ms | 130ms | 1038ms | +60% |
| BS=2 naive | 30ms | 190ms | 220ms | 1762ms | **+171%** |

The forward was fast (~27ms regardless of block size) but the backward scaled
superlinearly with total source references across all routing calls.

Attempted mitigations that did NOT help:
- `torch.utils.checkpoint` on routing calls: +20% worse (compilation overhead)
- Precomputed block logits: +20% worse (same reason)
- Disabling autocast in routing: no effect (overhead is structural, not dtype-related)

### Two-phase block-runner performance

The block-runner restructure dramatically reduced backward cost by amortizing
inter-block work across all sublayers in a block:

| Config | Forward | Backward | Total | Est step (×8) | vs baseline |
|---|---|---|---|---|---|
| Baseline | — | — | ~81ms | 650ms | — |
| BS=8 block-runner | 26ms | 65ms | 91ms | 728ms | +12% |
| BS=4 block-runner | 29ms | 71ms | 100ms | 803ms | +24% |
| BS=2 block-runner | 39ms | 94ms | 133ms | 1061ms | +63% |

The forward regression from the naive version is minimal (+2-9ms) while the
backward improved substantially (BS=2: 190ms→94ms, BS=4: 103ms→71ms).

Activation checkpointing on `_inter_block_phase` had zero additional effect —
`torch.compile` already handles intermediate storage efficiently.

### FP32 query storage

Queries are stored in fp32 via `CONTROL_TENSOR_NAME_PATTERNS` and cast to bf16
in `_inter_block_phase` and `_merge_with_partial`. This helps optimizer stability
without contaminating the forward pass with fp32 compute.

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

## Current status (2026-03-29)

### Where we are

Block-runner two-phase implementation is integrated into `train_gpt.py` and
verified (numerical equivalence + gradcheck). Baseline 500-step proxy (PR #549,
no warmdown) completed: **val_bpb=1.3925, step_avg=650ms**.

### Next: LR sweep needed

The `ATTN_RES_LR` value needs tuning. Previous runs used LR=0.5 with
`WARMDOWN_ITERS=3500`, which meant the effective LR at step 1 was
`0.5 * 500/3500 ≈ 0.07`. With warmdown now disabled for proxy runs, LR=0.5 may
be too aggressive. Suggested sweep for BS=4:

- `ATTN_RES_LR=0.5` (may be too high without warmdown)
- `ATTN_RES_LR=0.1` (conservative, closer to effective LR from earlier runs)
- `ATTN_RES_LR=0.25` (middle ground)

Then take the best LR to BS=2 and BS=8 for block-size comparison.

### Open questions / future work

**Performance:**
- BS=8 is +12% overhead (728ms vs 650ms), BS=4 is +24% (803ms). The remaining
  gap is genuine routing compute (norm, matmul, softmax, weighted sum). A fused
  triton kernel for `_inter_block_phase` could eliminate kernel launch overhead
  between these ops. A custom `torch.autograd.Function` could also help by
  collapsing routing into a single autograd node (fewer backward dispatch calls).
- The record's removed techniques (skip connections, resid_mix, attn/mlp scale,
  ln_scale) had near-zero overhead, so AttnRes should ideally match that. BS=8
  is close but not there yet.

**Correctness / training dynamics:**
- Zero-init gives uniform routing (equal-weight average of all blocks), NOT
  standard residual (which is an unnormalized sum). This is a known difference
  from the paper. The model learns to adjust weights away from uniform, but
  early training dynamics may differ.
- The `ATTN_RES_LR` interacts with the main optimizer schedule. In full runs
  with warmdown, the routing LR also decays. May want to consider whether the
  routing queries should have their own independent schedule.

**Scaling behavior:**
- The paper used 48-layer MoE models where routing overhead is negligible
  relative to per-layer compute. Our 11-layer dense model has proportionally
  more routing overhead. The benefit of finer routing (more sources) needs to
  outweigh the step-time cost.
- Worth checking: does BS=4 or BS=2 actually improve BPB enough over BS=8 to
  justify the extra compute? If BS=8 matches BS=4 on loss, it's the clear winner.

**Files:**
- `attnres_block_runner.py`: standalone functions + reference impl + tests
- `test_attnres_block_runner.py`: correctness tests (numerical + gradcheck)
- `profile_attnres.py`, `profile_fwd_detail.py`, `profile_fwd_detail2.py`,
  `bench_attnres.py`: profiling scripts (can be cleaned up after experiments)
