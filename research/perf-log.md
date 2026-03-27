# Performance Engineering Log

## 1. BF16 + uint16 mantissa trick (commit e61cfb8)

**What:** Replaced fp32 parameter banks with bf16 storage + a uint16 mantissa sidecar
in Muon, giving fp32-equivalent update precision while halving the bank memory
footprint during the forward pass. Also removed redundant `.to(x.dtype)` casts in
attention and MLP forward paths (banks are now natively bf16). A compiled
`_wd_and_update_inplace` helper handles the bit-manipulation (required for
correctness; uint16 view ops aren't supported in eager CUDA mode).

The commit also added `@torch.compile` on `zeropower_via_newtonschulz5` (the
Newton-Schulz iterations inside Muon). We tested with and without this.

**Results (1000-step runs, seed=3, 8xH100, second run = warm cache):**

| Config | step_avg (ms) | tok/s @ 1000 | train_loss | val_bpb |
|--------|---------------|--------------|------------|---------|
| Mantissa + NS compile | 82.50 | 9,532,519 | 2.2750 | 1.3211 |
| Mantissa, no NS compile | 82.41 | 9,542,896 | 2.2775 | 1.3194 |
| Baseline (fp32 banks) | 83.08 | 9,465,759 | 2.2773 | N/A |

**Conclusion:** The mantissa trick gives a ~0.8% throughput improvement with no loss
regression. The NS compile adds nothing (marginally slower due to compile overhead).
Keeping: mantissa trick yes, NS compile no.

## 2. Fused relu^2 MLP Triton kernel

**What:** Copied the fused `linear_relu_square` Triton kernel from modded-nanogpt
(by @andrewbriand, @jrauvola) into `glaucon/triton_kernels.py`. This kernel fuses
the up-projection matmul and relu^2 activation into a single kernel launch,
eliminating a memory round-trip for the intermediate activation. The autograd
backward also fuses the relu^2 gradient with the down-projection matmul.

Changes:
- New file `glaucon/triton_kernels.py` with the kernel + `FusedLinearReLUSquareFunction`.
- `mlp_down_bank` transposed from `(n, model_dim, mlp_dim)` to `(n, mlp_dim, model_dim)`
  to match the kernel's convention (`post @ W2` instead of `F.linear(post, W2)`).
- `mlp()` now delegates to the fused kernel via `ReLUSqrdMLP`.
- Fixed backward to properly flatten 3D inputs (modded-nanogpt's version assumes bsz=1).

Initially tested with standard relu^2 (dropping the leaky negative_slope=0.5), then
added leaky support directly in the Triton kernel.

**Results (1000-step runs, seed=3, 8xH100, second run = warm cache):**

| Config | step_avg (ms) | tok/s @ 1000 | train_loss | val_bpb |
|--------|---------------|--------------|------------|---------|
| Before (eager leaky relu^2) | 82.41 | 9,542,896 | 2.2775 | 1.3194 |
| Fused relu^2 (no leaky) | 79.68 | 9,869,438 | 2.2899 | 1.3282 |
| Fused leaky relu^2 | 80.41 | 9,780,090 | 2.2776 | 1.3212 |

**Conclusion:** The fused kernel with leaky relu^2 gives a ~2.4% throughput improvement
(80.41 vs 82.41 ms/step) with no meaningful loss regression (val_bpb 1.3212 vs 1.3194,
within noise). Dropping the leaky slope caused a clear loss regression (+0.009 val_bpb)
that was fully recovered by adding it back. Keeping: fused leaky relu^2 kernel.

## 3. Triton NS kernels (XXT, XTX, ba_plus_cAA) — not keeping

**What:** Copied the symmetric matmul Triton kernels from modded-nanogpt (@byronxu99)
to replace the vanilla PyTorch matmuls in Newton-Schulz orthogonalization. These
exploit output symmetry (only compute upper triangle + mirror) and are tuned for H100.

**Results (1000-step runs, seed=3, 8xH100, second run = warm cache):**

| Config | step_avg (ms) | tok/s @ 1000 | train_loss |
|--------|---------------|--------------|------------|
| Without Triton NS | 80.41 | 9,780,090 | 2.2776 |
| With Triton NS | 80.35 | 9,787,579 | 2.2818 |
| With Triton NS + compile | 85.20 | 9,230,872 | 2.2915 |

**Conclusion:** No meaningful speedup at our model dimensions (512×512, 256×512,
1536×512 intermediates). The symmetric matrices are too small for the Triton kernels
to outperform cuBLAS. Re-enabling `@torch.compile` on the NS function made things
significantly worse (85.20 ms). Reverted — not worth the complexity.

## 4. TODO: Symmetric Muon aspect ratio scaling for kv_bank

The Muon update scale factor (line 149 in train_gpt.py) is currently
`max(1, M/N)^0.5`, which only scales tall matrices. This means the kv_bank
(256×512, wide) gets scale=1.0 while it should get scale=sqrt(2)≈1.41 to
equalize update RMS with square matrices.

After the mlp_down_bank transpose (now both MLP banks are 1536×512, both tall),
only kv_bank is affected by this asymmetry. The fix is to change the scale to
`max(M/N, N/M)^0.5` — symmetric across tall and wide.

Should be tested in isolation (not bundled with LR scaling experiments).

## 5. Experiment 3: Depth/width sweep (MHA, no XSA, no VE)

**Goal:** Find the optimal depth/width tradeoff at ~27M MHA params under the 600s
wall-clock constraint.

**Results (seed=3, 8xH100, 600s wall-clock cutoff):**

| Config | D | L | mlp | params | steps | step_avg | val_bpb | ema_bpb | rt_bpb | compressed |
|--------|---|---|-----|--------|-------|----------|---------|---------|--------|------------|
| Baseline | 512 | 10 | 1536 | 27.1M | 7,561 | 79.35ms | 1.1505 | 1.1497 | 1.1612 | 16.2MB |
| A | 448 | 12 | 1536 | 26.9M | 7,037 | 85.27ms | 1.1588 | 1.1580 | 1.1699 | 16.5MB |
| B | 384 | 17 | 1280 | 27.5M | 6,523 | 91.98ms | 1.1749 | 1.1743 | 1.1854 | 17.0MB |
| C | 320 | 25 | 1024 | 27.3M | 5,395 | 111.32ms | 1.2055 | 1.2053 | 1.2165 | 16.8MB |

**Conclusion:** Wider/shallower wins decisively. The D=512/L=10 baseline achieves
the best val_bpb (1.1505) by a wide margin. Deeper models are slower per step AND
converge to worse loss — the train loss curves are nearly identical per step, but
the baseline gets ~40% more steps than D=320/L=25 in the same wall clock.

All configs exceed the 16MB compressed size limit, with deeper models slightly
worse (more scale/control params per layer).

## 6. LR scaling test (matrix_lr ∝ 1/sqrt(L), embed_lr ∝ 1/sqrt(D))

Tested on the L=25/D=320 config (most extreme scaling) to see if deeper models
were handicapped by hyperparameters.

| Config | matrix_lr | embed_lr | val_bpb | ema_bpb | compressed |
|--------|-----------|----------|---------|---------|------------|
| L25 unscaled | 0.025 | 0.035 | 1.2055 | 1.2053 | 16.8MB |
| L25 LR-scaled | 0.0158 | 0.0443 | 1.2009 | 1.2008 | 16.5MB |

**Conclusion:** Small but real improvement (~0.005 bpb). Not enough to close the
gap to the D=512 baseline (1.1505). Deeper models are fundamentally slower in
this regime — the throughput disadvantage dominates.

Applied LR scaling to all subsequent configs (matrix_lr=0.025*sqrt(10/L),
embed_lr=0.035*sqrt(512/D)).

## 7. Config D: D=576, L=8 (wider/shallower)

| Config | D | L | steps | step_avg | sw_bpb | rt_bpb | compressed |
|--------|---|---|-------|----------|--------|--------|------------|
| Baseline | 512 | 10 | 7,561 | 79.35ms | 1.1366 | 1.1612 | 16.2MB |
| **D** | **576** | **8** | **8,229** | **72.91ms** | **1.1320** | **1.1561** | **16.5MB** |

**Conclusion:** Config D beats the baseline on the competition metric (sliding
window BPB) by 0.0046. It's 8% faster per step, gets ~670 more steps in 600s,
and uses 1.7 GiB less memory. The wider/shallower trend is strictly monotonic
across all tested configs (L=25 through L=8).

Full sweep results (all with sw_bpb, the competition metric):

| D | L | sw_bpb | steps | step_avg |
|---|---|--------|-------|----------|
| 320 | 25 | 1.1913 | 5,395 | 111.32ms |
| 384 | 17 | 1.1604 | 6,523 | 91.98ms |
| 448 | 12 | 1.1450 | 7,037 | 85.27ms |
| 512 | 10 | 1.1366 | 7,561 | 79.35ms |
| 576 | 8 | 1.1320 | 8,229 | 72.91ms |

## 8. Configs E and E': Finding the depth floor (D=640)

| Config | D | L | mlp | params | steps | step_avg | sw_bpb |
|--------|---|---|-----|--------|-------|----------|--------|
| D | 576 | 8 | 1792 | 28.1M | 8,229 | 72.91ms | **1.1320** |
| E' | 640 | 7 | 1792 | 28.5M | 8,625 | 69.56ms | 1.1341 |
| E | 640 | 6 | 2048 | 26.2M | 9,567 | 62.72ms | 1.1395 |

**Conclusion:** The wider/shallower trend has plateaued. Going from L=8 to L=7
costs 0.002 sw_bpb despite gaining 400 steps. L=6 is decisively worse (+0.0075).
The sweet spot is **L=7-8 at D=576-640**. Config D (D=576, L=8) remains the best
at sw_bpb=1.1320.

Complete depth/width sweep results:

| D | L | sw_bpb | steps | step_avg |
|---|---|--------|-------|----------|
| 320 | 25 | 1.1913 | 5,395 | 111.32ms |
| 384 | 17 | 1.1604 | 6,523 | 91.98ms |
| 448 | 12 | 1.1450 | 7,037 | 85.27ms |
| 512 | 10 | 1.1366 | 7,561 | 79.35ms |
| **576** | **8** | **1.1320** | **8,229** | **72.91ms** |
| 640 | 7 | 1.1341 | 8,625 | 69.56ms |
| 640 | 6 | 1.1395 | 9,567 | 62.72ms |

## 9. Re-introduce GQA + XSA + VE (exp4)

Re-introduced GQA (9 heads, 3 kv_heads), XSA (last 3 layers), and VE (layers 6,7)
on the winning config D (D=576, L=8).

| Config | heads | kv | params | compressed | steps | sw_bpb |
|--------|-------|----|--------|------------|-------|--------|
| D (MHA) | 9 | 9 | 28.1M | 16.6MB | 8,229 | **1.1320** |
| exp4 (GQA+XSA+VE) | 9 | 3 | 24.7M | **14.8MB** | 8,528 | 1.1372 |

**Conclusion:** GQA brings us comfortably under 16MB (1.2MB headroom) and is 3.5%
faster, but sw_bpb regressed by 0.005. The 3:1 GQA ratio (9:3) may be too
aggressive vs the original 2:1 (8:4). D=576 with head_dim=64 gives 9 heads,
which doesn't divide evenly for 2:1 GQA.

**U-Net skip connections re-added** to the codebase (not yet tested). Previous
logs showed skip connections were worth ~0.014 sw_bpb on the L=11 baseline —
a very large effect.

## 10. Next steps

1. **Test skip connections** on the current config — expected to be a big win
   based on prior L=11 data (~0.014 sw_bpb improvement).

2. **GQA ratio tuning.** Options:
   - Find a nearby D where num_heads is even for clean 2:1 GQA
   - Keep D=576/9 heads but try 9:9 MHA and use the size headroom for larger MLP
   - Try non-64 head_dim (risky for throughput)

3. **d_vocab=2048** — sp2048 tokenizer spec is ready in
   `data/tokenizer_specs.json`. Needs `download_hf_docs_and_tokenize.py` run
   to train tokenizer and export shards.

4. **Batch size sweep** and **symmetric Muon kv_bank scaling** — still TODO.

5. Competition context: current frontier is 1.1194 sw_bpb. Our best is 1.1320
   (config D, MHA). Gap is 0.013 — skip connections alone could close most of it.
