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
