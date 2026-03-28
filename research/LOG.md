# Glaucon Development Log

## Cleanup

Starting point: PR #549 (record, 1.1194 BPB) copied into `glaucon/train_gpt.py` in commit `1e2934b14f2e6bf4dea755374272190d597a28c4`. PR #549's implementation and logs are available in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`. These changes go through commit `a2f177e0b69be89124d0de108053d12604cab9b1`.

- Removed features which were disabled in the record: DTG, gated attention, value residual, LAWA, MTP, standalone int8 quantization pipeline, unused compressor fallback imports.
- Removed dead SWA code — state was collected (`swa_enabled=1`) but never applied; final weights always came from EMA.
- Switched compression from `lzma` (preset 6) to `brotli` (quality 11).
- Fixed misleading log labels (`final_int8_zlib_roundtrip_exact` → correct `int6_sliding_window` labels).
- The `RMSNorm` and `MLP` classes were both stateless. Replaced with simpler `norm()` and `mlp()` functions.
- Hardcoded tied embeddings: removed `tie_embeddings`, `lm_head`, `optimizer_head`, `head_lr`. Renamed `tied_embed_lr` → `embed_lr`, `tied_embed_init_std` → `embed_init_std`.
- Simplified `CastedLinear`: always `bias=False`, removed QAT/STE branch (dead under `torch.compile` — class var flip after tracing has no effect).
- Removed late QAT activation logic from training loop (never active due to aforementioned `torch.compile` quirk).
- Changed `val_loss_every` default from 4000 → 0 (skip mid-training validation; still available via env var).
- Fixed RoPE bug: `Rotary` hardcoded `train_seq_len=1024` but we train at 2048, so NTK extension always fired with constant factor `2^(16/14)`. Collapsed to effective `rope_base=22082` and removed extension code entirely.
- Unified `train_seq_len` + `eval_seq_len` into single `seq_len` (env var `SEQ_LEN`, default 2048).
- Removed `muon_beta2` (defined, never used) and `num_layers_total` (computed, never used).
- Removed stale `CONTROL_TENSOR_NAME_PATTERNS` aliases (`attn_scales`, `mlp_scales`, `resid_mixes`, `skip_weight`) and stale `passthrough_fp16` branch in dequantization.
- Removed dead zero-inits on `bigram.proj` and `ve_shared.proj` (immediately overwritten by `_init_weights` orthogonal init).
- Removed `_classify_param` `lm_head` branch and `_zero_init` guard in `_init_weights`.
- Extracted `_build_model(args, device)` helper to deduplicate model construction between training and eval.
- Inlined `GPT._ve_target_dim` (used once, was just `kv_dim`). Removed `value_embeds` empty compat list.
- Meta device init refactor: `GPT.__init__` defines structure only (all `torch.empty`), `_init_weights()` sets all values after `to_empty()`. Eliminates wasted default inits from `nn.Embedding`/`nn.Linear` and co-locates all initialization in one method.
- Moved `rope_base` and `qk_gain_init` out of submodule constructors (Block, CausalSelfAttention, Rotary) — stored on GPT, applied in `_init_weights`.
- Fixed latent bug: CastedLinear init had `>= 64` size gate that would leave small projections uninitialized under meta device. Removed gate, always orthogonal-init.
- Converted config validation from `raise ValueError`/`RuntimeError` to `assert` throughout. Added new asserts for `rope_dims` bounds and `local_tokens % seq_len` divisibility.
- Removed unused `Muon._rank`, inlined `_world_size`.
- Replaced eager VE precomputation with lazy materialization (compute `ve_base` on first VE layer hit) to avoid holding ~150 MB/rank of activations through non-VE layers.
- Renamed `ValueEmbedding.__init__` param `model_dim` → `kv_dim` (it receives `kv_dim`, not `model_dim`).
- Fixed control tensor quantization bug: `mixed_quantize_int6` checked `numel() <= 65536` before control tensor patterns, so small control tensors (q_gain, attn_scale, mlp_scale, resid_mix, etc.) were silently downcast to fp16 instead of preserved in fp32.
- Removed SDPA backend configuration (we use FA3 directly, not `torch.nn.functional.scaled_dot_product_attention`).
- Added `local_grad_norm` to training log (rank-0 pre-communication norm, for rough monitoring of gradient clipping behavior).

## Baseline validation run (seed=3)

Cleanup seems performance-neutral, but hurts step time.

| Metric | Glaucon (cleanup) | PR #549 (record) |
|---|---|---|
| Sliding window BPB | 1.1222 | 1.1194 (with TTT) |
| Pre-TTT BPB | ~1.1222 | ~1.1217 |
| Step time | 87.08 ms | 83.4 ms |
| Steps completed | 6,891 | 7,185 |
| Artifact size | 15,514,703 bytes | ~15,950,000 bytes |
| Quant tax (pre→post) | 0.0083 BPB | similar |

- brotli-11 saves ~435KB vs record's lzma-6. ~414KB headroom under 16MB.
- The step time was 4.4% slower (87.08 vs 83.4 ms)! This might be hardware variance, but could also have to do with changes to the value embedding implementation.
- Grad clipping (0.3) heavily active in first ~200 steps (norms up to 3.5), inactive after step 500 (norms 0.07-0.13).

Note - the BPB change was probably due to using several script defaults instead of the PR's run command. The performance change may have been due to the value embedding cleanup in commit `9ba6376b97dee81e8a46a633e542af17190ec77e`. I'll experiment with reverting this.

## Performance improvements

Note that these were validated with short (1000 step) runs, so the `val_bpb` may not be completely representative.

### BF16 + uint16 mantissa trick (commit e61cfb8)

Adapted a trick from `modded-nanogpt`. Instead of using fp32 parameter banks, use bf16 storage and a separate uint16 mantissa sidecar for Muon. Improves memory usage and removes the need for casting in attention/MLP forward. A compiled `_wd_and_update_inplace` helper handles the bit-manipulation.

Results (1000-step runs, seed=3, 8xH100, second run = warm cache):

| Config | step_avg (ms) | tok/s @ 1000 | train_loss | val_bpb |
|--------|---------------|--------------|------------|---------|
| Mantissa | 82.41 | 9,542,896 | 2.2775 | 1.3194 |
| Baseline (fp32 banks) | 83.08 | 9,465,759 | 2.2773 | N/A |

Gives a ~0.8% throughput improvement with no loss regression.

### Fused relu^2 MLP triton kernel

Initially tested with standard relu^2 (dropping the leaky negative_slope=0.5), then added leaky support directly in the Triton kernel.

| Config | step_avg (ms) | tok/s @ 1000 | train_loss | val_bpb |
|--------|---------------|--------------|------------|---------|
| Before | 82.41 | 9,542,896 | 2.2775 | 1.3194 |
| Fused relu^2 | 79.68 | 9,869,438 | 2.2899 | 1.3282 |
| Fused leaky relu^2 | 80.41 | 9,780,090 | 2.2776 | 1.3212 |

The fused leaky relu^2 kernel gives a ~2.4% throughput improvement with no meaningful loss regression.

## Depth/width sweep

Motivated by MobileLLM (Meta), which found depth strongly preferred over width at sub-billion scales (optimal ~30 layers at 135M params). Our model is ~27M params at 11 layers — likely far from the depth optimum. I swept several simplified configs (XSA disabled, no skip connections, no VE). In hindsight I should have included skip connections here.

| Config | D | L | mlp | params | steps | step_avg | val_bpb | ema_bpb | rt_bpb | compressed |
|--------|---|---|-----|--------|-------|----------|---------|---------|--------|------------|
| Baseline | 512 | 10 | 1536 | 27.1M | 7,561 | 79.35ms | 1.1505 | 1.1497 | 1.1612 | 16.2MB |
| A | 448 | 12 | 1536 | 26.9M | 7,037 | 85.27ms | 1.1588 | 1.1580 | 1.1699 | 16.5MB |
| B | 384 | 17 | 1280 | 27.5M | 6,523 | 91.98ms | 1.1749 | 1.1743 | 1.1854 | 17.0MB |
| C | 320 | 25 | 1024 | 27.3M | 5,395 | 111.32ms | 1.2055 | 1.2053 | 1.2165 | 16.8MB |

Wider/shallower won decisively. The D=512/L=10 baseline achieves the best val_bpb (1.1505) by a wide margin. Deeper models are slower per step AND converge to worse loss — the train loss curves are nearly identical per step, but the baseline gets ~40% more steps than D=320/L=25 in the same wall clock.

I tried using a more principled LR scaling muon_lr ∝ 1/sqrt(L), adam_lr ∝ 1/sqrt(D) which produced a small improvement on the deepest configuration.

| Config | matrix_lr | embed_lr | val_bpb | ema_bpb | compressed |
|--------|-----------|----------|---------|---------|------------|
| L25 unscaled | 0.025 | 0.035 | 1.2055 | 1.2053 | 16.8MB |
| L25 LR-scaled | 0.0158 | 0.0443 | 1.2009 | 1.2008 | 16.5MB |

I applied LR scaling to all subsequent configs (`matrix_lr=0.025*sqrt(10/L)`, `embed_lr=0.035*sqrt(512/D)`). Since shallower configs had been performing better, I tried even shallower ones. An 8 layer config beat the baseline, but 7/6 layer configs regressed.

| Config | D | L | steps | step_avg | sw_bpb | rt_bpb | compressed |
|--------|---|---|-------|----------|--------|--------|------------|
| Baseline | 512 | 10 | 7,561 | 79.35ms | 1.1366 | 1.1612 | 16.2MB |
| D | 576 | 8 | 8,229 | 72.91ms | 1.1320 | 1.1561 | 16.5MB |

| Config | D | L | mlp | params | steps | step_avg | sw_bpb |
|--------|---|---|-----|--------|-------|----------|--------|
| Baseline | 512 | 10 | 1536 | 27.1 M | 7,561 | 79.35ms | 1.366 |
| D | 576 | 8 | 1792 | 28.1M | 8,229 | 72.91ms | **1.1320** |
| E | 640 | 7 | 1792 | 28.5M | 8,625 | 69.56ms | 1.1341 |
| F | 640 | 6 | 2048 | 26.2M | 9,567 | 62.72ms | 1.1395 |

I then re-introduced GQA (9 heads, 3 kv_heads), XSA (last 3 layers), and VE (layers 6,7) on the winning config D (D=576, L=8).

| Config | heads | kv | params | compressed | steps | sw_bpb |
|--------|-------|----|--------|------------|-------|--------|
| D (MHA) | 9 | 9 | 28.1M | 16.6MB | 8,229 | **1.1320** |
| exp4 (GQA+XSA+VE) | 9 | 3 | 24.7M | **14.8MB** | 8,528 | 1.1372 |

GQA brings us comfortably under 16MB (1.2MB headroom) and is 3.5% faster, but sw_bpb regressed by 0.005. The 3:1 GQA ratio (9:3) may be too aggressive vs the original 2:1 (8:4). D=576 with head_dim=64 gives 9 heads, which doesn't divide evenly for 2:1 GQA.

## Fixes

At this point:
1. I noticed that several defaults didn't align with the PR's run command and fixed these. Specifically, I moved bigram vocab size from 2048 -> 1536, enabled TTT, and removed TTT freezing.
2. I re-introduced u-net skip connections to the codebase.
3. Reverting the value embedding cleanup `9ba6376b97dee81e8a46a633e542af17190ec77e` yielded a small speedup.
