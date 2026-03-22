# Record: Glaucon

This PR adds gated attention, brotli compression, and symmetric Muon update scaling. It also tweaks AdamW eps and Muon/AdamW weight decay. The names were getting unwieldy so I've named it Glaucon.

Forked from [PR #315](https://github.com/openai/parameter-golf/pull/315) 11L Partial RoPE + LN Scale + EMA + Late QAT + XSA4, which achieved 1.1248 BPB.

[PR #70](https://github.com/openai/parameter-golf/pull/70) (9L, 1.1659) → [PR #164](https://github.com/openai/parameter-golf/pull/164) (9L, 1.1524) → [PR #198](https://github.com/openai/parameter-golf/pull/198) (11L, 1.1318) → [PR #287](https://github.com/openai/parameter-golf/pull/287) (11L, 1.1271) → [PR #315](https://github.com/openai/parameter-golf/pull/315) (11L, 1.1248) → this

## Summary

| Change | Sliding BPB | Delta |
|--------|-------------|-------|
| PR #315 baseline | 1.12440 | — |
| + gated attention | 1.12211 | -0.00229 |
| + symmetric Muon | 1.12090 | -0.00121 |
| + Adam eps=1e-10 | 1.12065 | -0.00025 |

## Change 1: gated attention

Per-head gated attention, inspired by [Qiu et al 2025](https://arxiv.org/abs/2505.06708) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

I project from the first 12 channels of the residual stream to a per-head SDPA output multiplier (before `W_O`). Unlike `modded-nanogpt`, I use `2 * sigmoid` which yields a slight improvement. That form means the multiplier is initialized at unit scale and can grow or shrink from there. Wiring aside, the implementation is just:
```python
# ATTN_GATE_DIM is 12
gate = 2 * sigmoid(x[..., :ATTN_GATE_DIM] @ attn_gate.T)  # (B, T, num_heads)
y = y * gate.unsqueeze(-1)
```

Gate params are added to the control tensors (AdamW, fp32). The gate adds ~4ms/step overhead (~167ms vs ~163ms), costing ~200 steps, but training efficiency more than compensates.

## Change 2: brotli compression

Switching to brotli-11 saves ~470 KB over zstd-22 and ~1.46 MB over zlib-9. Not noted above since this doesn't affect loss.

I was able to save another ~160 KB using ANS coding from the [constriction library](https://bamler-lab.github.io/constriction/), but left that out of this PR to avoid excessive complexity.

## Change 3: symmetric Muon update scale

Muon scales its update after the NS iters: `g *= max(1, g.size(0) / g.size(1)) ** 0.5`. This multiplier has a subtle quirk - its value depends on how you store your MLP matrices!

`modded-nanogpt` stores all MLP matrices the same way, producing a uniform 2x update multiplier (4 ** 0.5). `modded-nanogpt` applies an *additional* 2x LR multiplier on their `c_proj` weights (higher projection learning rates can help with muP-style inits).

This repo's baseline had neither change, making the effective `c_proj` learning rate[^1] `sqrt(3)` or `2 * sqrt(3)` smaller than what it should be (perhaps). Empirically, the `2*sqrt(3)` multiplier had ~no effect on BPB, but the `sqrt(3)` multiplier improved `val_bpb` to 1.12090. I incorporated this into the Muon class by making the aspect ratio scaling symmetric:

```python
# Before: only boosts tall matrices (i.e. c_fc)
g *= max(1, g.size(0) / g.size(1)) ** 0.5
# After: all MLP weight updates scaled by sqrt(3)
g *= max(g.size(0) / g.size(1), g.size(1) / g.size(0)) ** 0.5
```

## Change 4: AdamW epsilon

Lowering the AdamW epsilon from PyTorch's default `eps=1e-8` to `eps=1e-10` yielded a minor BPB improvement.

## Change 5: weight decay

Weight decay plays an interesting role in this challenge. The more weight decay we use, the more our model's weights can be compressed! This holds monotonically and the effect is quite large. I don't know why exactly this is, and will look into it later.

I ended up nudging the AdamW/Muon weight decay to 0.045 to fit within the 16 MB limit.

Earlier attempts:
1. I first tried out weight decay scheduling using `lr_mul`. Muon/AdamW already couple weight decay to the learning rate, so the net effect is a quadratic schedule on our effective weight decay. This increased our total bundle size to 17.6 MB, making it invalid, but yielded a substantial -0.00268 BPB improvement!
2. I tried adding [cautious weight decay](https://arxiv.org/abs/2510.12402). It had no impact on BPB but made the bundle ~2 MB larger.

I could see a run with increased Muon WD and decreased AdamW WD improving BPB.

## Negative results

1. GEGLU in the style of [gpt-oss](https://arxiv.org/abs/2508.10925) (clipping and `1 + up` scaling for linear init). Reduced throughput and BPB.
2. [Adaptive base frequency](https://arxiv.org/abs/2309.16039). No effect. I think this is worth revisiting, and would work if someone took the time to tune it.
3. `logit_softcap=15.0` and `10.0`, and modded-nanogpt's `mul * sigmoid((logits + offset) / temp)` - all hurt performance to some degree. I was surprised by this.

### Partial Key Offset

*If you're not familiar with induction heads, see [Olsson et al 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).*

[ClassicLarry's](https://github.com/ClassicLarry) [modded-nanogpt PR #169](https://github.com/KellerJordan/modded-nanogpt/pull/169) introduced a very clever trick to enable induction heads within a single layer - shift the keys forward! Specifically, he does this for stationary (non-RoPE) head dims:

```python
# shift keys forward for the stationary head dims. Enables 1-layer induction.
k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
```

I tried out several variants but couldn't quite get meaningful BPB improvements. I came closest when only offsetting *half* of the stationary head dims:

| Run | Sliding BPB | Steps | Step avg |
|---|---|---|---|
| baseline | 1.12097 | 6,923 | 86.67ms |
| key offset for 24/48 dims | 1.12097 | 6,873 | 87.31ms |

An extra ~0.6ms/step overhead, trains for fewer steps, learns more efficiently, yielding no effect on the BPB. I think something in this family could work with more tuning.

## Bugs

- Fixed a minor bug in the test-time sliding window method. Negligible impact.
- The Rotary class had a hard-coded `train_seq_len=1024`, but `TRAIN_SEQ_LEN` was set to 2048. This meant NTK rescaling was always active - effectively the same as having no NTK rescaling and changing the RoPE `theta_base` from 10k to ~22k. I removed the NTK code and set `theta_base` to 22K (functionally equivalent).

## Notes for Maintainers

Thank you for running this competition!

- Several records rely on `zstandard` and silently regress to `zlib` if it isn't present. Many also rely on FA3! I suggest requiring setup scripts for non-standard environments.
- The baseline `train_gpt.py` doesn't measure compression time. Maybe it should! If it did, brotli-1 would probably be best.

[^1]: Technically a true LR multiplier would affect weight decay whereas this update scale multiplier does not.
