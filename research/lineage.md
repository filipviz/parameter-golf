# Lineage to PR #549

As of 2026-03-24, the official upstream leaderboard record for `track_10min_16mb` is [PR #549](https://github.com/openai/parameter-golf/pull/549), `LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`, at `1.1194 BPB`.

This document reconstructs how the upstream frontier got from the naive baseline to `#549`, with two goals:

1. Recover the main record lineage, including important unmerged PRs that later merged records explicitly build on.
2. Separate large, durable improvements from small or stack-dependent ones.

## How To Read This Report

- Lower `val_bpb` is better.
- `bytes_total` is the submission artifact size: code plus compressed model. The hard cap is `16,000,000` bytes.
- Training is capped at `600s` on `8xH100 SXM`; evaluation gets a separate `600s`.
- Some deltas below are clean A/B comparisons reported by the PR author. Others are only package-level comparisons against the prior stack. I call this out when it matters.
- Several PRs in the lineage are not merged. They still matter because later merged PRs explicitly cite them as their base stack.
- Some TTT-era PRs were later ruled unsound. I only treat them as inspiration when the specific idea was re-used in a legal form.

## Executive Takeaways

- The early gains were large and structural. `Sliding window eval`, `int6/mixed quantization`, `MLP 3x`, longer context, and eventually `11 layers` delivered improvements on the order of `1e-2` to `5e-2` BPB.
- Mid-stage gains became incremental. `XSA + EMA` was worth about `-0.0047 BPB`; `Partial RoPE + LN Scale` was worth about `-0.0023 BPB`.
- Late-stage frontier work is mostly about stacking many small wins. `#374` was only `-0.0002 BPB` over `#315`, and `#414` was another `-0.0013 BPB` over `#374`.
- `Parallel Muon` is mostly a speed technique, not a direct BPB technique. It helps when the extra steps actually improve the averaged checkpoint; otherwise it can be neutral or even slightly harmful.
- Legal TTT is real, but its marginal value shrinks sharply on stronger bases. On weaker stacks it produced `-0.0068` to `-0.0165 BPB`; on the `#549` frontier stack it was only about `-0.0025 BPB` and consumed about `410s` of eval time.
- `LeakyReLU(0.5)^2` appears to be one of the cleaner late-stage ML wins: roughly `-0.002` to `-0.003 BPB` on strong stacks.

## Main Trunk At A Glance

This is the clearest ancestry chain from the naive baseline to `#549`.

| Stage | Status | BPB | Delta vs prev | Total bytes | Steps / speed | Main changes |
|---|---|---:|---:|---:|---|---|
| Naive baseline | merged | `1.2244` | — | `15,863,489` | `13,780`, `43.5ms` | 9L, 512d, 1024 ctx, MLP 2x, default eval |
| [#70](https://github.com/openai/parameter-golf/pull/70) | open ancestor | `1.1659` | `-0.0585` | `14,855,508` | `12,485`, about `48ms` | `MLP 3x`, int6 quant, sliding-window eval |
| [#164](https://github.com/openai/parameter-golf/pull/164) | open ancestor | `1.1524` | `-0.0135` | `15,401,594` | `8,390`, `68ms` | OrthoInit, SmearGate, BigramHash, seq2048, FA3 |
| [#198](https://github.com/openai/parameter-golf/pull/198) | open ancestor | `1.1318` | `-0.0206` | `15,689,380` | `7,412`, `81ms` | 11 layers, WD `0.04`, SWA, stride `64`, Bigram `2048` |
| [#287](https://github.com/openai/parameter-golf/pull/287) | merged | `1.1271` | `-0.0047` | `15,534,645` | `7,103`, `84ms` | XSA on last 4 layers, EMA replacing SWA |
| [#315](https://github.com/openai/parameter-golf/pull/315) | merged | `1.1248` | `-0.0023` | `15,612,308` | `7,051`, `85ms` | Partial RoPE `16/64`, LN Scale |
| [#374](https://github.com/openai/parameter-golf/pull/374) | open parent | `1.1246` claimed | `-0.0002` | `15,706,024` | `6,942`, `86.4ms` | Tight SWA, shared VE128, late QAT threshold `0.1` |
| [#414](https://github.com/openai/parameter-golf/pull/414) | merged | `1.1233` mean, `1.1228` best seed | `-0.0013` vs `#374` | `15,555,017` | about `7,100` | GPTQ-lite clip search, EMA, warmdown `3500`, QAT `0.15` |
| [#549](https://github.com/openai/parameter-golf/pull/549) | merged | `1.1194` mean | `-0.0039` vs `#414` mean | `15,990,006` | `7,185`, `83.4ms` | LeakyReLU^2, legal score-first TTT, Parallel Muon, final stack tuning |

## Detailed Notes On The Main Trunk

### Baseline

Source: `records/track_10min_16mb/2026-03-17_NaiveBaseline`.

- Reported `val_bpb`: `1.2244`.
- Reported pre-quant `val_bpb`: `1.2172`.
- Quantization penalty: about `+0.0072 BPB`.
- Total size: `15,863,489` bytes.
- Training volume: `13,780` steps, `43.54ms` step average, `7.22B` tokens seen.

Takeaway:

- The baseline is fast and under budget, but leaves large headroom in evaluation and architecture.

### Precursor: Sliding Window Evaluation

Source: `records/track_10min_16mb/2026-03-19_SlidingWindowEval`.

- Same training as baseline, but eval changes from non-overlapping chunks to stride-`64` sliding windows.
- Post-quant `val_bpb` improves from `1.2244` to `1.1925`, a huge `-0.0319 BPB`.
- Artifact size rises only from `15,863,489` to `15,874,829` bytes.
- Eval time grows from about `16s` to about `70s`, still far below the `600s` cap.

Takeaway:

- Sliding-window eval is one of the single highest-leverage ideas in the whole history.
- It became table stakes. Later frontier work should be assumed to use aggressive eval unless stated otherwise.

### #70: MLP 3x + int6 quant + sliding window

Source: [PR #70](https://github.com/openai/parameter-golf/pull/70). This PR is not merged, but later merged records explicitly cite it as their earliest ancestor.

- Reported `val_bpb`: `1.1659`.
- Improvement vs baseline: about `-0.0585 BPB`.
- Total size: `14,855,508` bytes, about `1.0MB` smaller than baseline.
- Steps remain high at `12,485`.

What changed:

- `MLP 3x` instead of `2x`.
- Per-row `int6` quantization on MLP and attention weights, with zstd-style compression.
- Sliding-window eval.

What likely mattered:

- This is a package change, not a clean ablation, but the biggest drivers are obvious: sliding-window eval plus capacity funded by low-precision weights.
- The combination simultaneously improves score and frees artifact budget.

### #164: OrthoInit + SmearGate + BigramHash + seq2048

Source: [PR #164](https://github.com/openai/parameter-golf/pull/164). Also unmerged, but explicitly cited by `#198`, `#287`, and `#315`.

- Reported `val_bpb`: `1.1524`.
- Improvement vs `#70`: `-0.0135 BPB`.
- Total size: `15,401,594` bytes, about `+0.55MB` vs `#70`.
- Steps fall from `12,485` to `8,390`; step time increases to `68ms`.

What changed:

- Orthogonal init with muP-style output scaling.
- SmearGate.
- BigramHash.
- `seq_len=2048` plus sliding-window eval.
- FlashAttention 3.
- Richer tuned Muon setup.

What likely mattered:

- This is another package jump, but the strongest pattern is "use saved bytes to buy more context and richer token-pair features."
- It sacrifices raw step count for better token efficiency.

### #198: 11 layers + WD 0.04 + SWA + stride 64

Source: [PR #198](https://github.com/openai/parameter-golf/pull/198). Unmerged, but the direct parent of `#287`.

- Reported `val_bpb`: `1.1318`.
- Improvement vs `#164`: `-0.0206 BPB`.
- Total size: `15,689,380` bytes.
- Steps: `7,412`, `81ms` step time.

What changed:

- Move from `9` to `11` layers.
- Increase Muon and AdamW weight decay to `0.04`.
- SWA over warmdown.
- Eval stride tightened to `64`.
- Bigram table reduced to `2048` buckets to fit the deeper model.

What likely mattered:

- Going to `11L` is one of the biggest pure modeling gains in the trunk.
- The author explicitly frames the extra two layers as the main capacity gain, funded by quantization headroom.
- `WD=0.04` shows up repeatedly after this point as a quantization-friendly default.

### #287: XSA + EMA

Source: `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`.

- Reported `val_bpb`: `1.1271`.
- Improvement vs `#198`: `-0.0047 BPB`.
- Total size drops from `15,689,380` to `15,534,645` bytes.
- Steps: `7,103`, `84ms`.

What changed:

- XSA on the last 4 layers.
- EMA (`0.997`) instead of SWA.

What likely mattered:

- This is a clean, high-value mid-stage improvement.
- The README attributes no other changes, so `-0.0047 BPB` is one of the more trustworthy isolated numbers in the lineage.
- EMA also made the artifact smaller, which matters because later stacks run close to the limit.

### #315: Partial RoPE + LN Scale

Source: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`.

- Reported `val_bpb`: `1.1248`.
- Improvement vs `#287`: `-0.0023 BPB`.
- Total size rises slightly to `15,612,308` bytes.
- Steps: `7,051`, `85ms`.

What changed:

- Partial RoPE on only `16` of `64` head dims.
- LN Scale: `1 / sqrt(layer_idx + 1)`.

Important footnote:

- The submitted code had a `Late QAT` flag, but the author later determined it was dead code under `torch.compile` constant folding. The README explicitly says the result is driven by Partial RoPE and LN Scale, not Late QAT.

What likely mattered:

- Another clean, trustworthy mid-stage win.
- Zero-parameter changes still had room left at this stage.

### #374: Tight SWA + shared VE128

Source: [PR #374](https://github.com/openai/parameter-golf/pull/374). This was not accepted as a new record because it was not statistically significant, but it is the direct parent named by `#414`.

- Reported `val_bpb`: `1.1246`.
- Claimed improvement vs `#315`: only `-0.0002 BPB`.
- Total size: `15,706,024` bytes.
- Steps: `6,942`, `86.4ms`.

What changed:

- Tight SWA: collect checkpoints only in the last `scale < 0.2` portion of training.
- Shared VE128 on deep layers.
- Late QAT threshold `0.1`.

What likely mattered:

- The PR body only isolates Tight SWA, not VE128.
- This looks like a small compression/averaging polish pass, not a major modeling jump.
- Since the result was not statistically significant, treat the `-0.0002 BPB` as directional, not definitive.

### #414: GPTQ-lite + EMA + warmdown3500 + QAT@0.15

Source: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`.

- Reported `val_bpb`: `1.1233` mean, `1.1228` best seed.
- Improvement vs `#374`: `-0.0013 BPB`.
- Total size: `15,555,017` bytes.
- Around `7,100` steps in `600s`.

The PR gives a clean internal breakdown:

- `GPTQ-lite` clip search: about `-0.0006 BPB`.
- `EMA` on top of the `#374` stack: about `-0.0006 BPB`.
- Warmdown `3000 -> 3500`: about `-0.0002 BPB`.
- Late QAT threshold `0.1 -> 0.15`: about `-0.0001 BPB`.

What likely mattered:

- This is quintessential late-stage frontier work: multiple `1e-4` to `1e-3` improvements stacked together.
- `GPTQ-lite` is attractive because it is eval/export-side work with zero training cost.

### #549: LeakyReLU^2 + legal score-first TTT + Parallel Muon

Source: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`.

- Reported `val_bpb`: `1.1194` mean, `0.0006` std.
- Improvement vs `#414` mean: `-0.0039 BPB`.
- Total size: `15,990,006` bytes, essentially at the limit.
- Training steps: `7,185`, `83.4ms`.
- Pre-TTT mean BPB: `1.1218`.
- Post-TTT mean BPB: `1.1194`.
- TTT gain on the final stack: `-0.0025 BPB`.
- Legal TTT alone costs about `409s` of eval time; total eval remains about `530s`.

The PR includes a useful incremental ablation:

- `+ Parallel Muon`: `+-0.0000 BPB` on the `#414` base in their single-seed ablation.
- `+ legal TTT (freeze=2)`: about `-0.0017 BPB`.
- `+ TTT freeze=0`: about `-0.0004 BPB`.
- `+ larger BigramHash`: about `-0.0009 BPB`.
- `+ LeakyReLU(0.5)^2`: about `-0.0021 BPB` at the end of the stack.

Important caveat:

- The `#549` README is internally inconsistent about BigramHash size. The architecture table and `submission.json` say `BigramHash(1536)`, the run command sets `BIGRAM_VOCAB_SIZE=1536`, but the ablation table mentions `2048 -> 3072`. Treat the BigramHash ablation as approximate or stale.

What likely mattered:

- The clearest ML gains here are legal TTT and LeakyReLU^2.
- Parallel Muon mostly buys speed headroom; on this stack it does not seem to directly improve BPB by itself.
- At this point the frontier is living on small, composable wins and precise engineering.

## Side Branches That Feed Into #549

### TTT Branch: #77 -> #456 -> #461 -> #549

This branch matters because `#549` explicitly says its legal TTT protocol is adapted from `#461`.

| PR | Status | Key result | What it taught |
|---|---|---|---|
| [#77](https://github.com/openai/parameter-golf/pull/77) | merged | mean `1.1928` | Introduced legal score-first LoRA TTT |
| [#456](https://github.com/openai/parameter-golf/pull/456) | open | `1.1600 -> 1.1532`, gain `-0.0068` | Full-model persistent score-first TTT can beat LoRA TTT |
| [#461](https://github.com/openai/parameter-golf/pull/461) | open | `1.1611 -> 1.1446`, gain `-0.0165` | SGD + momentum + 3 epochs + freezing early blocks is much stronger than flat AdamW |
| [#549](https://github.com/openai/parameter-golf/pull/549) | merged | `1.1218 -> 1.1194`, gain `-0.0025` | Same recipe still helps on the frontier, but much less |

Important detail from `#77`:

- Its own ablation says most of the early "TTT" win was actually evaluation protocol:
  - Cross-doc flat stream baseline: `1.2278`
  - Doc-isolated eval: `1.2168`
  - Sliding window: `1.1941`
  - LoRA TTT on top: `1.1910`

Takeaway:

- TTT is real, but it is easy to over-credit it if evaluation semantics are changing at the same time.
- The cleaner the base model gets, the smaller the remaining TTT headroom seems to become.

### Systems Branch: #399 -> #549

`#549` also explicitly cites [PR #399](https://github.com/openai/parameter-golf/pull/399) for Parameter Banking + Parallel Muon.

What `#399` reports:

- `81.87ms` step time, `7,330` steps, `1.1247 BPB`, about `15.8MB`, on an EMA-based stack close to `#315`.
- About `-3.4%` training time per step compared with the unbanked version.
- On that EMA stack, about `-0.0006 BPB`.

But the compatibility table in `#399` is the important part:

- On `#315` style EMA-only stacks: speedup translated into a small quality gain.
- On `#374` tight-SWA stacks: speed improved but BPB got slightly worse.
- On TTT-heavy stacks: more-converged models had less room for TTT adaptation, so quality could worsen.
- In `#549`'s own ablation, `+ Parallel Muon` is essentially `+-0.0000 BPB` by itself.

Takeaway:

- Parameter Banking + Parallel Muon is a good throughput tool.
- It is not a generally reliable direct score improver. It helps when the extra steps actually propagate into the exported checkpoint.

### Activation Branch: #180 -> #493 -> #549

`#549` credits both [#493](https://github.com/openai/parameter-golf/pull/493) and [#518](https://github.com/openai/parameter-golf/pull/518) for LeakyReLU^2, but only `#493` is a clean source here because `#518` was later ruled unsound.

Relevant points:

- [#180](https://github.com/openai/parameter-golf/pull/180) established a strong mixed int5/int6 sibling branch at `1.1428`.
- `#180`'s own ablation says its late tweaks were small:
  - `WD=0.04 + warmdown=3000`: about `-0.0001 BPB`
  - `SWA start_frac=0.4`: about `-0.0006 BPB`
  - `BigramHash 8192 -> 10240`: about `-0.0008 BPB`
- `#493` then introduced LeakyReLU^2 on a stronger 11-layer stack and cited roughly `-0.003 BPB`.
- `#549`'s ablation gives the best near-lineage number: about `-0.0021 BPB` from adding LeakyReLU^2 at the end of the stack.

Takeaway:

- LeakyReLU^2 looks like a genuine late-stage improvement, not just noise.
- The magnitude is modest, but at frontier scores `-0.002` to `-0.003 BPB` is a big deal.

### Bigram / Smear / Ortho / WD / SWA Branch

This branch is older than the clean `#70 -> #164 -> #198` trunk, but it explains where several long-lived ideas came from.

Relevant records:

- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd`:
  - `1.1556 BPB`
  - `15,878,809` bytes
  - `12,047` steps
  - Package: SmearGate, BigramHash, OrthoInit, `MLP 3x`, int6 STE QAT, sliding window
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`:
  - `1.1458 BPB`
  - `15,862,650` bytes
  - `7,379` steps
  - Same family, with Muon `WD=0.04` and aggressive SWA
  - The PR update says `WD 0.02 -> 0.04` plus `SWA every 200 -> 50` improved the score by about `-0.0025 BPB`

Takeaway:

- Before the later 11-layer frontier crystallized, this branch proved that token-pair features plus quantization-aware packing were worth a lot.
- `WD=0.04` and short-interval SWA emerged here before becoming standard defaults elsewhere.

## What Actually Seems To Move The Needle

### Large wins

- Sliding-window eval.
- Low-precision weights that free enough bytes to buy more model.
- `MLP 3x`.
- Longer context and eventually deeper stacks.
- 11 layers instead of 9.

### Medium wins

- SmearGate + BigramHash style token-pair features.
- XSA on top layers.
- EMA in place of naive SWA.
- Legal TTT on weaker models.

### Small but real wins

- Partial RoPE `16/64`.
- LN Scale.
- LeakyReLU^2.
- Bigram bucket retuning.
- GPTQ-lite clip search.
- Warmdown and QAT threshold tuning.

### Stack-dependent or easy to overestimate

- Parallel Muon as a direct BPB improver.
- TTT on already-strong stacks.
- Tight SWA unless the rest of the stack is set up to benefit.
- Claims from unsound TTT PRs such as `#518` or later-closed scheduling PRs like `#481`.

## Practical Guidance For Future Work

- If a new idea costs lots of complexity, it should probably beat a `1e-3 BPB` hurdle to be worth keeping. The easy `1e-2` gains are already gone.
- The strongest stable ingredients by `#549` are: 11L, long context, aggressive eval, strong quantization/compression, XSA, EMA, Partial RoPE, LN Scale, and a carefully engineered optimizer/runtime.
- TTT still matters, but it should be judged against its eval-time cost, legality risk, and diminishing returns on strong bases.
- When reading late-stage PRs, prefer ones that include clean ablations. The trunk after `#315` is mostly a story of tiny deltas, and those are easy to misread without controlled comparisons.
