## Key Context

We're competing in OpenAI's [Parameter Golf competition](https://github.com/openai/parameter-golf/). It's in the spirit of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), but with different rules:
- Code and compressed model are under 16 MB (16_000_000 bytes, not 16 MiB).
- Trains under 10 minutes on 8xH100 SXM.
- Evaluated by compression (nats of cross-entropy BPB) on the FineWeb validation set.
- They disqualify runs which are not in the spirit of challenge (e.g. sneaking in compute unfairly).
- To set a new record, runs must beat the current leader by 0.005 nats. Must show at `p < 0.01` due to inter-run variance.
- Evaluation is allowed at any sequence length.
- External dependencies are allowed. It is fair game to add third-party libraries or alternate toolchains/runtime versions (e.g. `flash-attn`, nightly PyTorch, `brotli`) as long as the result stays within the rules and spirit of the challenge.

## Repo Layout

- `README.md`: more about the challenge, FAQ, accepted leaderboard.
- `train_gpt.py`: organizer's baseline.
- `records/track_10min_16mb`: accepted submissions.
- `glaucon/`: our work. We'll move it to the records folder before submitting.
- `glaucon/train_gpt.py`: the Glaucon training script.
- `glaucon/run.sh`: the canonical Glaucon launcher and experiment queue for full remote runs; if we are lining up several serious experiments, make that explicit here.
- `research/`: writeups and documents related to our research.
- `research/agenda.md`: our lightweight source of truth for the current agenda and queued experiments.

Accepted submissions are merged into the repo, but:
- This takes time, so the current frontier is often in an upstream PR.
- The `origin` git remote is my fork, and the `upstream` git remote is the upstream repo.
- Use the `gh` CLI to view these PRs
- Be skeptical. Many of these submissions will be rejected because they made a mistake or cheated. Test time training PRs are particularly unreliable.

## Our Work and Philosophy

- Our submission is named Glaucon (like Plato's Republic).
- My first attempt started from PR #315 which achieved 1.1248 BPB.
- I got to 1.12065 BPB by adding per-head gated attention, symmetric Muon aspect ratio scaling, brotli compression, and tweaking AdamW eps (to 1e-10) and Muon/AdamW weight decay.
- Other tweaks like GEGLU, [partial key offset](https://github.com/KellerJordan/modded-nanogpt/pull/169), and modified logit softcapping parameters did not improve downstream validation BPB.
- If needed, there are more details on this attempt in `research/first-attempt.md`.

This was the top score at the time, but I didn't submit it. First of all, it fell short of the required 0.005 nat BPB improvement. But more fundamentally, I had been using the same approach everyone else had been using, which I think is fundamentally flawed. Looking at other PRs gives me the sense that everyone is competing in the same fashion - taking the best baseline, not deeply understanding it, then trying to apply techniques from `modded-nanogpt` or using something like Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) to automatically tweak hyperparameters until they get a minor improvement on validation BPB. This means that:
- People aren't thinking through the fundamentals. There is probably more low-hanging fruit in rigorous applications of scaling laws and analysis of weights, activations, and gradients throughout the network than there is in applying yet another random set of architectural tweaks.
- Many tweaks add complexity but don't work, or work for unexpected reasons (e.g. through an interaction with another component). For example, a buggy NTK extrapolation implementation never actually changed from train to test time, and was functionally equivalent to raising the RoPE `theta_base` from 10k to 22k. This *did* help validation BPB, but not for the reason the author thought.

Our philosophy is to beat other entrants by using rigorous, straightforward scientific approaches. Instead of blind guesswork, we are going to answer high-level questions using simple experimentation and use that to guide our work.

## Current Frontier

As of 2026-03-25, the official upstream leaderboard frontier is PR #549 at 1.1194 BPB (`LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`), and it has already been merged into `upstream/main`. Because new records must improve by at least 0.005 nats of BPB, the next record needs to achieve 1.1144 BPB or lower.

Among open upstream PRs, the strongest current 10min/16MB contender is PR #728 at 1.1142 BPB (`Val-Calibrated GPTQ + XSA-all + BigramHash 3072x112`), which at least clears that threshold on paper.

PRs progressively build on top of one another, so it can be helpful to take a look at the full lineage. A full report on the lineage of PR #549 is available in `research/lineage.md`.

## Key Insights and Considerations

- The 16 MB limit is clearly the practical bottleneck on performance. These models are 15-20x overtrained relative to Chinchilla-optimal.
- Most of the records so far are based on small one-off tweaks, and aren't motivated by unified theoretical premises, rigorous mathematics, or empirically derived scaling laws. We can win by being more rigorous.
- Records so far have an MFU of ~16-19%. We may be able to improve occupancy with larger, denser matrices.

## About Fineweb

- Mean document length ~430 words, but heavily right-skewed.
- Overwhelmingly English prose webtext.
- Code and math are scarce.
- PII is anonymized - it may be worth identifying these and creating dedicated tokens for `email@example.com`, IP addresses, and so on.
- Test-time sorting and cross-document attention could be worthwhile. Clever URL-domain grouping, topic clustering, and document packing with full attention.

## Compression Notes

- Treat compression as a first-class modeling problem, not just a postprocessing detail. What matters is the byte stream after quantization, not the fp32 checkpoint.
- brotli-11 tends to outperform other compressors.
- The tied embeddings are sensitive to quantization. Strong submissions often keep it at fp16 or fp8/int8.
- Good submissions often use weight decay around `0.04`. Think of it partly as a compression/quantization hyperparameter, not just regularization. Pushing it higher can hurt BPB, but reduce artifact size by making the weights more compressible.
- Mixed precision is already proven in the repo: int5 MLPs + int6 attention is real. Full int5 may be viable, but only with careful QAT / GPTQ and honest roundtrip validation.
- In repo history, big wins came from better quantized payloads: lower bit-width where possible, tensor-specific quantization, weight decay, EMA/SWA, GPTQ-lite / GPTQ, and artifact-aware architecture choices.
- Be wary of clever low-bit schemes that optimize RMSE per raw bit but flatten the code histogram. In some proxy experimentation, Hadamard/NF4/block-scaling style ideas look bad for LZ-style compression because they destroy the peaked, low-entropy code stream that compressors exploit.
- Always measure compression ideas end-to-end: `bytes_total`, per-tensor or per-family bytes, pre-quant vs roundtrip BPB, and eval-time impact. Do not trust a codec or quantization idea based on raw RMSE alone.
- Metric hygiene: the headline number is the final quantized roundtrip `val_bpb` together with total artifact bytes, not the raw checkpoint loss before quantization. In practice, pay attention to lines like `final_int8_zlib_roundtrip` in the baseline and `final_int6_roundtrip` / sliding-window roundtrip lines in Glaucon, plus the printed compressed-size totals.
- Several accepted PRs enabled late-QAT flags that were constant-folded away under `torch.compile` and never actually executed.

## Resources

If you get stuck or need more information, you have access to:
- The user, who has seen a decent amount of research in similar contexts and has good intuitions for what's worth trying and what isn't at a high level. The user can also help you with permissions issues and managing the environment. If part of the environment has poor ergonomics, you find things frustrating to use, or you are missing key details, which the user could have provided to you, tell the user about your issue so they can improve things going forwards.
- GPT 5.4 Pro, a model which can think for very long periods of time (30 minutes to an hour on average), but produces thorough and meticulous results. For detailed literature review, complex mathematical questions, and other things of that sort, you can make the case to the user why it's worth querying GPT 5.4 Pro.

We can get intuitions from:
- [modded-nanogpt](/Users/filip/Developer/modded-nanogpt): the challenge there is to reach `<= 3.28` FineWeb validation loss on `8xH100` as fast as possible, so it is primarily a speedrun over a fixed active-parameter budget rather than a compressed-model benchmark. Treat it as a monolithic speedrun repo: `README.md` is the fastest way to recover record lineage and accepted ideas, `run.sh` is just a thin wrapper, and the current training logic lives mainly in `train_gpt.py` plus `triton_kernels.py`. Also keep in mind that modded-nanogpt does not have our hard artifact-size constraint, which is part of why embedding-heavy techniques and other parameter-inefficient tricks show up so often there and may transfer poorly to Parameter Golf.
- [slowrun](/Users/filip/Developer/slowrun): basically Nanochat adapted to the finite-data regime. Most limited-track logic lives in `train.py`, with sibling variants in `tiny/train.py` and `unlimited/train.py`; it is a useful source of intuitions about overtraining, heavy regularization, EMA, and distillation.
- [nanochat](/Users/filip/Developer/nanochat): the cleanest scaling-law harness of the three. Start from `runs/scaling_laws.sh` or `runs/miniseries.sh`, then read `scripts/base_train.py`, `nanochat/gpt.py`, and `nanochat/optim.py`; `--depth` is the main complexity dial and many other hyperparameters are auto-derived.
- Other submissions and records for this repo.
- Existing small models like Qwen3-0.6B and [Baguettotron](https://huggingface.co/PleIAs/Baguettotron).

## Hardware

- Main development takes place on a MacBook M3 Pro. This is a decently powerful computer, meaning you can run small sanity-check experiments using the mps backend.
- The competition is scored on an 8xH100 instance, so the user will periodically rent such a machine for evaluations.
- That being said, to avoid undue spending, we can also do research on a 1xH100 machine. This is especially useful for performance engineering and profiling. If you think getting access to such a machine would meaningfully speed up your work, don't hesitate to ask the user for this.

## Implementing and Running Experiments

- Start by working with the user to put together a clear experimental design if you don't already have one. If you need to temporarily add code to calculate and log new metrics, that's completely fine. We can always remove that code later.
- When implementing experiments, your change should be disabled by default, with the option to enable it using an env variable.
- The script should remain functionally unchanged when your feature is not enabled.
- If this style of implementation is too onerous for your change because it's very involved or fundamental, explain this to the user and come to an agreement with them.
- If the results are good, we will probably hard-code the change in, or at least make it the new default.
- We will generally avoid using worktrees, reverting commits, or manipulating git history. If we need to make a change, we can make that change directly.
- Once your experiments is ready, you can queue it by adding the appropriate `torchrun` command to `glaucon/run.sh`. Pay attention to env vars and seeds. Clearly label command blocks so the intended run plan is obvious to the next agent.
- When editing `glaucon/run.sh`, keep each queued experiment explicit and self-contained. Avoid silently changing the canonical default unless that is the point; if multiple experiments are being lined up, label them with short comments and keep the script easy to prune after results come back.

## Miscellaneous

- Need upstream file: stage in `/tmp/`, then cherry-pick; never overwrite tracked.
- PRs: use `gh pr view/diff` (no URLs).
- Prefer end-to-end verify; if blocked, say what’s missing.

## Critical Thinking

- Fix root cause (not band-aid).
- Unsure: read more code; if still stuck, ask w/ short options.
- Conflicts: call out; pick safer path.
- Unrecognized changes: assume other agent; keep going; focus your changes. If it causes issues, stop + ask user.
- Leave breadcrumb notes in thread.
