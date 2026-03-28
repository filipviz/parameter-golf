First, a run with the current implementation to see where things are performance-wise.

1. Experimenting with `d_vocab`, non-tied embeddings, and vocab quantization. The `embedding`/`lm_head` matrix is an unusually small component of our overall parameter count. On one hand, this may be desireable given that `d_model` is so small. But this seems under-explored in general. I trained an sp2048 tokenizer, and we can experiment with that, the current sp1024 with untied embeddings/lm_head, and also leaving it in bf16 instead of quantizing to int8.
2. Label smoothing.
3. WSD lr schedule.
4. Batch sizes. The repo baseline used `524_288`, and a number of runs used `786_432`.
5. Grad norm clipping. First, the existing implementation is technically buggy - we clip before we all-reduce. That said, the difference is marginal in practice, and a proper implementation would be slightly tricky. I'm more interested in using a higher grad clip norm, or using other techniques to stabilize early training.

Re-introduce a few small tweaks:
- AdamW eps=1e-10
- symmetric muon aspect ratio scaling for kv bank.
- per-head gated attention

Start by reading AGENTS.md and glaucon/train_gpt.py to get your bearing. We're currently on an 8xH100 instance and the goal is to do some performance engineering, incorporating some techniques to see if they provide a speedup. We can use short (~1000 step) runs, profiling, and benchmarking to speed up our iteration cycle here.

1. One possible piece of low-hanging fruit is replacing the mutable rotary cache with fixed buffers (since sequence length is the same at train and test time). Want to give that a shot?
2. I remember seeing some test which showed a speedup from cudagraph marking the beginning of the step, but I'm not sure if that applies for us since we're using the default compile mode. Could you help me think through this?
3. The next thing we should investigate is "bucketed Muon" - stacking same-shape grads. As a disclaimer, I'm mentioning this because I was able to implement it on the repo baseline a while back. But I know that the muon implementation in the frontier record glaucon is branched off of is a bit different. So perhaps we should just do some investigation here - is there a possible speedup here, or have the conditions changed? I'm pretty sure there's a solid example implementation in Andrej Karpathy's nanochat (/workspace/nanochat/nanochat/optim.py)
4. Optimize EMA collection with foreach APIs. We should check every for loop for this possibility.
5. Compiling the NS iterations and AdamW step.
6. expandable_segments

The thing I'm really interested in is replacing the u-net skip connections, `resid_mix`, and MLP/attention scales with something like [Attention Residuals](https://arxiv.org/abs/2603.15031). I could see fixed learned scalars working well, but we should also try the learned query approach from the paper.

## What didn't work

- **FP8 MLP / FP8 head** — slower
- **Fused softcapped cross-entropy** — slower
- **Packed QKV projection** — slower
- **cuDNN-only SDPA** — slower (~477ms)
- **`torch.compile(mode="reduce-overhead")`** — blocked by CUDAGraph overwrite failures in both fused MLP and vanilla CastedLinear backward paths, even after multiple workaround attempts
- **CUTLASS-backed max-autotune** — impractical startup cost, repeated precompile failures
- **`torch.compile(mode="max-autotune-no-cudagraphs")`** — slight miss at 457ms + much slower validation
- **`expandable_segments` allocator** — helped standalone but not additive on the best config

