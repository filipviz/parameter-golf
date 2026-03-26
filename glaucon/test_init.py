"""Snapshot model parameter initialization for equivalence testing.

Usage:
    uv run python3 test_init.py > snapshot.txt
    # ... make changes ...
    uv run python3 test_init.py > snapshot2.txt
    diff snapshot.txt snapshot2.txt
"""
import sys, types, hashlib

# Mock flash_attn (only needed at forward time, not construction)
_mock = types.ModuleType('flash_attn_interface')
_mock.flash_attn_func = None
sys.modules['flash_attn_interface'] = _mock

import torch
sys.path.insert(0, '.')
import train_gpt as T

torch.manual_seed(42)

with torch.device('meta'):
    model = T.GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3.0, embed_init_std=0.005,
        logit_softcap=30.0, rope_base=22082.0, qk_gain_init=1.5,
        bigram_vocab_size=2048, bigram_dim=128, xsa_last_n=4,
        rope_dims=16, ln_scale=True, ve_enabled=True, ve_dim=128,
        ve_layers="9,10",
    )
model = model.to_empty(device=torch.device('cpu'))
model._init_weights()

for name, p in sorted(model.named_parameters()):
    h = hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()[:16]
    print(f"{name:50s} {str(list(p.shape)):>20s} {h}")

for name, b in sorted(model.named_buffers()):
    h = hashlib.sha256(b.detach().numpy().tobytes()).hexdigest()[:16]
    print(f"{name:50s} {str(list(b.shape)):>20s} {h}")
