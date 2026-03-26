"""Logging utilities for train_gpt.py."""
from __future__ import annotations
import torch
from torch import nn


def _quant_info(name: str, p: torch.Tensor) -> tuple[str, int]:
    """Return (quant_format, estimated_quant_bytes) for a named parameter.

    Mirrors the logic in mixed_quantize_int6 after unbanking:
    - 3D banks → per-slice int6 per-row: B * M * (N + 2)
    - 2D large (>65536) attn/mlp → int6 per-row: M * (N + 2)
    - 2D large (>65536) other → int8 per-row: M * (N + 2)
    - Small tensors (≤65536) → fp16 passthrough: 2 * numel
    - Control tensors → fp32 passthrough: 4 * numel (checked first in quantizer)
    """
    from train_gpt import CONTROL_TENSOR_NAME_PATTERNS
    # Control tensors preserved in fp32 (checked first in quantizer)
    if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
        return "fp32", p.numel() * 4
    # 3D bank tensors get unbanked into 2D slices, each quantized as int6
    if p.ndim == 3:
        B, M, N = p.shape
        return "int6", B * M * (N + 2)  # int8 data + fp16 row scales
    # Small tensors pass through as fp16
    if p.numel() <= 65536:
        return "fp16", p.numel() * 2
    # Large 2D: attn/mlp → int6, embeddings/other → int8
    is_attn_mlp = any(k in name for k in ("qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"))
    fmt = "int6" if is_attn_mlp else "int8"
    if p.ndim == 2:
        M, N = p.shape
        return fmt, M * (N + 2)
    return fmt, p.numel()  # fallback


def log_param_table(model: nn.Module, log_fn) -> None:
    """Log a parameter breakdown table by category with quantization info."""
    buckets: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, p in model.named_parameters():
        if "qo_bank" in name or "kv_bank" in name:
            cat = "Attention banks"
        elif "mlp_up_bank" in name or "mlp_down_bank" in name:
            cat = "MLP banks"
        elif "tok_emb" in name:
            cat = "Token embedding"
        elif "bigram" in name:
            cat = "Bigram"
        elif "ve_shared" in name or "ve_layer" in name:
            cat = "Value embedding"
        else:
            cat = "Scalars/control"
        buckets.setdefault(cat, []).append((name, p))
    # Fixed display order
    order = ["Attention banks", "MLP banks", "Token embedding",
             "Bigram", "Value embedding", "Scalars/control"]
    rows = []
    total_params = sum(p.numel() for p in model.parameters())
    total_qbytes = 0
    for cat in order:
        entries = buckets.get(cat, [])
        if not entries:
            continue
        n = sum(p.numel() for _, p in entries)
        dtype = entries[0][1].dtype
        # Collect quant formats and bytes
        qbytes = 0
        qfmts = set()
        for name, p in entries:
            fmt, qb = _quant_info(name, p)
            qfmts.add(fmt)
            qbytes += qb
        # Show dominant quant format (by byte count)
        fmt_bytes: dict[str, int] = {}
        for name, p in entries:
            fmt, qb = _quant_info(name, p)
            fmt_bytes[fmt] = fmt_bytes.get(fmt, 0) + qb
        qfmt = max(fmt_bytes, key=fmt_bytes.get)  # type: ignore
        if len(fmt_bytes) > 1:
            qfmt += "*"  # asterisk signals mixed formats
        total_qbytes += qbytes
        rows.append((cat, n, dtype, qfmt, qbytes))
    # Format table
    log_fn(f"{'Category':<20s} {'Params':>12s} {'%Params':>8s} {'Dtype':>8s} {'Quant':>6s} {'QBytes':>12s} {'%Quant':>7s}")
    log_fn(f"{'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*7}")
    for cat, n, dtype, qfmt, qbytes in rows:
        dstr = str(dtype).removeprefix("torch.")
        log_fn(f"{cat:<20s} {n:>12_} {n/total_params:>7.1%} {dstr:>8s} {qfmt:>6s} {qbytes:>12_} {qbytes/total_qbytes:>6.1%}")
    log_fn(f"{'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*7}")
    log_fn(f"{'Total':<20s} {total_params:>12_} {'100.0%':>8s} {'':>8s} {'':>6s} {total_qbytes:>12_} {'100.0%':>7s}")
