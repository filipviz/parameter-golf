"""Fused Triton kernels for Glaucon.

Fused leaky_relu(x @ W1.T, 0.5)^2 MLP kernel, adapted from modded-nanogpt by @andrewbriand, @jrauvola.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# -----------------------------------------------------------------------------
# Triton kernel for MLP: leaky_relu(x @ W1.T, 0.5)^2, by @andrewbriand, @jrauvola
# Leaky variant by Glaucon team.
# Forward:  f(x) = leaky_relu(x, 0.5)^2 = (x > 0 ? x : 0.5x)^2
# Backward: f'(x) = x > 0 ? 2x : 0.5x  =  2 * (x > 0 ? x : 0.25x)

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0.25 * c0_pre)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_leaky = tl.where(c0 > 0, c0, 0.5 * c0)
            c0_post = c0_leaky * c0_leaky
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0.25 * c1_pre)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_leaky = tl.where(c1 > 0, c1, 0.5 * c1)
            c1_post = c1_leaky * c1_leaky
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.view(-1, x.shape[-1])
        pre, post = linear_relu_square(x_flat, W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_flat = x.view(-1, x.shape[-1])
        grad_flat = grad_output.view(-1, grad_output.shape[-1])
        dW2 = post.T @ grad_flat
        dpre = linear_relu_square(grad_flat, W2, aux=pre)
        dW1 = dpre.T @ x_flat
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2
