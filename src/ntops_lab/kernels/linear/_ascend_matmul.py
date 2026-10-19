"""Compact Ascend lowering for matrix products, avoiding expanded tile masks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _activate(value, activation: tl.constexpr):
    if activation == "abs":
        return (tl.where(value < 0.0, 0.0 - value, value))
    elif activation == "cube":
        return (value * value * value)
    elif activation == "elu":
        return (tl.where(value > 0.0, value, tl.exp(value) - 1.0))
    elif activation == "gelu":
        return 0.5 * value * (1.0 + tl.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))
    elif activation == "hardshrink":
        return (tl.where(value > 0.5, value, tl.where(value < -0.5, value, 0.0)))
    elif activation == "hardsigmoid":
        return (tl.minimum(tl.maximum(value + 3.0, 0.0), 6.0) / 6.0)
    elif activation == "hardtanh":
        return (tl.minimum(tl.maximum(value, -1.0), 1.0))
    elif activation == "leaky_relu":
        return (tl.where(value > 0.0, value, 0.01 * value))
    elif activation == "logsigmoid":
        return (0.0 - tl.log(1.0 + tl.exp(0.0 - value)))
    elif activation == "neg":
        return (0.0 - value)
    elif activation == "out":
        return value
    elif activation == "relu":
        return (tl.maximum(value, 0.0))
    elif activation == "relu6":
        return (tl.minimum(tl.maximum(value, 0.0), 6.0))
    elif activation == "sigmoid":
        return ((1.0 / (1.0 + tl.exp(-(value)))))
    elif activation == "silu":
        return (value * (1.0 / (1.0 + tl.exp(-(value)))))
    elif activation == "softshrink":
        return (tl.where(value > 0.5, value - 0.5, tl.where(value < -0.5, value + 0.5, 0.0)))
    elif activation == "square":
        return (value * value)
    elif activation == "tanh":
        return tl.tanh(value)
    elif activation == "tanhshrink":
        return (value - (2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0))
    elif activation == "clamp":
        return (tl.minimum(tl.maximum(value, -1.0), 1.0))
    elif activation == "hardswish":
        return (value * tl.minimum(tl.maximum(value + 3.0, 0.0), 6.0) / 6.0)
    elif activation == "softplus":
        return (tl.log(1.0 + tl.exp(value)))
    elif activation == "softsign":
        return (value / (1.0 + tl.where(value < 0.0, -value, value)))
    else:
        return value


@triton.jit
def _matmul_kernel(
    a, b, out, bias, vector_bias, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    am: tl.constexpr, ak: tl.constexpr, bk: tl.constexpr, bn: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr, cm: tl.constexpr, cn: tl.constexpr,
    has_vector_bias: tl.constexpr, vector_stride: tl.constexpr,
    activation: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    reduction = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), tl.float32)
    for start in range(triton.cdiv(k, block_k)):
        ks = start * block_k + reduction
        left = tl.load(a + rows[:, None] * am + ks[None, :] * ak,
                       (rows[:, None] < m) & (ks[None, :] < k), other=0)
        right = tl.load(b + ks[:, None] * bk + cols[None, :] * bn,
                        (ks[:, None] < k) & (cols[None, :] < n), other=0)
        acc = tl.dot(left, right, acc)
    # References apply activation to a materialized product in the input dtype.
    # Preserve this rounding before discontinuous thresholds such as hardshrink.
    if activation is not None:
        acc = acc.to(out.dtype.element_ty).to(tl.float32)
    if has_bias:
        acc += tl.load(bias + rows[:, None] * cm + cols[None, :] * cn,
                       (rows[:, None] < m) & (cols[None, :] < n), other=0)
        if activation is not None:
            acc = acc.to(out.dtype.element_ty).to(tl.float32)
    if has_vector_bias:
        acc += tl.load(vector_bias + cols[None, :] * vector_stride,
                       cols[None, :] < n, other=0)
        if activation is not None:
            acc = acc.to(out.dtype.element_ty).to(tl.float32)
    if activation is not None:
        acc = _activate(acc, activation)
    tl.store(out + rows[:, None] * n + cols[None, :], acc,
             (rows[:, None] < m) & (cols[None, :] < n))


def matmul(a, b, bias=None, activation=None, vector_bias=None):
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError("matrix product inner dimensions must match")
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m and n:
        _matmul_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            a, b, out, bias, vector_bias, m, n, k, *a.stride(), *b.stride(), 32, 32, 32,
            bias is not None, *(bias.stride() if bias is not None else (0, 0)),
            vector_bias is not None, vector_bias.stride(0) if vector_bias is not None else 0,
            activation,
        )
    return out
