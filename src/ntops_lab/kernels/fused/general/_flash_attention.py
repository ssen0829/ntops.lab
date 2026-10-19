import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=64, upper_bound=128)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=32, upper_bound=64)
def arrangement(q, k, v, is_causal, softmax_scale, out, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
    def arrange_q_or_o(x):
        arranged = x.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))
        return arranged

    def arrange_k_or_v(x):
        arranged = (
            x.tile((1, 1, BLOCK_SIZE_N, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, q_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged

    q_arranged = arrange_q_or_o(q)
    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), is_causal, softmax_scale, arrange_q_or_o(out)


def application(q, k, v, is_causal, softmax_scale, out):
    q_loaded = (q * softmax_scale * 1.44269504089).to(q.dtype)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i])).to(ntl.float32)
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

        if is_causal:
            mask = q.offsets(-2)[:, None] >= k[i].offsets(-2)[None, :]
            qk = ntl.where(mask, qk, float("-inf"))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)
        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v[i].dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    out = acc


@functools.cache
def kernel(ascend=False):
    q, k, v, out = (
        Tensor(
            4,
            shape_options=(
                None,
                None,
                {"constexpr": True},
                {"constexpr": True, "upper_bound": 128},
            ),
        )
        for _ in range(4)
    )
    is_causal = Tensor(0, constexpr=True)
    softmax_scale = Tensor(0, constexpr=True)
    arrange = arrangement
    if ascend:
        def arrange(q, k, v, is_causal, softmax_scale, out):
            return arrangement(q, k, v, is_causal, softmax_scale, out, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32)

    return ninetoothed.make(
        arrange,
        application,
        (q, k, v, is_causal, softmax_scale, out),
        kernel_name="ntops_lab_flash_attention_ascend" if ascend else "ntops_lab_flash_attention",
    )


def run_flash_attention(q, k, v, *, is_causal=False, softmax_scale=1.0):
    out = torch.empty_like(q, dtype=v.dtype)
    kernel(q.device.type == "npu")(q, k, v, is_causal, float(softmax_scale), out)
    return out


def default_scale(q):
    return q.shape[-1] ** -0.5
