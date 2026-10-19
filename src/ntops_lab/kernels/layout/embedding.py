import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(index, weight, out, vocab, dim):
    index_arr = index[:, None, None].expand((-1, dim.value, vocab.value))
    weight_arr = weight.permute((1, 0))[None, :, :].expand((index.shape[0], -1, -1))
    out_arr = out.unsqueeze(2).tile((1, dim.value, 1))
    out_arr.dtype = out_arr.dtype.squeeze(2)
    return index_arr.tile((1, dim.value, vocab.value)), weight_arr.tile((1, dim.value, vocab.value)), out_arr, vocab, dim


def application(index, weight, out, vocab, dim):
    out = ntl.sum(weight * (index == weight.offsets(0)), axis=2)


@functools.cache
def _kernel(vocab, dim):
    vocab_tensor = Tensor(0, constexpr=True, value=vocab, name="vocab")
    dim_tensor = Tensor(0, constexpr=True, value=dim, name="dim")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(1), Tensor(2), Tensor(2), vocab_tensor, dim_tensor),
        kernel_name=f"ntops_lab_embedding_v{vocab}_d{dim}",
        max_num_configs=1,
    )


def run(*inputs, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    index, weight = inputs
    if padding_idx is not None or max_norm is not None or scale_grad_by_freq or sparse:
        raise ValueError("embedding currently supports the basic lookup path only")
    vocab, dim = weight.shape
    if weight.device.type == "npu":
        from ._ascend_lookup import embedding

        return embedding(index, weight)
    out = torch.empty((*index.shape, dim), device=weight.device, dtype=weight.dtype)
    _kernel(vocab, dim)(index, weight, out, vocab, dim)
    return out
