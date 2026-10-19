import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _kernel(in_l, out_l):
    def arrangement(x, indices, positions, out):
        x_arr = x[:, :, None, :].expand((-1, -1, out_l, -1))
        idx_arr = indices[:, :, None, :].expand((-1, -1, out_l, -1))
        pos_arr = positions.unsqueeze(3).expand((-1, -1, -1, in_l))
        out_arr = out.unsqueeze(3).expand((-1, -1, -1, in_l)).tile((1, 1, out_l, in_l))
        out_arr.dtype = out_arr.dtype.squeeze(3)
        return x_arr.tile((1, 1, out_l, in_l)), idx_arr.tile((1, 1, out_l, in_l)), pos_arr.tile((1, 1, out_l, in_l)), out_arr

    def application(x, indices, positions, out):
        out = ntl.sum(x * (indices == positions), axis=3)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(3), Tensor(3), Tensor(3), Tensor(3)),
        kernel_name=f"ntops_lab_max_unpool1d_in{in_l}_out{out_l}",
        max_num_configs=1,
    )


def run(*inputs, kernel_size=2, stride=2, padding=0, output_size=None):
    x, indices = inputs
    if kernel_size != 2 or stride != 2 or padding != 0:
        raise ValueError("max_unpool1d currently supports kernel_size=2, stride=2, padding=0")
    out_l = int(output_size[-1]) if output_size is not None else x.shape[2] * stride
    out = torch.empty((x.shape[0], x.shape[1], out_l), device=x.device, dtype=x.dtype)
    positions = torch.arange(out_l, device=x.device, dtype=torch.int64).view(1, 1, out_l).expand(x.shape[0], x.shape[1], out_l)
    _kernel(x.shape[2], out_l)(x, indices, positions, out)
    return out
