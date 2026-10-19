import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ._upsample_common import normalize_scale_factor


def _align_corners_ratio(in_size, out_size):
    return 0.0 if out_size <= 1 else (in_size - 1.0) / (out_size - 1.0)


@functools.cache
def _horizontal_kernel(in_w, out_w):
    ratio = _align_corners_ratio(in_w, out_w)

    def arrangement(x, temp, ratio):
        x_arr = x[:, :, :, None, :].expand((-1, -1, -1, out_w, -1)).flatten(end_dim=4)
        return x_arr.tile((1, in_w)), temp.flatten().tile((1,)), ratio

    def application(x, temp, ratio):
        src = temp.offsets(3) * ratio
        diff = x.offsets(3) - src
        dist = ntl.maximum(diff, 0.0 - diff)
        weight = ntl.maximum(0.0, 1.0 - dist)
        temp = ntl.sum(x * weight, axis=1)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(4), Tensor(4), Tensor(0, constexpr=True, value=ratio, name="ratio")),
        kernel_name=f"ntops_lab_upsample_bilinear_horizontal_w{in_w}_to_{out_w}",
        max_num_configs=1,
    )


@functools.cache
def _vertical_kernel(in_h, out_h):
    ratio = _align_corners_ratio(in_h, out_h)

    def arrangement(temp, out, ratio):
        temp_arr = temp[:, :, None, :, :].expand((-1, -1, out_h, -1, -1)).permute((0, 1, 2, 4, 3)).flatten(end_dim=4)
        return temp_arr.tile((1, in_h)), out.flatten().tile((1,)), ratio

    def application(temp, out, ratio):
        src = out.offsets(2) * ratio
        diff = temp.offsets(2) - src
        dist = ntl.maximum(diff, 0.0 - diff)
        weight = ntl.maximum(0.0, 1.0 - dist)
        out = ntl.sum(temp * weight, axis=1)

    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(4), Tensor(4), Tensor(0, constexpr=True, value=ratio, name="ratio")),
        kernel_name=f"ntops_lab_upsample_bilinear_vertical_h{in_h}_to_{out_h}",
        max_num_configs=1,
    )


def _output_hw(x, size, scale_factor):
    if size is not None:
        if len(size) != 2:
            raise ValueError("upsample_bilinear size must have 2 values")
        return int(size[0]), int(size[1])
    scale_h, scale_w = normalize_scale_factor(scale_factor, 2)
    return x.shape[2] * scale_h, x.shape[3] * scale_w


def run(*inputs, size=None, scale_factor=2):
    (x,) = inputs
    out_h, out_w = _output_hw(x, size, scale_factor)
    if x.device.type == "npu":
        from ._ascend_lookup import bilinear

        return bilinear(x, out_h, out_w)
    temp = torch.empty((x.shape[0], x.shape[1], x.shape[2], out_w), device=x.device, dtype=x.dtype)
    out = torch.empty((x.shape[0], x.shape[1], out_h, out_w), device=x.device, dtype=x.dtype)
    _horizontal_kernel(x.shape[3], out_w)(x, temp, _align_corners_ratio(x.shape[3], out_w))
    _vertical_kernel(x.shape[2], out_h)(temp, out, _align_corners_ratio(x.shape[2], out_h))
    return out
