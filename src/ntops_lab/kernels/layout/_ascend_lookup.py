"""Direct Ascend lookup kernels avoiding expanded reduction layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _embedding(index, weight, out, count: tl.constexpr, dim: tl.constexpr,
               row_stride: tl.constexpr, col_stride: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count * dim
    rows = tl.load(index + offsets // dim, valid, other=0)
    values = tl.load(weight + rows * row_stride + offsets % dim * col_stride, valid, other=0)
    tl.store(out + offsets, values, valid)


def embedding(index, weight):
    index = index.contiguous()
    out = torch.empty((*index.shape, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    if out.numel():
        _embedding[(triton.cdiv(out.numel(), 256),)](
            index, weight, out, index.numel(), weight.shape[1], *weight.stride(), 256)
    return out


@triton.jit
def _bilinear(x, out, count: tl.constexpr, ih: tl.constexpr, iw: tl.constexpr,
              oh: tl.constexpr, ow: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count
    plane = offsets // (oh * ow)
    sy = (offsets // ow % oh).to(tl.float32) * ((ih - 1.0) / (oh - 1.0) if oh > 1 else 0.0)
    sx = (offsets % ow).to(tl.float32) * ((iw - 1.0) / (ow - 1.0) if ow > 1 else 0.0)
    y0, x0 = sy.to(tl.int32), sx.to(tl.int32)
    y1, x1 = tl.minimum(y0 + 1, ih - 1), tl.minimum(x0 + 1, iw - 1)
    fy, fx = sy - y0, sx - x0
    a = tl.load(x + plane * ih * iw + y0 * iw + x0, valid, other=0).to(tl.float32)
    b = tl.load(x + plane * ih * iw + y0 * iw + x1, valid, other=0).to(tl.float32)
    c = tl.load(x + plane * ih * iw + y1 * iw + x0, valid, other=0).to(tl.float32)
    d = tl.load(x + plane * ih * iw + y1 * iw + x1, valid, other=0).to(tl.float32)
    tl.store(out + offsets, (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy, valid)


def bilinear(x, out_h, out_w):
    x = x.contiguous()
    out = torch.empty((*x.shape[:2], out_h, out_w), device=x.device, dtype=x.dtype)
    if out.numel():
        _bilinear[(triton.cdiv(out.numel(), 256),)](
            x, out, out.numel(), *x.shape[2:], out_h, out_w, 256, enable_fp_fusion=False)
    return out
