import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(x, residual, weight, out, hidden, approximate):
    weight = weight[None, :].expand((x.shape[0], -1))
    return x.tile((1, hidden.value)), residual.tile((1, hidden.value)), weight.tile((1, hidden.value)), out.tile((1, hidden.value)), hidden, approximate

def application(x, residual, weight, out, hidden, approximate):
    y = x + residual
    rrms = ntl.rsqrt(ntl.sum(y * y, axis=1) / hidden + 1.0e-5)
    value = y * rrms[:, None] * weight
    if approximate == 1:
        out = 0.5 * value * (1.0 + ntl.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))
    else:
        out = 0.5 * value * (1.0 + ntl.erf(value * 0.7071067811865476))

@functools.cache
def _kernel(hidden, approximate=0):
    hidden_tensor = Tensor(0, constexpr=True, value=hidden, name="hidden")
    approximate_tensor = Tensor(0, constexpr=True, value=approximate, name="approximate")
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(1), Tensor(2), hidden_tensor, approximate_tensor),
        kernel_name=f"ntops_lab_add_rms_norm_gelu_h{hidden}_approx{approximate}", max_num_configs=1,
    )

def run(*inputs):
    x, residual, weight = inputs
    out = torch.empty_like(x)
    hidden = x.shape[-1]
    # torch_npu GELU uses the tanh approximation even with its default arguments.
    approximate = int(x.device.type == "npu")
    _kernel(hidden, approximate)(x, residual, weight, out, hidden, approximate)
    return out
