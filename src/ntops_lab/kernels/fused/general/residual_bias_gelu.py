import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(a, b, c, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), c.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, c, out):
    out = 0.5 * (a + b + c) * (1.0 + ntl.erf((a + b + c) * 0.7071067811865476))

def application_npu(a, b, c, out):
    out = 0.5 * (a + b + c) * (1.0 + ntl.tanh(0.7978845608028654 * ((a + b + c) + 0.044715 * (a + b + c) * (a + b + c) * (a + b + c))))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_residual_bias_gelu")
kernel_npu = ninetoothed.make(arrangement, application_npu, (Tensor(1), Tensor(1), Tensor(1), Tensor(1)), kernel_name="ascend_ntops_lab_residual_bias_gelu")

def run(*inputs):
    a, b, c = inputs
    out = torch.empty_like(a)
    selected_kernel = kernel_npu if a.device.type == "npu" else kernel
    selected_kernel(a, b, c, out)
    return out
