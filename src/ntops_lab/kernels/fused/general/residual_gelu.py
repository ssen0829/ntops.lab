import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(a, b, out):
    return a.tile((BLOCK_SIZE,)), b.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(a, b, out):
    out = 0.5 * (a + b) * (1.0 + ntl.erf((a + b) * 0.7071067811865476))

def application_npu(a, b, out):
    out = 0.5 * (a + b) * (1.0 + ntl.tanh(0.7978845608028654 * ((a + b) + 0.044715 * (a + b) * (a + b) * (a + b))))

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ntops_lab_residual_gelu")
kernel_npu = ninetoothed.make(arrangement, application_npu, (Tensor(1), Tensor(1), Tensor(1)), kernel_name="ascend_ntops_lab_residual_gelu")

def run(*inputs):
    a, b = inputs
    out = torch.empty_like(a)
    selected_kernel = kernel_npu if a.device.type == "npu" else kernel
    selected_kernel(a, b, out)
    return out
