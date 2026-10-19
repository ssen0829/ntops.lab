import torch
import ninetoothed
from ninetoothed import Tensor, block_size

BLOCK_SIZE = block_size()

def arrangement(x, out):
    return x.tile((BLOCK_SIZE,)), out.tile((BLOCK_SIZE,))

def application(x, out):
    out = x == 0

kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)), kernel_name="ntops_lab_logical_xor_scalar_out")

def run(*inputs):
    x, = inputs
    out = torch.empty((x.numel(),), device=x.device, dtype=torch.bool)
    kernel(x, out)
    return out
