import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = block_size(upper_bound=64)
BN = block_size(upper_bound=64)
BK = block_size(upper_bound=64)

def arrangement(a, b, out):
    out_arr = out.tile((BM, BN))
    a_arr = a.tile((BM, BK)).tile((1, -1)).expand((-1, out_arr.shape[1]))
    a_arr.dtype = a_arr.dtype.squeeze(0)
    b_arr = b.tile((BK, BN)).tile((-1, 1)).expand((out_arr.shape[0], -1))
    b_arr.dtype = b_arr.dtype.squeeze(1)
    return a_arr, b_arr, out_arr

def application(a, b, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)
    for k in range(a.shape[0]):
        acc += ntl.dot(a[k], b[k])
    value = acc
    out = (0.0 - ntl.log(1.0 + ntl.exp(0.0 - value))).to(ntl.float16)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)), kernel_name="ntops_lab_mm_logsigmoid")

def run(*inputs):
    a, b = inputs
    if a.device.type == "npu":
        from ntops_lab.kernels.linear._ascend_matmul import matmul

        return matmul(a, b, activation="logsigmoid")
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    kernel(a, b, out)
    return out
