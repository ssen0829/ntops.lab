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
    out = acc.to(ntl.float16)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)), kernel_name="ntops_lab_bmm_out_2d")

def run(*inputs):
    a, b = inputs
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype)
    if a.device.type == "npu":
        from ntops_lab.kernels.linear._ascend_matmul import matmul

        for i in range(a.shape[0]):
            out[i].copy_(matmul(a[i], b[i], activation="out"))
        return out
    for i in range(a.shape[0]):
        kernel(a[i], b[i], out[i])
    return out
