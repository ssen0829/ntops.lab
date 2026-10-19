import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BM = block_size(upper_bound=64)
BN = block_size(upper_bound=64)
BK = block_size(upper_bound=64)

def arrangement(a, b, bias, out):
    out_arr = out.tile((BM, BN))
    a_arr = a.tile((BM, BK)).tile((1, -1)).expand((-1, out_arr.shape[1]))
    a_arr.dtype = a_arr.dtype.squeeze(0)
    b_arr = b.tile((BK, BN)).tile((-1, 1)).expand((out_arr.shape[0], -1))
    b_arr.dtype = b_arr.dtype.squeeze(1)
    bias_arr = bias[None, :].expand((out.shape[0], -1)).tile((BM, BN))
    return a_arr, b_arr, bias_arr, out_arr

def application(a, b, bias, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)
    for k in range(a.shape[0]):
        acc += ntl.dot(a[k], b[k])
    value = acc + bias
    out = (ntl.maximum(value, 0.0)).to(ntl.float16)

kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(1), Tensor(2)), kernel_name="ntops_lab_fused_gemm_bias_relu")

def run(*inputs):
    a, b, bias = inputs
    if a.device.type == "npu":
        from ntops_lab.kernels.linear._ascend_matmul import matmul

        return matmul(a, b, vector_bias=bias, activation="relu")
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    kernel(a, b, bias, out)
    return out
