"""Triton math primitives missing from the Ascend 3.2 libdevice module."""

import triton
import triton.language as tl


@triton.jit
def fmod(x, y):
    ratio = x / y
    quotient = tl.where(ratio < 0, tl.ceil(ratio), tl.floor(ratio))
    # FMA keeps the rounding residual at boundaries such as 1.0 / 0.1.
    remainder = tl.fma(-quotient, y, x)
    wrong_sign = (remainder != 0) & ((remainder < 0) != (x < 0))
    correction = tl.where(x < 0, -tl.abs(y), tl.abs(y))
    remainder = tl.where(wrong_sign, remainder + correction, remainder)
    remainder = tl.where(tl.abs(x) < tl.abs(y), x, remainder)
    return tl.where(remainder == 0, copysign(0.0, x), remainder)


@triton.jit
def copysign(x, y):
    x_value = (tl.full((), 0, tl.float32) + x).to(tl.float32)
    y_value = (tl.full((), 0, tl.float32) + y).to(tl.float32)
    magnitude = x_value.to(tl.int32, bitcast=True) & 0x7fffffff
    sign = y_value.to(tl.int32, bitcast=True) & -2147483648
    return (magnitude | sign).to(tl.float32, bitcast=True)
