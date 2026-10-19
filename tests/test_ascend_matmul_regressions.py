import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("NTOPS_RUN_OPERATOR_VALIDATION") != "1",
    reason="set NTOPS_RUN_OPERATOR_VALIDATION=1 to run NPU regressions",
)


@pytest.mark.parametrize("with_bias", [False, True])
def test_hardshrink_half_rounding(with_bias, npu_torch, ascend_backend):
    torch = npu_torch
    from ntops_lab.kernels.linear._ascend_matmul import matmul

    a = torch.ones((1, 4), device="npu", dtype=torch.float16)
    b = torch.tensor(
        [[0.1251220703125, -0.1251220703125, 0.12548828125],
         [0.125, -0.125, 0.125],
         [0.125, -0.125, 0.125],
         [0.125, -0.125, 0.125]],
        device="npu", dtype=torch.float16,
    )
    bias = torch.full((1, 3), 0.0001220703125, device="npu", dtype=torch.float16) if with_bias else None
    expected = a @ b
    if bias is not None:
        expected = bias + expected
    expected = torch.nn.functional.hardshrink(expected)
    actual = matmul(a, b, bias=bias, activation="hardshrink")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert expected[0, 0].item() == 0
    assert expected[0, 1].item() == 0
    assert expected[0, 2].item() > 0.5
