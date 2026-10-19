import os

import pytest

from ntops_lab import ops

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    os.environ.get("NTOPS_RUN_OPERATOR_VALIDATION") != "1",
    reason="set NTOPS_RUN_OPERATOR_VALIDATION=1 to run GPU regressions",
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(20260917)


@pytest.mark.parametrize("width", [1, 7, 33, 257])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_row_domain(width, dtype, operator_device):
    x = torch.randn((3, width), device=operator_device, dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 0.01
    torch.testing.assert_close(ops.softmax(x), x.softmax(-1), atol=tol, rtol=tol)
    torch.testing.assert_close(
        ops.log_softmax(x), x.log_softmax(-1), atol=tol, rtol=tol
    )


def test_safe_softmax_fully_masked_rows(operator_device):
    x = torch.randn((3, 33), device=operator_device)
    x[0] = float("-inf")
    expected = x.softmax(-1)
    expected[0] = 0
    torch.testing.assert_close(ops.get_op("_safe_softmax")(x), expected)


@pytest.mark.parametrize("width", [7, 33, 257])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_normalization_full_hidden_axis(width, dtype, operator_device):
    x = torch.randn((3, width), device=operator_device, dtype=dtype)
    w = torch.randn(width, device=operator_device, dtype=dtype)
    b = torch.randn_like(w)
    tol = 2e-5 if dtype == torch.float32 else 0.02
    expected = torch.nn.functional.layer_norm(x, (width,), w, b, 1e-5)
    torch.testing.assert_close(ops.layer_norm(x, w, b), expected, atol=tol, rtol=tol)
    expected_rms = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        * w.float()
    ).to(dtype)
    torch.testing.assert_close(ops.rms_norm(x, w), expected_rms, atol=tol, rtol=tol)


def test_layer_norm_large_offset(operator_device):
    x = torch.randn((3, 33), device=operator_device) + 10000
    w = torch.ones(33, device=operator_device)
    b = torch.zeros_like(w)
    expected = torch.nn.functional.layer_norm(x, (33,), w, b, 1e-5)
    torch.testing.assert_close(ops.layer_norm(x, w, b), expected, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("width", [7, 33, 257])
def test_full_axis_and_multiple_reductions(width, operator_device):
    x = torch.randn((3, width), device=operator_device)
    torch.testing.assert_close(ops.sum(x), x.sum(-1))
    actual_min, actual_max = ops.aminmax(x)
    torch.testing.assert_close(actual_min, x.amin(-1))
    torch.testing.assert_close(actual_max, x.amax(-1))
    actual_var, actual_mean = ops.var_mean(x)
    torch.testing.assert_close(actual_var, x.var(-1, correction=0))
    torch.testing.assert_close(actual_mean, x.mean(-1))
    y = torch.randn(width, device=operator_device)
    torch.testing.assert_close(ops.dot(x[0], y), torch.dot(x[0], y).reshape(1))


@pytest.mark.parametrize("offset", [-100, 0, 100])
def test_cross_entropy_stability(offset, operator_device):
    x = torch.randn((3, 33), device=operator_device) + offset
    target = torch.tensor([0, 7, 32], device=operator_device)
    expected = torch.nn.functional.cross_entropy(x, target, reduction="none")
    torch.testing.assert_close(
        ops.cross_entropy(x, target), expected, atol=2e-5, rtol=2e-5
    )


def test_round_out_ties_to_even(operator_device):
    x = torch.tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, float("inf"), float("-inf"), float("nan")],
        device=operator_device,
    )
    torch.testing.assert_close(ops.round_out(x), x.round(), equal_nan=True)


def test_floor_divide_integer_boundaries(operator_device):
    x = torch.tensor([1.0, -1.0, 0.6, -0.6, 0.0, -0.0, 1.0], device=operator_device)
    y = torch.tensor([0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 0.0], device=operator_device)
    actual = ops.floor_divide_out(x, y)
    expected = torch.floor_divide(x, y)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
