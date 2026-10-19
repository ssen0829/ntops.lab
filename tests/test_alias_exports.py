import os

import pytest

from ntops_lab import list_operators
from ntops_lab.aliases import ALIAS_TO_OP
from ntops_lab.registry import OPERATOR_MODULES


def test_aliases_resolve_without_manifest_duplicates():
    manifest_ops = {item.op for item in list_operators()}
    assert set(ALIAS_TO_OP).isdisjoint(manifest_ops)
    assert set(ALIAS_TO_OP).isdisjoint(OPERATOR_MODULES)
    assert all(target in manifest_ops for target in ALIAS_TO_OP.values())


def test_alias_get_op_and_call_match_canonical(request):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif os.environ.get("NTOPS_RUN_OPERATOR_VALIDATION") == "1":
        device = request.getfixturevalue("operator_device")
    else:
        pytest.skip("set NTOPS_RUN_OPERATOR_VALIDATION=1 for NPU alias execution")

    from ntops_lab import call_torch_op, get_torch_op
    import ntops_lab.ops as ops

    x = torch.tensor([-1.0, 0.0, 1.0], device=device)
    y = torch.tensor([2.0, 3.0, 4.0], device=device)
    assert torch.equal(call_torch_op("multiply", x, y), call_torch_op("mul", x, y))
    assert get_torch_op("multiply") is ops.multiply
    assert ops.canonical_name("multiply") == "mul"
    assert "multiply" in dir(ops)
