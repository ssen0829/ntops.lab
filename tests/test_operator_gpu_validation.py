import functools
import importlib
import inspect
import os

import pytest

from ntops_lab import runnable_operators


RUN_OPERATOR_VALIDATION = os.environ.get("NTOPS_RUN_OPERATOR_VALIDATION") == "1"


class _NpuSpecTorch:
    """Keep legacy spec synchronization on NPU without changing global torch."""

    def __init__(self, torch):
        self._torch = torch

    @property
    def cuda(self):
        # Existing spec.check functions use this namespace for synchronization.
        return self._torch.npu

    def __getattr__(self, name):
        return getattr(self._torch, name)


def spec_module_name(record) -> str:
    spec_path = record.file.replace("kernels/", "specs/")[:-3].replace("/", ".")
    return "ntops_lab.testing." + spec_path


@pytest.mark.parametrize("record", runnable_operators(), ids=lambda item: item.op)
def test_operator_matches_pytorch(record, request, monkeypatch):
    if not RUN_OPERATOR_VALIDATION:
        pytest.skip("set NTOPS_RUN_OPERATOR_VALIDATION=1 to run NPU operator validation")
    torch = request.getfixturevalue("npu_torch")
    request.getfixturevalue("ascend_backend")
    module = importlib.import_module(spec_module_name(record))
    make_inputs = module.make_inputs
    signature = inspect.signature(make_inputs)
    if "device" not in signature.parameters:
        pytest.fail(f"{module.__name__}.make_inputs must accept an NPU device")

    @functools.wraps(make_inputs)
    def make_npu_inputs(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs)
        arguments.arguments["device"] = torch.device("npu")
        return make_inputs(*arguments.args, **arguments.kwargs)

    monkeypatch.setattr(module, "make_inputs", make_npu_inputs)
    # Creation specs such as eye invoke run directly with no input tensors.
    for name in ("run", "run_pytorch"):
        function = getattr(module, name)
        if "device" in inspect.signature(function).parameters:
            @functools.wraps(function)
            def run_npu(*args, _function=function, **kwargs):
                kwargs["device"] = torch.device("npu")
                return _function(*args, **kwargs)

            monkeypatch.setattr(module, name, run_npu)
    monkeypatch.setattr(module, "torch", _NpuSpecTorch(torch))
    module.check()
    torch.npu.synchronize()
