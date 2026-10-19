import functools
import os
from importlib.metadata import version
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def npu_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.fail("Ascend NPU validation requires access to an available NPU")
    return torch


@pytest.fixture
def ascend_backend(npu_torch, monkeypatch):
    # Triton Ascend's launcher cache does not include the PyTorch ABI/version.
    # Separate it from artifacts built with other installed torch versions.
    # ALL_BLOCKS_PARALLEL changes compiler output as well as launch behavior;
    # artifacts compiled before enabling it must not be reused.
    cache = (
        Path.home() / ".cache" / "ntops-lab" / "ascend"
        / f"torch-{npu_torch.__version__}-triton-{version('triton-ascend')}"
        / "all-blocks-parallel-1"
    )
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache))
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "ascend")
    # Row-wise specs include 65536 rows, exceeding Ascend's default grid limit.
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")

    import ninetoothed
    from ninetoothed.backends.ascend import activate_triton_backend

    activate_triton_backend()
    from triton.language.extra.ascend import libdevice
    from ntops_lab.ascend_math import copysign, fmod

    monkeypatch.setattr(libdevice, "fmod", fmod)
    monkeypatch.setattr(libdevice, "copysign", copysign)
    make = ninetoothed.make

    @functools.wraps(make)
    def make_ascend(*args, **kwargs):
        kwargs["backend"] = "ascend"
        kwargs.setdefault(
            "platform", os.environ.get("NTOPS_NPU_PLATFORM", "ascend-910b4")
        )
        return make(*args, **kwargs)

    monkeypatch.setattr(ninetoothed, "make", make_ascend)


@pytest.fixture
def operator_device(request):
    torch = pytest.importorskip("torch")
    if torch.version.cuda is not None:
        if not torch.cuda.is_available():
            pytest.fail("CUDA validation requires an available CUDA device")
        return torch.device("cuda")
    request.getfixturevalue("ascend_backend")
    return torch.device("npu")
