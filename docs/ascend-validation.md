# Ascend NPU validation

The operator validation entry point uses `backend="ascend"` and NPU inputs.
It keeps each spec's reference implementation and comparison tolerances, and
routes legacy spec synchronization calls to NPU without globally mocking CUDA.

Validated environment (2026-10-13):

- Ascend 910B4, 8 devices
- CANN 9.0.0, Python 3.10
- torch 2.10.0+cpu, torch_npu 2.10.0.post6
- triton-ascend 3.2.0, pybind11 3.1.0
- Local NineToothed checkout at `/home/xcy/ninetoothed`, with its Ascend backend

Standard Triton and Triton Ascend both install the `triton` Python package.
Remove standard Triton before installing the Ascend distribution:

```bash
python -m pip uninstall -y triton
python -m pip install triton-ascend==3.2.0 pybind11==3.1.0
python -c 'from triton.backends import backends; print(backends.keys())'
```

The backend list must include `ascend`. The local wheel used for verification is
`/tmp/ninetoothed-wheels/triton_ascend-3.2.0-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`.
It already uses `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE` in `npu_utils.cpp`.
If another 3.2.0 wheel fails to compile against CANN 9.0 because
`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` is undefined, back up that file and replace
the old enum with the CANN 9.0 name above.

Run with access to the NPU device files and a writable compiler cache:

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_operator_gpu_validation.py -k 'matches_pytorch[add]' -v -s
```

The default platform is `ascend-910b4`; set `NTOPS_NPU_PLATFORM` to another
NineToothed-supported profile when appropriate.

Compatibility fixes in this checkout include compact Ascend matrix-product
kernels, materialized half-precision rounding before activation thresholds,
GELU reference alignment, missing Ascend math primitives, boolean scalar XOR,
fully masked softmax rows, and smaller attention tiles to fit local memory.
The matrix-product path runs Triton kernels on NPU.

The fixture sets `TRITON_ALL_BLOCKS_PARALLEL=1` for grids exceeding 65535
blocks. This environment variable also changes compiler output. Keep its
compiled artifacts separate from kernels compiled without it: reusing the old
cache produced incorrect outputs in basic operators such as `abs` and `add`.
The fixture uses an `all-blocks-parallel-1` cache subdirectory, additionally
partitioned by PyTorch and Triton Ascend versions.

For the 1D and 2D unpool specs, input pooling and the PyTorch reference run on
CPU because Ascend pooling returns packed masks instead of standard integer
indices. Inputs are copied to NPU before the custom unpool kernel runs; the
reference result is copied back for comparison with the original tolerances.

If a sandbox reports `Can't get ascend_hal device count` while the host can use
NPU, run validation with device access. Faking CUDA availability does not resolve
this access restriction or supply an Ascend compiler.

## Full validation results (2026-10-14)

All **1124 operator checks passed**, with **0 failures and 0 skips**, on five
Ascend NPU devices after the compatibility and cache fixes. Existing compiler
and matrix-product regressions also passed: **33 checks**. These results cover
the catalog specs and regression inputs, not every possible shape or dtype.

The per-operator results are in [validation/ascend-operator-results.json](validation/ascend-operator-results.json);
JUnit reports and the regression report are in `docs/validation/`.

Reproduce the full operator validation from the repository root:

```bash
NTOPS_VALIDATION_DEVICES=0,1,5,6,7 python scripts/validate_ascend.py
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_current_compiler_regressions.py tests/test_ascend_matmul_regressions.py
```

Set `NTOPS_VALIDATION_OUTPUT` to choose the output directory. The runner exits
nonzero if any operator fails, skips, or is missing from the results.
`linear` uses the compact Ascend matrix-product path; embedding and bilinear
upsampling use direct Triton lookup/interpolation kernels on NPU to avoid
compiler crashes in the expanded reduction layouts.

The complete repository test run also finished successfully:

```text
1173 passed, 1 warning in 5236.77s (1:27:16)
```

The only warning came from torch_npu's internal-format handling in
`cosine_embedding_loss`; it did not affect the result.
