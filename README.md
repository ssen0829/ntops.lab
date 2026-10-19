# ntops.lab

`ntops.lab` is an experimental operator lab for
[NineToothed](https://github.com/InfiniTensor/ninetoothed). It collects
generated and hand-shaped NineToothed kernels, keeps a runnable operator
catalog, and records the current gaps that need compiler/runtime support.

The project is intentionally a lab: many kernels are produced with LLM-assisted
operator development, then smoke-tested against PyTorch on CUDA.

## Current Coverage

The canonical catalog contains:

- 1124 runnable operator implementations
- 0 unsupported/scaffold source files included in the commit-ready tree
- gap analysis retained in documentation only

Primary categories:

- `pointwise`
- `reduction`
- `linear`
- `layout`
- `creation`
- `normalization`
- `fused/general`
- `fused/fla`

See [docs/operator-coverage.md](docs/operator-coverage.md) for the full list.

## Repository Layout

```text
src/ntops_lab/
  catalog.py                 # manifest loading and query helpers
  cli.py                     # ntops-lab CLI
  operator_manifest.json     # canonical runnable operator catalog
  kernels/                   # NineToothed kernel implementations
  ops/                       # PyTorch-facing callable wrappers
  testing/specs/             # input generation and PyTorch references
docs/
  operator-coverage.md       # runnable coverage
  runnable-operators.md      # runnable operator list
  unrunnable-operators.md    # excluded unsupported operator notes
  support-analysis.md        # missing capability analysis
tests/
  test_catalog.py            # CPU-only metadata tests
```

## Install

For metadata inspection and CPU-only tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For running kernels, use an environment that already has CUDA, PyTorch, and
NineToothed installed:

```bash
pip install -e ".[dev]"
```

NineToothed and PyTorch are kept as optional GPU dependencies because this
repository is often inspected on machines without a GPU.

## Use The Catalog

```bash
ntops-lab summary
ntops-lab list --status done
ntops-lab list --category reduction
```

From Python:

```python
from ntops_lab import list_operators, runnable_operators

print(len(runnable_operators()))
print(list_operators(category="linear"))
```

## Use Operators From PyTorch

Operator wrappers accept and return `torch.Tensor` objects while dispatching to
the underlying NineToothed kernels:

```python
import torch
from ntops_lab import ops

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
z = ops.add(x, y)

mm = ops.get_op("mm")
out = mm(torch.randn(32, 32, device="cuda", dtype=torch.float16),
         torch.randn(32, 32, device="cuda", dtype=torch.float16))
```

## Run Operator Tests

For Ascend NPU setup, execution commands, and verified results, see
[Ascend validation](docs/ascend-validation.md).
中文适配过程、错误修复与验证结果见
[昇腾算子适配报告](docs/ascend-porting-report.zh-CN.md)。

On a CUDA machine with NineToothed available:

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_operator_gpu_validation.py -k add
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_operator_gpu_validation.py -k softmax
```

Kernel modules expose `run`; test specs expose:

```python
make_inputs()
run(...)
run_pytorch(...)
check()
```

`check()` generates representative inputs, runs the NineToothed kernel, runs
the PyTorch reference, and compares results.

## Run Repository Tests

Default repository checks:

```bash
python -m pytest
```

Full GPU operator validation is available through pytest and is disabled by
default because it compiles and runs every registered kernel:

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest tests/test_operator_gpu_validation.py
```

Running every GPU check can take a while because many operators compile kernels
on first use.

## Compilation Cache

NineToothed and Triton already use content-addressed disk caches. Generated
NineToothed source is stored under `~/.ninetoothed`, while compiled Triton
artifacts are stored under `${TRITON_CACHE_DIR:-~/.triton/cache}`. Re-running an
operator in a new Python process reuses these artifacts when the generated
kernel source and compilation environment are unchanged.

Inspect the current caches:

```bash
ntops-lab cache status
```

Changing a kernel implementation or its NineToothed compilation configuration
changes the generated-source hash and causes the affected kernel to compile
again. Unchanged kernels continue to reuse their cached artifacts.

## What Is Not Included Yet?

Unsupported scaffold source files are intentionally excluded from this commit-ready tree. Remaining gaps mostly need one of these capabilities:

- complex manual layout/indexing
- dynamic gather/scatter, cache writes, or masked updates
- top-k/sort/histogram/atomic primitives
- cross-tile scan, recurrent state, or triangular solve
- RNG/stateful random distribution support
- normalization backward or running-stat APIs

See [docs/support-analysis.md](docs/support-analysis.md).

## Project Philosophy

This is a practical operator workshop:

- keep each operator in its own file
- keep PyTorch references close to implementations
- make every runnable operator independently smoke-testable
- record why excluded operators are blocked
- prefer clear templates over pretending incomplete kernels are done

## License

Standard MIT License. The MIT license is not versioned, so the repository uses
the canonical MIT text and SPDX-compatible `MIT` package metadata.
