# ntops.lab 昇腾算子适配与验证记录

记录日期：2026-10-14。本文依据当前代码差异、保存的 JUnit/JSON 报告及执行日志整理。中断前的环境操作以已有环境文档为依据，不将版本兼容建议写成每一步都已重新执行的安装记录。

## 1. 目标与实际结果

目标是让目录中的 1124 个可运行算子在 Ascend NPU 上实际执行，并与各自 PyTorch 参考结果比较，补齐环境和修复暴露的问题。

| 验证范围 | 结果 | 证据 |
| --- | --- | --- |
| 全目录算子，五个 pytest 进程分片 | 1124 passed，0 failed，0 skipped | [逐项 JSON](validation/ascend-operator-results.json)、`validation/npu-1.xml` 至 `npu-5.xml` |
| 已有编译器和矩阵乘边界回归 | 33 passed | [回归 JUnit](validation/ascend-regressions.xml) |
| 未启用硬件验证的仓库检查，不含算子验证文件 | 15 passed，34 skipped | 当次 `/tmp/ntops-final-repo.log` |
| 一次性执行整个仓库的完整 pytest | **1173 passed，0 failed，1 warning**，5236.77 秒 | `/tmp/ntops-all-pytest.log` |

因此可以确认“1124 个算子用例全部通过”“33 项回归全部通过”，并且一次性执行整个仓库的 pytest 也已完成且全部通过。分片运行使用的也是 pytest，覆盖了全部目录算子。

完整 pytest 的唯一 warning 来自 `torch_npu` 的 `cosine_embedding_loss`：
`Cannot create tensor with interal format while allow_internel_format=False`。这是 torch_npu 内部格式提示，不是测试失败。

结果覆盖现有 spec 的输入形状、dtype 和比较容差；不是所有任意输入都已验证，也不是性能达标报告。比较没有通过整体放宽容差或将失败项跳过来达成。

## 2. 验证环境与依赖

| 项目 | 验证环境 |
| --- | --- |
| NPU | Ascend 910B4，机器上有 8 张设备 |
| CANN | 9.0.0 |
| Python | 3.10，解释器 `/home/xcy/venv/bin/python` |
| PyTorch | 2.10.0+cpu，配合 torch_npu 使用 NPU |
| torch_npu | 2.10.0.post6 |
| Triton Ascend | triton-ascend 3.2.0 |
| pybind11 | 3.1.0 |
| NineToothed | 本地 `/home/xcy/ninetoothed`，包含 Ascend 后端 |
| 默认平台配置 | `ascend-910b4` |

标准 Triton 和 Triton Ascend 都提供 `triton` Python 包，混装可能导致后端识别和运行时不一致。环境记录中的安装方式如下，重新部署时应在专用虚拟环境执行：

```bash
python -m pip uninstall -y triton
python -m pip install triton-ascend==3.2.0 pybind11==3.1.0
python -c 'from triton.backends import backends; print(backends.keys())'
```

后端需要包含 `ascend`。本次环境使用过的本地 wheel 路径记录在 [英文环境说明](ascend-validation.md)。该 wheel 的 `npu_utils.cpp` 已使用 CANN 9.0 对应的 `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`；若其他 wheel 仍使用旧枚举 `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`，会遇到头文件兼容编译错误。

需要能够访问 NPU 设备文件及可写的编译缓存。沙箱中出现 `Can't get ascend_hal device count` 时，应修正执行权限，而不是伪造 CUDA 可用性。`torch` 版本带 `+cpu` 不代表安装了 `torch_npu` 后不能使用 NPU。

## 3. 测试入口如何适配昇腾

主要修改：[tests/conftest.py](../tests/conftest.py) 和 [算子验证入口](../tests/test_operator_gpu_validation.py)。

1. 导入 `torch_npu`，要求真实 NPU 可用。
2. 激活 NineToothed 的 Ascend Triton 后端，设置 `TRITON_DEFAULT_BACKEND=ascend`。
3. 在测试 fixture 内包装 `ninetoothed.make`，指定 `backend="ascend"`，默认使用 `platform="ascend-910b4"`；可通过 `NTOPS_NPU_PLATFORM` 覆盖。
4. 包装 spec 的 `make_inputs`，将 device 设为 `npu`。创建类算子的 `run`、`run_pytorch` 若带 device 参数，同样显式指定 NPU，修复 `eye`、`eye_m` 等仍尝试创建 CUDA 张量的问题。
5. 仅在 spec 模块内用代理把原先 `torch.cuda.synchronize()` 路由到 NPU 同步；没有全局伪装 CUDA。
6. 别名执行测试与现有回归测试通过 fixture 选择可用设备，让原本限定 CUDA 的执行路径能够在 NPU 上验证。

大多数算子仍由 NineToothed 生成 Ascend Triton 内核。部分编译器不支持的布局使用手写 Triton Ascend 内核；这些仍在 NPU 上实际计算。

## 4. 错误类型与修复方法

### 4.1 矩阵乘及融合矩阵乘的编译器崩溃

现象：`mm`、`gemm_bias`、融合 GEMM 等在 `ConvertTritonIRToLinalgIR` 阶段失败，`triton-adapter-opt` 发生 SIGSEGV。原实现扩展 tile、broadcast 和 mask 的布局在当前 Ascend 编译器上存在兼容问题。

修复：新增 [linear/_ascend_matmul.py](../src/ntops_lab/kernels/linear/_ascend_matmul.py)。使用明确的二维 tile、边界 mask 和 K 维循环，通过 `tl.dot` 累加到 float32；支持矩阵 bias、向量 bias 和激活。相关 `mm*`、`addmm*`、`gemm_bias*`、`bmm*`、`addbmm`、`baddbmm`、`grouped_mm` 及融合系列在 NPU 输入时选择该路径。

`linear` 在隔离缓存后的验证中也暴露同类崩溃，随后接入同一个内核，使用转置权重和向量 bias。多头注意力依赖 `linear` 做投影，因此该修复也解决了其投影编译失败。两者专项复测均通过。

这属于内核兼容绕行，未修改 Triton 编译器使其支持所有原始扩展布局。

### 4.2 半精度舍入导致 hardshrink 边界错误

参考实现会先把矩阵乘结果物化为输入 dtype，再加 bias、再执行激活。融合内核若一直保留 float32，靠近 `±0.5` 的数值可能在阈值另一侧，与参考不一致。

修复：存在激活时，矩阵乘结果以及相应 bias 相加结果按输出 dtype 舍入，再转为 float32执行激活。这保留参考计算的中间精度语义。

验证：[test_ascend_matmul_regressions.py](../tests/test_ascend_matmul_regressions.py) 包含有 bias/无 bias 的半精度 hardshrink 边界，要求 `atol=0, rtol=0`。

### 4.3 GELU 数值路径与 NPU 参考不一致

现象：GELU 门控、残差、归一化融合等用例数值比较失败。原内核使用 erf 路径，与本次 torch_npu 参考的数值行为不一致。

修复：NPU 专用 application 使用 tanh 近似：

```text
0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
```

相关 `gate_gelu*`、`residual*_gelu`、`layer_norm_gelu`、`rms_norm_gelu` 以及 add-normalization 融合路径按 NPU 设备选择适配实现。矩阵乘激活辅助函数也使用对应公式。CUDA 原路径保留。

### 4.4 缺少 fmod 与 copysign 数学原语

现象：部分 `fmod`、`remainder`、`floor_divide` 系列在 Ascend libdevice 路径上编译失败。

修复：新增 [ascend_math.py](../src/ntops_lab/ascend_math.py)，在 fixture 中为 Ascend libdevice 注册这两个 Triton JIT 实现。

- `fmod` 采用向零截断商，通过 FMA 计算余数，并修正浮点边界下的符号与小于除数的情况。
- `copysign` 通过 float32 位表示组合幅值与符号位，保留零的符号。

该实现当前按代码中的 float32 算法工作；通过目录中的现有输入不等于已经验证所有极端值和任意 dtype。

### 4.5 布尔标量 XOR 错误

现象：`logical_xor_scalar` 和 out 版本比较失败。

修复：将 `(x != 0.0) != True` 化简为 `x == 0`。对于固定标量 True，这直接表达逻辑取反，并避开当前编译路径对原表达式的异常处理。

### 4.6 全掩码 softmax 行

现象：`_safe_softmax` 在整行都是 `-inf` 时比较失败，直接做 `x-max(x)` 会产生 `-inf-(-inf)`。

修复：[safe_softmax.py](../src/ntops_lab/kernels/reduction/safe_softmax.py) 先转 float32，在行最大值为 `-inf` 时使用 0 作为减数，让指数为零；分母为零时替换为 1，输出零行。已有全掩码回归用例验证该语义。

### 4.7 注意力局部内存溢出

现象：因果 flash attention 和相关 SDPA 用例编译时报告 `ub overflow`，一个记录要求 1991680 bits，超过 1572864 bits 的限制。

修复：[fused/general/_flash_attention.py](../src/ntops_lab/kernels/fused/general/_flash_attention.py) 为 NPU 使用 `BLOCK_SIZE_M=32, BLOCK_SIZE_N=32`，减少 tile 中间结果占用，并按设备缓存不同内核。原注意力算法仍实际在 NPU 上执行。

### 4.8 昇腾池化索引与标准 PyTorch 索引格式不同

现象：`max_unpool1d` 输入 indices 变成 rank 4，而内核要求 rank 3；`max_unpool2d` 对 indices.dtype 创建 arange 时遇到 `DT_INT8` 不支持。根因是当前 torch_npu max-pool 返回打包掩码，而 spec 假定标准整数位置索引。

修复：两个 unpool spec 在 NPU 场景下用 CPU pooling 生成标准 values/indices，再复制到 NPU。PyTorch unpool 参考同样在 CPU 上计算后复制结果比较；自定义 unpool 内核仍在 NPU 上执行。1D 内核的位置张量使用 int64。

这里明确存在 CPU 输入准备和 CPU 参考计算，并非全部步骤都在 NPU 上；没有将实际被测 unpool 内核替换成 CPU/PyTorch 结果。

### 4.9 编译配置与旧缓存不一致，基础算子大面积错误

现象：全量顺序执行的开头出现 `abs`、`acos`、pooling、`add` 等 16 项大误差，但专项运行可以全部通过。这一规律提示运行/缓存问题而非 16 个算子同时出现独立数值 bug。

分析：`TRITON_ALL_BLOCKS_PARALLEL=1` 用于允许超过默认 65535 blocks 的网格，也会改变编译输出（启用 auto-blockify-loop）。旧缓存未充分区分该环境配置，可能复用不匹配的二进制；launcher 缓存还需要隔离 PyTorch ABI/版本。

修复：fixture 将缓存按 PyTorch、Triton Ascend 版本及并行配置分目录：

```text
~/.cache/ntops-lab/ascend/
  torch-<torch版本>-triton-<triton-ascend版本>/
    all-blocks-parallel-1/
```

隔离后问题区段通过，最终分片全量通过。首次新缓存编译较慢，后续同一生成源和配置可以复用。该处理隔离本项目测试缓存，没有直接修复上游缓存键。

### 4.10 embedding 与双线性上采样布局崩溃

隔离缓存后，`embedding` 在顺序测试中暴露编译器 SIGSEGV；`upsample_bilinear` 在分片测试中也失败。原始实现将查表/插值转换成大规模展开和归约。

修复：新增 [layout/_ascend_lookup.py](../src/ntops_lab/kernels/layout/_ascend_lookup.py)。

- embedding：直接读取 token 索引，再按权重行列 stride 查表，保留输出 dtype 和输入索引形状。
- bilinear：按 align_corners=True 计算源坐标，读取四个邻点并进行双线性插值；NPU 路径使用连续输入。

两个入口仅在 NPU 时选择这些 Triton 内核。专项与最终全量均通过。

## 5. 修复与验证过程

已有记录中较早一轮全量是 1028 passed、96 failed。随后矩阵乘、融合算子和数学函数进行了分类修复及专项验证。

恢复任务后的一轮全量为 1106 passed、18 failed：16 项基础算子缓存相关问题，另两项为 unpool 的输入适配问题。隔离缓存及 unpool 修复后，五卡分片结果为 1121 passed、3 failed，失败项为 linear、多头注意力和 bilinear；额外顺序执行发现 embedding 编译崩溃。

上述问题全部修复并专项通过后，重新启动同一版本代码的五卡全量复测，最终：

```json
{"passed": 1124, "failed": 0, "skipped": 0, "total": 1124}
```

五个分片分别覆盖 225、225、225、225、224 项。现有编译器和矩阵乘回归另行得到 33 passed。

随后执行完整仓库 pytest，命令为：

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest \
  --tb=short --junitxml=/tmp/ntops-all-pytest.xml \
  > /tmp/ntops-all-pytest.log 2>&1
```

最终结果为 **1173 passed，0 failed，1 warning**，耗时 `5236.77s`（约 1 小时 27 分钟）。

## 6. 复现命令

在已配好上述环境、具备 NPU 设备访问权限的情况下，从仓库根目录运行。

### 全目录算子，五卡分片（本次全通过采用的方法）

```bash
NTOPS_VALIDATION_DEVICES=0,1,5,6,7 python scripts/validate_ascend.py
```

脚本为每张卡创建独立 pytest 进程，按排序后的目录轮转分配算子，设置 `ASCEND_RT_VISIBLE_DEVICES`。卡号可调整，也可使用单卡分片模式：

```bash
NTOPS_VALIDATION_DEVICES=0 python scripts/validate_ascend.py
```

默认输出目录为 `/tmp/ntops-final-parallel-validation`；建议每次指定新的目录，避免中断后遗留报告造成混淆：

```bash
NTOPS_VALIDATION_DEVICES=0,1,5,6,7 \
NTOPS_VALIDATION_OUTPUT=/tmp/ntops-validation-new-run \
python scripts/validate_ascend.py
```

脚本输出各分片日志、JUnit XML 和汇总 results.json。日志中的 `npu-1` 等是分片序号，不一定是物理卡号。正常完整执行时，脚本在失败、跳过或结果总数不符时返回非零。

### 回归测试

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest \
  tests/test_current_compiler_regressions.py \
  tests/test_ascend_matmul_regressions.py --tb=short
```

### 全仓库统一 pytest（已完成并通过）

```bash
NTOPS_RUN_OPERATOR_VALIDATION=1 python -m pytest --tb=short \
  --junitxml=/tmp/ntops-all-pytest.xml \
  > /tmp/ntops-all-pytest.log 2>&1
```

另一个终端查看进度：

```bash
tail -f /tmp/ntops-all-pytest.log
```

仅执行 `python -m pytest` 时，默认不会启动全目录硬件验证，会产生预期 skip。开启验证环境变量后才执行算子比较与相关 NPU 回归。本次启用验证后完整结果为 **1173 passed，0 failed，1 warning**。

## 7. 主要文件与交付物

| 文件/范围 | 用途 |
| --- | --- |
| `tests/conftest.py` | NPU fixture、Ascend 编译后端、数学原语注册和缓存隔离 |
| `tests/test_operator_gpu_validation.py` | 1124 项目录参数化验证，spec device 和同步适配 |
| `src/ntops_lab/ascend_math.py` | fmod/copysign 的 Ascend 实现 |
| `src/ntops_lab/kernels/linear/_ascend_matmul.py` | 矩阵乘、bias、激活的直接 Ascend 内核 |
| `kernels/linear/` 与融合矩阵乘系列 | NPU 调用路径接入 |
| `kernels/fused/general/` 中 GELU 系列 | NPU 数值路径适配 |
| `kernels/fused/general/_flash_attention.py` | 注意力 tile 尺寸适配 |
| `kernels/reduction/safe_softmax.py` | 全掩码行处理 |
| `kernels/pointwise/logical_xor_scalar*.py` | 标量 XOR 表达式修复 |
| `kernels/layout/_ascend_lookup.py` 及对应入口 | embedding 与 bilinear 直接 NPU 内核 |
| `testing/specs/layout/max_unpool1d.py`、`max_unpool2d.py` | 标准索引生成与参考计算适配 |
| `tests/test_alias_exports.py`、现有回归测试 | 设备适配及回归验证 |
| `scripts/validate_ascend.py` | 多卡 pytest 分片与结果汇总 |
| `docs/validation/` | 1124 项逐项结果、五份 JUnit 和 33 项回归报告 |

环境还依赖仓库外的本地 NineToothed 及 Triton Ascend 安装。仅复制本仓库代码，不会自动复制这些依赖、wheel 或 CANN，因此换机器需要按环境说明重新部署。当前成果保证现有用例的正确性；性能优化和更多形状/dtype 覆盖仍需单独评估。
