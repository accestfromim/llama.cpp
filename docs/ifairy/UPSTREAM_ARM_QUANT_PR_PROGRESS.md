# ARM / Fairy2i upstream port progress

> 开发进度记录。本文只记录已在当前分支完成并验证的 A0-A7、A9、B0-B1 与 B3 适配；不把通用上游 ARM quant 代码直接套到 Fairy2i/iFairy 自定义布局。

## 当前状态

- 分支：`lwt/merge_master`
- 最新迭代：A9 row-contiguous quantized concat
- 提交粒度：每个目标保持独立。
- 状态：A0-A7、A9、B0-B1 与 B3 实现完成；A9 已通过 assertion 复现、五格式/四维/view-mask 回归、代码审查和完整 CPU 矩阵。

| 批次 | 内容 | 提交 |
|---|---|---|
| A0 | 修正 `ggml_vec_mad1_f32()` 的 SIMD FMA 参数顺序；加入长向量 `SCALE` 回归用例 | `64cd4f3bc`, `8c351ee68` |
| A1 | 适配 ARM `-mcpu`/`-march` 探测、宏验证和所需 feature flag 传递；保留 Fairy2i source split 与 DOTPROD gate | `6028b4b64` |
| A2 | 增加 Linux/aarch64 缺失 HWCAP fallback；FP16 runtime 能力要求 `HWCAP_FPHP` 与 `HWCAP_ASIMDHP` 同时存在 | `1c8e8bf8d` |
| A3 | 适配 #15928、#19861、#20460，初始化 IQ scratch/fallback 状态并保护退化缩放 | `7ea3c22ed` |
| A4 | 适配 #24305、#25247，修正 `RMS_NORM_BACK` 原地别名和量化 `CONCAT` block 索引 | `03f3ed5a2` |
| A5 | 适配 #26838，使 Android 进入已有 Linux `sched_setaffinity()` 分支 | `c719d8b4a` |
| A6 | 适配 #16520，隔离 Linux-only SVE header 并使 non-Linux SVE 明确失败 | `c71df67f9` |
| A7 | 适配 #17788、#18451，移除无效架构断言并阻止 unsplit/split 输出覆盖输入 | `90e2586b6` |
| A9 | 适配 #25678，使量化 `CONCAT` 接受 row-contiguous 高维 views | `98327d467` |
| B0 | 移植 #17748 的 packed threadpool graph/active-thread 状态发布；加入 barrier 与 Fairy2i 同 backend/threadpool 线程切换回归 | `d8fe14ad4`, `89e5045d2`, `66b095e15` |
| B1 | 移植 #17133 的 CPU 空算子跳过路径，避免 metadata-only node 进入 worker barrier | `70465c5` |
| B3 | 参考 #23595 的并行初始化方法，将 Fairy2i tile64 LUT encode/pack 按完整 16-row tile 分片并行化 | `1d4c22ad5` |

## A0 证据

- 修复前，SIMD-only build 的长向量 `SCALE` case 在 Metal backend 对比中失败，CPU 路径通过。
- 修复后执行：

  ```bash
  cmake --build build-a0-noaccel --target test-backend-ops --config Release -j 8
  ./build-a0-noaccel/bin/test-backend-ops test -b Metal -o SCALE
  ```

  结果：`2/2 backends passed`，`SCALE` 两个 case 通过。
- `tests/test-backend-ops.cpp` 的 scale case 从 `n=10` 扩展到 `n=100`，覆盖超过 `GGML_F32_STEP` 的 SIMD 分块和尾部路径。
- Fairy2i 图仍在 `src/llama-graph.cpp` 使用通用 `ggml_scale()`；因此该修复覆盖 LoRA scaling 和 attention scaling 的共享 CPU 路径，不改变 Fairy2i 自定义权重布局。

## A1 证据

- `ggml/src/ggml-cpu/CMakeLists.txt`：
  - 优先使用可验证的显式 `-march`，否则使用显式 `-mcpu`，最后回退到 `-mcpu=native`。
  - 对 `-U__ARM_FEATURE_*` 和编译器实际宏做 feature 检查，避免 host feature 泄漏到目标 variant。
  - 将 `ARCH_FLAGS` 与 `ARCH_DEFINITIONS` 分开传递，保持 Fairy2i DOTPROD source 和 compile definition gate 不变。
- 已验证配置：
  - 显式 `-march=armv8-a` baseline；
  - `GGML_CPU_ALL_VARIANTS=ON` 的 apple-m1 / apple-m2-m3 / apple-m4 variants；
  - dotprod-enabled Fairy2i direct build；
  - dotprod-disabled Fairy2i build，`test-fairy2i` 全部通过。
- 失败的 Fairy2i CMake test-only 配置是旧 target 选择问题（`tests` 子目录未被配置），随后用 `test-fairy2i` / `test-backend-ops` 目标重新配置并通过；不是源代码失败。

## A2 证据

- `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` 仅补充缺失宏：`HWCAP_FPHP`、`HWCAP_ASIMDHP`、`HWCAP_ASIMDDP`、`HWCAP_SVE`、`HWCAP2_SVE2`、`HWCAP2_I8MM`、`HWCAP2_SME`。
- 保留本地 backend score ABI 和 Fairy2i runtime gate；没有引入后续 `ggml-feats.h` 重构。
- 通过伪造 Linux `sys/auxv.h` 的语法/score harness 验证缺少系统 HWCAP 宏时仍可编译，并验证 FP16 / DOTPROD score 条件。

## A3 IQ 量化退化输入正确性（已完成）

- **上游链：** #15928 初始化 IQ3_XXS `is_on_grid`；#19861 补齐 IQ2/IQ3 scratch、IQ1 shift 和失败搜索 fallback；#20460 在 IQ4_NL/XS 的 `sumq2 == 0` 时生成零缩放，避免 NaN。
- **复现：** 非零输入配合零 importance weights 会让旧 IQ2_XXS 路径把未初始化的 `L` 编码为 off-grid point，并在 `ggml-quants.c` 的 grid lookup 处 abort。该输入是合法 imatrix 边界：量化接口要求权重指针存在，但没有要求每个权重严格大于零。
- **回归：** `test-quantize-fns` 对 IQ2_XXS、IQ2_XS、IQ3_XXS、IQ1_S、IQ4_NL、IQ3_S、IQ2_S、IQ4_XS、IQ1_M 使用同一 block 内 mixed zero/nonzero imatrix，检查两次不同目标预填充得到完全相同的字节并可有限值反量化。IQ2/IQ3/IQ1 七种格式另以独立 subgroup grid/index/shift oracle 检查 fallback 打包位。
- **验证：** `test-quantize-fns -v` 与 CTest 通过；`bash scripts/ci-fairy2i-cpu.sh` 的 baseline、Fairy2i direct/LUT、LUT required/disabled、W2 `14578/14578`、legacy direct/LUT 全矩阵通过。
- **指定模型：** `qwen3-row4-int8-v1-final-bos.gguf` 继续加载 436 tensors，8 threads、BF16 KV、CPU-only、固定 seed 的 16-token smoke 输出可读文本，eval 为 `31.03 t/s`。模型使用 `ROW4_CODES` 而非 IQ 类型，因此该结果是共享量化核心的无回归证据，不代表 IQ 性能。

## A4 共享 CPU 算子正确性（已完成）

- **上游链：** #24305 修正 `RMS_NORM_BACK` 目标与梯度输入别名时的读写次序；#25247 修正量化 `CONCAT` 把逻辑元素索引误当成 block 索引。
- **复现：** 旧 RMS backward 路径在 `dst == dz` 时先覆盖梯度再用于后续乘加，四组 epsilon 均由独立 reference/sentinel 图检测到 NaN。旧量化 concat 的 Q4_0 case 出现 NMSE、NaN、未写 sentinel，随后进程退出 139。
- **实现：** RMS backward 在单个元素表达式中先读取 `dz[i]` 与 `x[i]`，再写 `dx[i]`；量化 concat 仅将 dimension 0 的 offset 和循环次数转换为完整 block 单位，dimensions 1-3 保持逻辑元素偏移，并要求同类型、连续输入和 block 对齐。
- **回归：** `RMS_NORM_BACK_ALIAS` 的四组 epsilon 全部通过；`CONCAT` 对 Q4_0、Q4_1、Q5_0、Q5_1、Q8_0 在 dimensions 0-3 的 20 个新增 case 全部通过。回归显式确认 alias tensor 与 `dz` 共用 data pointer，并用有限 reference 差值到 NaN 的 sentinel 捕获相同 CPU 实现之间的别名错误。
- **验证：** `bash scripts/ci-fairy2i-cpu.sh` 的 baseline、Fairy2i direct/LUT、LUT required/disabled、W2 `14602/14602`、legacy direct/LUT 全矩阵通过。
- **指定模型：** `qwen3-row4-int8-v1-final-bos.gguf` 加载 436 tensors，8 threads、BF16 KV、CPU-only、固定 seed 的 16-token smoke 输出可读文本，eval 为 `30.86 t/s`。该模型不执行 concat 或 backward graph，因此该结果只证明共享 CPU/loader 无回归。

## A5 Android CPU affinity portability（已完成）

- **上游目标：** #26838 将线程亲和力实现的条件从 glibc 专用 `__gnu_linux__` 改为内核平台宏 `__linux__`。Android bionic 定义后者而不定义前者，因此旧代码落入不执行任何操作的 unsupported stub。
- **实现边界：** 只修改 affinity/priority 平台分支。分支内现有 `__ANDROID__` 路径继续调用 `sched_setaffinity()`；NUMA 与 `syscall.h` 的 `__gnu_linux__` 条件保持不变，避免把 bionic 不支持的 NUMA/glibc 接口带入 Android。
- **平台探针：** `clang --target=aarch64-linux-android24 -dM -E` 证明 `__ANDROID__`、`__linux__` 存在且 `__gnu_linux__` 不存在；同一 target 的条件编译断言通过。Apple 分支优先于 Linux，BSD 不定义 `__linux__`。
- **验证：** 独立审查未发现平台覆盖或头文件问题；`bash scripts/ci-fairy2i-cpu.sh` 的 baseline、Fairy2i direct/LUT、LUT required/disabled、W2 `14602/14602`、legacy direct/LUT 全矩阵通过。
- **验证边界：** 当前机器没有 Android NDK、Android device、QEMU 或远程 ARM host，因此未执行 Android link/runtime affinity probe；这项缺口已保留为目标环境验证，而不是推断运行成功。

## A6 non-Linux SVE portability boundary（已完成）

- **上游目标：** #16520 隔离 `sys/prctl.h` 与 `PR_SVE_GET_VL` 的 Linux 依赖。旧代码只要编译器定义 `__ARM_FEATURE_SVE` 就包含 Linux header；AppleClang 的 macOS `-march=armv9-a` 正会定义 SVE/SVE2。
- **复现与实现：** 修复前，standalone `ggml-cpu-impl.h` probe 在 macOS `-march=armv9-a` 下因缺失 `sys/prctl.h` 失败。修复后该 header probe 通过；Linux aarch64/SVE 继续由 `prctl()` 初始化 vector length，non-Linux aarch64/SVE 在 backend source 中得到明确 compile-time diagnostic，避免 `ggml_cpu_get_sve_cnt()` 返回未初始化的零值。
- **验证：** 显式 `GGML_NATIVE=OFF`、`GGML_CPU_ARM_ARCH=armv9-a` 配置到达预期 non-Linux SVE diagnostic；正常 M4 native feature probe 加入 `+nosve` 与 `-U__ARM_FEATURE_SVE`，backend build 通过。独立审查和 `bash scripts/ci-fairy2i-cpu.sh` 全矩阵通过。
- **边界：** 该批次只修复平台 header/runtime 初始化不变量，不声称实现 macOS SVE。SVE vector arithmetic 正确性链仍需要可运行的 Linux SVE target。

## A7 quantization tool safety（已完成）

- **上游目标：** 最终 #17788 删除无法描述 hybrid/recurrent/expert 模型的 attention-layer count assertion；#18451 在量化前拒绝 input/output 文件碰撞。
- **实现：** 删除仅服务于失效 assertion 的 `pruned_attention_w`，保留用于 tensor type 策略的 `n_attention_wv/i_attention_wv`。CLI 对普通单文件在加载前快速拒绝 alias；库在知道 `n_split` 后先生成全部实际 output paths，再将每个输出与单输入或全部 input shards 逐一 `std::filesystem::equivalent()`，所有检查完成后才允许首个 `ofstream` 打开。
- **审查修正：** 初始上游式 guard 只比较 keep-split prefix，可能遗漏生成 shard 的碰撞；修正后由库检查实际 shard paths。CLI guard 仅用于非 keep-split，避免合法无扩展名 prefix 的假阳性。最终复审覆盖相对路径、符号链接、硬链接和多 shard。
- **验证：** `llama-quantize` build 通过；真实 `qwen3-row4-int8-v1-final-bos.gguf` 同路径 Q8_0 请求在模型加载前返回错误，前后 SHA-256 均为 `306b3086b28251cf662c462e0bc2d4e153b2a517593a651cf95f3887a62b5deb`。`bash scripts/ci-fairy2i-cpu.sh` 全矩阵通过。
- **边界：** 当前没有 hybrid/expert/recurrent source GGUF fixture，故未伪造量化输出；assert 删除依据其结构性错误和上游最终状态，构建与现有模型矩阵证明无回归。

## A9 row-contiguous quantized concat（已完成）

- **上游目标：** #25678 将量化 concat 的输入要求从 globally contiguous 放宽为 contiguous rows。A4 已完成 #25247 的 block-unit 索引修复，因此本批次只扩展合法 stride 组合。
- **复现：** 先加入 row-contiguous 但 dim3 有 gap 的 Q4_0 view；旧 `ggml_is_contiguous()` guard 在首个 `v=4` case 明确 assertion。
- **实现与回归：** `ggml_compute_forward_concat_any` 已按输入各自的 `nb[0..3]` 寻址，故只将 guard 改为 `ggml_is_contiguous_rows()`，保留 dim0 整 block 断言。Q4_0、Q4_1、Q5_0、Q5_1、Q8_0 × dimensions 0-3 × view masks 0/4/8/12 全部通过；v=4/8/12 分别覆盖 only-a、only-b、both 高维 stride gaps。
- **验证：** `CONCAT` 定向运行 `14662/14662`；独立审查确认 dim0 block offset 和 dim1-3 logical offsets 保持一致；`bash scripts/ci-fairy2i-cpu.sh` 全矩阵通过。
- **模型边界：** 指定 Row4 模型不构造量化 concat graph；A4 的模型 smoke 已覆盖共享 loader/runtime，无需把 tok/s 归因于本次 stride capability。

## Fairy2i / iFairy 模型兼容性

当前模型格式对应不同 CPU gate，不能把两条路径混成一个标准 quant kernel：

| 模型 | 可验证配置 | 结果 |
|---|---|---|
| `models/Fairy-plus-minus-i-700M/ifairy.gguf` | `build-ifairy-legacy`，`GGML_IFAIRY_LUT=1` | CLI 生成 smoke PASS；加载 267 tensors，输出可读文本 |
| `models/Fairy2i-W2/fairy2i-w2.gguf`（旧 `IFairy` 权重、`fairy2i` architecture） | 同时启用 `GGML_FAIRY2I_CPU=ON` 与 `GGML_LEGACY_IFAIRY_CPU=ON` 的 compatibility build，`--device none` | CLI 生成 smoke PASS；加载 966 tensors，输出可读文本 |
| Fairy2i tile64/bundle fixtures | `build-rel-fairy2i-direct` / `build-rel-fairy2i` | loader、CPU direct、LUT、Metal fixture gates PASS |

`fairy2i-w2.gguf` 不能在只启用新 Fairy2i CPU kernel、未启用 legacy iFairy CPU kernel 的配置中加载；其 tensor type 仍是 `GGML_TYPE_IFAIRY`。这属于现有格式/compile gate 约束，不由 A0-A7、A9 隐式改变。

## CPU CI 证据

完整脚本：

```bash
bash scripts/ci-fairy2i-cpu.sh
```

结果：通过。

覆盖内容：

- CPU-only baseline `ggml-base` / `ggml-cpu` build；
- Fairy2i direct loader/test；
- Fairy2i LUT loader/test、LUT required/disabled runtime gates；
- Fairy2i W2 backend-op 14662/14662 cases；
- legacy iFairy direct 与 LUT loader/test。

另行通过的 targeted checks：

```bash
./build-a0-noaccel/bin/test-backend-ops test -b Metal -o SCALE
./build-a1-fairy2i-nodot/bin/test-fairy2i
```

两项均通过。

## B0 调度器移植决策（已完成）

- **目标：#17748**（threadpool 中把 `n_graph` 与 active-thread count 合并为一个原子状态）；#17133 已在后续的独立 B1 提交中完成。
- 选择依据：修复前，同一个 `ggml_threadpool` 交替执行 `1` 与 `N` 线程图会出现 `SIGSEGV`、挂死或极端延迟；这是共享 CPU runtime 状态一致性缺陷，不是 Fairy2i tile claim 算法缺陷。
- B0 提交：`d8fe14ad4`（barrier active/multi-graph 回归）、`89e5045d2`（threadpool packed state）、`66b095e15`（同 backend/threadpool 的 Fairy2i W1/W2 LUT stress）。

### 实现边界

1. **`ggml/src/ggml-cpu/ggml-cpu.c`**：只适配 packed `n_graph`/active-thread count、barrier 读取、worker ready/poll、kickoff 发布、OpenMP 分支，以及 threadpool 线程数状态字段。保留 Fairy2i/legacy `prepare_graph`、extension work-size、affinity 和 pause 逻辑。
2. **`tests/test-barrier.cpp`**：active-thread graph 与 two-graph back-to-back regression 在同一 threadpool 上反复切换 `1`/`N` 线程并重新 plan。
3. **`tests/test-fairy2i.cpp`**：固定 `M=256,N=1,K=256`，复用同一个 CPU backend/threadpool，按 `1,2,1,3,1,4,1,6,1,8,1,10,1,12` 切换 3 轮；W1/W2 LUT 每轮检查 BF16 输出、dynamic-tile hit 和 `batch=1`。

以下文件不属于 B0 源码改动：`fairy2i/wide-linear*.cpp`、LUT qgemm/pack layout、模型 loader、通用 `ggml_op_is_empty` 重构。Fairy2i dynamic tile batch 未因 scheduler patch 改写。

### 验证

- 修复前的临时同-threadpool harness 已复现 `10` 线程 active graph `SIGSEGV`，以及 `12` 线程无输出挂死。
- 修复后 `timeout 120 ./build-rel-fairy2i/bin/test-barrier 10 1000` 正常返回：1000 次 2000-node graph、10000 次 active graph、1000 次双 graph；graph compute 5.04 s。
- 修复后 `timeout 30 ./build-rel-fairy2i/bin/test-barrier 12 1` 正常返回；M4 仅有 10 个逻辑 CPU，12-thread case 为过量线程诊断，不作为性能配置。
- `./build-rel-fairy2i/bin/test-fairy2i` 全部通过；新增同 backend/threadpool stress 为 42 对图，W1/W2 输出与静态首轮基线一致，dynamic hit/batch 约束通过。
- `./build-ifairy-legacy/bin/test-legacy-ifairy` 与 `test-legacy-ifairy-direct` 全部通过；`GGML_IFAIRY_LUT=1` 的 legacy ctest 也通过。
- `bash scripts/ci-fairy2i-cpu.sh` 通过：CPU baseline、Fairy2i direct/LUT loader/test、LUT required/disabled、Fairy2i W2 backend-op `14578/14578`、legacy direct/LUT gates 全部通过。

### 真实模型 smoke 与速度记录

- Qwen3 Row4 模型 `models/qwen3-row4-int8-v1-final-bos.gguf` 的 direct-mode `llama-cli`（`-t 8`, BF16 KV, `-no-cnv`, 固定 seed）输出可读中文/ASCII，无乱码；eval 为 `27.26 t/s`。
- 完成的 focused `llama-bench` 命令：

  ```bash
  ./build-rel-fairy2i/bin/llama-bench \
    -m ./models/qwen3-row4-int8-v1-final-bos.gguf -pg 128,256 -t 8,10 \
    -b 64 -ub 1 -ctk bf16 -ctv bf16 -ngl 0 -dev none -r 1 --no-warmup -o md
  ```

  同一命令结果：`t=8` 的 `tg128=26.22 t/s`、`pp512=27.50 t/s`、`pp128+tg256=23.85 t/s`；`t=10` 的 `tg128=13.14 t/s`。本次实际最高 decode 为 **26.22 t/s**，未达到预期的约 40 t/s。
- `t=11/12` 的额外 CLI probe 分别只有 `0.45/0.24 eval t/s`；结合 M4 的 10-CPU 拓扑，暂判定为过量线程诊断结果，不把它们作为推荐配置。此前包含 `t=12` 的完整 benchmark 在 900 s 截止前未完成。
- `models/Fairy-plus-minus-i-700M/ifairy.gguf` 在 `GGML_IFAIRY_LUT=1`、CPU-only、固定 seed 下也输出可读文本；eval 为 `92.30 t/s`。

### 后续

- B0 已稳定；#17133 的 NOP/barrier 优化已作为独立 B1 迭代实现。

## B1 空算子 barrier 优化（已完成）

- CPU worker loop 直接跳过 `NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE`，这些节点只修改 tensor metadata，不执行数值计算，因此不再为它们进入线程 barrier。
- 保留 `ggml_compute_forward()` 中的显式 no-op cases，但删除 `ops.cpp`/`ops.h` 中无实际工作的四个 compute wrapper；没有扩大到全局 `ggml_op_is_empty` API 重构。
- `test-barrier` 的 active-thread graph 在两组 matmul 后插入 reshape nodes，同时覆盖空节点跳过与 B0 的 `1/N` active-thread 切换。
- x86_64 Release 定向验证通过：`test-barrier 8 1` 完成 2000-node 基准 graph、10 轮含 reshape 的 active graph，以及 10 轮双 graph 切换。M4 ARM 的规定模型检查使用 `-pg 128,256 -t 8 -b 64 -ub 1 -ctk bf16 -ctv bf16 -ngl 0 -dev none -r 3 --no-warmup`，空闲态结果为 `pp512=28.78 ± 0.70 t/s`、`tg128=27.25 ± 0.26 t/s`、`pp128+tg256=25.32 ± 0.08 t/s`；相关 Row4/Fairy2i 调优环境变量均未设置。数据只证明当前实现可运行，不把 tok/s 变化归因于 B1。

## B3 Fairy2i LUT 权重变换并行化（已完成）

- **目标与边界：** #23595 并行化的是通用 IQ quantizer LUT 初始化；B3 只借用“独立工作分片”的方法，没有移植 IQ 表、OpenMP 配置或通用 quant 路径。变更局限于 `GGML_TYPE_FAIRY2I_TILE64_V2` 的一次性 CPU LUT weight transform。
- **分片语义：** `ggml-fairy2i-lut-transform.cpp` 以完整 16-row tile 为最小所有权单位。每个 worker 对自己的连续 row range 调用既有 encoder，再只写自己的 packed tiles；scale、code byte 和最后一个不满 tile 的零填充均无交叉写。
- **线程策略：** packed 输出小于 2 MiB 时保持串行；更大张量的线程数为硬件线程（最多 16）、tile 数与每 1 MiB 一个有效 worker 三者的最小值，调用线程参与工作。线程创建失败时已启动 worker 会正常 join，未启动分片由调用线程串行完成。生产路径没有新增调优接口；回归使用 `GGML_FAIRY2I_TEST_TRANSFORM_THREADS=4` 确定性覆盖并行分支。
- **缓存与内存：** 原有单 buffer 分配、临时 indexes、第二次加锁发布、cache-key 复用和 duplicate-builder 丢弃路径保持不变。五个零填充 `4096 x 4096` tile64 张量的 synthetic max-RSS 增量由串行基线 `65.75 MiB` 变为 `58.08 MiB`，未观察到峰值增加；该数包含五份保留的 cache buffer 和分配器高水位。
- **定向正确性：** 新增 `2047 x 4096` fixture，逐行逐 block 检查所有 packed code 与 FP16 scale，检查 64-byte alignment、最后 partial tile 的零 lane，以及第二次 transform 复用同一 packed pointer。`build-rel-fairy2i/bin/test-fairy2i` 的 LUT、W1/W2 variant、dynamic tile、42-pair thread-switch 和 bundle gates 全部通过。
- **完整回归：** `bash scripts/ci-fairy2i-cpu.sh` 通过 CPU baseline、Fairy2i direct/LUT、LUT required/disabled、W2 backend-op `14578/14578`、legacy direct/LUT 全矩阵；`GGML_FAIRY2I_LUT=1`、legacy direct 与 `GGML_IFAIRY_LUT=1` 三组显式门禁也全部通过。
- **首次变换延迟：** 同一 M4、空闲 CPU、五个 `4096 x 4096` 张量的中位数从串行 `10.00 ms` 降至 `3.52 ms`，为 `2.84x`；这是 weight transform 一次性延迟，不是 token throughput。
- **指定模型兼容性：** `qwen3-row4-int8-v1-final-bos.gguf` 加载 436 tensors，并以 8 threads、BF16 KV、CPU-only exact path 完成固定 seed 的 32-token CLI smoke，eval 为 `30.69 t/s`；同一最终构建的三次性能检查为 `pp512=28.78 ± 0.70 t/s`、`tg128=27.25 ± 0.26 t/s`、`pp128+tg256=25.32 ± 0.08 t/s`。该模型使用 `ROW4_CODES`，不经过 Fairy2i tile64 LUT transform，因此这里只证明 B3 没有造成共享 CPU/loader 回归，不能作为 B3 加速证据。

## KleidiAI optional-backend screening（blocked）

- `GGML_CPU_KLEIDIAI=ON` 的当前 v1.13 integration 可在 M4 构建，但 CPU `MUL_MAT` 定向矩阵仅 `14583/14602` 通过；reachable Q4_0 GEMM（GEMV 除外）产生 NaN。
- 试验性升级到 v1.14，并完整替换 #16460 的 kernel interface refactor 后，结果仍只有 `14585/14602`，Q4_0 GEMM 继续产生 NaN。该试验未满足 correctness gate，源码和 CMake 改动已全部撤回。
- 因为 #20620 只放宽 3D input 支持，在基础 Q4_0 GEMM 已错误时单独移植会扩大错误可达面。后续 Q8、SME/SME2 hybrid scheduling、v1.24 dependency 和 runtime feature detection 形成一个版本化 backend migration，不能拆成安全的一行 backport。
- 当前默认 Fairy2i/Row4 build 未启用 KleidiAI；完整 CPU 矩阵不经过该可选 backend。解除 blocker 需要整链迁移，并让 KleidiAI-enabled `test-backend-ops` 零失败后再讨论性能。

## 暂不移植

- 标准 Q4/Q5/Q6/Q8 ARM repack：布局不兼容 Fairy2i tile64。
- KleidiAI / SME / SME2：单独的可选 ARM 后端路线，不能从通用 benchmark 推导 Fairy2i 收益。
- 通用 RMS_NORM、FlashAttention 和 IQ LUT 优化：不直接移植；A4 仅移植有独立失败复现的 RMS backward 别名正确性修复，B3 仅在 Fairy2i 私有 transform 中采用独立分片方法。
- 通用 GGUF/model-quant PR：需要 Fairy2i-specific fixture 证明 tensor type、stride、scale 和 bundle metadata 后再处理。
