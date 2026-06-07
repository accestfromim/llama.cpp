# iFairy / Fairy2i ARM CPU Optimization Plan

Status: Draft (2026-06-05)

本文记录后续针对 Fairy2i / iFairy ARM CPU 推理路径的三个优化方向：

1. wide linear 图结点融合
2. `IFAIRY64` / 权重量化块结构优化
3. SVE/SVE2/SME/SME2 专用算子

本文是 V2 增量规划，不修改 legacy 文档。所有方案都必须保持现有语义不变：`w * conj(x)`。

---

## 1. 背景与当前瓶颈

### 1.1 当前 wide linear 图形态

Fairy2i 的一个 widely-linear 层当前在 `llm_build_fairy2i()` 中构造为：

```text
x_conj -> U.s0 matmul --+
                        +-> ifairy_add -> u
x_conj -> U.s1 matmul --+

x      -> W.s0 matmul --+
                        +-> ifairy_add -> w
x      -> W.s1 matmul --+

u ----------------------+
                        +-> ifairy_add -> y -> optional bias
w ----------------------+
```

无 LoRA 时，核心是 4 个 `GGML_OP_MUL_MAT` 和 3 个 `GGML_OP_IFAIRY_ADD`。有 LoRA 时，每个 `build_lora_mm()` 分支还会展开出额外的 `A*x`、`B*(A*x)`、`scale`、`add`。

数学依赖上，`U.s0`、`U.s1`、`W.s0`、`W.s1` 四个 matmul 彼此独立，只共享输入 `x` / `x_conj`，最后才汇合。

### 1.2 当前 CPU backend 执行模式

ggml CPU backend 当前按 `cgraph->nodes[]` 拓扑序逐 node 执行。一个 node 内部会使用多线程；node 之间有 barrier。因此现状是：

```text
全部线程跑 U.s0
全部线程跑 U.s1
全部线程跑 ifairy_add(U.s0, U.s1)
全部线程跑 W.s0
全部线程跑 W.s1
全部线程跑 ifairy_add(W.s0, W.s1)
全部线程跑 ifairy_add(u, w)
```

它不会自动把四个独立 matmul 分配给不同线程组并发运行。decode 场景下，单个 GEMV-like matmul 生命周期短，node 调度、barrier、中间结果写回和重复激活量化都会成为额外成本。

### 1.3 当前 ISA 使用情况

iFairy 专用 vecdot 当前主要使用 ARM NEON DOTPROD：

```text
GGML_TYPE_IFAIRY   -> ggml_vec_dot_ifairy_q16_K
GGML_TYPE_IFAIRY64 -> ggml_vec_dot_ifairy64_q16_K
```

仓库已有 SVE/SVE2/SME/SME2 backend variant 和 KleidiAI SME2 kernel 集成，但 Fairy2i / iFairy 专用 vecdot 和 LUT qgemm 热路径尚未充分使用这些扩展。

---

## 2. 目标与非目标

### 2.1 目标

- 保持 `w * conj(x)` 语义完全一致。
- 优先优化 decode / small-N 路径，同时不破坏 prefill。
- 降低 wide linear 内部 node 数、barrier 次数和中间 tensor 写回。
- 保留 `IFAIRY64` 的 64 粒度 scale 精度意图，同时减少热路径的小块开销。
- 为 SVE/SVE2/SME/SME2 专用 kernel 预留清晰后端接口。
- 所有性能结论必须用固定命令、固定模型、固定线程数复现。

### 2.2 非目标

- 不先改 legacy 文档。
- 不先做通用 ggml 图级并行调度器。
- 不在第一阶段处理 LoRA fused 路径。
- 不把 `IFAIRY64` 的文件格式变更作为第一选择；优先做 compute-only packed layout。
- 不在没有 benchmark 的情况下把 SVE/SME 设为默认快路。

---

## 3. 方向 A：wide linear 图结点融合

### 3.1 核心想法

把当前：

```text
y = U0*x_conj + U1*x_conj + W0*x + W1*x + bias
```

从多个 graph node 融合为一个 Fairy2i 专用 op 或 CPU extra op。

目标图：

```text
x, x_conj, U0, U1, W0, W1, optional bias
        |
        v
FAIRY2I_WIDE_LINEAR_FUSED
        |
        y
```

### 3.2 预期收益来源

- 四个 matmul 只做一次 fused 调度。
- 减少 `IFAIRY_ADD` node。
- 减少中间输出 tensor 写回。
- `x` / `x_conj` 的激活量化可以复用。
- decode 下减少 node barrier。
- kernel 内可按输出 row range 直接累计四路结果。

### 3.3 第一阶段范围

第一版只覆盖：

- 无 LoRA。
- `U.s0`、`U.s1`、`W.s0`、`W.s1` 均存在。
- decode / `N == 1` 优先。
- no-LUT baseline vecdot 优先。
- bias 可先不融合，第二步再接入。

第一版 kernel 伪代码：

```text
for output row assigned to thread:
    acc = 0
    acc += vecdot(U.s0[row], x_conj)
    acc += vecdot(U.s1[row], x_conj)
    acc += vecdot(W.s0[row], x)
    acc += vecdot(W.s1[row], x)
    if bias:
        acc += bias[row]
    store acc as bf16-pair complex
```

### 3.4 分阶段计划

#### A0. Baseline profiling

- 记录每个 wide linear 的 node 数、matmul 数、add 数。
- 使用 `llama-bench` 拆分 prefill / decode。
- 用 `xctrace` 或现有 profiling 工具确认 decode 下热点分布：
  - `ggml_compute_forward_mul_mat`
  - `ggml_vec_dot_ifairy*_q16_K`
  - `ggml_compute_forward_ifairy_add`
  - barrier / scheduler overhead

#### A1. no-LoRA / no-bias fused decode op

- 新增 graph 构造入口，用 fused op 替换 `build_wide_linear()` 的 4 matmul + 3 add。
- 新增 CPU compute implementation。
- 只支持 `N == 1` 或 small-N decode。
- 保留 fallback：不满足条件时走原图。

#### A2. bias 融合

- 将 `build_ifairy_bias()` 的 split/add/merge 逻辑融合到输出 store 前。
- 保留无 bias 分支。

#### A3. 激活量化复用

- 在 fused op 内分别量化一次 `x` 和 `x_conj`。
- U0/U1 复用 `x_conj_q16`。
- W0/W1 复用 `x_q16`。

#### A4. prefill / small batch 扩展

- 将 fused op 从 `N == 1` 扩展到 small-N。
- 评估是否按 row 切、按 col 切或 row/col 二维切。

#### A5. LoRA 支持

- 初期不融合 LoRA。
- 后续可考虑：
  - LoRA 仍走 fallback。
  - 或将 base fused 与 LoRA delta 分开计算再 add。

### 3.5 当前 graph-only 实验入口

- 新增环境变量：`LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1`。
- 默认关闭；未设置时 Fairy2i wide linear 继续构造旧图。
- 打开后，仅当 `U.s0`、`U.s1`、`W.s0`、`W.s1` 四个权重均存在且均为 `GGML_TYPE_IFAIRY64` 时，构造 `GGML_OP_IFAIRY_WIDE_LINEAR_W2`。
- 该 op 的 graph 语义包含 optional bias，等价于当前 `build_ifairy_bias()` 的 split/add/merge。
- 当前阶段只添加 graph op、builder 和 CPU dispatch stub；CPU/GPU 后端 fused kernel 尚未实现，实际执行会显式报 `GGML_OP_IFAIRY_WIDE_LINEAR_W2 not implemented`。

### 3.6 风险

- 新 op 的 graph buffer / backend scheduler 适配成本。
- 多输入 op 的 tensor 生命周期和 backend placement。
- LUT 路径和 no-LUT 路径差异较大，第一阶段应只做 no-LUT。
- fused op 会减少图可观察性，需要补充 debug/profiling hooks。

---

## 4. 方向 B：权重量化块结构优化

### 4.1 当前 `IFAIRY64` 的结构性成本

当前：

```text
IFAIRY:   K=256 一个 block，共用一组 d_real/d_imag
IFAIRY64: K=64  一个 block，共用一组 d_real/d_imag
```

激活侧 `block_ifairy_q16` 仍是 K=256。因此 `IFAIRY64` 在热路径中表现为：

```text
4 个 K64 权重 block 共享 1 个 K256 激活 block
```

这保留了 64 粒度 scale，但带来：

- 每 K256 有 4 次权重 scale 读取与合成。
- 每 K256 有 4 次小块循环收尾。
- 权重存储中 scale 开销更高。
- LUT preprocess 会把同一个激活 scale 复制到 4 个 K64 权重块。

### 4.2 优先方案：compute-only super-block

当前实现备注：

- 已新增 `GGML_TYPE_IFAIRY64_Q16` / `block_ifairy64_q16`，作为 `GGML_TYPE_IFAIRY64` vecdot 的专用激活格式。
- 新格式每个 activation block 包含 64 个 complex activation，`IFAIRY64` vecdot 不再从旧 `GGML_TYPE_IFAIRY_Q16` 的 K256 block 中按 `i / 4` 切片。
- 下面的 compute-only super-block 仍是后续权重 packed layout 优化方向；它需要基于新的 K64 activation 格式重新评估。

不先修改 GGUF 权重格式，而是在 CPU transform / packed layout 阶段把连续 4 个 `block_ifairy64` 组合为计算态 super-block：

```text
ifairy64x4_compute_block:
    qs[4][...]
    d_real[4]
    d_imag[4]
```

语义保持：

```text
每 64 个权重仍使用独立 scale
每 256 个 K 作为一个计算循环单元
```

### 4.3 预期收益来源

- 以 K256 为单位复用同一个激活 q16 block。
- 减少循环控制和索引计算。
- 减少重复读取激活 scale。
- 便于 fused wide-linear kernel 一次处理 4 个 K64 subblock。
- 便于后续 SVE/SVE2 kernel 按更大粒度展开。

### 4.4 分阶段计划

#### B1. baseline vecdot super-loop

- 不改变数据结构，只重写 `ggml_vec_dot_ifairy64_q16_K()` 内部循环。
- 将 `i += 1` 的 K64 循环展开为 `i += 4` 的 K256 super-loop。
- 每个 super-loop 内处理 4 个 K64 subblock。
- 先保持输出 bit-level 或误差行为与原实现一致。

#### B2. LUT packed layout super-block

- 新增 compute-only packed tile，例如：

```text
ifairy64x4_lut_wtile_16:
    qs[4][QK_IFAIRY64_GROUPS_PER_BLOCK / 2][16]
    d_real[4][16]
    d_imag[4][16]
```

- transform 阶段将原始 `block_ifairy64` 打包成 super-block。
- qgemm 阶段以 K256 为外层 block，内部保留 4 组 scale。

#### B3. 文件格式变更评估

仅当 compute-only super-block 收益不足时，评估 GGUF 层面的新类型或新 block：

- `GGML_TYPE_IFAIRY64X4`
- 二级 scale：K256 base scale + 4 个 subscale
- `block_ifairy64_q16` 激活格式

这些方案都会影响转换、加载、测试与兼容性，不作为第一阶段。

### 4.5 风险

- `IFAIRY64` 原始设计可能依赖 64 粒度 scale 的数值行为，不能合并 scale。
- super-block 只能合并计算调度，不能改变 scale 语义。
- packed layout 变更会影响 LUT transform、preprocess、qgemm 和测试。

---

## 5. 方向 C：SVE/SVE2/SME/SME2 专用算子

### 5.1 当前状态

仓库已有 ARM feature 检测和 backend variants：

- DOTPROD
- SVE
- SVE2
- MATMUL_INT8 / I8MM
- SME / SME2

但 Fairy2i / iFairy 专用 vecdot 目前主要是 NEON DOTPROD。SVE/SME 应作为专用 kernel backend，而不是期待编译开关自动加速现有 NEON kernel。

### 5.2 SVE/SVE2 优化方向

适合目标：

- `ggml_vec_dot_ifairy_q16_K`
- `ggml_vec_dot_ifairy64_q16_K`
- LUT preprocess / qgemm 的 table lookup 和 int8 accumulation

可能策略：

- SVE2 vector-length agnostic 实现。
- 使用更宽 vector 一次处理更多 2-bit/4-bit code。
- 保持 NEON DOTPROD fallback。
- 对 `SVE_CNT == 128` 的机器谨慎启用，避免无收益。

### 5.3 SME/SME2 优化方向

SME/SME2 更适合大块矩阵乘，尤其是 prefill / large-N。Fairy2i 权重是 2-bit complex code，不能直接套普通 int8 GEMM，需要专用数据布局。

候选方案：

1. fused wide-linear op 的 SME2 后端。
2. `IFAIRY64x4` compute layout 的 SME2 tile kernel。
3. prefill-only SME2 qgemm，decode 仍用 NEON/SVE2。

### 5.4 预期收益范围

以下为实现前的工程预估，不作为性能承诺：

```text
SVE2 vecdot only:
    decode overall: 5% - 30%
    kernel-only:    1.2x - 2.0x, 取决于 SVE vector length

SVE2 + fused wide-linear:
    decode overall: 15% - 50%

SME2 prefill kernel:
    prefill overall: 20% - 80%
    取决于 N、线程数和内存带宽

SME2 decode without fusion:
    不确定，可能收益很小
```

### 5.5 分阶段计划

#### C1. ISA capability and dispatch

- 在现有 CPU backend variant / feature 检测基础上增加 Fairy2i kernel dispatch。
- 保留 NEON DOTPROD baseline。

#### C2. SVE2 vecdot prototype

- 先实现 `IFAIRY64`，因为 K64 小块更需要重构。
- 结合 B1 的 K256 super-loop，避免只扩大 vector 而不减少小块开销。

#### C3. SVE2 LUT qgemm prototype

- 优先 small-N decode path。
- 对比 no-LUT baseline，避免 LUT 在高线程下回退。

#### C4. SME2 prefill prototype

- 只覆盖 prefill / large-N。
- 如果需要新 packed layout，应与 B2 合并设计。

---

## 6. 推荐实施顺序

优先级建议：

```text
P0: 建立稳定 benchmark / profiling 基线
P1: wide-linear fused no-LoRA no-bias decode op
P2: fused op 内复用 x / x_conj 激活量化
P3: IFAIRY64 K256 super-loop baseline vecdot
P4: fused bias
P5: IFAIRY64 LUT packed super-block
P6: SVE2 vecdot backend
P7: SVE2/SME2 fused backend
P8: LoRA/fallback 策略细化
```

理由：

- 融合减少 node 数和重复工作，decode 下通常比单纯换 ISA 更先见效。
- `IFAIRY64` super-block 保留量化语义，风险小于文件格式变更。
- SVE/SME 最好服务于 fused kernel，而不是在四个串行 matmul node 上分别替换 NEON。

---

## 7. 验证与验收

### 7.1 正确性

必须通过：

```text
./build-rel/bin/test-ifairy
./build-rel-lut/bin/test-ifairy
```

新增 fused op 后需要补充：

- no-LoRA / no-bias fused vs 原图逐元素对比。
- bias fused vs 原图逐元素对比。
- `IFAIRY` / `IFAIRY64` 分别覆盖。
- decode `N == 1` 和 small-N 覆盖。
- fallback 条件覆盖。

### 7.2 功能 smoke

固定命令：

```text
./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```

LUT 构建：

```text
GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv
```

### 7.3 性能

至少记录：

```text
./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 1,2,4,6,8 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3
```

decode 专项：

```text
./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf -t {1,2,4,6,8} -ngl 0 --device none -p 1 -n 512 -r 3 -o jsonl
```

记录内容：

- raw command
- commit / build type
- CPU model / OS
- threads
- pp tok/s
- tg tok/s
- env vars
- 是否开启 LUT / fused / SVE / SME

### 7.4 质量

如果任何方案改变量化格式或输出层行为，需要跑 perplexity：

```text
./build-rel/bin/llama-perplexity -m <model> -f <wikitext2> -c 2048 --chunks 16 -t 4 -ngl 0 --device none --no-warmup
```

纯 compute-only fused / super-block 理论上应与原路径数值等价；若出现差异，需要解释误差来源。

---

## 8. 待细化问题

1. fused wide-linear op 是新增 `GGML_OP_*`，还是作为 CPU extra op / pattern rewrite？
2. fused op 的 tensor 输入如何编码，是否超过 `GGML_MAX_SRC` 限制？
3. `x` / `x_conj` 的 Q16 激活量化 workspace 如何分配，是否复用 `params->wdata`？
4. `IFAIRY64x4` packed layout 是否只用于 LUT，还是 baseline vecdot 也引入 transform？
5. SVE2 kernel 是否采用 vector-length agnostic 写法，还是按 `SVE_CNT` 做专门版本？
6. SME2 是否只做 prefill，decode 是否仍以 fusion + NEON/SVE2 为主？
7. LoRA 打开时是否强制 fallback 原图？

---

## 9. 当前结论

后续优化应先解决图和数据复用问题，再进入更激进的 ISA 后端：

```text
先融合 wide linear，减少 node 和重复工作；
再把 IFAIRY64 的 4 个 K64 block 组织成 K256 计算态；
最后基于 fused / super-block 形态实现 SVE2 或 SME2 kernel。
```

这条路线能尽量保留当前模型格式和数值语义，同时给 ARM CPU 路径留下明确的性能演进空间。
