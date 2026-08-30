# Upstream x86 CPU 优化筛选与开发计划

## 1. 目标和范围

本文记录 `lwt/merge_master` 分支的 x86 CPU 上游优化筛选结果和开发顺序。

筛选基线：

- fork 分歧点：`4b8560ab56fdd9819358b47c338bbc8ec357c57e`
- 冻结上游：`upstream/master`，`34af94cd9ab277632e27caeec2d41de2fd091b31`
- first-parent 范围：3951 个 commits
- 本地基线：`79715a1fc02763f9a5cf33fe53c2ed755e0f4dd1`
- GitHub 状态补充截止时间：2026-08-30。冻结点之后合并的 #24575 单独标注，不计入 3951-commit 范围

本计划覆盖：

- x86 CPU build、ISA 检测和运行时 dispatch
- AVX、AVX2、AVX-VNNI、AVX512、AVX512-VNNI、AVX512-BF16、AVX512-FP16
- 标准 Q4、Q5、Q6、Q8、IQ 和 MXFP4 CPU kernel
- CPU repack、FlashAttention、NORM、RMS fusion、concat 和量化初始化
- Intel AMX 现有实现及未合并安全修复
- AMD ZenDNN 和 NUMA 候选的适用边界

不在本计划内：

- Row4 x86 backend。当前 Row4 loader 明确拒绝 x86，这属于新增 backend，不是优化移植
- CUDA、SYCL、Vulkan、OpenCL 和其他 GPU backend
- Fairy2i 和 legacy iFairy 私有布局重写
- 未解决公共格式 ID 冲突前的 Q1_0、NVFP4、Q2_0 导入
- 将多个未合并的实验性 PR 混入一个实现批次

## 2. 结论

x86 优化可以继续，但必须按独立行为批次移植，不能批量同步 `upstream/master`。

优先顺序：

1. 修复 build 和 ISA dispatch 边界。
2. 移植 AVX2 Q6_K 和 MXFP4 E8M0 LUT，两项改动窄，当前本地路径可达。
3. 处理 NORM、concat、IQ LUT init 和 RMS fusion，每项单独提交。
4. 分阶段移植 tiled FlashAttention 完整链。
5. 目标机支持 AMX 时，先完成 X8 correctness 链，再处理其他 AMX 性能候选。
6. 只在匹配硬件上处理 P/E affinity、AVX512-FP16 和 MXFP4 repack。
7. Q1_0、NVFP4、Q2_0 进入独立公共格式项目，不进入当前性能批次。
8. ZenDNN 和 NUMA 只在明确的 AMD EPYC 或双 socket 需求下立项。

最小可执行版本是 X0、X1、X2。完成三批后先比较整模型结果，再决定是否扩大范围。

## 3. 分级规则

| 级别 | 含义 |
|---|---|
| P0 | correctness、build 或 dispatch 前置条件。缺失时可能编译失败、错误 dispatch 或产生错误结果 |
| P1 | 已有明确上游性能证据，当前本地路径存在，可建立独立正确性 oracle |
| P2 | 硬件、模型或运行模式相关。收益有限，或存在明显回退区间 |
| Present | 上游行为已在当前分支存在，不重复移植 |
| Defer | 依赖公共格式、外部 backend 或当前缺失的基础设施 |
| Drop | 已 revert、被替代、不可达、回退明确，或与当前目标无关 |

上游性能数字只用于候选排序。最终收益必须在目标 x86 开发机重新测量。

## 4. 推荐实施批次

每个批次必须独立可审查、可验证、可提交和可回滚。

### X0：build 和 dispatch 安全

上游 PR：

- #17270，`cb623de3fc61011e5062522b4d05721a22f2e916`
- #18623，`e75ee11024befa163cbc0398f9e697e4b32c5f2c`
- #19609，`ae2d3f28a86f7132af742e89e212fcd874cf27f2`

改动：

- 将需要 AVX512BW 和 AVX512DQ 的 repack fast path 从过宽的 AVX512F guard 中分离。
- 修复 AVX512-BF16 build 所需的 `immintrin.h` 和 cast。
- 对 CPU feature 检测目标禁用 LTO，避免 ISA 指令被跨模块内联到 dispatch score 路径。

上游 PR #18186，`94de74e7b1b9ee6404b54ebb0df273a3ef35a555`，作为 X0 的独立 all-variants 后续提交，不与上述三项混合。完成 X0 需要该提交的 build matrix 通过，但不要求在 native 单机性能基线中启用所有 variants。

验收：

- GCC 和 Clang 构建。
- baseline、AVX2、AVX512F-only、AVX512BW/DQ、AVX512-BF16 组合。
- `GGML_BACKEND_DL=ON` 和 `GGML_CPU_ALL_VARIANTS=ON`。
- 低 ISA 主机不得因高 ISA backend 检测而触发 `SIGILL`。

### X1：AVX2 Q6_K vec-dot

上游 PR：#22345，`2dd84169d1225c2a56ddd1902d718f4514d7536a`

当前状态：本地 `ggml/src/ggml-cpu/arch/x86/quants.c` 仍是旧 AVX2 Q6_K 实现。

上游证据：

- Llama 1B Q6_K，4 threads，`pp512` 从 `49.75` 提升到 `63.15 t/s`，约 27%。
- `tg128` 从 `15.53` 到 `15.63 t/s`，基本持平。
- microkernel 的 `n=4` 从 `80.91` 到 `111.72 GFLOPS`，`n=8` 从 `97.28` 到 `128.53 GFLOPS`。

验收：

- Q6_K 和 Q8_K 随机 block differential test。
- 奇偶 block、不同 K 长度和 tail。
- scalar、AVX、AVX2 的允许误差一致。
- Q6_K 标准模型 `pp512` 和 `tg128`。
- 非 Q6_K operator 不回退。

### X2：MXFP4 E8M0 scale LUT

上游 PR：#19288，`2ceda3f6622661af839c19767705f33fb9f6cdd2`

当前状态：本地 MXFP4 x86 vec-dot 仍直接执行 E8M0 到 F32 转换，没有 CPU LUT。

上游证据：gpt-oss 20B MXFP4 的 prefill 提升约 6% 到 8%。

验收：

- 256 个 E8M0 输入值与 reference 完全一致。
- SSSE3、AVX、AVX2 和 generic fallback。
- MXFP4 和 Q8_0 vec-dot differential test。
- gpt-oss MXFP4 CPU-only `pp512`、`pp2048`、`tg128`。
- LUT 初始化时间和常驻内存。

### X3：NORM SIMD 和 scalar correctness

上游 PR：

- #15953，`1deee0f8d494981c32597dca8b5f8696d399b0f2`
- #16558，`c515fc577166042234241c6bd0da9b08dcbe2bb9`

#16558 修复 #15953 引入的 scalar bug，两项必须按最终语义一起评审和提交。

上游证据：AVX2 NORM bandwidth 从 `7.48` 提升到 `12.89 GB/s`。

验收：

- scalar、SSE、AVX2 和 AVX512 output differential。
- 不同 row size、epsilon、view 和 tail。
- TTS 和普通 transformer NORM 图回归。
- operator perf 和整模型 no-regression。

### X4：共享 CPU 窄优化

X4 由三个独立提交组成。

#### X4a：row-level concat copy

上游 PR：#24575，merge commit `369e1cd6140b7cbdb552cc3f87613aeb9e122422`。该 PR 于 2026-08-22 合并，晚于冻结上游，实施前需要先对当前 upstream 重新冻结和 rebase。

要求：

- 在现有量化 concat block、stride 和 row-contiguous 检查上手工适配。
- 不得覆盖 A4 和 A9 已加入的正确性 guards。

上游 isolated benchmark 为 83x 到 133x，但没有整模型数据，不能把该数字当作 token throughput。

#### X4b：IQ LUT 初始化并行化

上游 PR：#23595，`826539ce590fe294642db0acd54ea5e0a2fcd739`

当前标准 IQ2 和 IQ3 LUT 初始化仍串行。上游初始化时间约缩短 20x。

要求：

- OpenMP ON 和 OFF 生成的表必须 byte-identical。
- 保留无 OpenMP 构建。
- 记录 startup latency 和 RSS。
- 不把该收益归因于已完成的 Fairy2i B3 transform。

#### X4c：RMS_NORM + MUL fusion

上游 PR：#22423，`f08f20a0e330da9530ded3e800dff9fbb05a3288`

上游 operator 提升 1.72x 到 2.07x，EPYC 整模型 prefill 提升约 3% 到 7%。

要求：

- 只覆盖标准 F32 graph。
- Fairy2i exact custom RMS/MUL 路径不得进入该 fusion。
- fusion enabled/disabled、broadcast、多 add、非相邻节点和 graph reuse 均需覆盖。

### X5：tiled FlashAttention 完整链

#### X5a：tiled prefill 和 split-KV decode

上游 PR：

- #19012，`bcb43163aed6a8986cf3d66e90848c9c258d4936`
- #19209，`9f682fb640765ff79ee13a7a00cdbaa15c1ed07a`

上游长上下文 prefill 最高约 3.87x 到 4.39x，decode 在上下文增长时报告超过 2x。

#19012 和 #19209 分为两个提交。第一批建立 tiled prefill，第二批加入 split-KV 和 reference plumbing。

#### X5b：最终修正的 SIMD GEMM

上游 PR：

- #19422，`684b36101c9eeb7e89c9e602f9ded05f1353a0c6`
- #19642，`08e6d914b8fef477d2a5aeeb89e08d42709981cf`
- #25390，`57b50e1f6b50e01eb14c43fc1253602af74c1870`

当前本地没有 `ggml/src/ggml-cpu/simd-gemm.h`。#19422 新增微内核，#19642 修复 UB 和 compiler warning，#25390 修复 #19642 后的 scalar tail A-index。三项必须以最终修正状态一起落地。

验收：

- BS=75。
- full-row 和 tail-column。
- SSE、AVX2、AVX512 和 scalar reference。
- mask 全为负无穷、GQA、MQA、DV/DK tails、F16/F32 KV。
- `pp512` 和 `tg128`，上下文 1K、2K、4K、8K、16K。

#### X5c：F16 V-cache bulk conversion

上游 PR：#26947，`eeae28b67e94cbce01f016576803509dbad11d09`

只有 tiled FA 建立后才移植。上游 Qwen3 4B 报告 prefill 提升 17% 到 31%。

验收：

- F16C available 和 unavailable。
- stride、tail 和不同 DV。
- 转换结果与 scalar reference 一致。
- tiled FA output 和长上下文 prefill。

### X6：MXFP4 repack，显式 opt-in

上游 PR：

- #19738，`d903f30e25f3024c37d1eedd4b46ed0f5b13ff88`
- #20692，`78d550b541a12eb473f1bca4cf9ac6920ebdb42b`

上游 CPU-only 证据：

- prefill 提升约 20% 到 23%。
- token generation 提升约 14%。

上游 mixed offload 测试曾退化到 0.63x 到 0.64x。该批次必须保持默认关闭，只允许 CPU-only 显式 opt-in。#20692 只随 #19738 处理。

验收：

- repack OFF 和 ON。
- CPU-only 和 mixed offload。
- GEMV、GEMM、GET_ROWS、3D 和 MoE。
- PPL、确定性输出、load time、RSS 和额外 buffer 顺序。
- mixed offload 不得回退。

### X7：硬件条件分支

#### OpenMP P/E affinity

上游 PR：#16164，`4e29084ba4104c4ea529fd3163bb6e76f64383df`

只在 Intel P/E 混合核且使用 OpenMP 时实施。当前本地 affinity 调用位于非 OpenMP worker 路径，OpenMP graph branch 没有应用 cpumask。

验收必须读取实际 worker CPU affinity，不能只验证 API 返回成功。

#### AVX512-FP16

上游 PR：#20529，`d0b79aaa2f6e7b7d3c26b1845b43cef158697540`

只在支持 AVX512-FP16 的真实 CPU 上实施。上游端到端只提升约 0.8%，若目标机没有稳定收益则不保留。

## 5. X8：AMX correctness 和性能链

当前本地 AMX 模块早于以下上游修复，不能标为已存在：

- `amx.cpp` 仍只接受 2D tensor，没有 #19925 的 per-type K alignment。
- `mmq.cpp` 的 Q4_0/Q4_1 unpack 路径仍在第二个 tile 使用 `B_blk0`，而不是 `B_blk1`。
- 无 OpenMP 时，`common.h` 仍串行执行，没有 #20074 的 `std::thread` fallback。
- TLS tile 和 accumulator buffers 没有 #21058 的 64-byte alignment。
- activation quantization 仍按 `M` 分片，没有 #24806 的 `n_batch * M` partition。

因此 AMX 需要独立的条件链：

| 批次 | PR | Frozen SHA | 作用 | 优先级 |
|---|---|---|---|---|
| X8a | #19925 | `4e76d24f282e0fa591f1eb87ae6fd9174c6ed998` | 吸收 #16315 的边界修复，加入 per-type alignment、Tile1 修复和 batched support | P0 correctness |
| X8b | #20074 | `66199c9f03af450df47440998ebf005eae202163` | 无 OpenMP 时并行 weight conversion | P1 load time |
| X8c | #21058 | `f84270ea10b2413b4a56b91cfa40ddec27f7300f` | 64-byte aligned tile buffers | P0 safety |
| X8d | #24806 | `37a77fb0579be9d71e2c73da0553cfd42b7b103a` | 按 `n_batch * M` 分配 activation quantization 工作 | P1 prefill |

上游 PR #16315，`a23b9bdbd3b64ce172f9962249f432d01aea7437`，不单独移植。#19925 已包含并替换其过严 guard。

X8 只在真实 AMX 主机上执行。X8a correctness 通过后，X8b、X8c、X8d 分别提交，不能合成一个性能补丁。

AMX open watchlist：

- #20940：OSXSAVE 和 XCR0 dispatch gate，当前首选方案。
- #26954：每个 worker 请求 XTILEDATA 权限，但当前 PR 在 syscall 失败后仍执行 tile 指令，必须改为 fail-closed。
- #27024：Q8_K VNNI，只有外部 differential gist，暂不采用。

AMX 验收需要：

- CPUID、OSXSAVE、XCR0 和 Linux XFD。
- 每个 worker 的 XTILEDATA permission。
- Q4_0、Q4_1、Q8_0、Q4_K、Q5_K、Q6_K、IQ4_XS。
- batch、MoE、tail、thread stress、PPL 和 reference output。

## 6. 公共格式 ABI 阻塞

GGML/GGUF quantization type 和 `llama_ftype` 是独立的 ABI namespace，必须分别处理：

| 上游格式 | 上游 GGML/GGUF type | 本地 GGML type 占用 | 上游 `llama_ftype` | 本地 `llama_ftype` 占用 |
|---|---:|---|---:|---|
| NVFP4 | 40 | `GGML_TYPE_IFAIRY` | 39 | 未占用 |
| Q1_0 | 41 | `GGML_TYPE_IFAIRY_Q16` | 40 | `LLAMA_FTYPE_MOSTLY_IFAIRY` |
| Q2_0 | 42 | `GGML_TYPE_IFAIRY64` | 41 | `LLAMA_FTYPE_MOSTLY_FAIRY2I_TILE64_V2` |

三个格式的 GGML/GGUF type 均冲突。`llama_ftype` 只有 Q1_0 和 Q2_0 冲突，NVFP4 的 `llama_ftype=39` 当前未占用。重编号和兼容策略不能把两个 namespace 合并处理。

受阻链：

| 格式链 | 上游 PR | 上游 x86 证据 | 决策 |
|---|---|---|---|
| Q1_0 foundation | #21273，`2e1f0a889e19a3922db57452268f4574c35c36e5` | foundation 无 x86 kernel | Defer |
| Q1_0 x86 | #21636，`7f251fdbce614a50141005dc70ce3787b7777a8e` | Ryzen 7640HS AVX2，pp `13.07→131.03`，tg `9.38→73.85` | ABI 项目后列 P1 |
| NVFP4 foundation | #19769，`5eae9cb1d9ecf0bbe031352da61b8b22a3e10bbb` | foundation 还包含 converter、loader 和 scale2 graph 语义 | Defer |
| NVFP4 x86 | #23961，`6dbc1174b8f46a8e064259985d84c81a472beea4` | i9-7900X，pp512 `2.85→30.48` | ABI 项目后列 P1 |
| Q2_0 foundation | #24448，`bec4772f6a2527d371557b5d2032641e5ff7619c` | generic 和 ARM，无 x86 SIMD | Defer |

公共格式项目必须先决定：

1. 是否重编号本地私有类型。
2. 是否增加 GGUF schema/version 隔离。
3. 旧 IFAIRY、Fairy2i 和 Row4 文件如何识别和拒载。
4. converter、loader、quantizer 和 Python constants 如何同步。
5. 新旧模型能否在同一 binary 中安全共存。

这些问题没有解决前，禁止 cherry-pick #21273、#19769 或 #24448。

## 7. ZenDNN 决策

ZenDNN 不进入标准 x86 第一阶段。

上游链：

- #17690，backend foundation
- #19133，dependency pin
- #19159，dynamic backend symbol fix
- #19923，LowOHA API 和 static/shared build
- #21315，MUL_MAT_ID
- #22681，small-batch adaptive fallback
- #23414，Q8_0
- #20964，rename only
- #25918，group matmul API

当前本地没有 `ggml/src/ggml-zendnn`、`GGML_ZENDNN` option 或 backend registration。初始导入约 19.7k lines，其中大部分是生成的 ops CSV。backend 还需要 ZenDNN、AOCL-DLP、oneDNN、LibXSMM、AOCL utils、OpenMP 和 Linux 专用构建维护。

只有明确的 AMD EPYC 或 Ryzen AI 产品需求时，才把 ZenDNN 作为独立 optional backend 项目。它不能与通用 AVX kernel 批次混合。

## 8. Open PR watchlist

Open PR 不进入批准的实施批次。必须等 upstream merge 或在本地完成独立设计审查、rebase 和完整验证。

### 8.1 ISA 和 build 安全

| PR | 方向 | 决策 |
|---|---|---|
| #20940 | OSXSAVE/XCR0 feature gate | 首选安全基线，merge 后适配；保守处理 Darwin |
| #19514 | XCR0 gate | 被 #20940 替代 |
| #20388 | XCR0 + Darwin sysctl | 与 #20940 竞争，rewrite 过大，不混合 |
| #25346 | MSVC native AVX-VNNI detection | 独立 native build-time probe；必须在目标机执行，不提供 portable binary runtime 保护 |
| #26187 | clang-cl `-mavxvnni` | 独立 build fix，merge 后评估 |
| #23593 | clang-cl AVX512-BF16 build | Draft，等待 CI |
| #24094 | Apple/MSVC AMX variant gate | merge 后适配 |
| #26954 | AMX per-worker XFD permission | 当前 fail-open，修正前不采用 |
| #27024 | AMX Q8_K VNNI | 暂不采用 |

注意：这些 XCR0 PR 只影响 `GGML_BACKEND_DL` 的 dynamic backend score。静态单 variant 或手工强制 ISA 不受该 gate 保护，实施时必须明确禁止不匹配的静态 ISA 配置，或补独立 runtime dispatch。

### 8.2 x86 quant 和 repack

| PR | 方向 | 上游证据 | 决策 |
|---|---|---|---|
| #22331 | x86 Q8 producer SIMD | Q8_K 约 `5→25–29 GB/s` | Watch，先验证非默认 MXCSR rounding |
| #27590 | Q5_K/Q6_K AVX512/VNNI | dot 1.20x/1.44x，pp +16.7% | 当前最成熟 direct open 候选 |
| #26348 | Q2_0 VNNI | pp/tg 3.0x 到 3.64x | 受 Q2 ABI 阻塞，另有 FMA/PPL policy drift |
| #22181 | AVX2 Q4_K/Q5_K reduction | 约 +4% 到 +12% | Watch，证据较薄 |
| #23309 | Q4_K GEMV VNNI | tg +12% 到 +23% | 与 #23793 统一 rebase 后评估 |
| #23793 | Q4_K GEMV prefetch | tg +6% 到 +13% | #23309 同函数后续，不单独落地 |
| #22525 | Q4_K GEMM | geomean 1.10x 到 1.20x | Watch，与 #27851 竞争 |
| #18495 | Q4_K runtime | 公开复测 pp/tg 回退 | Drop，被 #23309/#22525 替代 |
| #22250 | Q5_0 block interleave | pp +42% 到 +129% | 改动约 978 x86 lines，后置 |
| #19707 | Q5_K block interleave | pp +90% 到 +140% | 改动约 2121 x86 lines，后置 |
| #19706 | Q6_K block interleave | pp 有收益，tg 约 -3% | Drop，已有 3D assert 报告，改动约 3986 lines |
| #27402 | IQ runtime panel | dense 3.4x 到 8.4x | 高潜力，独立 IQ 项目 |
| #27851 | tiled K-quant GEMM | VNNI tall GEMM 3x 到 7x | 高潜力，与现有 repack GEMM 逐一比较 |
| #23439 | TQ zero-scale skip | 无 benchmark | P2 watch，只在零 block 多时可能有收益 |

同一函数或同一 shape 区间的 open PR 不得一起移植：

- #18495 的 GEMV 方向被 #23309 替代，GEMM 方向被 #22525 替代。
- #23793 是 #23309 的同函数 prefetch 后续。
- #27851 只在非 repacked tall GEMM 区间与 #22250、#19707、#22525 竞争。
- #27590 是直接 vec-dot，不等价于 repack GEMM/GEMV。
- #19706 在重新设计 3D 和 GET_ROWS 前不可采用。

### 8.3 shared CPU 和 runtime

| PR | 方向 | 决策 |
|---|---|---|
| #27478 | batch-1 FA + heap THP | Watch，必须拆成 attention 和 allocator 两批；部分 DV/pp 场景回退 |
| #26468 | SOFT_MAX sweeps | Watch，短 F16/T5 场景曾回退 |
| #26948 | quantized KV tiled FA | 当前本地没有 tiled FA，等待 X5 完成 |
| #22022 | mmap MADV_HUGEPAGE | 证据弱，只有内存压力场景可能有收益 |
| #27986 | NUMA mirror | 双 socket EPYC 专用，内存按 node 复制，另立项目 |
| #16000 | broad NUMA mirror prototype | Drop，范围约 100 files，已有 hybrid 回退 |
| #14232 | NUMA migrate | Drop，已报告 hybrid crash 和乱码 |
| #13319 | GQA loop reorder | Drop，收益不稳定，正确性证据不足 |
| #17113 | multi-ISA mega optimization | Drop，范围过宽，已有 CI/crash 和 tail 风险 |

## 9. 硬件准入

实施前记录：

- CPU 型号、microarchitecture 和 stepping
- physical cores、SMT、P/E cores、socket 和 NUMA nodes
- L1/L2/L3 cache 和内存通道
- SSE4.2、AVX、F16C、FMA、AVX2、AVX-VNNI
- AVX512F、BW、DQ、VL、VNNI、BF16、FP16
- AMX_TILE、AMX_INT8、OSXSAVE、XCR0 和 Linux XFD
- 操作系统、kernel、GCC/Clang/MSVC/clang-cl、CMake 和 OpenMP
- THP mode、CPU governor、NUMA policy 和 affinity

计划假设目标机至少支持 AVX2：

- 只有 SSE 或 AVX 时，只执行 build、安全和 shared fallback 批次。
- Intel P/E 混合核优先验证 #16164。
- Sapphire Rapids 或更新 Xeon 优先处理 XCR0、XFD 和 AMX worker safety。
- 双 socket EPYC 才讨论 NUMA mirror 或 ZenDNN。

## 10. 正确性门禁

每批必须先运行最小行为测试，再运行完整相关矩阵。

构建矩阵：

- x86 baseline
- SSE4.2
- AVX + F16C + FMA
- AVX2 + F16C + FMA
- AVX-VNNI
- AVX512F-only
- AVX512F + BW + DQ + VL
- AVX512-VNNI、BF16、FP16，仅在真实支持机
- AMX，仅在 OS state 和每 worker permission 都可验证时
- `GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON`
- `GGML_NATIVE=ON`，只用于目标验证机

行为测试：

```bash
./build-x86/bin/test-quantize-fns -v
./build-x86/bin/test-backend-ops test -b CPU -o MUL_MAT
./build-x86/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT
./build-x86/bin/test-backend-ops test -b CPU -o NORM
./build-x86/bin/test-backend-ops test -b CPU -o CONCAT
bash scripts/ci-fairy2i-cpu.sh
```

如果修改 GGUF、converter 或公共量化格式，再运行：

```bash
bash scripts/test-gguf-py.sh
```

`scripts/ci-fairy2i-cpu.sh` 不覆盖 Row4。x86 运行 standalone scalar oracle：

```bash
cmake -B build-x86-row4 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-x86-row4 --target test-row4 -j 8
ctest --test-dir build-x86-row4 -R '^test-row4$' --output-on-failure
```

`test-row4-loader` 只在 ARM64 注册，因此 loader 回归在当前 M4 上独立执行：

```bash
cmake -B build-arm64-row4 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-arm64-row4 --target test-row4 test-row4-loader -j 8
ctest --test-dir build-arm64-row4 -R '^test-row4' --output-on-failure
```

测试必须覆盖：

- scalar/reference 与所有启用 ISA 的 differential output
- 奇偶 block、tail、unaligned、view、stride 和 3D
- `n_threads=1`、physical cores 和 SMT
- GEMV、GEMM、MUL_MAT_ID 和 MoE
- NaN、Inf、zero block、negative zero 和 scale extremes
- graph reuse、mixed offload 和额外 buffer 路由
- 标准模型的 PPL、确定性 generation 和 loader smoke
- 完整 Fairy2i 和 legacy iFairy 矩阵、x86 Row4 scalar oracle，以及 ARM64 Row4 loader 回归。三个入口共同保证 shared CPU 改动没有越过私有路径边界

## 11. 性能门禁

固定以下条件：

- source commit 和 build flags
- compiler 和 linker
- model 文件及 SHA-256
- threads、CPU affinity 和 NUMA policy
- KV dtype、batch、ubatch 和 context
- THP、OpenMP 和所有相关环境变量
- 同一台机器、空闲状态、一次只运行一个 benchmark process

执行方法：

1. baseline 和 candidate 使用 ABBA 顺序，执行四个完整 ABBA blocks。每侧得到 8 个独立样本。
2. 每个进程只运行一次有效测量，预热结果不计入。
3. 分别保存 operator、`pp512`、`tg128`、长上下文、load time 和 RSS 的原始结果。
4. 从 8 个独立样本计算 arithmetic mean、sample standard deviation 和 coefficient of variation。`scripts/compare-llama-bench.py` 只用于生成平均吞吐对比表，不承担置信区间判断。
5. 该比较脚本需要 `GitPython` 和 `tabulate`。性能环境在执行前安装这两个工具依赖，不把它们加入项目运行时依赖。

保留门槛：

- 主指标 mean 至少提升 3%。
- baseline 和 candidate 的 coefficient of variation 均不超过 2%。超过时重新控制机器噪声并重测，不增加样本选择规则。
- 四个 ABBA blocks 中至少三个 block 的 candidate mean 高于对应 baseline mean。
- 非目标 `pp/tg` point regression 不超过 1%。
- load time 或 RSS 增加超过 5% 时，必须有明确、稳定的目标收益。
- correctness 和 PPL 不允许退化。
- 未达到门槛的优化删除，不增加 runtime knob 掩盖结果。

推荐模型集合：

- 标准 Llama 或 Qwen 1B Q6_K，用于 #22345。
- gpt-oss 20B MXFP4，用于 #19288 和 #19738。
- Llama 8B Q4_K_M 和 Qwen3 4B，用于长上下文 FA。
- F16/BF16 小模型，用于 AVX512-FP16/BF16。
- IQ2/IQ3 模型或量化工具 workload，用于 #23595。
- Q1_0、NVFP4、Q2_0 模型只在公共格式 ABI 项目获批后加入。

## 12. Stop conditions

出现以下情况时停止当前批次并拆分设计：

- 引入新的公共量化 type 或 llama ftype。
- 需要改变现有私有 GGUF type ID。
- 引入新 backend、外部依赖或自动下载。
- fast path 没有 generic/reference oracle。
- open PR 相互覆盖同一函数或同一 dispatch 区间。
- 需要默认开启在 mixed offload 上有明确回退的 repack。
- CPUID 支持但 OS state、XCR0 或每线程权限无法验证。
- 上游性能数字不能在目标 x86 开发机复现。

## 13. 开发顺序

建议按以下顺序执行：

1. X0 build 和 dispatch 安全。
2. X1 AVX2 Q6_K。
3. X2 MXFP4 E8M0 LUT。
4. 比较整模型结果，决定是否继续。
5. X3 NORM SIMD 和 scalar fix。
6. X4a concat、X4b IQ init、X4c RMS fusion，三项分别提交。
7. X5a tiled FA、X5b 最终 SIMD GEMM、X5c F16 conversion。
8. 目标机支持 AMX 时，先执行 X8a correctness，再分别执行 X8b、X8c、X8d。
9. 目标机匹配时执行 X7 affinity 或 AVX512-FP16。
10. CPU-only 有明确需求时执行 X6 MXFP4 repack opt-in。
11. 保持 open PR watchlist，不在 merge 前混入稳定批次。
12. 公共格式、ZenDNN 和 NUMA 只作为独立项目启动。

完成 X0 到 X2 后，重新冻结 binary、模型和目标机信息，再决定下一轮。这样可以先获得直接 x86 收益，同时把公共格式、optional backend 和大规模 runtime 重构留在独立决策边界内。
