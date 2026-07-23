# Fairy2i bundle-v1 NEON 优化实验报告

日期：2026-07-18
基线：`e7675ec3`
候选：本报告所在提交

## 1. 结论

最终方案直接以 `bundle_m64k64_v1` 为 NEON 的执行布局，不生成 CPU 私有重排副本：

- W1 保留已有的双 branch NEON 结构，并优化整数累加依赖链、scale 读取、地址递推和预取。
- W2 补齐原先缺失的 bundle-v1 NEON 路径，将 `U0/U1` 与 `W0/W1` 分成两个相邻 branch pair 执行；旧版本在 AArch64 上会完整回退到逐行逐 branch 标量内核。
- W1 bundle-v1 的 `N == 1` 默认改用动态 tile claim，默认 claim batch 为 4；旧的非 bundle 布局仍保持 opt-in，默认 batch 仍为 2。
- 没有加入 Apple、M4、SME 或 dotprod 专属分支；生产快路径只要求现有的 AArch64 NEON 条件，其他平台继续使用原有 AVX2 或标量回退。

Apple M4、CPU-only、8 线程、`R=3`、ABBA、相邻测试间隔 15 秒的整模型结果如下：

| 模型 | 负载 | 基线 A 均值 | 候选 B 均值 | 变化 |
|---|---:|---:|---:|---:|
| W1 Qwen3 bundle-v1 QKV | pp512 | 42.4810 tok/s | 43.5825 tok/s | +2.5928% |
| W1 Qwen3 bundle-v1 QKV | tg128 | 16.2074 tok/s | 20.4651 tok/s | +26.2706% |
| W2 Llama2 bundle-v1 | pp512 | 1.7411 tok/s | 23.1522 tok/s | +1229.7330%，13.2973x |
| W2 Llama2 bundle-v1 | tg128 | 1.4858 tok/s | 15.3419 tok/s | +932.5447%，10.3254x |

W2 的数量级提升主要来自补齐通用 AArch64 NEON 算法路径，而不是针对 M4 的窄化调参。W1 pp 的小幅提升来自内核细化，tg 的主要额外收益来自 bundle-v1 动态 tile 调度。

## 2. bundle-v1 与 NEON 的对应关系

bundle-v1 的 code 逻辑次序为：

```text
[physical_m64_k64][q4][branch][16 lanes]
```

scale 次序为：

```text
[physical_m64_k64][branch][real, imag]
```

一个逻辑 `m16` tile 映射到：

```text
physical_tile = (global_m16 / 4) * k_blocks + k_block
slot_base     = (global_m16 % 4) * 16
```

因此同一 `q4` 下相邻两个 branch 正好是连续的两个 16-byte 向量，两个 branch 的四个 FP16 scale 也连续。最终内核据此执行：

```text
activation LUT (shared)
        |
        +--> vld1q_u8_x2 -> branch 0/1 查表与 int16 累加 -> FP32 累加 -> store
        |
        +--> vld1q_u8_x2 -> branch 2/3 查表与 int16 累加 -> FP32 累加 -> vector add/store
```

W1 只执行第一种 pair，模板参数为 `<0, true, false, false>`；W2 依次执行：

- `U0/U1`：`<0, true, true, false>`；
- `W0/W1`：`<2, false, false, true>`。

布尔模板参数在编译期固定 branch 起点、虚部符号和 add 语义，避免热循环中的运行时分支。

## 3. 最终实现细节

### 3.1 W2 的成对 NEON 内核

新增统一的 `ggml_fairy2i_bundle_lut_qgemm_pair_neon<>`，同时服务 W1/W2：

- 每个 `q4` 用一次 `vld1q_u8_x2` 读取 32-byte 相邻 branch code。
- activation LUT 只加载一次，然后供两个 branch 的 `vqtbl1q_s8` 查表共享。
- 每个 branch 保持独立的 `ac/bc/ad/bd` int16 向量和，保证 codebook 语义不变。
- W2 的第二个 pair 用 `vld2q_f32` 读取交错 complex 输出，向量相加后用 `vst2q_f32` 写回；尾 tile 和 BF16 container 仍有完整回退处理。
- 非 AArch64 NEON、测试强制 scalar、无效 shape 等情况仍走原有路径。

### 3.2 缩短整数累加依赖链

原逻辑对同一 accumulator 连续执行两次 widening add。最终实现先用 `vaddl_s8` / `vaddl_high_s8` 合并两个 LUT 结果，再做一次 `vaddq_s16`：

```text
旧：sum = sum + a; sum = sum + b
新：sum = sum + widen(a + b)
```

在二补数 int16 语义下结果逐位一致，同时缩短同一 accumulator 的串行依赖链。pp128 ABBA 微实验为 `+0.7809%`。

### 3.3 scale 与地址访存

- 四个相邻 FP16 branch scale 用一次 64-bit load 和一次向量 FP16→FP32 转换读取，再按 lane broadcast。
- `physical_tile0`、`slot_base` 和首地址移出 K-block 循环。
- code 与 scale 地址用固定 stride 递推，不在每个 block 重算多层乘加索引。
- 每个 block 仅预取下一 block 的一条 code cache line；更远预取和第二条预取都没有稳定收益。

scale 向量化在 pp128 ABBA 中为 `+1.3131%`，一条下一 block 预取为 `+1.0481%`。

### 3.4 W1 bundle decode 调度

W1 的 `N == 1` 以前默认静态按线程切 tile。对于不同层宽度，尾部线程容易提前空闲。最终策略为：

- bundle-v1 W1：默认动态 claim，batch=4；
- 非 bundle W1：仍默认关闭动态 claim；显式启用时默认 batch=2；
- `GGML_FAIRY2I_W1_DYNAMIC_TILES=0` 可关闭，`=1` 可显式开启；
- `GGML_FAIRY2I_W1_DYNAMIC_TILE_BATCH=1|2|4` 仍可用于诊断覆盖，不增加新开关。

微实验中，动态 batch=2 相对静态为 `+14.6009%`，batch=4 再比 batch=2 快 `+3.6767%`；batch=1 比 batch=4 慢 `18.6606%`。

## 4. Profiling 证据

macOS `sample` 的旧 W2 bundle-v1 profile 中，collapsed top-of-stack 有 21758 个样本落在 `ggml_fairy2i_bundle_lut_qgemm_branch_scalar`。补齐 pair NEON 后，21726 个样本转移到 `ggml_fairy2i_bundle_lut_qgemm_pair_neon`，证明路由和主热点均按设计变化，而不是由其他图算子偶然提速。

原始 profile：

- `/tmp/fairy2i-neon-opt/path/w2-baseline.sample.txt`
- `/tmp/fairy2i-neon-opt/path/w2-pair.sample.txt`
- `/tmp/fairy2i-neon-opt/path/w2-pair.disasm.txt`
- `/tmp/fairy2i-neon-opt/path/vector-scale.disasm.txt`

## 5. 候选实验

除最终整模型结果外，候选均使用同模型、同负载的 A1→B1→B2→A2，测试之间至少间隔 15 秒。

| 实验 | B 相对 A | 决策 |
|---|---:|---|
| W2 scalar → 初版 pair NEON，pp16 | +912.4167% | 保留算法方向 |
| pair → 单 branch NEON，pp128 | -8.0429% | 拒绝单 branch |
| 运行时 pair 参数 → 模板特化，pp128 | +1.2651% | 保留模板特化 |
| 串行 widening add → 先 pair widen，pp128 | +0.7809% | 保留 |
| 标量 scale 转换 → pair scale 向量读取，pp128 | +1.3131% | 保留 |
| 强制 q4 unroll=2，pp128 | -2.0436% | 拒绝 |
| 无预取 → 下一 block 两条预取，pp128 | +1.0481% | 保留预取方向 |
| 预取距离 1 → 距离 2，pp128 | -0.4524% | 保留距离 1 |
| 两条预取 → 一条预取，pp128 | -0.0064% | 视为等价，选择更温和的一条 |
| W1 原版 → 当前内核，pp128 | +2.7823% | 保留 |
| W2 tg batch=4 → batch=1 | -14.9034% | 保留 batch=4 |
| W2 tg batch=4 → batch=2 | -2.1149% | 保留 batch=4 |
| W1 静态 → 动态 batch=2 | +14.6009% | 保留动态调度 |
| W1 动态 batch=2 → batch=4 | +3.6767% | 默认 batch=4 |
| W1 动态 batch=4 → batch=1 | -18.6606% | 拒绝 batch=1 |

`GGML_RESTRICT` 候选生成的 `libggml-cpu.dylib` 与前一候选字节相同，说明编译器已经得到等价 alias 信息，因此不保留无效标注，也没有为它重复跑 benchmark。

对应原始目录均位于 `/tmp/fairy2i-neon-opt/abba-*`。

## 6. 正式 ABBA 明细

环境：Apple M4，macOS 26.3，Apple clang 21.0.0，8 CPU 线程，CPU-only，LUT16，默认 warmup，`R=3`。A 为冻结的 `e7675ec3`，B 为最终候选。

| 模型/负载 | A1 | B1 | B2 | A2 | A 均值 | B 均值 | 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 pp512 | 42.576235 | 43.650645 | 43.514288 | 42.385787 | 42.481011 | 43.582466 | +2.5928% |
| W1 tg128 | 16.194452 | 20.373454 | 20.556810 | 16.220269 | 16.207361 | 20.465132 | +26.2706% |
| W2 pp512 | 1.741871 | 23.154163 | 23.150318 | 1.740368 | 1.741119 | 23.152240 | +1229.7330% |
| W2 tg128 | 1.493170 | 15.346646 | 15.337153 | 1.478498 | 1.485834 | 15.341900 | +932.5447% |

原始 JSON：

- `/tmp/fairy2i-neon-opt/final-abba-w1-pp512/`
- `/tmp/fairy2i-neon-opt/final-abba-w1-tg128/`
- `/tmp/fairy2i-neon-opt/final-abba-w2-pp512/`
- `/tmp/fairy2i-neon-opt/final-abba-w2-tg128/`

## 7. 正确性与误差

### 7.1 单元测试

`test-fairy2i` 全部通过，包括：

- ARM accumulate/quantize；
- LUT add、same-lane extremes、layout guardrails；
- W1/W2 各 336 个 variant cases；
- W1 768 个 N=1 动态和 384 个 N>1 fallback cases；
- W2 1664 个 N=1 动态和 832 个 N>1 fallback cases；
- 15 个 bundle LUT 执行 case，含 W1 默认 bundle dynamic batch=4、scalar/ISA 对照。

### 7.2 W1 PPL

WikiText-2 raw、8 chunks、4×512、CPU-only：

| 实现 | PPL |
|---|---:|
| 冻结 CPU 基线 | 39.3324 ± 3.01197 |
| 最终 NEON | 39.3324 ± 3.01197 |

八个逐 chunk 累计值也完全相同。原始日志：

- `/tmp/fairy2i-neon-opt/correctness/w1-baseline-chunks8.log`
- `/tmp/fairy2i-neon-opt/correctness/w1-neon-chunks8.log`

### 7.3 W2 scalar/NEON

同一当前二进制、同一个 512-token chunk，用 `GGML_FAIRY2I_TEST_FORCE_SCALAR=1` 生成标量参考：

| 实现 | PPL |
|---|---:|
| scalar | 17.7752 ± 4.55811 |
| NEON | 17.7353 ± 4.55160 |

NEON 相对 scalar 为 `-0.2245%`，没有 PPL 退化。压缩全词表 log-prob 的 KL 对照结果为：

- mean KLD：0.001928 ± 0.000270；
- median KLD：0.000640；
- same top probability：99.608%；
- mean PPL ratio：1.001513 ± 0.006771。

差异来自浮点归并顺序：scalar 逐 branch 完成全部 K-block 后写回，NEON 为利用 bundle 相邻存储按 branch pair 在 K-block 内交错归并；整数 LUT 和、code/scale 映射均由逐 case 测试覆盖。原始日志位于 `/tmp/fairy2i-neon-opt/correctness/`。

## 8. ARM 兼容性与回归门

除本机 native build 外，额外配置了不使用本机 ISA 的通用构建：

```bash
cmake -S . -B build-rel-fairy2i-armv8 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8-a \
  -DGGML_FAIRY2I=ON \
  -DGGML_FAIRY2I_CPU=ON \
  -DGGML_FAIRY2I_CPU_LUT=ON \
  -DGGML_FAIRY2I_CPU_ARM_DOTPROD=OFF \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF
```

该构建的 `test-fairy2i` 全部通过，path capability 显示 `NEON`，没有 dotprod/SME。Legacy direct 与 legacy LUT 回归门也全部通过；针对本次触及 C++ 文件的 `git clang-format --diff` 和 `git diff --check` 均无输出。本机工具链没有可用的 `clang-tidy`，因此没有伪报该项检查。由此确认最终代码没有把 M4 特性变成隐式最低要求。

## 9. 复现最终 benchmark

W1 pp512：

```bash
env GGML_FAIRY2I_LUT=1 \
    GGML_FAIRY2I_LUT_IMPL=lut16 \
    LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W1=1 \
    DYLD_LIBRARY_PATH=/Users/a1806/.codex/worktrees/0a47/llama.cpp/build-rel-fairy2i/bin \
  /Users/a1806/.codex/worktrees/0a47/llama.cpp/build-rel-fairy2i/bin/llama-bench \
    -m /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1-qkv.gguf \
    -ngl 0 --device none -t 8 -p 512 -n 0 -r 3
```

W2 pp512：

```bash
env GGML_FAIRY2I_LUT=1 \
    GGML_FAIRY2I_LUT_IMPL=lut16 \
    LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 \
    DYLD_LIBRARY_PATH=/Users/a1806/.codex/worktrees/0a47/llama.cpp/build-rel-fairy2i/bin \
  /Users/a1806/.codex/worktrees/0a47/llama.cpp/build-rel-fairy2i/bin/llama-bench \
    -m /Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-bundle-v1.gguf \
    -ngl 0 --device none -t 8 -p 512 -n 0 -r 3
```

将 `-p 512 -n 0` 改为 `-p 0 -n 128` 即为 tg128。正式比较应继续使用 A1→B1→B2→A2，并在相邻测试间等待至少 15 秒。

## 10. 保留边界

- 没有为 M4 加 CPU 型号判断、SME kernel、固定 cache-size blocking 或 Apple 专用调度。
- 没有保留单 branch、强制展开、预取距离、多预取线等实验开关。
- 没有为 CPU 生成 bundle-v1 的二次重排副本，加载阶段的内存占用不增加。
- 当前默认 claim batch 是 Apple M4、8 线程、这两份模型上的结果；其他 ARM 设备可用现有环境变量诊断覆盖，但不应在没有 ABBA 数据时继续添加设备特判。
