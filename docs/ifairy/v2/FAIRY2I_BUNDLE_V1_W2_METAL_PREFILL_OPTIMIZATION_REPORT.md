# Fairy2i bundle_v1 W2 Metal prefill 优化实验报告

日期：2026-07-18

分支：`codex/metal`

实验基线：`affba11124c6e4f1b87d97d72a2713dbba9ae136`

机器：Apple M4，10 核 CPU，24 GiB 统一内存，macOS 26.3

## 1. 结论

本轮把 W1 优化中可迁移的思路逐项应用到 W2 `bundle_v1` Metal kernel，并针对 W2 的四分支、K8 MMA
映射做了额外实验。最终只保留一项能够稳定复现的优化：

- 将 W2 bundle code 的行、q4 子组和 M16 子块索引移出 K8 内层循环；
- 每个 W64 physical tile 只构造一次 `code_ptr`；
- 后续八个 K8 step 通过固定偏移读取四个 branch byte，并将指针递增 128 B；
- 保持现有 32 行 × 16 token × K8 MMA、threadgroup activation 共享、FP32 累加和输出顺序不变。

最终有效 ABBA 中：

| workload | 原始内核 | 最终内核 | 变化 |
| --- | ---: | ---: | ---: |
| pp512 | 173.036 tok/s | 174.402 tok/s | **+0.790%** |
| tg128，两组 ABBA 汇总 | 21.7961 tok/s | 21.7942 tok/s | **-0.0086%** |

tg128 使用独立 decode kernel，本轮修改对它没有路径影响；两组 tg ABBA 的方向相反，汇总结果等价于零。

质量验证中，双方 8-chunk WikiText-2 PPL 都为 `27.6237 +/- 2.28037`；1-chunk 全量 logits 文件逐字节
相同，固定 seed 的 32-token 生成也逐字节相同。

未保留 W1 式 activation 直读、scale hoist、K8 强制展开、四 SIMD 解包、FP16 系数生成、双缓冲、
function constant、输出展开等实验代码。生产代码中没有新增环境变量、布局选项、权重副本或模型文件。

## 2. 实验对象与协议

模型：

```text
/Users/a1806/llama/llama2_7b_int8_activate/fairy2i-llama2-7b-checkpoint5998-bundle-v1.gguf
```

模型 SHA-256：

```text
160116d3d8959cb500eaa47ffaa82c3de4cbc4ccf52c8b1604a878b921d9259d
```

构建配置：Release、shared libraries、Metal、embedded Metal library、Fairy2i；CPU Fairy2i LUT 未启用。

公共 benchmark 参数：

```text
LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1
threads=8
batch=2048
ubatch=512
gpu_layers=99
flash_attn=1
mmap=1
```

性能对比全部遵守以下协议：

1. 顺序固定为 A1 → B1 → B2 → A2；
2. 每条正式腿前先运行同一 binary 的 R1 warm leg；
3. 正式腿使用 R5；
4. 任意两次 benchmark 至少间隔 15 秒；
5. 不并发运行 `llama-bench`；
6. 出现腿内突降或腿间硬件状态阶跃时，整组作废并重跑；
7. 只用同一 ABBA 组内的相对值判断候选，不跨时段比较绝对 tok/s。

驱动脚本：

```text
perf/scripts/run_fairy2i_metal_bench.sh
```

原始 A binary 位于独立 worktree：

```text
/tmp/fairy2i-w2-abba/baseline/build-rel-metal/bin/llama-bench
```

最终 B binary：

```text
/Users/a1806/.codex/worktrees/0a47/llama.cpp/build-rel-metal/bin/llama-bench
```

## 3. W2 bundle 内存访问与最终实现

W2 `bundle_v1` 的每个 M64 × K64 physical tile 使用：

```text
codes[physical_tile][m16][q4][branch][row_lane]
```

其中：

- `m16`：M64 中的 16 行子块；
- `q4`：K64 中连续四个 K code；
- `branch`：`U0,U1,W0,W1`；
- `row_lane`：M16 内的行；
- 每个 byte 保存同一行、同一 branch 的四个连续 2-bit K code。

W2 prefill 每个 threadgroup 计算 M32 × N16，K tile 为 8。前 64 个线程按
`coeff_row × q4_local = 32 × 2` 展开完整 M32 × K8 系数 tile。

### 3.1 原实现的内层地址计算

原实现每个 K8 step 都重新计算：

```text
coeff_row
q4_local
row_in_m64
m16
row_lane
q4
slot
code_base
```

`code_base` 包含 physical tile、slot、branch 和 row lane 的多级 64-bit 乘加。一个 W64 block 有八个 K8
step，所以相同的行映射和大部分地址链会被重复八次。

### 3.2 最终实现

最终实现将与 K8 无关的索引移到 `wb` 循环之外，并在每个 physical tile 开始时建立：

```metal
device const uchar * code_ptr =
    codes + physical_tile * 64 * 4 * 16 +
    (m16 * 16 + q4_local) * 4 * 16 + row_lane;
```

每个 K8 step 仍读取：

```metal
code_ptr[0]
code_ptr[16]
code_ptr[32]
code_ptr[48]
```

这四个地址对应相同 q4/row 的 `U0,U1,W0,W1`。一个 K8 step 消费两个相邻 q4，因此随后执行：

```metal
code_ptr += 2 * 4 * 16; // 128 B
```

该改变降低了前 64 个解包线程每个 K8 step 的整数和 64-bit 地址生成开销，同时保持原有 coalescing、code
读取次数、scale 读取、threadgroup scratch、barrier 和 MMA 序列不变。它不要求转换器或加载器重排权重，
也不创建第二份 GPU 权重。

### 3.3 测试覆盖补充

原测试已有 W2 bundle decode 与 `N=17` 尾块 prefill。本轮新增 `M=128,K=64,N=16` 的完整 prefill tile
用例，确保最终热点路径被直接执行，而不是只依赖尾块用例间接覆盖。

## 4. Profiling 结果

使用 `ifairy-w2-m4-shader` Instruments 模板采集当前 code-pointer 版本 pp512 R1。原始 trace 在完成汇总后因
体积约 1.4 GiB 已清理，保留的 Shader Timeline XML 为：

```text
/tmp/fairy2i-w2-opt/profile-codeptr/profile-codeptr.pp.shader-intervals.xml
```

该 profile 在 Instruments 开销下为 153.228 tok/s。Shader Timeline 汇总：

| shader | 合计时间 | interval 数 | 占采样 shader 时间 |
| --- | ---: | ---: | ---: |
| `kernel_fairy2i_bundle_w2_half_mma32x16` | 774.495 ms | 737 | 81.6% |
| flash attention | 37.298 ms | 43 | 3.9% |
| `kernel_mul_fuse_1` | 34.846 ms | 62 | 3.7% |
| activation half staging | 19.057 ms | 161 | 2.0% |

结论是 W2 主计算 kernel 明确占主导，而一次性 activation staging 只占约 2%。因此继续复制 activation、
修改 staging 布局或牺牲共享来减少 barrier 都不合理；应优先降低主 kernel 的地址/解包或 MMA 开销。

完整 GPU counter XML 导出会产生超过 1.3 GiB 的单个中间文件，因此在确认 shader 热点后中止该导出。
该不完整导出和另一份约 712 MiB 的候选 profile 原始 trace 也已清理；没有把大体积 trace、counter XML 或
模型文件放入仓库。

## 5. 候选实验结果

下表只列有效 ABBA；标为“噪声”的候选即使算术均值略正，也因幅度、异常样本或最终组合复验而被删除。

| 候选 | A 均值 | B 均值 | pp512 变化 | 决策 |
| --- | ---: | ---: | ---: | --- |
| W1 式 K-major activation 直读 | 152.951 | 121.763 | -20.391% | 删除 |
| code 索引外提 + pointer 递增，首次有效复验 | 152.712 | 154.446 | +1.135% | 暂留 |
| scale 从 K8 提到 W64 生命周期 | 151.214 | 151.365 | +0.100% | 噪声，删除 |
| 强制展开八个 K8 step | 157.109 | 157.055 | -0.034% | 删除 |
| `blocks/act_rows` function constants | 153.455 | 154.027 | +0.373% | 单个 158.499 高值驱动，删除 |
| 四 SIMDgroup 平分系数展开 | 140.768 | 140.846 | +0.055% | 噪声，删除 |
| FP16 系数构造 | 166.777 | 166.635 | -0.085% | 删除 |
| coefficient/activation 双缓冲，有效重跑 | 146.446 | 146.392 | -0.037% | 删除 |
| activation staging 单线程单元素 | 143.803 | 143.932 | +0.090% | 噪声，删除 |
| 输出四项固定展开，有效重跑 | 144.119 | 144.165 | +0.032% | 噪声，删除 |
| 整 tile `full_cols` 专用化 | 145.817 | 145.983 | +0.114% | 最终组合未复现，删除 |
| 所有微专用化组合对 code-pointer，有效重跑 | 177.651 | 177.569 | -0.046% | 全部删除 |
| 原始内核对最终 code-pointer，最终正式组 | 173.036 | 174.402 | **+0.790%** | **保留** |

### 5.1 W1 activation 直读为何在 W2 失败

W1 的 K16 direct-activation 路径让每个 SIMDgroup 从 K-major activation 直接读自己的 N16 tile，可以删除
共享 activation scratch。W2 每个 K8 的 MMA 工作更少，四个 SIMDgroup 对同一 activation 的四次读取无法被
足够计算隐藏。有效 ABBA 为 `-20.391%`，与此前 W2 SIMD-local activation 实验的方向一致。因此 W2 必须
保留 threadgroup activation 共享，不能机械复制 W1 的最终内存路径。

### 5.2 为什么其他小优化没有保留

- scale hoist：同一 W64 scale 很可能已由 L1/compiler 缓存；延长八个 float scale 的寄存器生命周期抵消收益。
- K8 unroll：编译器已能处理固定八次循环，强制展开只增加指令窗口。
- 四 SIMD 解包：减少每线程工作，但相邻线程重复 code/scale 读取，最终只有 +0.055%。
- FP16 系数：scratch 最终虽为 FP16，但 M4 上 half 构造没有降低热点，且会增加中间舍入风险。
- 双缓冲：理论上减少 barrier，但额外 2.5 KiB threadgroup scratch 和更长生命周期抵消收益，说明 barrier
  不是当前 limiter。
- function constant、边界删除和输出展开：单项均低于 0.12%，组合对 code-pointer 为 -0.046%。为避免
  pipeline 数量和代码复杂度，不保留。

以下组因硬件状态阶跃作废，没有用于排名：首次 code-pointer ABBA、首次双缓冲 ABBA、首次输出展开 ABBA、
首次最终微优化组合 ABBA。对应 artifact 仍保存在 `/tmp/fairy2i-w2-opt/` 便于审计。

## 6. 最终性能

### 6.1 pp512

artifact：

```text
/tmp/fairy2i-w2-opt/abba-original-vs-final-pp-20260718/
```

| leg | binary | tok/s |
| --- | --- | ---: |
| A1 | 原始 `affba111` | 173.126149 |
| B1 | 最终 code-pointer | 174.262906 |
| B2 | 最终 code-pointer | 174.541769 |
| A2 | 原始 `affba111` | 172.945836 |

两腿均值：

```text
A = 173.035993 tok/s
B = 174.402337 tok/s
B/A - 1 = +0.7896%
```

### 6.2 tg128

由于 tg 绝对值在长时间测试中缓慢漂移，执行了两组完整 ABBA：

```text
/tmp/fairy2i-w2-opt/abba-original-vs-final-tg-20260718/
/tmp/fairy2i-w2-opt/abba-original-vs-final-tg-rerun-20260718/
```

| 组 | 原始 A | 最终 B | 变化 |
| --- | ---: | ---: | ---: |
| 第一组 | 21.802653 | 21.730317 | -0.3318% |
| 第二组 | 21.789566 | 21.858170 | +0.3149% |
| 两组汇总 | 21.796110 | 21.794244 | **-0.0086%** |

方向在重跑中反转，汇总差异小于 0.01%，符合“prefill-only 修改不影响 decode”的预期。

## 7. Correctness 与质量验证

### 7.1 单元与加载测试

```text
LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1 build-rel-metal/bin/test-fairy2i
All tests PASSED
```

```text
ctest --test-dir build-rel-metal --output-on-failure -R fairy2i
2/2 passed
```

新增 W2 bundle `N=16` 完整 tile 与既有 `N=17` 尾块均通过。

### 7.2 WikiText-2 PPL

数据：

```text
/tmp/fairy2i-w1-ppl-compare-20260718/data/wikitext-2-raw/wiki.test.raw
SHA-256 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
```

参数：`ctx=512,batch=2048,ubatch=512,chunks=8,threads=8,ngl=99,FA=on`。

双方逐 chunk 序列完全一致：

```text
[1]17.5726,[2]29.7276,[3]27.1913,[4]29.8912,
[5]28.6302,[6]28.2372,[7]27.5877,[8]27.6237
```

最终结果均为：

```text
PPL = 27.6237 +/- 2.28037
```

日志：

```text
/tmp/fairy2i-w2-opt/ppl-final-20260718/A-original-ppl.log
/tmp/fairy2i-w2-opt/ppl-final-20260718/B-final-ppl.log
```

### 7.3 Logits 与固定生成

1-chunk 全 logits：

```text
A SHA-256 = c209ab0f6d71b85eb77a4b724ac966aafe5cb4e43d73afed3d850d37bbc1dddf
B SHA-256 = c209ab0f6d71b85eb77a4b724ac966aafe5cb4e43d73afed3d850d37bbc1dddf
cmp = IDENTICAL
```

固定 `seed=1234,temp=0` 的 32-token 生成：

```text
A SHA-256 = a2603103bef2500884ead595c2976b876bb0a8a6a783515a6cab1422cdeea072
B SHA-256 = a2603103bef2500884ead595c2976b876bb0a8a6a783515a6cab1422cdeea072
cmp = IDENTICAL
```

## 8. 静态检查

- `git diff --check`：通过；
- 针对 `tests/test-fairy2i.cpp` 的 `git clang-format --diff`：无修改；
- Homebrew LLVM 22.1.6 `clang-tidy`：退出码 0；报告的是该大型既有测试文件和公共 header 中的历史 warning，
  新增单行测试 case 没有产生新诊断；
- 对 merge-base 的全分支 clang-format 检查还显示 `ggml-metal-device.cpp` 有一个既有空行差异，本轮未触碰
  或修改该无关文件。

## 9. 后续若要继续提升的边界

当前 `bundle_v1` kernel 内不改变数值和布局的低风险空间已经很小。下一阶段若要获得明显增益，应把重点放在
转换布局，而不是继续堆叠亚 0.1% 的 shader 微专用化：

1. 为 Metal 定义按 M32 × K8 消费顺序连续的 W2 packet，使每个 SIMD 解包子块能够用对齐 vector load
   获取 joint branch code，进一步减少 gather/address 指令；
2. 评估将每个 physical tile 的 8 个 scale 内联到首个 packet，权衡重复 scale 与独立 buffer 访问；
3. 对 joint8 `U0|U1<<2|W0<<4|W1<<6` 布局做端到端转换，而不是运行时重排；
4. 只有新布局能减少实际 code load 或 coefficient 指令时，才重新考虑 M32/K8 以外的 tile；现有 row64、K16、
   K32 历史 sweep 已没有稳定优势；
5. 预解码 FP16 系数会显著扩大模型并增加带宽，不应在没有端到端容量/带宽模型的情况下采用。

这些方向都需要新的 GGUF layout/version、转换器和位级等价验证，超出本轮“在现有 bundle_v1 W2 kernel 内
迁移并筛选 W1 类优化”的范围。
