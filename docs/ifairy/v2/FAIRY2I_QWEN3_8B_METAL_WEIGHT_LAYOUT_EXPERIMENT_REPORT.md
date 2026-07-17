# Fairy2i Qwen3-8B W1 Metal 权重布局综合实验报告

日期：2026-07-17

分支：`codex/metal`

生产 winner 基线提交：`aa2eebf2d1503ca4b9c73d20180416b503f07dac`

## 1. 最终结论

最终保留的 Metal 布局仍是当前 `bundle_v1` 的：

```text
m16_q4_branch_lane
```

也就是每个 M64×K64 tile 使用：

```text
codes : [m16][q4][branch U/W][lane16] = 2048 B
scales: [branch U/W][real/imag]        =    8 B
total                                       2056 B/tile
```

没有新候选同时稳定提高 pp512 和 tg128。最接近的 `m64_row_q4_lane_joint` 在受控 ABBA 中只有
pp512 `+0.257%`，但 tg128 `-2.469%`；差异低于 1.5% 时应选择更简单、更小、地址计算更少的布局，因此
不替换现行方案。

清理后的最终生产二进制对 winner 运行 3 个独立 R5 进程、共 15 个样本，结果为：

| workload | 最终均值 |
|---|---:|
| pp512 | **191.799 tok/s** |
| tg128 | **25.160 tok/s** |

绝对数值受到本机当前沙箱/频率状态影响；布局排名以同 binary、相邻运行及 ABBA 为准。实验中曾出现
M8 joint 的 R3 筛选值 `201.03 / 27.79`，但 R5 复测降为 `190.70 / 25.16`，证明该高值是瞬态，不能作为
布局收益。

## 2. 实验范围和硬约束

本轮只排名 Qwen3-8B learned-scale W1 的 Metal 路径，ARM、W2 和改变数值语义的量化方案不参与排名。

所有进入全模型性能表的候选满足：

- 来自同一 checkpoint：`/Users/a1806/llama/qwen_1bit_scale/checkpoint-5639`；
- 252 个 Fairy2i linear 全部完成转换；
- 除 inline16 外，均保持 `2048 B codes + 8 B scales`；
- code 与 scale 回解后和同一 V2 GGUF 位级一致；
- 不在模型加载后做全量转码或保留第二份 runtime packed weights；
- 用真实完整 GGUF、真实模型图、真实 Metal kernel 测 pp512/tg128；
- 默认 warmup，`-ngl 99 -t 8 -fa 1`，不并发运行多个 `llama-bench`。

作为数值基准的 V2 文件是：

```text
/Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-learned-scale.gguf
```

生产 winner 文件是：

```text
/Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1.gguf
```

## 3. winner 的具体存储设计

### 3.1 tile 内排列

每个 code byte 打包同一输出行连续的四个 K 位置：

```text
byte = c[k+0] | c[k+1] << 2 | c[k+2] << 4 | c[k+3] << 6
```

其中 `c` 是 2-bit phase code。一个 M64×K64 tile 的索引是：

```text
m16  = row_in_tile / 16
lane = row_in_tile % 16
q4   = k_in_tile / 4
slot = m16 * 16 + q4

codes[tile][slot][branch][lane]
```

W1 的 branch 顺序固定为 `U0,W0`。每个 tile 的四个 FP16 scale 采用：

```text
[U.real, U.imag, W.real, W.imag]
```

scale 独立放在 side tensor 中，作用域是 M64×K64。GGUF tensor 起始地址按 64 B 对齐。

### 3.2 文件与内存成本

| 项目 | 数值 |
|---|---:|
| 完整 GGUF | 3,368,144,832 B（3.1368 GiB） |
| Fairy2i codes | 868,220,928 B |
| Fairy2i scales | 3,391,488 B |
| scale 占 Fairy2i payload | 0.3891% |
| V2 完整 GGUF | 3,581,797,824 B |
| 相对 V2 减少 | 213,652,992 B（5.965%） |

生产路径直接 mmap/上传文件中的 codes 和 scales，不分配整权重重排副本。

## 4. Metal kernel 如何使用 winner 布局

### 4.1 tg128 / decode

decode 使用 `kernel_fairy2i_bundle_w1_bf16_tile8x1_w8_full_nobias_fc_simd`：

- 一个 128-thread threadgroup 处理 8 个输出行；
- `q4` 直接对应一个 byte 中连续四个 K code；
- 对一个 8-row 子块，U 和 W 分别用连续 `uint2` 载入；
- 每行只需从相应 32-bit word 中做 byte shift，再做 2-bit phase 提取；
- scale 用一次连续 `half4` 访问得到，并拆成 `wr/wi`；
- activation 的四个连续复数值与同一 code byte 的四个 phase 正好对应。

branch-plane 看起来比 U/W joint 多一条 load stream，但它允许 U、W 各自成为自然对齐的 8-row 向量，
避免 joint 候选中的 16-bit lane 选择和额外地址计算。最终 TG 的瓶颈也不在这 8 B/tile scale 上。

### 4.2 pp512 / prefill

prefill 使用 `kernel_fairy2i_bundle_w1_half_mma32x16_k16`：

- threadgroup 每次计算 M32×N16×K16；
- 一个 thread 拥有一个 `(row, q4)`，读 U/W 两个 byte；
- 每个 byte 展开四个连续 K phase，直接填入 simdgroup matrix coefficient tile；
- M16 lane 分组使相邻线程访问相邻 byte，且不需要把 row-major 权重先转成临时 MMA panel；
- K64 tile 在文件中对同一 M block 连续，kernel 的 K loop 顺序读权重。

## 5. 完整候选定义

| ID | `code_order` | 物理含义 | bytes/tile |
|---|---|---|---:|
| S0 | `m16_q4_branch_lane` | 现行 M16、U/W branch-plane | 2056 |
| S1 | `m16_q4_lane_joint` | M16 内每 lane 紧邻 U/W | 2056 |
| S2 | `m8_q4_lane_joint` | M8 decode 子块内紧邻 U/W | 2056 |
| S3 | `m32_k16_m8_q4_branch_lane` | M32×K16 原生 256B packet，branch-plane | 2056 |
| S4 | `m32_k16_m8_q4_lane_joint` | M32×K16 原生 packet，U/W joint | 2056 |
| S5 | `m8_q4_branch_bitplane` | M8×K4 的 sign/axis bitplane | 2056 |
| S6 | `m64_row_q4_lane_joint` | 同一输出行 K64 连续，U/W joint | 2056 |
| S7 | `m32_k16_m8_q4_branch_lane_inline16` | S3 + 16B inline scale header | 2064 + 2B/tensor dummy |

S7 的 header 前 8 B 保存四个 half scale，后 8 B 是对齐 padding；side scale tensor 只保留一个 dummy half
以便临时实验图维持统一接口，kernel 的 scale 实际从 header 读取。

## 6. 正确性证据

每个完整 GGUF 都通过 `validate_fairy2i_bundle_v1.py`：

```text
linear_count       = 252
common_tensor_count= 147
tensor_count       = 651
canonical_sha256   = ae4e8890c419d637e3fd7b339ba7f730e401f14de00e5a2e89ba002cbb449709
```

校验器执行：

1. dense/common tensor 逐 byte 比较；
2. 每个候选按自己的物理 byte order 回到 canonical bundle；
3. canonical codes 与 V2 的 2-bit symbols 逐 tile 比较；
4. FP16 scale 按 uint16 bit pattern 比较；
5. 检查 tensor type、shape、64B 对齐、branch order 和 metadata。

临时实验 runtime 还对全部布局运行了：

- N=1 decode 算子测试；
- N=17 prefill 算子测试；
- Metal 输出与 scalar reference 比较；
- `test-fairy2i` 全部通过。

完成排名后，失败布局的 Metal function-constant 分支、runtime enum、dispatch 和临时 C++ tests 已删除；
最终生产 runtime 回到只接受 S0。

## 7. 测试环境和命令

```text
SoC:       Apple M4
RAM:       24 GiB
OS:        macOS 26.3 (25D125)
binary:    Release + Metal + Accelerate
commit:    aa2eebf2d1503ca4b9c73d20180416b503f07dac
threads:   8
GPU layers:99
FA:        enabled
```

最终 sweep 命令：

```bash
./build-rel-metal/bin/llama-bench \
  -m MODEL.gguf \
  -ngl 99 -t 8 -fa 1 \
  -p 512 -n 128 -r 5 -o json
```

方法：

- 先用 R3 淘汰明显失败方案；
- 全部完整模型再跑 R5；
- 运行间留出 10–20 秒；
- sweep 前后插入 S0 control；
- 对唯一可能的 challenger S6 做 `S0 → S6 → S6 → S0` ABBA；
- winner 清理后重新跑 3 个独立 R5 进程；
- `pmset -g therm` 全程没有 thermal/performance warning；
- 没有并发 `llama-bench`。

## 8. 全部完整 GGUF 的最终 R5 结果

下表是同一次顺序 sweep 的 R5 均值。百分比只以 sweep 开头的 control A 为顺序参考；最终判定使用后面的
ABBA，不把百分比当作严格配对置信区间。

| 布局 | pp512 | Δ vs control A | tg128 | Δ vs control A | 结论 |
|---|---:|---:|---:|---:|---|
| S0 control A | 192.306 | — | 25.183 | — | 基线 |
| S1 M16 joint | 192.213 | -0.049% | 25.034 | -0.592% | 无收益 |
| S2 M8 joint | 190.697 | -0.837% | 25.163 | -0.082% | 无收益 |
| S3 native branch | 187.776 | -2.356% | 25.144 | -0.155% | PP 回退 |
| S4 native joint | 188.800 | -1.823% | 25.265 | +0.323% | TG 差异不复现，PP 回退 |
| S5 bitplane | 186.321 | -3.112% | 25.099 | -0.334% | 最差 PP |
| S6 row joint | 193.857 | +0.806% | 24.875 | -1.225% | 进入 ABBA |
| S7 native branch inline16 | 187.910 | -2.286% | 25.111 | -0.289% | 与 S3 等价但更大 |
| S0 control B | 194.664 | — | 25.593 | — | 证明时段漂移 |

S3 和 S7 的直接 side/inline 对比：

| scale placement | 文件大小 | pp512 | tg128 |
|---|---:|---:|---:|
| side M64×K64 | 3,368,144,832 B | 187.776 | 25.144 |
| inline 16B header | 3,371,544,384 B | 187.910 | 25.111 |
| inline 相对 side | +3,399,552 B | +0.072% | -0.134% |

这证明 scale placement 对 end-to-end 性能没有可测收益；side 更小、更简单。

## 9. ABBA 最终判定

ABBA 顺序：

```text
S0 control B -> S6 row-joint #1 -> S6 row-joint #2 -> S0 control #2
```

| 布局 | 两进程 pp512 均值 | 两进程 tg128 均值 | 相对 S0 |
|---|---:|---:|---:|
| S0 `m16_q4_branch_lane` | 193.260 | 25.434 | — |
| S6 `m64_row_q4_lane_joint` | 193.757 | 24.806 | pp +0.257%，tg -2.469% |

S6 不是通用 winner：PP 收益小于 1.5% 选择阈值，TG 明确回退，且 row-joint 的每行 accessor 比 S0 的
M16 branch-plane vector load 更复杂。

## 10. 为什么 R3 曾错误指向 M8 joint

R3 筛选曾得到：

| 布局 | pp512 | tg128 |
|---|---:|---:|
| S0 | 192.182 | 25.001 |
| S1 M16 joint | 192.727 | 25.836 |
| S2 M8 joint | 201.033 | 27.792 |
| S3 native branch | 198.637 | 26.608 |
| S4 native joint | 189.641 | 25.100 |
| S5 bitplane | 187.329 | 25.280 |
| S6 row joint | 194.149 | 25.046 |

但 S2 的 TG 三个样本已经从 `28.26, 28.17` 降到 `26.95`；R5 后四个样本稳定在约 25.0。control B
也出现前两个 TG 样本 26.4、随后回到 25.0 的模式。由此可知首轮 GPU 频率/系统状态会显著抬高短测，
必须使用 R5、重复进程和 ABBA。

## 11. 每种布局的解释

### S1：M16 joint

把 U/W 从两个 16-lane plane 改成每 lane 一个 ushort。理论上可用单一 stream，但 decode 需要从
joint word 中选择 U/W byte；最终 PP/TG 都没有改善。branch-plane 的两次自然向量载入更适合现行 kernel。

### S2：M8 joint

物理块与 decode 的 8 行输出对应，筛选时看似最佳，但 R5 完全不复现。M8 增加了 subtile/地址层级，
prefill 的 M32 协同也被切得过细。

### S3/S4：M32×K16 native packet

一次 prefill 工作包恰好 256 B，并在其中分四个 M8 decode subtile。它在纸面上最贴近 kernel，但地址要
同时分解 `m32/k16/m8/q4`；现行 prefill thread mapping 读取单 `(row,q4)`，并不会一次消费整个连续 packet，
所以物理 256B 并没有转化为更少事务，反而增加索引算术。

### S5：bitplane

用 sign/axis mask 代替 packed q4 byte。虽然消除了部分 2-bit extraction，但需要 mask shift、组合和更多
寄存器，pp512 为所有候选最低。说明 code unpack 不是当前主要瓶颈。

### S6：row-joint

同一输出行的 K64 完全连续，直觉上适合 TG；实际 TG 反而稳定下降。现行 decode 是 8 行协同而不是单行
串行，row-major 破坏了跨行向量载入。PP 有约 0.26% ABBA 增益，但在噪声/选择阈值内。

### S7：inline scale

scale 只有 payload 的 0.3891%。把它放入 16B header 会增加 padding，并改变 2048B tile stride；side 与
inline 的 end-to-end 差异不足 0.2%，最终保留 side。

## 12. 本轮全部激进想法及处置

| 想法 | 是否产生完整 pp512/tg128 | 结果或未晋级原因 |
|---|---|---|
| branch-plane → joint-phase | 是，S1/S2/S4/S6 | 没有通用收益；row-joint TG 回退 |
| M8/M16/M32/M64 panel 扫描 | 是 | S0 的 M16 综合最好；M8/M32 不稳或 PP 回退；M64 TG 回退 |
| M32×K16 原生 256B packet | 是，S3/S4 | PP 回退 1.8–2.4% |
| sign/axis bitplane | 是，S5 | pp512 回退 3.1% |
| side scale → inline tile header | 是，S7 | +0.07% / -0.13%，文件更大，淘汰 |
| M/K traversal order 重排 | 否 | 只允许 M5/M6 winner 晋级；native packet 未胜出。当前 M-major 已使 K loop 连续 |
| 更细 packet 邻域搜索 | 已由 S1–S6 覆盖主要邻域 | 没有中心候选胜出，不继续做笛卡尔积 |
| expanded int8 selector（M9） | 否 | codes 从 868 MB 增至约 3.47 GB，估算完整 GGUF 5,972,807,616 B；未满足先有 ≥10% kernel 上限的前提 |
| PP/TG 双 codes（M10） | 否 | 最佳可组合上限仅约 PP +0.26%、TG +0%，远低于 15% 门槛；估算文件 4,236,365,760 B |
| runtime 全量转码 | 否 | 违反“文件布局直接被 Metal 消费”和零整权重副本约束 |
| 每 M32 复制 scale | 否 | scale 已仅占 0.389%，inline M64 都无收益；复制只增加带宽和文件大小 |
| scale 压缩/精度变化 | 否 | 改变数值语义，不属于本轮 byte-order 对比 |
| branch 单独采用不同数学顺序 | 否 | 会扩大变量；joint/branch 主问题已经被隔离测试 |

### 仍可作为独立 kernel 项目继续研究的方向

这些方向不再是“只改权重 byte order”，应另立基线：

1. 调整 prefill thread mapping，使一个 threadgroup 真正整块消费 256B native packet；本轮 S3 证明只改文件
   而不改计算映射不够。
2. 对 decode 做 GPU counter profile，确认 25 tok/s 稳态是内存带宽、occupancy 还是其它 graph op 限制；
   所有紧凑布局稳态都约 25 tok/s，暗示 wide-linear code extraction 不是主瓶颈。
3. 让 scale 由单 simd lane 读取并 `simd_broadcast`；这是 kernel load 策略实验，不需要再改 GGUF。
4. 对 PP-only 部署可重新研究 row-joint，但当前 +0.26% 小于噪声和维护成本，不应进入默认格式。
5. 若未来设计完全不同的 fused complex GEMM，可重新定义 packet；届时应从 kernel 消费顺序反推文件布局，
   而不是继续围绕当前 kernel 枚举排列。

## 13. 保留的复现代码

为避免把失败布局留在生产 runtime，本轮只保留离线转换/回解能力：

- `gguf-py/fairy2i/quant/tile64_v3_metal.py`：全部候选 pack/canonicalize/unpack；
- `gguf-py/experimental/convert_fairy2i_qwen3_metal_layout.py`：完整 checkpoint → 候选 GGUF；
- `gguf-py/validate_fairy2i_bundle_v1.py`：候选与 V2 的逐 tensor 位级校验；
- `gguf-py/tests/fairy2i/test_tile64_v3_metal.py`：不同 shape、全部 code order round-trip；
- `FAIRY2I_METAL_WEIGHT_LAYOUT_RESULTS_20260717.json`：原始 R5 样本、模型哈希和 ABBA 汇总。

实验转换示例：

```bash
cd gguf-py
../.venv/bin/python experimental/convert_fairy2i_qwen3_metal_layout.py \
  /Users/a1806/llama/qwen_1bit_scale/checkpoint-5639 \
  /tmp/qwen3-w1-m8-joint.gguf \
  --weight-layout bundle_v1 \
  --bundle-code-order m8_q4_lane_joint \
  --verbose
```

校验示例：

```bash
cd gguf-py
../.venv/bin/python validate_fairy2i_bundle_v1.py \
  /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-learned-scale.gguf \
  /tmp/qwen3-w1-m8-joint.gguf
```

非默认候选是离线实验产物，清理后的生产 runtime 会按设计拒绝它们。若要重新测 Metal，需要按 packer
中的 offset 重新接入临时 experimental reader；生产用户只应转换/使用默认 S0。

## 14. 最终推荐命令

```bash
cd /Users/a1806/llama/llama.cpp

./build-rel-metal/bin/llama-bench \
  -m /Users/a1806/llama/qwen_1bit_scale/qwen3-fairy2i-w1-bundle-v1.gguf \
  -ngl 99 -t 8 -fa 1 \
  -p 512 -n 128 -r 5
```

为了得到机器原生数据，建议在终端中关闭其它 GPU-heavy 应用，连续运行 3 个独立进程，并保留每次输出，
不要只使用第一个 TG 样本。

## 15. 原始结果位置

结构化结果随仓库保留在：

```text
docs/ifairy/v2/FAIRY2I_METAL_WEIGHT_LAYOUT_RESULTS_20260717.json
```

本地详细日志位于：

```text
tmp/fairy2i-layout-metal/baseline/
tmp/fairy2i-layout-metal/screen/
tmp/fairy2i-layout-metal/final/
tmp/fairy2i-layout-metal/abba/
tmp/fairy2i-layout-metal/winner/
tmp/fairy2i-layout-metal/conversion/
```

最终选择不是“没有找到新排列”，而是确认现行 S0 已处在当前 Metal kernel 的正确局部最优：它在保持最小
payload、零 runtime 转码和简单地址公式的同时，没有任何主要 workload 的稳定回退。
