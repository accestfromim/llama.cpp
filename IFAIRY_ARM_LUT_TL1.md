# iFairy 2-bit 复数模型 · ARM NEON LUT (TL1) 适配设计

本文档参考 BitNet 仓库的 ARM TL1 LUT 方案（见 `../BitNet/docs/lut-arm.md`），给出在 llama.cpp 中为 iFairy 2-bit 复数量化模型接入 LUT 路径的整体设计、函数拆分与数据流说明。

## 1. 背景与目标
- 现状：iFairy 在 CPU 路径上已实现 2-bit 复数权重（`block_ifairy`）与 q16 激活的 NEON `sdot` 点积（`ggml_vec_dot_ifairy_q16_K`，位置 `ggml/src/ggml-cpu/arch/arm/quants.c`）。当前 `mul_mat` 仍逐块解码权重/激活，吞吐受限。
- 目标：复用 BitNet TL1 的“激活预量化 + 查表内核”思路，降低解码与访存开销，在 ARMv8.2+NEON/DOTPROD 设备上为 iFairy 提供高吞吐的 matvec/matmul 内核。
- 约束：保持 iFairy 复数语义（{−1,+1,−i,+i} × `d_real`/`d_imag`），输入/输出仍为 GGML 现有的 `GGML_TYPE_IFAIRY`（权重）与 `GGML_TYPE_IFAIRY_Q16`（激活）。

## 2. 现有量化与计算路径回顾
- 权重量化（`ggml/src/ggml-quants.c:quantize_row_ifairy_ref`）
  - `block_ifairy` 布局：`qs[QK_K/4]` 存 256 个 2-bit 码，`d_real`、`d_imag` 为 FP16 缩放。
  - 编码规则：`00→-1`、`01→+1`（作用于实部），`10→-i`、`11→+i`（作用于虚部），每行共享同一对缩放因子。
- 激活量化（`ggml/src/ggml-cpu/arch/arm/quants.c:quantize_row_ifairy_q16`）
  - `block_ifairy_q16`：`x_real[256]`、`x_imag[256]` 为 int8，`d_real`、`d_imag` 为 FP16。
  - 量化标尺分别取 real/imag 的 `127/max_abs`，保证共轭乘法时实部与虚部各自对齐。
- 当前乘法（同文件的 `ggml_vec_dot_ifairy_q16_K`）
  - 运行时用 `tbl` 把 2-bit 码映射到 `{wr, wi}`，与 `x_real/x_imag` 做 4 路 `sdot`，最后按：
    - `real = d_w_r * (wr·ar) - d_w_i * (wi·ai)`
    - `imag = d_w_r * (wr·ai) + d_w_i * (wi·ar)`
  - 优势：语义正确；劣势：每次都要解码权重且未做激活重排。

## 3. BitNet ARM TL1 LUT 核心思路（摘自 `BitNet/docs/lut-arm.md`）
1. 预处理：线程 0 先对激活做 `per_tensor_quant` 得到单个 `lut_scale`，随后 `lut_ctor<K>` 生成 `QLUT`（大小约 `K/2*32` 字节，按 nibble 排列）。
2. 工作区：`QLUT` + `lut_scales`（可选 fp16 缓存）；K 维与内核严格匹配。
3. 内核：`tbl_impl_*` 用 `vqtbl1` 直接查 `QLUT`，避免在主循环解码权重；`qgemm_lut_*` 负责 K 分块与反量化。
4. 入口：`ggml_bitnet_can_mul_mat` 限制类型与批次，`ggml_bitnet_mul_mat_get_wsize` / `ggml_bitnet_transform_tensor` 提前写好 `n_tile_num`、`lut_scales_size`、权重指针等到 `tensor->extra`。

## 4. iFairy LUT 适配总体架构
**数据流（类比 BitNet）：**
1. 构图/加载时：`ggml_ifairy_transform_tensor`（新）读取 `block_ifairy`，设置 `extra`（`qweights`、`scales_real/imag`、`lut_scales_size=2`、`n_tile_num` 等）。
2. 前向 `mul_mat`：若命中 ARM TL1 条件（类型、形状、批次），走 `ggml_ifairy_mul_mat_lut`。
3. 线程 0 预处理：调用 `ggml_ifairy_preprocessor(m, k, act, lut_scales, qlut_r, qlut_i)` 构造查表。
4. 各线程：按 tile 调用 `ggml_qgemm_ifairy_lut` → `ifairy_tbl_impl_*`，累计 `int32`，最后用 `d_real/d_imag` 与 `lut_scales_*` 反量化得到 fp32 结果。

## 5. 关键组件设计

### 5.1 权重转换与 `extra` 布局
- 参考 `BitNet/src/ggml-bitnet-lut.cpp:ggml_bitnet_transform_tensor`。
- 需要写入的字段（可直接复用 `bitnet_tensor_extra` 结构或新增）：
  - `lut_scales_size = 2`（real/imag 各一）。
  - `n_tile_num = m / bm`（bm 取决于生成的内核，建议沿用 1536/4096 的分块参数）。
  - `qweights = tensor->data`（2-bit 主体），`scales = {d_real, d_imag}`（来自行尾）。
  - `tile_stride`、`c_tile_size` 与 BitNet 相同，保证线程切分一致。
- 选择形状：iFairy 典型矩阵与 BitNet 的 TL1 内核一致（1536×4096、1536×1536、4096×1536），可直接生成同尺寸内核。
- 实现进度：`ggml/src/ggml-ifairy-lut.c` 添加 `ggml_ifairy_transform_tensor`（受 `GGML_IFAIRY_ARM_LUT` 控制），将每 block 的 `d_real/d_imag` 抽取到 `scales`（float 双通道），并按形状映射设置 `bm/bk/n_tile_num/tile_stride/c_tile_size`；`ggml/include/ggml-ifairy.h` 暴露 `ggml_ifairy_tensor_extra` 供前向读取。

### 5.2 工作区规划
- 计算：`ggml_ifairy_mul_mat_get_wsize` 针对单列 matvec 返回工作区大小，按列累计 `QLUT_R/QLUT_I`（各 `k/2*32` → 总 `k*32` 字节）与 `lut_scales`（2×`sizeof(float)`），最终 64B 对齐；未命中形状直接返回 0（回退旧路径）。
- 布局（约定）：连续缓冲顺序为 `qlut_real` → `qlut_imag` → `lut_scales`，与后续预处理/内核偏移保持一致；如需 fp16 缓冲可追加在头部扩展（当前未启用）。
- 集成：`ggml_graph_plan`（CPU，`GGML_IFAIRY_ARM_LUT`）在 MUL_MAT 估算阶段优先调用 iFairy wsize，为 LUT 预处理预留空间。
- 实现进度：`ggml/src/ggml-ifairy-lut.c` 添加 `ggml_ifairy_transform_tensor`（受 `GGML_IFAIRY_ARM_LUT` 控制），将每 block 的 `d_real/d_imag` 抽取到 `scales`（float 双通道），并按形状映射设置 `bm/bk/n_tile_num/tile_stride/c_tile_size`；`ggml/include/ggml-ifairy.h` 暴露 `ggml_ifairy_tensor_extra` 供前向读取。

### 5.2 工作区规划
- 建议顺序（与 BitNet 对齐，便于偏移复用）：
  1. 可选 fp16 缓存（当输入是 fp32 时，用于转存 `act`）。
  2. `qlut_real`：`m_tile * k_tile /2 * 32` 字节。
  3. `qlut_imag`：同上大小。
  4. `lut_scales`：2 × `ne11` 个 `float/half`（分别对应 real/imag）。
- 估算公式（单 tile）：
  ```
  qlut_bytes = 2 /*real+imag*/ * (k / 2 * 32);
  scales_bytes = 2 * sizeof(bitnet_float_type);
  wsize = qlut_bytes * (ne11 / n_tile_num) + scales_bytes * (ne11 / n_tile_num);
  ```
  再做 64B 对齐；若使用 fp16 转存，再加 `max(ne10, ne01) * ne11 * sizeof(bitnet_float_type)`。

### 5.3 激活预处理 `ggml_ifairy_preprocessor`
- 入口签名（与 BitNet 风格一致）：
  ```c
  void ggml_ifairy_preprocessor(int m, int k,
      const void *B, void *lut_scales, void *qlut_real, void *qlut_imag);
  ```
- 逻辑：
  1. `partial_max_reset`：将 `lut_scales_r/lut_scales_i` 置零。
  2. `per_tensor_quant`（复数版）：扫描 `block_ifairy_q16`，按 `x_real/x_imag` × `d_real/d_imag` 计算 max_abs，写入 `lut_scales_r = 127/max_r`，`lut_scales_i = 127/max_i`。
  3. `ifairy_lut_ctor`：将激活乘以对应 `lut_scales` 量化到 int8，当前实现顺序写入前 `k` 字节（其余填零，布局后续与内核联动时再替换为正式转置/nibble 排列）。
- 形状分支：与内核一一对应，生成 `preprocessor_k<4096>`、`preprocessor_k<1536>` 等模板实例。

### 5.4 LUT 内核 `ifairy_tbl_impl_*` / `ggml_qgemm_ifairy_lut`
- `tbl_impl_*`（核心循环）
  - 输入：`uint8_t *a`（2-bit 权重）、`int8_t *qlut_r`、`int8_t *qlut_i`、`int32 *c_real/int32 *c_imag`。
  - 查表：`idx = nibble & 0xF`，`wr = tbl(lut_wr, idx)`，`wi = tbl(lut_wi, idx)`，其中 `lut_wr={-1,1,0,0}`，`lut_wi={0,0,-1,1}`（重复填满 16B）。
  - 激活：`ar = vqtbl1_s8(qlut_r, idx_vec)`，`ai = vqtbl1_s8(qlut_i, idx_vec)`。
  - 累计（复数共轭乘法）：
    - `acc_real += sdot(wr, ar) - sdot(wi, ai)`
    - `acc_imag += sdot(wr, ai) + sdot(wi, ar)`
  - 每个 `BBK` 循环后把 `int16` 扩展到 `int32`，累加到 `CBits_real/CBits_imag`。
- `qgemm_lut_*`
  - 外层遍历 K：`for k_outer += BBK*…` 调用 `tbl_impl_*`。
- 反量化：
    ```
    scale_wr = d_real / lut_scale_r;
    scale_wi = d_imag / lut_scale_i;
    C_real[i] = scale_wr * acc_rr - scale_wi * acc_ii;
    C_imag[i] = scale_wr * acc_ri + scale_wi * acc_ir;
    ```
  - 输出格式：与现有路径一致，实/虚交错或双通道由上游张量决定（通常以两行存储）。

### 5.4 参考实现现状
- `ggml_ifairy_qgemm_lut_ref`（`ggml-ifairy-lut.c`）已实现解码 + 复数 dot + 反量化的参考内核，用于功能正确性验证和后续 NEON 内核对齐。当前读取的 QLUT 为顺序 int8 布局（尚未转置/nibble 化），输出布局为 `[real, imag]` 交错。
- 未来将以 TL1 风格生成/手写 `tbl_impl_*` + `qgemm_ifairy_lut_*`，替换参考内核并同步调整 QLUT 生成与工作区偏移。

### 5.5 前向调度
- `ggml_compute_forward_mul_mat` 在 `GGML_IFAIRY_ARM_LUT` 下优先走 iFairy LUT：线程 0 先 `ggml_ifairy_transform_tensor`（填充 `tensor->extra`），再用 `ggml_ifairy_preprocessor` 将激活写入工作区（顺序：`qlut_r` → `qlut_i` → `lut_scales`），`ggml_barrier` 同步后由线程 0 调用参考 LUT 内核写回结果，其余线程仅同步；未命中条件即回退原路径。
- 当前仅支持单列 matvec，后续 NEON TL1 内核接入时可在同一调度框架内替换 `qgemm`、扩展批次与 tile 分配。

### 5.6 容错与对比
- 环境变量 `GGML_IFAIRY_ARM_LUT_DISABLE=1` 可强制关闭 LUT 路径；形状/类型不符时自动回退。
- 若工作区不足（`params->wsize` 小于预估值），前向直接回退原始 mul_mat 路径，避免断言或越界。
- 参考内核仅线程 0 执行，便于与旧路径做 A/B 数值对比；NEON 版本接入后可继续保留该回退逻辑作为安全网。
- 自测：`tests/test-ifairy-lut.cpp` 构造随机 ifairy 权重/激活，调用 `ggml_ifairy_preprocessor` + `ggml_ifairy_qgemm_lut_ref`，并与浮点解码后的复数乘结果对比（1% 相对误差阈值）验证数值正确性。

### 5.5 调度与入口判定
- `ggml_ifairy_can_mul_mat`（类似 BitNet）：
  - `src0->type == GGML_TYPE_IFAIRY`、`src1->type == GGML_TYPE_IFAIRY_Q16` 或 fp32（需转存）、`dst->type == F32`。
  - `src1->ne[1] <= 1`（仅 matvec），后续再扩展批次。
  - `(m, k)` 命中已生成的内核组合。
- `ggml_mul_mat` 中新增分支：优先检查 ifairy LUT → 回退到现有 `ggml_vec_dot_ifairy_q16_K`。

## 6. 与现有点积路径的差异与收益
- 主要差异：激活提前重排到 `QLUT`，主循环不再重复解码权重/激活，降低访存与分支；权重解码通过 `tbl` 共享 LUT。
- 预期收益：在 1536/4096 典型形状上，比逐块 `sdot` 的实现更易向内存带宽对齐，性能类似 BitNet TL1（通常 1.3×~1.6×）。
- 精度保持：使用独立的 `lut_scales_r/i` 与 `d_real/d_imag`，公式与现有复数乘法一致，无需修改训练/转换格式。

## 7. 测试与验证建议
- 单元测试：
  - 复用 `tests/test-ifairy-ref.py` 生成的小规模 matmul 用例，增加“LUT on/off”对比（可在内核中加开关）。
  - 针对 1536×4096、1536×1536、4096×1536 做逐元素误差对比（real/imag 独立对齐）。
- 性能基准：
  - 在 Apple M 系列或 ARMv8.2+ 平台，对 `ggml_vec_dot_ifairy_q16_K` 与 LUT 路径分别跑 1000 次，记录 tok/s 或 ms。
  - 关注工作区尺寸是否命中 L2（可调 `bm/bbk`）。
- 回退验证：关闭 `GGML_IFAIRY_ARM_LUT` 宏应回落到旧实现，结果一致。

## 8. 后续扩展
- 支持批量（`src1->ne[1] > 1`）：参考 BitNet TL2 的双 LUT（two/three 分支）做批量 LUT 构造。
- Metal/GPU：当前设计 CPU-only，后续可在 Metal Backend 复用同样的 LUT 逻辑。
- 自动生成内核：仿照 `preset_kernels/bitnet-lut-kernels-tl1.h`，根据模型维度脚本化生成 `ifairy-lut-kernels-tl1.h`，减少手写汇编。
