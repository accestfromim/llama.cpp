# iFairy 2-bit 复数模型 · ARM NEON LUT (TL1) 适配设计

本文档参考 BitNet 仓库的 ARM TL1 LUT 方案（见 `../BitNet/docs/lut-arm.md`），给出在 llama.cpp 中为 iFairy 2-bit 复数量化模型接入 LUT 路径的整体设计、函数拆分与数据流说明。

## 1. 背景与目标
- 现状：iFairy 在 CPU 路径上已实现 2-bit 复数权重（`block_ifairy`）与 q16 激活的 NEON `sdot` 点积（`ggml_vec_dot_ifairy_q16_K`，位置 `ggml/src/ggml-cpu/arch/arm/quants.c`）。当前 `mul_mat` 仍逐块解码权重/激活，吞吐受限。
- 目标：复用 BitNet TL1 的“激活预量化 + 查表内核”思路，降低解码与访存开销，在 ARMv8.2+NEON/DOTPROD 设备上为 iFairy 提供高吞吐的 matvec/matmul 内核。
- 约束：保持 iFairy 复数语义（{−1,+1,−i,+i} × `d_real`/`d_imag`），输入/输出仍为 GGML 现有的 `GGML_TYPE_IFAIRY`（权重）与 `GGML_TYPE_IFAIRY_Q16`（激活）。

## 当前性能瓶颈（2025-02-05）
- Xcode Time Profiler（`m=1536,k=4096`）热点：`ggml_ifairy_unpack_block_codes` 37%（权重解码 + 4 路 buffer 写）、`ggml_ifairy_qgemm_lut_neon_slice` 33.6%、`ggml_vec_dot_ifairy_q16_K` 7.9%、`ggml_vec_dot_f16` 6%。LUT 路径仍被解码+访存拖慢。
- 根因拆解：
  - 解码阶段为每个 block 写入 4 份 buffer（wr_even/wr_odd/wi_even/wi_odd，合计 512B），再在主循环重新读回，访存次数翻倍。
  - 每 8 对权重执行 6×`vqtbl1`（wr/wi + even/odd 拆分），主循环对同一块又做 8 次 `vdot`（偶/奇分别累计 rr/ii/ri/ir），ILP/吞吐不足。
  - 激活压缩为偶/奇 4 份 packed 缓冲，内核需多次 gather/组合才能参与点积，cache pressure 偏高。
- 优化思路（本轮落地）：
  1) **权重解码瘦身**：`ggml_ifairy_unpack_block_codes` 改为直接生成“偶→奇”拼接的 16-lane 向量（`[e0..e7|o0..o7]`），wr/wi 各 1 次 `vqtbl1` 完成映射和重排，减少 tbl 次数并删掉 4 路写回。
  2) **qgemm 融合偶/奇**：内核按 8 对为单位加载 `[even|odd]` 激活向量，将偶/奇 dot 合并为单次 `vdot`，rr/ii/ri/ir 各 1 条指令，指令/访存减半，常量表保持寄存器驻留并加权重/QLUT 预取。
  3) **激活预打包**：预处理阶段直接写出 `[even|odd]` 16B pack，内核不再做 `vcombine` 重组；打包同时利用权重取值的对称性（±1/±i）共享同一权重查表和 pack 布局。
  4) **解码-计算融合 + 32B 展开**：`qgemm` 内核在解码 2-bit 权重时即时查表并 `vdot`，移除 wr/wi 临时 buffer；按 32B 展开同时维护双路累加（交替流水）并预取下一 pack 的激活，减小循环/访存开销。
  5) **形状专门化**：保持 256 权重块的步长，按 k=4096/1536 的固定块数量展开循环，利于编译器展开与寄存器复用。
  6) **实测**：`./build/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 512 --n-gen 128 -ngl 0`（Apple M4，Metal+BLAS，4 线程）得到 pp512 48.57 tok/s、tg128 34.82 tok/s（Metal/调度存在波动，建议补 CPU-only A/B）。

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
  2. `per_tensor_quant`（复数版）：扫描 `block_ifairy_q16`，按 `x_real/x_imag`（以 int8 符号解释）× `d_real/d_imag` 计算 max_abs，NEON 同步归约 real/imag，写入 `lut_scales_r = 127/max_r`，`lut_scales_i = 127/max_i`。
  3. `ifairy_lut_ctor`：将激活乘以对应 `lut_scales` 量化到 int8，使用 NEON 分块量化（8 对激活）生成 TL1 nibble 布局：每对 (even, odd) 激活写入两张 16 项查表（偶/奇各一份），单通道大小 `k/2*32`，real/imag 各占一份。
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
- `ggml_ifairy_qgemm_lut_ref`（`ggml-ifairy-lut.c`）已实现解码 + 复数 dot + 反量化的参考内核，用于功能正确性验证和后续 NEON 内核对齐。当前读取的 QLUT 已改为 nibble 布局并由 NEON 构表（每对激活 2×16 表，偶/奇拆分），乘法仍标量解码权重并输出 `[real, imag]` 交错。
- `ggml_ifairy_qgemm_lut_neon`（`ggml-ifairy-lut.c`）新增 DOTPROD 路径：`vqtbl1` 将权重码映射到 `wr/wi`，按偶/奇激活块取常数表，`vdotq_s32` 同步累计 `rr/ii/ri/ir`，最后用 `w_r/w_i` 与 `lut_scales` 反量化。前向在支持 DOTPROD 时优先调用，环境/形状不符回退标量；行区间接口支持线程切分（`row_start/row_end`），后续仍可迭代内核特化。

### 5.5 前向调度
- `ggml_compute_forward_mul_mat` 在 `GGML_IFAIRY_ARM_LUT` 下优先走 iFairy LUT：线程 0 先 `ggml_ifairy_transform_tensor`（填充 `tensor->extra`），再用 `ggml_ifairy_preprocessor` 将激活写入工作区（顺序：`qlut_r` → `qlut_i` → `lut_scales`），`ggml_barrier` 同步后由线程 0 调用参考 LUT 内核写回结果，其余线程仅同步；未命中条件即回退原路径。
- 当前仅支持单列 matvec，后续 NEON TL1 内核接入时可在同一调度框架内替换 `qgemm`、扩展批次与 tile 分配。

### 5.6 容错与对比
- 环境变量 `GGML_IFAIRY_ARM_LUT_DISABLE=1` 可强制关闭 LUT 路径；形状/类型不符时自动回退。
- 若工作区不足（`params->wsize` 小于预估值），前向直接回退原始 mul_mat 路径，避免断言或越界。
- 参考内核仅线程 0 执行，便于与旧路径做 A/B 数值对比；NEON 版本接入后可继续保留该回退逻辑作为安全网。
- 自测：`tests/test-ifairy-lut.cpp` 构造随机 ifairy 权重/激活，先比对 LUT 构造结果（NEON vs 标量参考），再调用 `ggml_ifairy_qgemm_lut_ref`，与浮点解码后的复数乘结果对比（2% 相对误差阈值）验证数值正确性；DOTPROD 平台额外比对 `ggml_ifairy_qgemm_lut_neon` 与参考输出，防止偶/奇 pack 对齐回归。

## 6. 标量 LUT → ARM TL1 SIMD 转换拆解
- 标量现状：`ggml_ifairy_qgemm_lut_ref` 逐 block 标量解码 `wr/wi`，读取 NEON 构表的 nibble 布局 `qlut_r/qlut_i`（偶/奇查表各 16 项），做 4 路累加（`rr/ii/ri/ir`），最后按 `w_r/w_i` 与 `1/lut_scales` 反量化。
- NEON 现状：`ggml_ifairy_qgemm_lut_neon` 用 `vqtbl1` 查权重码、`vdotq_s32` 累积 `rr/ii/ri/ir`，与参考共享 nibble QLUT；当前仍为单列 matvec，必要时回退标量。已修复偶/奇错位：权重查表结果按 `idx_pack={0,2,...,14,1,3,...,15}` 重排为 `[even|odd]` 后再参与点积，与预打包激活布局一致。
- 与 BitNet TL1 差异：QLUT 尺寸/偏移已对齐且构表 NEON 化，权重查表/累加已用 NEON，但尚未做 K/M 特化或批次切分，仍使用行首尺度 `d_real/d_imag`（单 block 场景无误）。
- 迁移步骤（按依赖排序）：
  1) **QLUT 布局定版**：仿 BitNet TL1，单通道大小 `k/2*32`，real/imag 各占一份；更新 `wsize` 注释与偏移，删除现有 `memset(qlut, k*32)` 占位写，确保构表和内核共享同一 nibble 布局。（已完成：工作区按 nibble 计算，预处理生成偶/奇各一张 16 项表并被参考内核消费）
  2) **量化/构表 SIMD 化**：`ggml_ifairy_per_tensor_quant` 用 NEON 同时求 real/imag `max_abs`（`vld1_s8`→`vmovl`→`vcvtq_f32`×`d_*`→`vmaxq`）；`ggml_ifairy_lut_ctor` 参考 `Transpose_8_8`+`tbl_mask` 生成 nibble 友好布局的 `vec_lut_r[16]`、`vec_lut_i[16]`，一次写完 real/imag QLUT。
  3) **权重查表 SIMD 化**：构造 16B `tbl_wr`/`tbl_wi`（码 `0/1`→±1, `2/3`→±i），`tbl_impl_*` 中对 16B 权重用 `vshrq_n_u8`/`vandq` 拆高低 nibble，通过 `vqtbl1q_s8` 同步取 real/imag LUT，使用 `vdotq_s32`（或 `vmlal_s8`）分别累计 `rr/ii/ri/ir`，注意虚部符号在 `wi` 查表中编码。
  4) **TL1 内核包装**：按 `(m,k)` 特化 `qgemm_ifairy_lut_{m}_{k}`，与 `bm/bk` 对齐 K 步长；反量化沿用 `w_r/w_i` 与 `lut_scales_r/i` 的共轭公式。`ggml_compute_forward_mul_mat` 中将参考内核替换为 NEON 版本，保留 scalar 版本做回退。（已完成：通用 matvec 版本支持 DOTPROD，按线程切分行区间；后续可进一步按 `bm/bk` 特化）
  5) **验证与基线**：更新 `ggml_ifairy_mul_mat_get_wsize`/预处理偏移与新布局一致，`tests/test-ifairy-lut` 校验 LUT 构造（NEON vs 标量）与参考解码（2% 容忍）对齐；`ctest -R ifairy-lut --test-dir build --output-on-failure` 通过，性能对比待在目标设备补充 tok/s。

## 7. 工程化与维护
- 构建开关：CMake 选项 `GGML_IFAIRY_ARM_LUT`（默认 OFF）控制整个路径，运行时可用环境变量 `GGML_IFAIRY_ARM_LUT_DISABLE=1` 强制关闭，便于 A/B。
- 接口：公共头 `ggml/include/ggml-ifairy.h` 暴露 LUT 相关 API（transform、preprocess、wsize、参考 qgemm），前向在 `ggml-cpu` 中受宏保护。
- 工作区与回退：`ggml_ifairy_mul_mat_get_wsize` 估算并 64B 对齐，前向若 wsize 不足则回退旧路径；形状/类型校验与环境开关也会触发回退。
- 测试：新增 `test-ifairy-lut`，后续 NEON 内核加入后可复用该测试对齐 LUT 与 baseline。
- 未来：可将 QLUT 布局改为脚本生成（仿 BitNet preset_kernels），并在文档中附上启用示例命令（`-DGGML_IFAIRY_ARM_LUT=ON` + 可选 `GGML_IFAIRY_ARM_LUT_DISABLE` 环境变量）。
- 验证示例：`cmake -B build -DGGML_IFAIRY_ARM_LUT=ON && cmake --build build -j`，执行 `./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -p "I believe life is" -n 128 -no-cnv` 正常输出，CPU eval 12.38 ms/token（~80.8 tok/s）。

### 5.7 调度与入口判定
- `ggml_ifairy_can_mul_mat`（类似 BitNet）：
  - `src0->type == GGML_TYPE_IFAIRY`、`src1->type == GGML_TYPE_IFAIRY_Q16` 或 F32（F32 激活会在 wdata 先量化为 `IFAIRY_Q16`）、`dst->type == F32`，`k % QK_K == 0`。
  - `src1->ne[1] <= 1`（仅 matvec），后续再扩展批次，tile 选择按权重维度 `(m=src0->ne[1], k=src0->ne[0])`。
  - `(m, k)` 命中已生成的内核组合；`GGML_IFAIRY_ARM_LUT_DEBUG=1` 可打印首次命中/拒绝原因，`GGML_IFAIRY_ARM_LUT_DISABLE=1` 强制回退。
- `ggml_mul_mat` 中新增分支：优先检查 ifairy LUT → 回退到现有 `ggml_vec_dot_ifairy_q16_K`。
- 输出格式与 baseline 对齐：`qgemm_lut_{ref,neon}` 将 real/imag 打包为单个 float 内的两个 bf16（同 `vec_dot_ifairy_q16_K`）。

## 8. 与现有点积路径的差异与收益
- 主要差异：激活提前重排到 `QLUT`，主循环不再重复解码权重/激活，降低访存与分支；权重解码通过 `tbl` 共享 LUT。
- 预期收益：在 1536/4096 典型形状上，比逐块 `sdot` 的实现更易向内存带宽对齐，性能类似 BitNet TL1（通常 1.3×~1.6×）。
- 精度保持：使用独立的 `lut_scales_r/i` 与 `d_real/d_imag`，公式与现有复数乘法一致，无需修改训练/转换格式。

## 9. 测试与验证建议
- 单元测试：
  - 复用 `tests/test-ifairy-ref.py` 生成的小规模 matmul 用例，增加“LUT on/off”对比（可在内核中加开关）。
  - 针对 1536×4096、1536×1536、4096×1536 做逐元素误差对比（real/imag 独立对齐）。
- 性能基准：
  - 在 Apple M 系列或 ARMv8.2+ 平台，对 `ggml_vec_dot_ifairy_q16_K` 与 LUT 路径分别跑 1000 次，记录 tok/s 或 ms。
  - 关注工作区尺寸是否命中 L2（可调 `bm/bbk`）。
  - 近期实测：`./build-arm64-apple-clang-release/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -b 1 -t 4 -p "I believe life is" -n 128` 启用 LUT 时 eval ≈ 290.8 ms/token（≈3.44 tok/s，参考内核），`GGML_IFAIRY_ARM_LUT_DISABLE=1` 基线 ≈ 31.7 ms/token（≈31.6 tok/s），功能正常但性能尚待优化。
- 回退验证：关闭 `GGML_IFAIRY_ARM_LUT` 宏应回落到旧实现，结果一致。

## 10. 后续扩展
- 支持批量（`src1->ne[1] > 1`）：参考 BitNet TL2 的双 LUT（two/three 分支）做批量 LUT 构造。
- Metal/GPU：当前设计 CPU-only，后续可在 Metal Backend 复用同样的 LUT 逻辑。
- 自动生成内核：仿照 `preset_kernels/bitnet-lut-kernels-tl1.h`，根据模型维度脚本化生成 `ifairy-lut-kernels-tl1.h`，减少手写汇编。

## 11. 多权重组合 LUT 可行性（3 权重示例）
- 现状：TL1 的 nibble 表只覆盖“偶/奇”两权重，`ggml_ifairy_fill_pair_tables` 对 16 项表填同一常量，内核运行时用 `wr_tbl/wi_tbl` 展开 2-bit 权重后逐权重做 4 路 `vdot`；BitNet TL1 亦是两权重粒度，没有 3 权重同查表。
- 方案 A（6bit/三权重查表）：每 3 个激活构成一组，预处理生成 `4^3=64` 项查表并提前折叠 wr/wi 符号，运行时按 6bit 直接 `vqtbl` → 4×`vdot` 覆盖三权重，可减少约 1/3 的 `vqtbl`/`vdot` 次数。
  - 代价：QLUT 单通道从 `k/2*32` 增到 `k/3*64`（k=4096 时 real+imag ≈ 342 KB，约 +33%），预处理写表与访存同样上升；权重仍以 4 权重/byte 存储，三权重跨字节，需在 `ggml_ifairy_transform_tensor` 侧额外 repack（或内核内拼 6bit，可能吞掉收益）；需处理 k%3 尾块。
  - 风险：工作集变大影响 L2 命中，预处理耗时上升；需为 LUT/非 LUT 路径补充一致性测试与基准。
- 方案 B（轴/符号分离压缩，建议优先验证）：将三权重拆为 `(axis, sign)`，其中 `axis ∈ {R,I}` 决定是 wr 还是 wi，sign 仍为 ±1。预处理仅按 3bit 轴组合生成 8 项“无符号”表（单通道约 `k/3*16`，低于现有 wsize），内核再用 3bit 符号 mask 做 `eor/bsl` 翻转。
  - BitNet TL2 参考：`three_lut_ctor` 每 24 激活输出 8×32B 表（覆盖 6bit 组合），先将三路量化值以 int32 求和再转置/pack，额外维护 `sign` 缓冲（3 权重 1bit，8 权重/byte），`three_tbl_impl_*` 用 LUT + sign 翻转。我们的轴/符号拆分可沿用“无符号表 + sign buffer”结构，轴表对应 BitNet LUT，sign buffer 对应其 sign 流。
  - 内核思路：解码 6bit（低 3bit sign，高 3bit axis）→ `axis_pack = vqtbl1(axis_tbl, axis_codes)`（axis_tbl 两份 real/imag 互斥掩码），`sign_pack` 来自 sign buffer 或 codes 衍生，`wr_pack = vbsl(axis_is_r, sign_pack, vzero)`，`wi_pack = vbsl(axis_is_i, sign_pack, vzero)`，随后与 `[even|odd]` 激活做 4 路 `vdot`。符号翻转用 `veor`/`vbsl`/`vnegq_s8`，避免额外访存。
  - 工作区/接口：`ggml_ifairy_qlut_bytes_3w(k) = k/3*16`（单通道轴表），可选 `sign_bytes = k/3/8`（每 3 权重 1bit，向 16B 对齐）；工作区布局改为 `qlut_axis_r/qlut_axis_i[/sign] → lut_scales`，`mul_mat_get_wsize` 按列累加上述大小；尾部 k%3 可回退到两权重或补零。
  - 权重处理：保持 4 权重/byte 原布局，内核用 `vext`/`tbl` 拼 6bit（2 个 byte 覆盖 3 组；用移位/与掩码方式裁出 6bit 序列），或在 `ggml_ifairy_transform_tensor` 侧预先写一个 6bit 索引 buffer（仅在 `GGML_IFAIRY_ARM_LUT_3W` 打开时分配）。
  - Gate：新增编译/运行时开关（如 `GGML_IFAIRY_ARM_LUT_3W=1` 或 env）控制 3 权重路径；默认沿用两权重，确保未验证设备不受影响。
- 风险：符号翻转仍有 3 次 mask，但访存/表尺寸低于方案 A；吞吐增益需以实测 tok/s 验证；需扩展单测覆盖 3 权重 LUT 构造与标量/NEON 一致性。
- 结论：当前实现仅支持两权重粒度。三权重/对称压缩可作为后续 gated 实验方向，推进时需同步更新 QLUT 布局、`mul_mat_get_wsize` 估算、预处理构表、权重转换或 repack 缓冲、NEON 内核与单测/基准记录；方案 B 先行，方案 A 作为备选。
- **现状（2025-02-??）**：实现了方案 A 的标量参考版并接入 gated 前向：`GGML_IFAIRY_ARM_LUT_3W`（默认 OFF）+ `GGML_IFAIRY_ARM_LUT_3W=1` 时启用 6bit 三权重 LUT。预处理复用顺序量化的 `qr/qi`，每 block 以 3 权重为组生成 64 项 int16 表（条目存 `{wr·ar, wr·ai, wi·ar, wi·ai}`，大小 `(k/QK_K)*(QK_K/3)*64*4*2`），每 block 余下 1 权重在 qgemm 内直接用顺序 `qr/qi` 累加。`test-ifairy-lut` 增加 3W 对齐（4% 容忍）通过；性能上 `GGML_IFAIRY_ARM_LUT_3W=1 ./build/bin/llama-cli ... -n 128` 在本机 CPU 仅 ~6.78 tok/s，较默认 2W LUT (~50.8 tok/s) 明显回退，需后续轴/符号压缩或 NEON 版优化。
- **新的结论与路线**（2025-02-XX 更新）：
  - **256 % 3 尾块处理**：当前参考实现按 block（256 权重）构 3 权重组，尾部不足 3 的 1 个权重直接使用顺序量化的 `qr/qi` 在 qgemm 中累加（不进入 6bit LUT），功能正确但吞吐受影响。
  - **对称压缩进展**：尚未做 3 权重轴/符号分离压缩，现有 6bit LUT 直接存 {wr,wi}×{ar,ai} 的 int16 求和，没有额外压缩或 sign buffer；下阶段应优先验证方案 B 的轴/符号分离以缩小工作区并减少访存。
  - **优先级建议**：先推进方案 B（轴/符号分离 + sign buffer）的原型与 NEON 路径，目标是降低 wsize 和表访存，再评估 tok/s；方案 A 的 NEON 优化可作为备选（若轴/符号分离收益有限再补充），避免在当前膨胀的表尺寸上做 SIMD 化而收益有限。
  - **环境变量说明**：在 fish 中需 `set -x GGML_IFAIRY_ARM_LUT_3W 1` 才会被子进程继承，否则 `llama-cli` 仍走 2 权重 LUT，出现 ~65 tok/s 的速度是因为未实际启用 3W 路径（回落到 2W NEON LUT）。***

## 12. LUT 构建的溢出/截断处理
- 现状：`ggml_ifairy_lut_ctor` 在量化偶/奇激活时使用 `vcvtn` + `vmin/vmax` 把每个 int16 结果夹到 `[-127,127]`，写入 nibble 表；`ggml_ifairy_fill_pair_tables` 将单值复制为 16 项，不存在多值求和。多权重 LUT（3 权重）若提前折叠激活/符号，可能出现 3×int8 相加溢出。
- 设计：预处理中的每个表项在写入前先用 int16/32 累加，再统一 clip 到 int8。NEON 路径可用 `vqaddq_s16`/`vqmovn_s16` 或在构表时用 `vaddl_s8`→`vqmovn_s16` 实现饱和；标量路径用 `ggml_ifairy_clamp_s8`（`MAX(-127, MIN(127, val))`）。
- 代码落点建议：
  - 在未来的 3 权重 LUT 构造函数里，对同一表项的 3 路贡献使用 int16 累加，最终经 `vqmovn_s16`（或标量 clamp）写回。
  - 现有两权重路径保持不变（每表项单值），但在文档中明确 clip 策略，避免后续改动时遗漏饱和处理。
  - 若构表需要预乘 `lut_scales`，确保乘法结果先转换为 int16/32，再饱和下采样到 int8，避免中间 int8 溢出。
