# iFairy ARM NEON LUT 开发计划（TL1）

> 参考 `IFAIRY_ARM_LUT_TL1.md`（复数 2-bit 方案）与 `BitNet/docs/lut-arm.md`（ARM TL1 LUT 流程），作为后续所有开发与对齐的主线文档。

## 目标与范围
- 在 ARMv8.2+NEON/DOTPROD 平台为 `GGML_TYPE_IFAIRY` × `GGML_TYPE_IFAIRY_Q16/F32` 推理引入 LUT 加速路径，保持复数语义与现有权重/激活格式。
- 支持主流形状（仿 BitNet TL1 的 1536×4096、1536×1536、4096×1536 等 matvec/matmul 热点），并允许后续脚本化扩展。
- 保持可回退：宏关闭或不匹配形状时回落到 `ggml_vec_dot_ifairy_q16_K` 现有实现，数值一致。

## 主要参考与落点
- 设计文档：`IFAIRY_ARM_LUT_TL1.md`、`BitNet/docs/lut-arm.md`。
- 内核示例：`BitNet/preset_kernels/*/bitnet-lut-kernels-tl1.h`（`per_tensor_quant`、`lut_ctor`、`tbl_impl_*`、`qgemm_lut_*`）。
- 入口与工作区：`BitNet/src/ggml-bitnet-lut.cpp`、`ggml/src/ggml.c`（ARM BitNet 路径）。
- 当前 iFairy 路径：`ggml/src/ggml-cpu/arch/arm/quants.c`（`quantize_row_ifairy_q16`、`ggml_vec_dot_ifairy_q16_K`）、`ggml/src/ggml-quants.c`（`quantize_row_ifairy_ref`）。
- 其它：`IFAIRY_INFERENCE_PIPELINE.md` 确认数据流，`include/llama.h`/`include/ggml.h` 中类型声明。

## 设计原则与约束
- 复用 BitNet TL1 的工作区布局（可选 fp16 缓存 → QLUT → LUT scales），并新增 imag 通道；保持 64B 对齐。
- 仅在 `src1->ne[1] <= 1`（matvec）时启用，后续再扩展批量；`src0->type == GGML_TYPE_IFAIRY`、`src1->type ∈ {GGML_TYPE_IFAIRY_Q16, GGML_TYPE_F32}`、`dst == F32`。
- 形状受限于已生成的 LUT 内核集合；未命中时立即回退。
- 编译期开关独立（如 `GGML_IFAIRY_ARM_LUT`），不影响 BitNet 宏。

## 开发里程碑与步骤
1. **现状梳理与编译守护**
   - 通读 `ggml/src/ggml-cpu/arch/arm/quants.c` 的 ifairy 分支，确认 `block_ifairy`/`block_ifairy_q16` 布局、当前点积流程及宏保护。
   - 在现有构建命令（`cmake -B build && cmake --build ...`）下复现 baseline，记录 tok/s 作为对照。
   - 评估是否需要新的编译选项（CMake cache 变量）以隔离 LUT 代码生成。
   - 评估：`block_ifairy` = `qs[QK_K/4] + d_real + d_imag`，`block_ifairy_q16` = `x_real[256] + x_imag[256] + d_real + d_imag`；`ggml_vec_dot_ifairy_q16_K` 仅在 `__ARM_NEON && __ARM_FEATURE_DOTPROD` 下走 NEON inline asm，未加独立宏，输出通过 `((ggml_bf16_t *) s)[0/1]` 写回，else 回退泛型路径。
   - 基线：`cmake --build build --config Release -j $(sysctl -n hw.ncpu)` 成功；`./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -p "I believe life is" -n 512 -no-cnv`（CPU 路径，Metal 不可用）得到 `eval 17.39 ms/token ≈ 57.5 tok/s`，可作为后续 LUT 对比；加载阶段提示 `unknown type ifairy` 但推理正常。
   - 编译选项：新增 `GGML_IFAIRY_ARM_LUT`（默认 OFF，见 `ggml/CMakeLists.txt` + `ggml/src/CMakeLists.txt`），后续 LUT 路径均受该宏控制，与 BitNet 宏解耦。
2. **宏与类型接口定义**
   - 在 `ggml/src/ggml-cpu/arm/` 或公共头中添加 `GGML_IFAIRY_ARM_LUT` 宏开关，与 BitNet 宏解耦。
   - 若需扩展 `bitnet_tensor_extra` 风格的 `extra`，决定是复用现有结构还是新建 `ifairy_tensor_extra`（含 `qweights`、`scales_real/imag`、`lut_scales_size=2`、`n_tile_num` 等）。
   - 现状：已在 `ggml/include/ggml-ifairy.h` 定义 `ggml_ifairy_tensor_extra`（`qweights`、`scales_bytes`、`scales` 双通道、`lut_scales_size=2`、`n_tile_num`、`bm/bk`、`tile_stride/c_tile_size`），并预留 `ggml_ifairy_lut_{init/free}` 接口，方便后续生命周期管理。
3. **权重转换与 `extra` 填充**
   - 已实现：`ggml/src/ggml-ifairy-lut.c` 新增 `ggml_ifairy_transform_tensor`，提取每 block 的 `d_real/d_imag` 写入 `scales`（float 双通道），基于形状映射选择 `bm/bk`（覆盖 1536×4096、1536×1536、4096×1536，默认回退整行 bm），计算 `n_tile_num`、`tile_stride`（按 block 行大小 × bm）与 `c_tile_size`。
   - 调用时机：在 `ggml_compute_forward_mul_mat`（CPU）中，若开启 `GGML_IFAIRY_ARM_LUT` 且 `src0` 为 `GGML_TYPE_IFAIRY`、`extra` 为空，则自动调用转换以填充 `tensor->extra`；后续 LUT 内核可直接读取。
4. **工作区尺寸计算与布局**
   - 已实现：`ggml_ifairy_mul_mat_get_wsize`（`ggml-ifairy-lut.c`）在形状命中时返回工作区大小：每列分配 `QLUT_R/QLUT_I`（各 `k/2*32`，总 `k*32` 字节）加 2×`sizeof(float)` 的 LUT scales，再对 `m` 列累加并 64B 对齐；`ggml_ifairy_can_mul_mat` 限定 `src0=IFAIRY`、`src1=IFAIRY_Q16`、`dst=F32` 且 `ne1<=1`。
   - 集成：`ggml_graph_plan`（CPU）在 MUL_MAT 分支优先调用 iFairy wsize（受 `GGML_IFAIRY_ARM_LUT` 控制），为后续 LUT 预处理/内核预留连续缓冲；布局默认顺序为 QLUT_real → QLUT_imag → LUT scales（后续内核按此约定取偏移）。
5. **激活预处理与 LUT 生成**
   - 已实现：`ggml_ifairy_preprocessor`（`ggml-ifairy-lut.c`）对 `GGML_TYPE_IFAIRY_Q16` 激活进行 per-tensor quant（独立 real/imag max_abs → `lut_scales={127/max_r,127/max_i}`），再将 block 内 256 个 int8 激活按缩放写入 `qlut_r/qlut_i`；当前布局简单按顺序写入前 `k` 字节并将剩余工作区清零，后续内核接入时可替换为正式 nibble/转置布局。
   - 辅助：`ggml_ifairy_partial_max_reset`/`ggml_ifairy_per_tensor_quant`/`ggml_ifairy_lut_ctor` 拆分逻辑，便于后续替换生成代码或按 K 特化。
6. **LUT 内核实现**
   - 已实现参考内核：`ggml_ifairy_qgemm_lut_ref`（`ggml-ifairy-lut.c`）解码 2-bit 权重块（`qs`→`wr/wi` 映射 {-1,+1,0,0}/{0,0,-1,+1}），对 `qlut_r/qlut_i` 做复数共轭乘积累计 `acc_rr/acc_ii/acc_ri/acc_ir`，并按 `scale_wr = d_real / lut_scale_r`、`scale_wi = d_imag / lut_scale_i` 反量化生成 real/imag 输出（当前输出布局：`dst[2*i]=real`，`dst[2*i+1]=imag`）。用于功能验证与后续 NEON 内核对齐。
   - 形状/Tile：按行步长 `blocks_per_row = k/QK_K` 迭代，预留与 `bm/bk` 映射兼容；后续可替换为特化的 `tbl_impl_*` + `qgemm_ifairy_lut_*`（TL1 向量化）以覆盖 1536×4096、1536×1536、4096×1536。
   - 当前 QLUT 布局仍为顺序 int8 写入（未做 8×8 转置/nibble 排列），内核按此读取；待最终内核生成时可切换到 BitNet 风格布局并同步预处理与 wsize 偏移。
7. **前向调度接入**
   - 添加 `ggml_ifairy_can_mul_mat` 判定入口（类型/形状/批次/后端），并在 `ggml_mul_mat` ARM 分支优先检查 iFairy LUT，再回退。
   - 在线程 0 调用 `ggml_ifairy_preprocessor` 构造 LUT，`ggml_barrier` 同步后各线程按 tile 调用 `ggml_qgemm_ifairy_lut`。
   - 若 `src1` 为 F32，复用 BitNet 的 fp16 转存逻辑（或保持 F32 路径，注意工作区大小）。
   - 已接入：`ggml_compute_forward_mul_mat` 在 `GGML_IFAIRY_ARM_LUT` 下优先判定 `ggml_ifairy_can_mul_mat`，线程 0 先触发 `ggml_ifairy_transform_tensor`（填充 extra）与 `ggml_ifairy_preprocessor`（写入 wdata：可选 `act_q16` → `qlut_r → qlut_i → lut_scales`），全线程 barrier 后调用 LUT 内核写回 `dst`（输出按 bf16 打包，与 `vec_dot` 写法一致），再 barrier 结束，未命中则回退原有路径。当前仍为单列 matvec（`ne1<=1`），wsize/布局与预处理保持一致，后续可替换为 NEON 内核与多线程切分。
8. **数值校验与容错**
   - 在内核上加可编译时/运行时开关以强制回退，方便 A/B 对比。
   - 保证未匹配形状或缺少 `extra` 时立即回退旧路径，避免 crash。
   - 已落地：运行时环境变量 `GGML_IFAIRY_ARM_LUT_DISABLE=1` 可整体关闭 LUT，`GGML_IFAIRY_ARM_LUT_DEBUG=1` 输出首次命中/拒绝原因；`ggml_ifairy_can_mul_mat` 用权重维度判定 tile，允许 F32 激活（在 wdata 先量化为 IFAIRY_Q16），形状/类型不符或宏/环境关闭时直接回退；`ggml_compute_forward_mul_mat` 若工作区不足则直接回退原路径，避免触发断言。当前参考内核仅线程 0 执行，后续 NEON 内核接入时可保留同样的回退逻辑。
9. **测试计划**
   - 单元/比对：扩展 `tests/test-ifairy-ref.py` 或新增 C 测试，构造固定 seed 的 matvec（1536×4096、1536×1536、4096×1536），比较 LUT on/off 的逐元素误差（real/imag 分开）。
   - 性能：在目标 ARM 设备上运行 `./build/bin/llama-cli ...` 与 `ggml_vec_dot_ifairy_q16_K` baseline，对比 tok/s；若需可加微基准（重复 1000 次）。
   - 工作区验证：在调试模式打印/断言 `wsize` 与偏移，确保无越界。
   - 现状：新增 `tests/test-ifairy-lut.cpp`（构建标签 `test-ifairy-lut`）固定 seed 生成 ifairy quant 数据，跑 `ggml_ifairy_preprocessor` + `ggml_ifairy_qgemm_lut_ref` 与高精度浮点解码结果对比（相对误差 1% 容忍，避免二次量化导致的饱和差异）；`ctest -R ifairy-lut` 通过（输出已改为 bf16 打包格式）。工作区不足/宏关闭时依赖回退逻辑保证安全。实测：`./build-arm64-apple-clang-release/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -b 1 -t 4 -p "I believe life is" -n 128` 在 LUT 打开时 eval ≈ 290.8 ms/token（≈3.44 tok/s，参考内核仍需性能优化），`GGML_IFAIRY_ARM_LUT_DISABLE=1` 回退基线时 eval ≈ 31.7 ms/token（≈31.6 tok/s）。
10. **工程化与维护**
    - 构建/配置：新增 CMake 选项 `GGML_IFAIRY_ARM_LUT`（默认 OFF），运行时可用 `GGML_IFAIRY_ARM_LUT_DISABLE=1` 关闭；公共头 `ggml/include/ggml-ifairy.h` 提供 API；前向受宏保护。
    - 测试/验证：新增 `tests/test-ifairy-lut` 覆盖参考 LUT 与浮点解码对比；后续内核变更复用此测试做回归。
    - 代码生成：后续可借鉴 BitNet `preset_kernels` 做 QLUT 布局/内核脚本化生成，覆盖更多形状；当前仍使用参考内核与顺序 QLUT 布局。
    - 回退/兼容：wsize 不足、形状不匹配或环境关闭时自动回退旧路径；设计字段（`lut_scales_size`、`n_tile_num` 等）预留批量/其他后端扩展。
    - 现网验证：`cmake -B build -DGGML_IFAIRY_ARM_LUT=ON && cmake --build build -j` 后，`./build/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -p "I believe life is" -n 128 -no-cnv` 成功运行（CPU-only，Metal 关闭），eval 12.38 ms/token（~80.8 tok/s），可作为启用 LUT 后的 sanity。

## 标量 → NEON TL1 迁移任务拆解
- 目标：将现有标量参考 LUT（顺序 QLUT + 标量 wr/wi 解码）替换为 ARM NEON TL1 SIMD 路径，复用 BitNet LUT 工作区与查表模式，保持 iFairy 复数语义。
- 待办步骤：
  1. **QLUT 布局与 wsize**（已完成）：把 `qlut_r/qlut_i` 改为 TL1 nibble 布局（单通道 `k/2*32`，每对激活输出偶/奇两张 16 项表），`ggml_ifairy_mul_mat_get_wsize`/CPU 前向偏移与之对齐，移除原有顺序写入与占位 memset。
  2. **SIMD 量化/构表**（已完成）：`ggml_ifairy_per_tensor_quant` 用 NEON 同时归约 real/imag 最大值；`ggml_ifairy_lut_ctor` 用 NEON 对每 8 对激活分块量化并写入 nibble 友好 QLUT（real/imag 独立，偶/奇各 16 项）。
  3. **权重查表 SIMD 化**（已完成）：准备 16B `tbl_wr`/`tbl_wi`（映射码 0/1→±1，2/3→±i），`ggml_ifairy_qgemm_lut_neon` 用 `vqtbl1` 查权重码、`vdotq_s32` 累计 `rr/ii/ri/ir`，对未开启 DOTPROD 或工作区不足回退标量。
  4. **TL1 内核封装**（已完成）：添加行区间版本的 NEON/标量 LUT 内核，前向按线程切分 `row_start/row_end` 并调用 DOTPROD 内核（不支持则回退标量），保持 BM/BK 选择与回退逻辑一致；当前仍是单列 matvec，后续可特化 `(m,k)` 内核进一步提速。
  5. **验证与基准**（已完成）：扩展 `tests/test-ifairy-lut` 校验 LUT 构造（NEON vs 标量）与参考解码对齐（2% 容忍），当前 `ctest -R ifairy-lut --test-dir build --output-on-failure` 通过；性能对比待在目标设备上补充 tok/s。

## 近期性能优化计划（NEON LUT 热点）
- **现状复盘**：Xcode profiler 显示 `ggml_ifairy_unpack_block_codes`（37%）和 `ggml_ifairy_qgemm_lut_neon_slice`（33.6%）吞吐落后 `ggml_vec_dot_ifairy_q16_K`。瓶颈集中在 4 路权重解码写回 + 偶/奇拆分后 8 次 `vdot`。
- **目标**：在不改动 QLUT 布局的前提下，将解码/点积访存次数减半，针对 k=4096/1536 常见形状展开，力争让 LUT 路径超越 `vec_dot` 基线。
- **执行步骤**：
  1) 重写 `ggml_ifairy_unpack_block_codes`：输出 `[even|odd]` 拼接的 16B 向量（wr/wi 各一份），单次 `vqtbl1` 完成重排，移除对 4 个 buffer 的写入。
  2) 改造 `ggml_ifairy_qgemm_lut_neon_slice` 主循环：以 8 对为粒度加载 `[even|odd]` 激活（`vcombine_s8`），rr/ii/ri/ir 各 1 次 `vdot`，加权重/QLUT 预取与最小化寄存器重载。
  3) 保留形状选择/回退逻辑，完成构建 + `./build/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 512 --n-gen 128 -ngl 0` 基准记录 tok/s。
- **验收**：功能正确（与参考内核结果一致），LUT 路径在目标形状下优于现有 `vec_dot`，文档同步记录方案与基准。
- **执行记录（2025-02-05）**：完成权重解码瘦身与 qgemm 偶/奇融合，构建 `cmake --build build --config Release -j $(nproc)` 通过，`llama-bench`（同上参数）输出 pp512 49.06 tok/s、tg128 22.77 tok/s（Apple M4, 4 线程, Metal+BLAS）。

## 交付物与完成判据
- 新的 LUT 内核头（或源码）与前向接入代码，受 `GGML_IFAIRY_ARM_LUT` 宏控制。
- `ggml_mul_mat` 路径能在匹配形状时使用 LUT，未匹配时保持旧行为，数值一致。
- 通过新增/扩展的单测与至少一组性能对比记录；文档更新描述使用方式与已知限制。

## 风险与缓解
- **形状覆盖不足**：优先支持 1536/4096 热点，预留生成脚本；回退逻辑保证正确性。
- **工作区尺寸错误**：在计算/使用处加入断言与对齐检查；重用 BitNet 的偏移顺序。
- **精度偏差**：独立 real/imag scale，添加测试覆盖；必要时保留 debug 模式打印中间尺度。
- **构建膨胀**：宏隔离 + 可选头文件生成；仅在 ARM NEON/DOTPROD 打开。
