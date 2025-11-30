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
   - 编译选项：当前无 iFairy 专用 CMake 选项/宏，后续需新增 `GGML_IFAIRY_ARM_LUT`（默认 OFF）隔离 LUT 代码生成，与 BitNet 宏完全解耦。
2. **宏与类型接口定义**
   - 在 `ggml/src/ggml-cpu/arm/` 或公共头中添加 `GGML_IFAIRY_ARM_LUT` 宏开关，与 BitNet 宏解耦。
   - 若需扩展 `bitnet_tensor_extra` 风格的 `extra`，决定是复用现有结构还是新建 `ifairy_tensor_extra`（含 `qweights`、`scales_real/imag`、`lut_scales_size=2`、`n_tile_num` 等）。
   - 评估：`ggml` 侧尚未使用 `tensor->extra` 支持 iFairy；现有 `bitnet_tensor_extra` 位于 `BitNet/include/ggml-bitnet.h`（字段：`lut_scales_size/BK/n_tile_num/qweights/scales`），不含复数双尺度与 tile_stride 等信息，且引入会产生跨仓依赖。倾向在 `ggml` 内新增 `ifairy_tensor_extra`（含 `qweights`、`scales[2]`、`lut_scales_size=2`、`n_tile_num`、`tile_stride/c_tile_size`），并配套 `ggml_ifairy_{init/free}` 风格的生命周期管理。
3. **权重转换与 `extra` 填充**
   - 实现 `ggml_ifairy_transform_tensor`（类比 `ggml_bitnet_transform_tensor`），读取 `block_ifairy` 末尾的 `d_real/d_imag` 并写入 `extra`。
   - 选择 tile 参数（参考 BitNet `BM1536_4096` 等），计算 `n_tile_num = m / bm` 与 `tile_stride/c_tile_size`，保证 `m` 可整除。
   - 在图构建阶段调用转换，确保 `tensor->extra` 在前向可用。
4. **工作区尺寸计算与布局**
   - 新增 `ggml_ifairy_mul_mat_get_wsize`：QLUT_real/QLUT_imag 按 `k/2*32` 字节、scales 2×`bitnet_float_type`，可选 fp16 缓存；做 64B 对齐。
   - 在 `ggml_mul_mat` 调度时复用 BitNet 的 `wdata` 布局顺序（fp16 缓冲 → qlut_r → qlut_i → lut_scales），明确偏移计算。
5. **激活预处理与 LUT 生成**
   - 编写 `per_tensor_quant` 复数版：分别求 real/imag max_abs，写入 `lut_scales_r/i = 127/max`，提供 `partial_max_reset`。
   - 实现 `ifairy_lut_ctor<K>`：量化 real/imag 到 int8，完成 8×8 转置与 nibble 排列，输出 `QLUT_R/QLUT_I` 两份查表。
   - 封装 `preprocessor_k<K>` 与泛化入口 `ggml_ifairy_preprocessor(m, k, B, lut_scales, qlut_r, qlut_i)`；覆盖至少 K∈{4096,1536} 组合。
6. **LUT 内核实现**
   - 设计 `tbl_impl_*` 复数版：用 `vqtbl1` 查 `wr/wi` 与 `QLUT_R/QLUT_I`，执行复数共轭乘法累加到 `int32`。
   - 依据目标形状生成 `qgemm_ifairy_lut_*`：K 维分块（`BBK*`）、M 维 tile（`BM*`），完成反量化：
     - `scale_wr = d_real / lut_scale_r`，`scale_wi = d_imag / lut_scale_i`
     - `C_real = scale_wr * acc_rr - scale_wi * acc_ii`；`C_imag = scale_wr * acc_ri + scale_wi * acc_ir`
   - 参考 `BitNet/preset_kernels/*/bitnet-lut-kernels-tl1.h` 的向量化套路，决定是手写还是脚本生成 `ifairy-lut-kernels-tl1.h`。
7. **前向调度接入**
   - 添加 `ggml_ifairy_can_mul_mat` 判定入口（类型/形状/批次/后端），并在 `ggml_mul_mat` ARM 分支优先检查 iFairy LUT，再回退。
   - 在线程 0 调用 `ggml_ifairy_preprocessor` 构造 LUT，`ggml_barrier` 同步后各线程按 tile 调用 `ggml_qgemm_ifairy_lut`。
   - 若 `src1` 为 F32，复用 BitNet 的 fp16 转存逻辑（或保持 F32 路径，注意工作区大小）。
8. **数值校验与容错**
   - 在内核上加可编译时/运行时开关以强制回退，方便 A/B 对比。
   - 保证未匹配形状或缺少 `extra` 时立即回退旧路径，避免 crash。
9. **测试计划**
   - 单元/比对：扩展 `tests/test-ifairy-ref.py` 或新增 C 测试，构造固定 seed 的 matvec（1536×4096、1536×1536、4096×1536），比较 LUT on/off 的逐元素误差（real/imag 分开）。
   - 性能：在目标 ARM 设备上运行 `./build/bin/llama-cli ...` 与 `ggml_vec_dot_ifairy_q16_K` baseline，对比 tok/s；若需可加微基准（重复 1000 次）。
   - 工作区验证：在调试模式打印/断言 `wsize` 与偏移，确保无越界。
10. **工程化与维护**
    - 补充 CMake 选项与文档片段（README/IFAIRY_ARM_LUT_TL1.md 更新）说明如何启用/禁用 LUT。
    - 考虑脚本生成内核（参考 BitNet `preset_kernels` 模板）以覆盖更多形状，便于未来扩展。
    - 预留后续批量/Metal 扩展的接口占位（如 `lut_scales_size`、`n_tile_num` 设计兼容）。

## 交付物与完成判据
- 新的 LUT 内核头（或源码）与前向接入代码，受 `GGML_IFAIRY_ARM_LUT` 宏控制。
- `ggml_mul_mat` 路径能在匹配形状时使用 LUT，未匹配时保持旧行为，数值一致。
- 通过新增/扩展的单测与至少一组性能对比记录；文档更新描述使用方式与已知限制。

## 风险与缓解
- **形状覆盖不足**：优先支持 1536/4096 热点，预留生成脚本；回退逻辑保证正确性。
- **工作区尺寸错误**：在计算/使用处加入断言与对齐检查；重用 BitNet 的偏移顺序。
- **精度偏差**：独立 real/imag scale，添加测试覆盖；必要时保留 debug 模式打印中间尺度。
- **构建膨胀**：宏隔离 + 可选头文件生成；仅在 ARM NEON/DOTPROD 打开。
