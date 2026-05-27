# IFAIRY64 — 状态 / 性能记录

Status: Draft (2026-04-23)

本文件用于记录 `IFAIRY64` 相关的实现状态、性能数据和专项变更，避免继续混在 `IFAIRY_ARM_3W_LUT_V2_STATUS.md` 中。

相关文档：
- `IFAIRY64_LUT_IMPLEMENTATION_PLAN.md`
- `IFAIRY64_X86_ADAPTATION_EXECUTION_GUIDE.md`
- `IFAIRY_ARM_3W_LUT_V2_STATUS.md`（旧的 ARM 3W LUT V2 总状态）

---

## 变更记录（Changelog）

按日期追加（YYYY-MM-DD）：

### 2026-05-23 (x86 synthetic-model repro + LUT auto routing)
- 准备工作：
  - 解压 `generate_model_scripts_20260521.tar.gz` 后，使用本仓库 `models/ggml-vocab-llama-spm.gguf` 作为 tokenizer 参考，无需下载大模型即可生成低配置 smoke GGUF：
    - `python3 generate_model/generate_ifairy64_gguf_direct.py --name smoke256 --n-layer 1 --n-embd 256 --n-ff 256 --n-head 1 --vocab-size 32000 --ctx 256 --output-dir /tmp/ifairy64_smoke256 --ref-gguf models/ggml-vocab-llama-spm.gguf`
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel/bin/llama-cli -m /tmp/ifairy64_smoke256/ifairy64-smoke256/ifairy64.gguf -p hi -n 1 --no-warmup -c 256 -t 4 -ngl 0 -no-cnv`: smoke PASS
- 遗留 bench 修复：
  - `--ifairy64-lut-backend-bench` 现在用相同 warmup/iters 计时 `GGML_IFAIRY_LUT=0` baseline，并打印 `baseline_ms_per_iter`、`lut16_ms_per_iter` 和 `speedup_vs_baseline`。
- x86 routing:
  - `GGML_IFAIRY_LUT_IMPL=auto` 在 x86 上默认选择 `lut16`；aarch64+NEON 维持 Fairy2i F32 默认 `lut_c`。
- x86 观测（Machine: x86_64, AVX2/AVX512 available; synthetic `smoke256`, threads=4, `tg256`, `-b 1 -ub 1 -r 3`）：
  - vecdot baseline: `870.13 ± 63.38 tok/s`
  - explicit `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16`: `175.84 ± 13.24 tok/s`
  - explicit `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut_c`: `176.55 ± 15.59 tok/s`
  - core bench (`M=4096 N=1 K=1536 threads=4 warmup=5 iters=50`): `baseline_ms_per_iter=0.257044`, `lut16_ms_per_iter=1.400920`
  - 结论：x86 LUT 仍慢于 vecdot baseline；本轮先补齐 baseline 观测，并减少 x86 LUT 连续 bf16 输出的 store 开销。

### 2026-04-22 (working tree; base build `abcaafef`)
- 变更摘要：
  - `IFAIRY64` 在 `GGML_IFAIRY_LUT=1` 时于模型加载阶段提前完成 LUT transform/prepack，避免 decode 首轮再做 transform。
  - `IFAIRY64` 的 packed weight tile 不再把每 lane scale 展开成 `f32`，改为保留 `fp16` scale 并在 kernel 内按需转 `f32`，将 packed footprint 从 `384B/block-tile` 压回 `320B/block-tile`。
- Correctness:
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
- microbench（Machine: Mac16,12 / Apple M4）：
  - `./build-rel-lut/bin/ifairy-microbench --type ifairy64 --mode fused --m 3456 --k 2560 --iters 200 --warmup 20`
  - Result:
    - `ns/iter=322370.0`
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`; threads=4; `-dev none -ngl 0 -b 32 -ub 32 -fa 0 --no-warmup -p 0 -n 32 -r 1 -o md`）：
  - vecdot baseline:
    - `./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.22 ± 0.00 tok/s`
  - explicit LUT:
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench -m ~/fairy2i_32b/fairy2i_32b.gguf ...`
    - `tg32`: `2.46 ± 0.00 tok/s`
  - Delta vs vecdot:
    - `tg32`: `+10.8%`

### 2026-04-23 (working tree, Fairy2i 32B output-only merged IFAIRY64)
- 变更：
  - 新增 `LLAMA_FAIRY2I_MERGED_OUTPUT=1` opt-in。
  - load 阶段从 `output.{U,W}.s{0,1}` 逐行 `dequant + sum + requant` 合成两块 output-only `IFAIRY64` 权重：
    - `output.U.merged`
    - `output.W.merged`
  - 最终 lmhead 从 4 次 Fairy2i output matmul 缩成 2 次；中间层不变。
- 额外权重开销：
  - `load_tensors: prepared merged FAIRY2I output weights in 0.523 sec (2 x 58.01 MiB)`
- 验证：
  - `./build-rel-lut/bin/test-ifairy --ifairy-lut-only`: PASS
  - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-cli -m ~/fairy2i_32b/fairy2i_32b.gguf -dev none -ngl 0 -t 4 -c 8196 -b 32 -ub 32 -fa off --no-warmup -no-cnv --temp 0.2 --top-k 20 --top-p 0.9 -n 1 -p '<｜begin▁of▁sentence｜> You are a helpful AI assistant. <｜User｜> Where is China?'`: smoke PASS
- `llama-bench`（model=`~/fairy2i_32b/fairy2i_32b.gguf`, threads=4, `-p 0 -n 32 -r 1`）：
  - 默认 vecdot：
    - `./build-rel-lut/bin/llama-bench ...`: `tg32=2.24 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.26 tok/s`
    - 提升：`+0.02 tok/s`（`+0.9%`）
  - 显式 LUT：
    - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.42 tok/s`
    - `LLAMA_FAIRY2I_MERGED_OUTPUT=1 GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 ./build-rel-lut/bin/llama-bench ...`: `tg32=2.49 tok/s`
    - 提升：`+0.07 tok/s`（`+2.9%`）
- 结论：
  - 这条 output-only 合并对 LUT 和 vecdot 都是正收益，但收益量级符合 lmhead 占比，属于小幅提速。
  - 当前只做了 smoke 和 tok/s 验证，未做质量/困惑度回归。
