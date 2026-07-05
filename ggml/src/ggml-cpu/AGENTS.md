# ggml/src/ggml-cpu/AGENTS.md (Fairy2i / legacy iFairy CPU rules)

This directory is hot-path CPU code. Prefer minimal diffs, keep routing conditions explicit, and always preserve correctness.

## Non-negotiable semantic invariant

Must match the ggml baseline exactly: compute `w * conj(x)` (NOT `w * x`). See `docs/ifairy/legacy/IFAIRY_ARM_3W_LUT_DESIGN.md`.

## Current routing constraints (as implemented)

- Compile-time, Fairy2i: `GGML_FAIRY2I_CPU`, `GGML_FAIRY2I_CPU_LUT`, `GGML_FAIRY2I_CPU_AVX512`,
  `GGML_FAIRY2I_CPU_ARM_DOTPROD`
- Compile-time, legacy iFairy: `GGML_LEGACY_IFAIRY_CPU`, `GGML_LEGACY_IFAIRY_CPU_LUT`,
  `GGML_LEGACY_IFAIRY_CPU_AVX512`, `GGML_LEGACY_IFAIRY_CPU_ARM_DOTPROD`
- Deprecated aliases: `GGML_IFAIRY_LUT_CPU` maps only to `GGML_LEGACY_IFAIRY_CPU_LUT`, and
  `GGML_IFAIRY_FUSE_AVX512` maps only to `GGML_LEGACY_IFAIRY_CPU_AVX512`
- CPU LUT options are CPU-only; CMake disables accelerator backends when either Fairy2i or legacy iFairy
  CPU LUT is enabled
- Platform: LUT route requires `__aarch64__` + `__ARM_NEON` (otherwise fall back)
- Shape gate: `K % QK_K == 0` with `QK_K=256`
- Supported activations: `GGML_TYPE_F32` (bf16-pair complex container) or `GGML_TYPE_IFAIRY_Q16`
- Output type: `GGML_TYPE_F32` (written as bf16-pair when `pack_bf16=true`)

Primary integration points:
- Fairy2i CPU module: `ggml/src/ggml-cpu/fairy2i/`
- Legacy iFairy CPU module: `ggml/src/ggml-cpu/legacy-ifairy/`
- Fairy2i LUT helpers: `ggml/src/ggml-cpu/fairy2i/lut/`
- Legacy iFairy LUT helpers: `ggml/src/ggml-cpu/legacy-ifairy/lut/`
- CPU execution LUT/QGEMM sources are listed from
  `ggml/src/ggml-cpu/CMakeLists.txt`, not `ggml-base` sources
- Index encoding: `ggml/src/ggml-quants.c` (3W 6-bit pattern)

## Runtime toggles (current implementation)

- Fairy2i: `GGML_FAIRY2I_LUT=0/1`, `GGML_FAIRY2I_LUT_DEBUG=0/1`,
  `GGML_FAIRY2I_LUT_IMPL=auto|lut16|lut_c`
- Fairy2i W2 LUT16 dynamic tile claiming is enabled by default:
  `GGML_FAIRY2I_W2_DYNAMIC_TILES=0` disables it; it only affects W2 LUT with `N==1`;
  `GGML_FAIRY2I_W2_DYNAMIC_TILE_BATCH=1|2|4` controls claim batch size. Unset uses `2`, invalid values use `1`.
  Current M4 tg128 evidence prefers `batch=1` for low thread counts (`nth<=4`); consider `batch=2`
  only for higher-thread tg after path-validated measurement. Do not present `batch=2` as generally safe.
- Legacy iFairy: `GGML_IFAIRY_LUT=0/1`, `GGML_IFAIRY_LUT_DEBUG=0/1`,
  `GGML_IFAIRY_LUT_IMPL=auto|lut16|lut_c`
- Legacy tensor-scale vecdot policy remains under `GGML_IFAIRY_VEC_DOT_ACT_TENSOR`, but the route is
  owned by `ggml-cpu/legacy-ifairy/`, not the generic matmul policy in `ggml-cpu.c`

V2 keeps a single production LUT path and removes layout/kernel/tiling knobs to reduce surface area. Do not add new knobs unless strictly necessary.

If adding a new knob, document it in `docs/ifairy/v2/IFAIRY_ARM_3W_LUT_V2_STATUS.md` and keep a safe default.

## Formatting & Static Analysis (required)

Follow repo-root `AGENTS.md` for `git clang-format` / `clang-tidy` (diff-only, and only on the `.c/.cpp` files you touched).

## Validation gates (required for any LUT change)

1) Fairy2i release build/test:
- `cmake --build build-rel-fairy2i --target test-fairy2i -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`
- `GGML_FAIRY2I_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i`

2) Legacy direct build/test:
- `cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`
- `ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct`

3) Legacy LUT/full build/test:
- `cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`
- `GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy`

4) CLI sanity (quick smoke) + bench tok/s baseline when old iFairy weights are available:
- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`
- `GGML_IFAIRY_LUT=1 ./build-rel-lut/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 3`

Edge-case regression coverage is in `tests/test-fairy2i.cpp`, `tests/test-legacy-ifairy-direct.cpp`,
and `tests/test-legacy-ifairy.cpp`.

For the full CPU feature matrix, run `scripts/ci-fairy2i-cpu.sh`.

## Performance claims

- Use `eval tok/s` only, and always include the full command + env.
- Record results in `docs/ifairy/v2/IFAIRY_ARM_3W_LUT_V2_STATUS.md` (or link to raw logs/TSV paths).

## Current V2 core path (lut16)

- LUT layout: 16 entries × 4 channels × int8 per group (`k_ifairy_lut_group_bytes==64`)
- Weight layout: packed 16-row tiles (`struct ifairy_lut_wtile_16`), cached in `tensor->extra->packed_w`
- Kernel entrypoints:
  - `ggml_ifairy_lut_preprocess_ex_lut16()` (build per-column LUT tables)
  - `ggml_ifairy_lut_qgemm_lut16()` (mul_mat core; consumes packed tiles)
