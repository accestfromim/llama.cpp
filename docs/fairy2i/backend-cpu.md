# Fairy2i CPU Backend

Fairy2i CPU support is now behind explicit CMake feature gates. A normal CPU
build does not compile the experimental Fairy2i fused or LUT sources unless the
feature is enabled.

## CMake Options

```cmake
GGML_FAIRY2I=OFF
GGML_FAIRY2I_CPU=${GGML_FAIRY2I}
GGML_FAIRY2I_CPU_LUT=OFF
GGML_FAIRY2I_CPU_AVX512=OFF
GGML_FAIRY2I_CPU_ARM_DOTPROD=ON
```

Legacy iFairy aliases are still accepted, but only as legacy options:

```cmake
GGML_IFAIRY_LUT_CPU      -> GGML_LEGACY_IFAIRY_CPU_LUT
GGML_IFAIRY_FUSE_AVX512 -> GGML_LEGACY_IFAIRY_CPU_AVX512
```

They do not enable Fairy2i CPU features.

When `GGML_FAIRY2I_CPU_LUT=ON`, CMake keeps the historical CPU-only LUT
behavior and disables accelerator backends.

## Behavior Matrix

| Build options | Fairy2i W2 | Fairy2i LUT | legacy iFairy W2/vecdot | legacy iFairy LUT |
| --- | --- | --- | --- | --- |
| `GGML_FAIRY2I=OFF`, `GGML_LEGACY_IFAIRY_CPU=OFF` | no | no | no | no |
| `GGML_FAIRY2I=ON`, `GGML_FAIRY2I_CPU=ON` | yes | no | no unless legacy is also enabled | no |
| `GGML_FAIRY2I_CPU_LUT=ON` | yes | yes, defaults to LUT16; `GGML_FAIRY2I_LUT=0` disables | no unless legacy is also enabled | no |
| `GGML_LEGACY_IFAIRY_CPU=ON`, `GGML_FAIRY2I=OFF` | no | no | yes | no |
| `GGML_LEGACY_IFAIRY_CPU_LUT=ON`, `GGML_FAIRY2I=OFF` | no | no | yes | yes, via `GGML_IFAIRY_LUT*` |

Deprecated aliases only affect the legacy columns. They are accepted so old
build scripts keep working, but they must not enable Fairy2i CPU code.

## Build Examples

Feature-off smoke:

```bash
cmake -B build-cpu-clean \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=OFF
cmake --build build-cpu-clean --target ggml-base ggml-cpu -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

Feature-on CPU LUT:

```bash
cmake -B build-rel-fairy2i \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=ON

cmake --build build-rel-fairy2i --target test-fairy2i test-fairy2i-loader -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
GGML_FAIRY2I_LUT=0 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
GGML_FAIRY2I_LUT=0 ./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
```

Legacy iFairy CPU direct-only:

```bash
cmake -B build-ifairy-direct \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF

cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct
```

Legacy iFairy CPU LUT/full:

```bash
cmake -B build-ifairy-legacy \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=ON

cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
```

## Current Structure

The CPU backend now has a narrow Fairy2i shim:

```text
ggml/src/ggml-cpu/fairy2i/fairy2i-cpu.h
ggml/src/ggml-cpu/fairy2i/fairy2i-cpu.cpp
```

`ggml-cpu.c` calls this shim for `GGML_OP_FAIRY2I_WIDE_LINEAR_W1` and
`GGML_OP_FAIRY2I_WIDE_LINEAR_W2` work-size, graph prepare, free, and compute
dispatch. Fairy2i LUT env parsing, impl selection, weight prepack, scratch
planning, and execution policy live under
`ggml/src/ggml-cpu/fairy2i/`; `ggml_threadpool` does not store Fairy2i LUT
configuration.

Legacy iFairy W2 fused, vecdot tensor-scale activation, and LUT execution
belong to `ggml/src/ggml-cpu/legacy-ifairy/` and must remain runnable when
`GGML_FAIRY2I=OFF`.

The LUT files are split by runtime surface:

```text
ggml/src/ggml-cpu/fairy2i/lut/ggml-fairy2i-lut*
ggml/src/ggml-cpu/fairy2i/*lut*
ggml/src/ggml-cpu/legacy-ifairy/lut/ggml-ifairy-lut*
ggml/src/ggml-cpu/legacy-ifairy/*lut*
```

Those LUT helper files are CPU backend sources. They are not compiled into
`ggml-base`; `ggml-base` keeps block formats, type traits, and reference
quantization only.

ARM files are also split by owner:

```text
ggml/src/ggml-cpu/fairy2i/arm/
ggml/src/ggml-cpu/legacy-ifairy/arm/
```

Fairy2i-only builds no longer compile `quants-ifairy.*`.

## Migration Map

| Old name | Current owner |
| --- | --- |
| `GGML_IFAIRY_LUT_CPU` | deprecated alias for `GGML_LEGACY_IFAIRY_CPU_LUT` |
| `GGML_IFAIRY_FUSE_AVX512` | deprecated alias for `GGML_LEGACY_IFAIRY_CPU_AVX512` |
| `GGML_IFAIRY_LUT*` | legacy iFairy runtime only |
| `GGML_FAIRY2I_LUT*` | Fairy2i runtime only |
| `GGML_TYPE_IFAIRY64` | legacy iFairy tile64 storage |
| `GGML_TYPE_FAIRY2I_TILE64_V2` | Fairy2i tile64_v2 storage |
| `GGML_OP_IFAIRY_WIDE_LINEAR_W2` | legacy iFairy W2 op |
| `GGML_OP_FAIRY2I_WIDE_LINEAR_W1` | Fairy2i W1 learned-scale op |
| `GGML_OP_FAIRY2I_WIDE_LINEAR_W2` | Fairy2i W2 op |

## Troubleshooting

- Fairy2i W1/W2 is unsupported: check `GGML_FAIRY2I=ON` and
  `GGML_FAIRY2I_CPU=ON`.
- Fairy2i LUT is not selected: check `GGML_FAIRY2I_CPU_LUT=ON` at build time
  and make sure `GGML_FAIRY2I_LUT=0` is not set at runtime.
- Fairy2i W1/W2 path or timing is unclear: set `GGML_FAIRY2I_CPU_DEBUG=1`
  for first-hit path logs and `GGML_FAIRY2I_CPU_TIMING=1` for per fused
  wide-linear timing logs.
- Legacy iFairy direct vecdot is unavailable: check
  `GGML_LEGACY_IFAIRY_CPU=ON`; old `GGML_IFAIRY_LUT_CPU` only enables the
  legacy LUT alias path.
- Legacy tensor-scale activation quantization is unavailable: check the
  legacy module is compiled and `GGML_IFAIRY_VEC_DOT_ACT_TENSOR` is set.
- A clean CPU build unexpectedly compiles LUT sources: inspect
  `ggml/src/ggml-cpu/CMakeLists.txt`, not `ggml/src/CMakeLists.txt`.

## Review Checklist

- `ggml-cpu.c` has no `fairy2i_lut_cfg`, `GGML_IFAIRY_VEC_DOT_ACT_TENSOR`, or
  direct `quantize_row_ifairy_q16_tensor()` policy.
- Fairy2i changes pass `test-fairy2i`.
- Legacy iFairy changes pass `test-legacy-ifairy-direct`, and LUT changes also
  pass `test-legacy-ifairy`.
- New runtime knobs are documented here and, for legacy iFairy LUT, in the V2
  status docs.

## Current Limits

- `GGML_TYPE_FAIRY2I_TILE64_V2` is the storage type for new tile64_v2 weights.
- New graph code defaults to Fairy2i W1/W2 fused wide-linear ops when the model,
  LoRA state, tensor types, and target CPU backend support the fused op. Set
  `LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W1=0` or
  `LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=0` to force the unfused graph.
- Fairy2i LUT defaults to LUT16 when `GGML_FAIRY2I_CPU_LUT=ON`; set
  `GGML_FAIRY2I_LUT=0` to force the direct CPU path. Legacy iFairy LUT
  configuration uses `GGML_IFAIRY_LUT*`.
- Reference/planner/dispatch split is started by the shim but not yet a full
  kernel registry.

## Correctness Gate

For changes touching Fairy2i CPU behavior, run:

```bash
cmake --build build-rel-fairy2i --target test-fairy2i test-fairy2i-loader -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
GGML_FAIRY2I_LUT=0 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
```

For changes touching legacy iFairy CPU behavior, also run:

```bash
cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct
cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
```

The semantic invariant remains `w * conj(x)`.

For the complete baseline/Fairy2i/legacy matrix:

```bash
scripts/ci-fairy2i-cpu.sh
```
