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

## Build Examples

Feature-off smoke:

```bash
cmake -B build-rel-clean -DCMAKE_BUILD_TYPE=Release -DGGML_FAIRY2I=OFF
cmake --build build-rel-clean --target llama -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

Feature-on CPU LUT:

```bash
cmake -B build-rel-fairy2i \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=ON

cmake --build build-rel-fairy2i --target test-fairy2i -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_FAIRY2I_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
```

Legacy iFairy CPU LUT:

```bash
cmake -B build-ifairy-legacy \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=ON

cmake --build build-ifairy-legacy --target test-legacy-ifairy -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
```

## Current Structure

The CPU backend now has a narrow Fairy2i shim:

```text
ggml/src/ggml-cpu/fairy2i/fairy2i-cpu.h
ggml/src/ggml-cpu/fairy2i/fairy2i-cpu.cpp
```

`ggml-cpu.c` calls this shim for `GGML_OP_FAIRY2I_WIDE_LINEAR_W2` work-size and
compute dispatch. Legacy iFairy W2 fused and LUT execution belong to a separate
legacy backend path and must remain runnable when `GGML_FAIRY2I=OFF`.

The LUT files are split by runtime surface:

```text
ggml/src/ggml-fairy2i-lut*
ggml/src/ggml-cpu/fairy2i/*lut*
ggml/src/ggml-ifairy-lut*
ggml/src/ggml-cpu/legacy-ifairy/*lut*
```

## Current Limits

- `GGML_TYPE_FAIRY2I_TILE64_V2` is the storage type for new tile64_v2 weights.
- New graph code uses `GGML_OP_COMPLEX_*` plus
  `GGML_OP_FAIRY2I_WIDE_LINEAR_W2`.
- Fairy2i LUT configuration uses `GGML_FAIRY2I_LUT*`; legacy iFairy LUT
  configuration uses `GGML_IFAIRY_LUT*`.
- Reference/planner/dispatch split is started by the shim but not yet a full
  kernel registry.

## Correctness Gate

For changes touching Fairy2i CPU behavior, run:

```bash
cmake --build build-rel-fairy2i --target test-fairy2i -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_FAIRY2I_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
```

For changes touching legacy iFairy CPU behavior, also run:

```bash
cmake --build build-ifairy-legacy --target test-legacy-ifairy -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
```

The semantic invariant remains `w * conj(x)`.
