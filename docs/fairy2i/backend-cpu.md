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

cmake --build build-rel-fairy2i --target test-ifairy -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
./build-rel-fairy2i/bin/test-ifairy
ctest --test-dir build-rel-fairy2i --output-on-failure -R test-ifairy
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

## Current Limits

- `GGML_TYPE_FAIRY2I_TILE64_V2` is the storage type for new tile64_v2 weights.
- New graph code uses `GGML_OP_COMPLEX_*` plus
  `GGML_OP_FAIRY2I_WIDE_LINEAR_W2`.
- LUT configuration and threadpool fields now use Fairy2i naming, while the
  low-level kernel files are still being migrated.
- Reference/planner/dispatch split is started by the shim but not yet a full
  kernel registry.

## Correctness Gate

For changes touching Fairy2i CPU behavior, run:

```bash
./build-rel-fairy2i/bin/test-ifairy
ctest --test-dir build-rel-fairy2i --output-on-failure -R test-ifairy
```

The semantic invariant remains `w * conj(x)`.
