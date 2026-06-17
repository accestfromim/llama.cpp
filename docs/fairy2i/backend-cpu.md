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

cmake --build build-rel-fairy2i --target test-fairy2i -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_FAIRY2I_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
GGML_FAIRY2I_LUT=1 ./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
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

`ggml-cpu.c` calls this shim for `GGML_OP_FAIRY2I_WIDE_LINEAR_W2` work-size,
graph prepare, free, and compute dispatch. Fairy2i LUT env parsing, impl
selection, weight prepack, scratch planning, and execution policy live under
`ggml/src/ggml-cpu/fairy2i/`; `ggml_threadpool` does not store Fairy2i LUT
configuration.

Legacy iFairy W2 fused, vecdot tensor-scale activation, and LUT execution
belong to `ggml/src/ggml-cpu/legacy-ifairy/` and must remain runnable when
`GGML_FAIRY2I=OFF`.

The LUT files are split by runtime surface:

```text
ggml/src/ggml-fairy2i-lut*
ggml/src/ggml-cpu/fairy2i/*lut*
ggml/src/ggml-ifairy-lut*
ggml/src/ggml-cpu/legacy-ifairy/*lut*
```

Those root-level LUT helper files are CPU backend sources. They are not compiled
into `ggml-base`; `ggml-base` keeps block formats, type traits, and reference
quantization only.

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
cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct
cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
```

The semantic invariant remains `w * conj(x)`.
