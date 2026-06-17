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

Deprecated aliases are still accepted:

```cmake
GGML_IFAIRY_LUT_CPU      -> GGML_FAIRY2I_CPU_LUT
GGML_IFAIRY_FUSE_AVX512 -> GGML_FAIRY2I_CPU_AVX512
```

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

`ggml-cpu.c` calls this shim for `GGML_OP_IFAIRY_WIDE_LINEAR_W2` work-size and
compute dispatch. The existing optimized implementation files remain in their
current locations for compatibility, but they are no longer compiled unless
`GGML_FAIRY2I_CPU` is enabled.

## Current Limits

- `GGML_TYPE_IFAIRY64` remains the storage type for tile64_v2 weights.
- Existing `GGML_OP_IFAIRY_*` op ids are retained for ABI stability.
- LUT configuration and threadpool fields are still in `ggml-cpu.c`; moving
  them into the Fairy2i CPU module is a follow-up refactor.
- Reference/planner/dispatch split is started by the shim but not yet a full
  kernel registry.

## Correctness Gate

For changes touching Fairy2i CPU behavior, run:

```bash
./build-rel-fairy2i/bin/test-ifairy
ctest --test-dir build-rel-fairy2i --output-on-failure -R test-ifairy
```

The semantic invariant remains `w * conj(x)`.
