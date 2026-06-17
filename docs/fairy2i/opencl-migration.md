# Fairy2i OpenCL Migration Guide

This page tracks the OpenCL-side Fairy2i / legacy iFairy split.

## Build Options

Fairy2i OpenCL and legacy iFairy OpenCL are independent feature gates:

```cmake
GGML_FAIRY2I_OPENCL=${GGML_FAIRY2I}
GGML_LEGACY_IFAIRY_OPENCL=OFF
```

`GGML_OPENCL_EMBED_KERNELS=ON` is the only mode that requires Python3 for
OpenCL kernel embedding. Non-embedded OpenCL builds no longer require Python3.

## Runtime Gates

Fairy2i OpenCL uses the new runtime names:

```text
GGML_OPENCL_FAIRY2I=1
GGML_OPENCL_FAIRY2I_TILE64_MUL_MAT_IMPL=auto|gemm|gemv2|gemv4|direct
```

Legacy iFairy OpenCL keeps the legacy names:

```text
GGML_OPENCL_IFAIRY64=1
GGML_OPENCL_IFAIRY64_MUL_MAT_IMPL=auto|gemm|gemv2|gemv4|direct
```

The legacy names do not enable Fairy2i OpenCL kernels.

## Kernel Names

Fairy2i-only builds copy or embed these specialized kernels:

```text
complex_add
complex_merge
complex_mul
complex_relu2
complex_rms_norm
complex_rope
complex_split
fairy2i_tile64
```

Legacy iFairy-only builds copy or embed only the legacy kernels:

```text
ifairy_add
ifairy_merge
ifairy64
ifairy_mul
ifairy_relu2
ifairy_rope
ifairy_rms_norm
ifairy_split
```

A regular OpenCL build with both specialized gates disabled does not copy or
embed either set.

## Current Host Structure

OpenCL now has module directories:

```text
ggml/src/ggml-opencl/fairy2i/
ggml/src/ggml-opencl/legacy-ifairy/
```

The module files own compile/runtime env naming and module policy constants.
The central `ggml-opencl.cpp` still owns most host-side program/kernel fields,
tensor upload/download, scratch allocation, and compute dispatch. Keep further
cleanup incremental: move one hook family at a time, test, then commit.

Support checks are routed through two internal capability entry points:

```text
ggml_opencl_fairy2i_supports(...)
ggml_opencl_legacy_ifairy_supports(...)
```

They preserve current restrictions: F32 packed-BF16 complex tensors, contiguous
no-view tensors, default complex RoPE parameters, F32 output, and tile64
matmul with OpenCL's current 256-wide activation staging.

## Validation

Build isolation:

```bash
cmake -B build-opencl-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=OFF \
  -DGGML_FAIRY2I_OPENCL=OFF \
  -DGGML_LEGACY_IFAIRY_OPENCL=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE
cmake --build build-opencl-clean --target ggml-opencl -j 4
```

Fairy2i OpenCL:

```bash
cmake -B build-opencl-fairy2i \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=OFF \
  -DGGML_FAIRY2I=ON \
  -DGGML_FAIRY2I_OPENCL=ON \
  -DGGML_LEGACY_IFAIRY_OPENCL=OFF
cmake --build build-opencl-fairy2i --target ggml-opencl test-backend-ops -j 4
```

Legacy iFairy OpenCL:

```bash
cmake -B build-opencl-ifairy \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=OFF \
  -DGGML_FAIRY2I_OPENCL=OFF \
  -DGGML_LEGACY_IFAIRY_OPENCL=ON
cmake --build build-opencl-ifairy --target ggml-opencl -j 4
```

Embedded kernels:

```bash
cmake -B build-opencl-fairy2i-embed \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_FAIRY2I=ON \
  -DGGML_FAIRY2I_OPENCL=ON
cmake --build build-opencl-fairy2i-embed --target ggml-opencl -j 4
```

If an OpenCL device is available, also run the relevant runtime smoke tests with
`GGML_OPENCL_FAIRY2I=1` or `GGML_OPENCL_IFAIRY64=1`.
