# Fairy2i OpenCL Migration Guide

This page records the current OpenCL policy after the Fairy2i / legacy iFairy
cleanup: OpenCL is Fairy2i-only. Legacy iFairy remains supported by the CPU
backend, but OpenCL no longer builds, loads, or dispatches legacy iFairy kernels.

## Build Options

Fairy2i OpenCL has one feature gate:

```cmake
GGML_FAIRY2I_OPENCL=${GGML_FAIRY2I}
```

`GGML_OPENCL_EMBED_KERNELS=ON` is the only mode that requires Python3 for
OpenCL kernel embedding. Non-embedded OpenCL builds do not require Python3.

## Runtime Gates

Fairy2i OpenCL uses only Fairy2i runtime names:

```text
GGML_OPENCL_FAIRY2I=1
GGML_OPENCL_FAIRY2I_TILE64_MUL_MAT_IMPL=auto|gemm|gemv2|gemv4|direct
```

Legacy iFairy OpenCL runtime routing has been removed. Legacy iFairy tensors
and ops are reported unsupported by OpenCL so the scheduler can leave them on a
supported backend.

## Behavior Matrix

| Build options | Specialized kernels copied or embedded | Runtime gate |
| --- | --- | --- |
| `GGML_FAIRY2I_OPENCL=OFF` | none | none |
| `GGML_FAIRY2I_OPENCL=ON` | `complex_*`, `fairy2i_tile64` | `GGML_OPENCL_FAIRY2I=1` |

A regular OpenCL build with Fairy2i disabled does not copy or embed any
Fairy2i or legacy iFairy kernel files.

## Kernel Names

Fairy2i OpenCL builds copy or embed these specialized kernels:

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

The OpenCL source directory should not contain legacy iFairy kernel files or
host-side legacy iFairy symbols.

## Activation Staging

`fairy2i_tile64` stages activations with the internal `fairy2i_act_q16_64`
format:

```text
qk = 64 complex values
q  = int8 real[64] + int8 imag[64]
d  = fp16 real_scale + fp16 imag_scale
```

Each 64-wide activation block maps one-to-one to a tile64 weight block. OpenCL
tile64 matmul requires `K % 64 == 0`.

## Host Structure

Fairy2i OpenCL module state lives under:

```text
ggml/src/ggml-opencl/fairy2i/
```

The module owns Fairy2i compile/runtime env naming and kernel handles. The
central `ggml-opencl.cpp` still owns generic OpenCL backend state, buffer
management, program cache, and dispatch glue.

Support checks route Fairy2i operations through the internal capability path.
Current restrictions are: F32 packed-BF16 complex carrier tensors, contiguous
no-view tensors, default complex RoPE parameters, F32 output, and tile64 matmul
with q16_64 activation staging.

## Troubleshooting

- No Fairy2i OpenCL execution: check `GGML_FAIRY2I_OPENCL=ON` at build time and
  `GGML_OPENCL_FAIRY2I=1` at runtime.
- Legacy iFairy runs on CPU only; OpenCL intentionally reports it unsupported.
- OpenCL rejects a complex op: check dtype is `F32`, tensors are contiguous,
  and none of the tensors are views.
- OpenCL rejects complex RoPE: only default RoPE parameters and no `src2` are
  currently supported.
- OpenCL rejects tile64 matmul: current staging requires `K % 64 == 0`.

## Review Checklist

- Clean OpenCL builds do not copy or embed `complex_*` or `fairy2i_tile64`
  kernels.
- Fairy2i OpenCL code uses `fairy2i`, `tile64`, or `complex` names.
- OpenCL source has no legacy iFairy path, kernel, runtime env, tensor-extra, or
  dispatch hook.
- Capability rejects identify build gate, runtime gate, dtype/view/contiguity,
  shape, RoPE parameter, staging-block, or missing-extra causes.
- Runtime smoke tests are skipped, not passed, when no OpenCL device exists.

## Validation

Build isolation:

```bash
cmake -B build-opencl-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_EMBED_KERNELS=OFF \
  -DGGML_FAIRY2I_OPENCL=OFF \
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
  -DGGML_FAIRY2I_OPENCL=ON
cmake --build build-opencl-fairy2i --target ggml-opencl test-fairy2i test-backend-ops -j 4
GGML_OPENCL_FAIRY2I=1 ctest --test-dir build-opencl-fairy2i --output-on-failure -R fairy2i
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

If an OpenCL device is available, also run Fairy2i runtime smoke tests with
`GGML_OPENCL_FAIRY2I=1`. Otherwise, the test suite should report the runtime
portion as skipped.
