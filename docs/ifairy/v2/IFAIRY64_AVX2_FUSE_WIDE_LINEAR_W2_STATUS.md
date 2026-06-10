# IFAIRY64 Fused Wide Linear W2 Status

Status: CPU fused implementation available (2026-06-11)

## 1. Implemented

`GGML_OP_IFAIRY_WIDE_LINEAR_W2` fuses the Fairy2i widely-linear layer:

```text
y = mul_mat(U.s0, x_conj) + mul_mat(U.s1, x_conj)
  + mul_mat(W.s0, x)      + mul_mat(W.s1, x)
  + optional_bias
```

The fused CPU path:

- replaces four matrix-multiplication graph nodes and intermediate additions
  with one graph node and one output write;
- quantizes the input activation once into K64 INT8 blocks;
- evaluates U0, U1, W0, and W1 in one K64 traversal;
- preserves the existing `weight * conj(input)` iFairy semantics;
- accumulates integer contributions before restoring per-block activation and
  weight scales;
- supports optional bias, multi-threaded execution, and a scalar fallback.

The graph path is opt-in:

```text
LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1
```

The dedicated CPU implementation lives in:

```text
ggml/src/ggml-cpu/ifairy-fuse.cpp
ggml/src/ggml-cpu/ifairy-fuse.h
```

## 2. ISA Kernels

### AVX2

The AVX2 kernel processes two weight branches in each 256-bit register. It
builds masks from packed 2-bit IFAIRY64 codes, selects signed INT8 activation
values with `_mm256_blendv_epi8`, and uses `_mm256_maddubs_epi16` for paired
INT16 accumulation.

### AVX512

An optional full-width AVX512 kernel processes U0, U1, W0, and W1 together in
one ZMM register. It reuses 64-bit k masks and uses 512-bit INT8
`vpmaddubsw`. The generated hot loop has no helper calls, `kmov` operations,
or YMM/ZMM stack spills.

Enable it at build time with:

```text
GGML_IFAIRY_FUSE_AVX512=ON
```

AVX512 performance is intentionally not recorded here until it is tested on a
CPU with native full-width 512-bit execution.

## 3. Correctness

Both CPU builds pass `test-ifairy`:

```powershell
.\build-rel-avx512\bin\test-ifairy.exe
.\build-rel-lut\bin\test-ifairy.exe
```

The fused W2 comparisons against the original four-matmul graph report maximum
packed-component differences of:

```text
0.000732, 0.001953, 0.001953, 0.001953
```

All are below the `0.010000` threshold.

A real Fairy2i-Llama 7B smoke test also produced identical text from the
original and fused graph:

```text
I believe life is too precious to
```

## 4. AVX2 Versus LUT16 Performance

Common configuration:

```text
CPU: AMD Ryzen 9 7940H
model: models/v1-20260605-115845/v1-20260605-115845.fairy2i.gguf
backend: CPU only
tests: pp64 and tg64
threads: 4 and 8
repetitions: 3
```

Common command:

```powershell
.\build-rel-lut\bin\llama-bench.exe `
  -m models/v1-20260605-115845/v1-20260605-115845.fairy2i.gguf `
  --threads 4,8 --n-prompt 64 --n-gen 64 `
  -ngl 0 --device none --repetitions 3
```

Environment configurations:

```powershell
# Non-fused LUT16
$env:GGML_IFAIRY_LUT='1'
$env:GGML_IFAIRY_LUT_IMPL='lut16'
$env:LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2='0'

# AVX2 fused W2
$env:GGML_IFAIRY_LUT='0'
$env:LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2='1'
```

Results:

| Path | Threads | pp64 tok/s | tg64 tok/s |
| --- | ---: | ---: | ---: |
| Non-fused LUT16 | 4 | 0.721312 | 0.724883 |
| AVX2 fused W2 | 4 | 3.223523 | 2.373005 |
| Non-fused LUT16 | 8 | 1.306274 | 0.785288 |
| AVX2 fused W2 | 8 | 5.045703 | 3.117279 |

AVX2 fused W2 is approximately `3.3x` to `4.5x` faster in this benchmark.

Raw logs:

```text
tmp/ifairy-bench/nonfused-lut16-performance-threads-4-8-pp64-tg64.jsonl
tmp/ifairy-bench/fused-avx2-performance-threads-4-8-pp64-tg64.jsonl
```

## 5. Next Work

- Investigate the current LUT16 performance issue before using it as a
  production baseline.
- Profile mask construction, scale restoration, and activation quantization in
  the fused path.
- Test the full-width AVX512 kernel on hardware with native 512-bit execution.
- Evaluate output-row tiling to reuse activation blocks across additional rows.
