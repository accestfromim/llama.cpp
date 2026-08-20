# Apple M5 Row4 Metal TensorOps

This document describes the Apple M5-specific Row4 prefill path, its numeric
contract, runtime gates, validation, and performance evidence. The path changes
how the existing Row4 dot product is executed. It does not change the GGUF
format, activation quantizer, codebook, scales, accumulation semantics, or BF16
output boundary.

## Summary

Metal 4 MPP TensorOps can execute signed `INT8 x INT4 -> INT32` matrix
products on Apple M5 GPUs. Row4 cannot feed its packed nibbles directly to this
operation: one Row4 nibble is a codebook index shared by four output rows, not
one signed INT4 weight. Schema v1 uses the two exact prefill pipelines below.
Pair2 schema v2 uses its layout-aware portable path below B512 and a dedicated
device pre-expansion feeding the same large-prefill MPP kernel at B512 and
above:

```text
32 <= B < 512:
packed Row4 code stream
    -> exact four-row codebook lookup
    -> numeric signed INT4 in 16 KiB threadgroup memory
    -> MPP INT8 x INT4, cooperative INT32 accumulation
    -> existing row4_finish_i32 scale and BF16-RNE boundary

B >= 512:
packed Row4 code stream
    -> one full lossless row-major numeric INT4 expansion
    -> direct-device MPP INT8 x INT4, cooperative INT32 accumulation
    -> existing row4_finish_i32 scale and BF16-RNE boundary
```

On the Apple M5 Max used for development, the final results are:

| workload | comparison baseline | M5-tuned path | change |
| --- | ---: | ---: | ---: |
| pp512 | 952.270 tok/s | 3861.411 tok/s | 4.055x (+305.5%) |
| tg128 | 97.596 tok/s initial M4-tuned path | 104.458 tok/s | +7.03% |

The original M4-tuned path measured on the same M5 Max was 953.375 tok/s for
pp512 and 97.596 tok/s for tg128. The final M5 prefill result is 4.050x the
initial pp512 result. Relative to the first online-threadgroup MPP result
(3568.788 tok/s), full pre-expansion adds 8.20% end-to-end.

After merging the Pair2/LUT16 work from master, the same release binary measured:

| format and path | pp512 | tg128 |
| --- | ---: | ---: |
| schema v1 M5 | 3942.799 tok/s | 106.282 tok/s |
| Pair2 portable prefill | 969.701 tok/s | - |
| Pair2 M5 prefill + LUT16 decode | 3737.879 tok/s | 148.803 tok/s |

The Pair2 M5 prefill route is 3.855x (+285.5%) faster than the same-binary
portable Pair2 route. It gives up 5.20% pp512 versus schema v1 while the Pair2
LUT16 decode path improves tg128 by 40.0%, making Pair2 the faster combined
prefill/decode production format on this machine.

## Numeric contract

### Row4 is not ordinary INT4

The authoritative format remains
[`ROW4_QUANT_FORMAT_NOTES.md`](ROW4_QUANT_FORMAT_NOTES.md). Each Row4 code
selects four integer weights whose values are in `{-2, -1, 0, 1, 2}`. The M5
shader's `k_row4_m5_int4_codebook` is element-for-element identical to the
portable Row4 codebook, expressed as `char4`. `row4_m5_pack_int4()` stores those
already-integral values in signed INT4 without quantization, clipping, or
rounding.

### INT32 accumulation is required

Activations are clamped to `[-127, 127]`, and the maximum Row4 weight magnitude
is 2. The worst-case exact sums for production shapes are:

```text
K=4096:   4096 * 127 * 2 = 1,040,384
K=12288: 12288 * 127 * 2 = 3,121,152
```

Both overflow INT16. A BK128 partial sum is at most 32,512, but MPP's supported
integer destination is INT32 and merging narrower partial sums would add
conversion and synchronization without changing the required final range.
The selector additionally enforces the conservative general bound
`K <= INT32_MAX / (127 * 2)`.

The W8A8 output head has an even larger worst-case sum. Replacing its INT32
accumulator with F32 would lose integer exactness above `2^24`, so the existing
INT32 decode kernel is intentional rather than wasted precision.

### Threadgroup INT4 layout

The MPP weight tensor has logical extents `{output_tile, 128}` and element
strides `{1, 256}`. The second stride is measured in INT4 elements, so every K
row has a 128-byte physical pitch and the BK128 staging area is always 16 KiB:

```text
128 K rows * 128 bytes = 16 KiB
```

Even N32 and N64 tiles retain this pitch. The shader writes each row at
`k * 128 + output_pair`, then uses barriers before TensorOps consumes the tile
and before the next BK128 block overwrites it.

The cooperative INT32 accumulator remains live across every BK128 block. The
existing `row4_finish_i32()` applies the activation scale and signed BF16 row
scale with the original exact helpers, then returns the original BF16-RNE
boundary represented as F32.

### Large-prefill device INT4 layout

For B at least 512, every token tile reuses the same weights often enough that
one complete lossless expansion is faster than repeating the codebook lookup in
each threadgroup. `kernel_row4_m5_preexpand_int4` expands schema v1, while
`kernel_row4_m5_preexpand_int4_pair2` applies the exact Pair2 inverse
permutation for schema v2. Both produce ordinary signed-INT4 bytes in row-major
`{K, O}` order. The direct MPP kernel uses M32N128/BK128 with four SIMDgroups,
retains INT32 across all K, and applies `row4_finish_i32()` directly from the
cooperative accumulator.

The temporary is `O*K/2` bytes: 12 MiB for qkv, 8 MiB for attention output,
48 MiB for gate/up, and 24 MiB for down. It is backend-private trailing scratch,
not a persistent expanded model copy. Its absolute Metal-buffer offset is
64-byte aligned; allocation-aware lookup verifies that both the output and the
complete scratch range fit in one mapped Metal view. A buffer-scope barrier
publishes expansion writes before TensorOps reads them.

## Runtime gating and fallback

The optimized path requires all of the following:

- OS availability for the APIs introduced in 26.4;
- `MTLGPUFamilyApple10` and `MTLGPUFamilyMetal4`;
- runtime MSL 4.0 source compilation;
- successful creation of all selected expansion and M5 Row4 TensorOps pipelines;
- `GGML_METAL_ROW4_M5_TENSOROPS` not set to `0`.

All pipeline states are probed during device initialization. A source
compile failure removes the MPP macro and recompiles the portable shader
library; a pipeline failure disables the entire optional path. A precompiled
portable metallib also keeps MPP disabled. Non-embedded builds may enable the
path when `GGML_METAL_FORCE_SOURCE=1` is used and runtime source compilation
succeeds.

The operator selector also requires a Row4 linear operation, no fused residual,
and complete M/N tiles. Schema v1 requires K divisible by 128. Pair2 schema v2
requires K divisible by 256 and only selects the device-preexpanded route for B
at least 512; its smaller batches stay on the layout-aware portable path because
the online MPP staging kernels decode the schema v1 physical layout.
Unsupported shapes use the existing portable Metal path. W8A8 is not routed
through this selector.

Set this environment variable to make an explicit same-binary comparison:

```sh
GGML_METAL_ROW4_M5_TENSOROPS=0 ./build-rel-metal/bin/llama-bench ...
```

## Prefill tile selection

The measured selector is:

| activation rows | output condition | M/N tile | SIMD groups | threads/TG |
| --- | --- | --- | ---: | ---: |
| divisible by 256 | `O % 32 == 0` | M256N32 | 8 | 256 |
| divisible by 128 | `O % 32 == 0` | M128N32 | 4 | 128 |
| divisible by 64 | `O % 64 == 0` | M64N64 | 4 | 128 |
| divisible by 32 | `O % 128 == 0` | M32N128 | 4 | 128 |

For B at least 512 and divisible by 32, `O % 128 == 0` selects layout-specific
device pre-expansion plus direct M32N128 TensorOps before this online-staging
table. The grid is `{O / N_tile, B / M_tile, 1}`. When a TensorOps tile is
available, the old gate/up producer fusion is deliberately bypassed so both
Row4 linear operations can use the faster M5 path; exact SiLU and multiply
semantics are unchanged. MPP shapes allocate only the A8 activation and its F32
scale, not the portable path's unused half transpose.

The choices came from real qkv, attention-output, gate/up, and down shape
sweeps. The pre-expanded path additionally swept all 32 combinations of
TM={32,64,128,256}, TN={32,64,128,256}, and SG={4,8}; M32N128/SG4 was the
common winner. Every route is covered by opt-in markers and bit-exact tests.

## Why decode stays on the Row4 codebook kernel

MPP cooperative TensorOps require M to be a legal multiple of at least 8 for
this integer combination. M1, M2, and M4 are rejected at compile time; the
per-thread execution mode does not support `INT8 x INT4 -> INT32`. Reusing one
activation row with stride zero makes a padded M8 experiment numerically exact,
but still performs eight rows of work.

Measured sustained kernel time for that best padded experiment is:

| projection | existing M1 decode | padded M8 MPP | MPP penalty |
| --- | ---: | ---: | ---: |
| qkv | 27.58 us | 67.73 us | 2.46x slower |
| attention output | 24.23 us | 49.72 us | 2.05x slower |
| gate/up | 82.58 us | 209.17 us | 2.53x slower |
| down | 57.27 us | 145.18 us | 2.54x slower |

Persistently expanding all Row4 codes to numeric INT4 would also grow the Row4
weight stream from about one bit to four bits per real weight. The resulting
minimum decode stream is about 4.099 GB/token instead of 1.495 GB/token. With a
measured sustained read roof of 546.1 GB/s, its memory-only upper bound is about
133 tok/s. It is therefore not a sound default for decode.

The compressed-code decode sweep selected
`kernel_row4_w1a8_decode_o16_o4_staged_act`: four SIMDgroups/128 threads cover
one O16 tile while preserving the branchless basis/I32/BF16 implementation.
The K loop statically interleaves four BK128 blocks into four independent I32
accumulator banks, then merges the exact integer sums before the unchanged SIMD
reduction and BF16-RNE epilogue. This gives the M5 load/store machinery four
independent weight reads and dependency chains to overlap. Across the four
real projections the final implementation lowers summed operator latency by
6.43% versus O32. Full-model tg128 reaches 104.458 tok/s, 3.17% above the
previous O16 path and 7.03% above the initial M4-tuned path.
W8A8 output continues through the INT32 O128 kernel.

The tempting load-overlap alternatives were measured and rejected. Software
prefetch distances 2/4/8 increased summed kernel latency by about 5-6%; staging
the private Row4 code bytes through threadgroup memory added barriers without
reuse and was about 20% slower; pre-blitted private buffers and untracked
hazards were within about 0.3% of the normal shared/tracked buffers. The public
macOS 26.5 Metal shader API does not expose a CUDA-style asynchronous
device-to-threadgroup copy primitive. M5's second-generation dynamic caching
and occupancy manager therefore benefit most from independent live load/math
chains, not a software-managed copy pipeline, for this M1 codebook kernel.
M5 lossless compute texture compression was also tested with the real Row4
streams through private R8Uint textures populated by blit. It remained
bit-exact but slowed the four decode shapes by about 0.97-7.91%; the scalar
texture-read/coordinate overhead outweighed any compression benefit, so model
weights remain ordinary buffers.

For continuous batching the conclusion changes. A bit-exact padded M8
TensorOps prototype is not competitive for one or two rows, but at eight real
independent rows the four main Row4 projections take about 551 us versus 1278
us for eight separate native rows. A future multi-sequence selector should use
TensorOps in complete eight-row groups while preserving the compressed native
path for single-stream latency.

## Correctness evidence

`tests/test-row4.cpp` uses bit comparisons, not floating-point tolerances. The
final test matrix covers:

- B = 1, 2, 8, 9, 16, 17, 31, 32, 33, 64, 96, 128, 256, and 512;
- all four schema v1 online M5 tiles plus both layout-specific pre-expanded
  paths, with a required Pair2 B512 `device-preexpand` marker;
- real qkv O6144/K4096, output O4096/K4096, gate/up O24576/K4096, and down
  O4096/K12288 shapes;
- signed BF16 scales, QAT SwiGLU/down, decode residual fusion, and the full
  O151936 W8A8 output head;
- a separate process with M5 TensorOps disabled to test the portable fallback.

For a full-model check, a 1043-token prompt was evaluated once with the M5 path
and once with the portable path. All 151,936 final F32 logits, 607,744 bytes,
were byte-identical. Both files have SHA-256:

```text
eb7c5da4772fe2f92714b7eaa5901f75bb91817692a41683d089f7ebdd2bd7d7
```

A 32-token greedy generation check of the online MPP implementation was also
byte-identical. This proves that
the M5 path adds no loss relative to the original Row4 Metal route. It does not
replace the separate H100/training-reference acceptance work documented in the
format notes.

The four-bank decode change was additionally isolated from the previous
single-bank O16 kernel in two detached builds. After the same 1043-token prompt,
both builds selected token 151668, decoded that token through all four real
Row4 projection shapes, and exported all 151,936 next-token F32 logits. The
607,744-byte files were identical with SHA-256:

```text
1b71472143bf0e1722720b12cd347d48dc226cc844e55b098258755618db78bb
```

Run the strict checks with:

```sh
LLAMA_ROW4_REQUIRE_METAL_TESTS=1 \
LLAMA_ROW4_REQUIRE_M5_TENSOROPS_TESTS=1 \
LLAMA_ROW4_REAL_SHAPE_TESTS=1 \
LLAMA_ROW4_FULL_LM_HEAD_TESTS=1 \
./build-rel-metal/bin/test-row4

./build-rel-metal/bin/test-backend-ops test \
  -b Metal -o ROW4_LINEAR,W8A8_LINEAR

GGML_METAL_ROW4_M5_TENSOROPS=0 \
LLAMA_ROW4_REQUIRE_METAL_TESTS=1 \
LLAMA_ROW4_REAL_SHAPE_TESTS=1 \
LLAMA_ROW4_FULL_LM_HEAD_TESTS=1 \
./build-rel-metal/bin/test-row4
```

## Performance ceiling

The online M256N32 Row4 kernels measured 49.20, 46.34, 56.99, and 49.34
TOPS for qkv, attention output, gate/up, and down. A direct-device numeric
INT4 MPP roof probe reached 95.96 TOPS on a much larger M8192/N8192/K4096
matrix. With production shapes and BK128, the best direct-device tiles measured
81.08, 84.79, 85.01, and 82.74 TOPS. The Row4 path reaches roughly 55-67% of
those shape-specific upper bounds while additionally performing codebook
decode, staging, barriers, scale application, and the BF16 boundary. The B512
pre-expanded production-shape probe reaches about 67-69 total TOPS while timing
the complete expand+MPP+exact-epilogue chain, about 81-82% of the corresponding
production-shape direct-device roofs.

The decode minimum weight stream is approximately 1.4946 GB/token. The local
sustained Metal read probe reached 546.1 GB/s, giving a memory-only empirical
roof of about 365 tok/s. This is a weight-stream-only upper bound, not a
full-model prediction. The current full-model tg128 result is 104.46 tok/s.
This gap does not make numeric-INT4 expansion attractive: the output-head kernel
alone already reaches roughly 560 GB/s in the operator microbenchmark, while
Row4 codebook decode, integer issue rate, attention/KV work, epilogues, and
dispatch overhead limit the full graph. No claim is made that decode has
reached the absolute chip limit; the measured evidence instead explains why
the available M5 TensorOps API is not the limiting path for M1 decode.

## Reproduction and artifacts

The guarded benchmark harness serializes Metal runs, applies cooldown, verifies
the GGUF inventory, and requires runtime path markers:

```sh
MODEL=/path/to/qwen3-row4-v1.gguf
RESULTS_DIR=$PWD/tmp/row4-m5/results \
PYTHON=$PWD/.venv/bin/python \
perf/scripts/run_row4_bench.sh metal pp "$MODEL" m5-combined-final

GGML_METAL_ROW4_M5_TENSOROPS=0 \
RESULTS_DIR=$PWD/tmp/row4-m5/results \
PYTHON=$PWD/.venv/bin/python \
perf/scripts/run_row4_bench.sh metal pp "$MODEL" m5-combined-portable

MODEL=/path/to/qwen3-row4-v2-pair2.gguf
RESULTS_DIR=$PWD/tmp/row4-m5/results \
PYTHON=$PWD/.venv/bin/python \
perf/scripts/run_row4_bench.sh metal pp "$MODEL" m5-pair2
```

Raw local evidence from the development machine is under these ignored paths:

- `tmp/row4-m5/results/m5-combined-ultra-final.metal.pp.20260812T032744Z.*`
- `tmp/row4-m5/results/m5-combined-ultra-final.metal.tg.20260812T032809Z.*`
- `tmp/row4-m5/results/m5-decode-bank4-final.metal.tg.20260813T025735Z.*`
- `tmp/row4-m5/results/m5-combined-portable-ultra-final.metal.pp.20260812T032838Z.*`
- `tmp/row4-m5/microbench/mpp-preexpand-row4-fused-b512-v2.txt`
- `tmp/row4-m5/microbench/mpp-preexpand-tile-sweep-v3.txt`
- `tmp/row4-m5/microbench/mpp-m256n32-prefill-v1.txt`
- `tmp/row4-m5/microbench/mpp-decode-sustained.txt`
- `tmp/row4-m5/microbench/row4-decode-dependency-v3.txt`
- `tmp/row4-m5/microbench/row4-decode-overlap-v3-interleaved.txt`
- `tmp/row4-m5/microbench/mpp-decode-batch-v1.txt`
- `tmp/row4-m5/microbench/row4-texture-compression-real-shapes.txt`
- `tmp/row4-m5/microbench/mpp-roofline-production.txt`
- `tmp/row4-m5/microbench/metal-read-roofline-sustained.txt`
- `tmp/row4-m5/quality/{m5,classic}-last-logits-recovered-final.f32`
- `tmp/row4-m5/quality/decode-bank4-ab-20260813/`
- `tmp/row4-m5/validation/test-row4-recovered-final.log`
- `tmp/row4-m5/validation/test-row4-decode-bank4-final.log`
- `tmp/row4-m5/validation/test-backend-ops-recovered-final.log`
- `tmp/row4-m5/validation/test-row4-portable-recovered-final.log`

Do not run concurrent `llama-bench` processes. These artifacts are machine-local
evidence and are not model or repository assets.

## Current limitations

- Only complete Row4 prefill tiles use MPP. Decode uses the tuned compressed O16
  kernel; small batches and W8A8 retain their integer kernels.
- The portable precompiled metallib does not contain MSL 4 TensorOps kernels;
  runtime source compilation is required.
- Online staging is fixed at 16 KiB. BK256 would consume the full 32 KiB
  threadgroup-memory budget on this device and reduced occupancy in experiments.
- MPP cannot consume the Row4 codebook nibble directly, so exact online staging
  or a lossless temporary numeric-INT4 expansion is unavoidable with this API.
- The measured MPP and memory roofs are empirical results for this machine,
  operating system, driver, and compiler, not published chip peak values.
