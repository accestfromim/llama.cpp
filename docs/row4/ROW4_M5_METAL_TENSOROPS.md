# Apple M5 Row4 Metal TensorOps

This document describes the Apple M5-specific Row4 prefill and decode paths, their numeric
contract, runtime gates, validation, and performance evidence. The path changes
how the existing Row4 dot product is executed. It does not change the GGUF
format, activation quantizer, codebook, scales, accumulation semantics, or BF16
output boundary.

## Summary

Metal 4 MPP TensorOps can execute signed `INT8 x INT4 -> INT32` matrix
products on Apple M5 GPUs. Row4 cannot feed its packed nibbles directly to this
operation: one Row4 nibble is a codebook index shared by four output rows, not
one signed INT4 weight. Both schema v1 and Pair2 schema v2 use the two exact
prefill pipelines below when their shape gates are satisfied. Pair2 uses a
layout-aware load for online staging and a dedicated device pre-expansion for
large prefills:

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

The following tables record the earlier M5 implementation measurements. The
2026-09-05 Pair2 follow-up, 2026-09-10 coalesced expansion and cooperative
M64 stores, 2026-09-11 blocked expansion, and 2026-09-12 RMSNorm/A8 fusion
and scheduling overlap are described separately below.

On the Apple M5 Max used for that development, the results were:

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
`{K, O}` order. Pair2 with B divisible by 64 and K divisible by 512 uses
M64N64/SG4, or M64N128/SG8 for O at least 16384. These full tiles use static
extents, short groups of token tiles for cache reuse, and cooperative output
stores. The exact `row4_finish_i32()` F32 result bits replace the final INT32
accumulator elements before an INT32 cooperative store; no numeric conversion
is performed by the store. Other large-prefill shapes retain M32N128/SG4,
with BK512 when K is divisible by 512, otherwise BK128. All routes retain
INT32 accumulation across K and the same BF16 output boundary.
Pair2 expansion uses an equivalent packed ushort codebook. Each 128-thread
group covers O128 x K32, with K blocks varying fastest in the two-dimensional
grid. A thread reads two adjacent K positions for two four-output groups,
using two aligned ushort loads. It packs each pair of codebook results into a
32-bit store for eight output coefficients; the four stores cover K, K+1,
K+8, and K+9. Nearby threads reuse source cache lines while keeping stores
contiguous along O. The selector guarantees complete O128/K32 blocks, so no
tail masking or persistent weight state is needed.

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

The operator selector also requires a Row4 linear operation and no fused residual.
Complete output tiles are required; selected small batches pad activation rows
in backend scratch. Schema v1 requires K divisible by 128. Pair2 schema v2
requires K divisible by 256. It selects device pre-expansion for B at least 512
when the large-prefill shape gates match, and layout-aware online MPP for
eligible B at least 32. Smaller Pair2 batches combine independent-row LUTs,
shared-weight LUTs, and shape-selected MPP as described below. Unsupported shapes
use the existing portable Metal path. The W8A8 output head uses direct-device
M8N16, M16N16, or M32N16 `INT8 x INT8 -> INT32` TensorOps for 4-32 rows.

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
| schema v1, 2-16 rows after independent-row gates | `O % 64 == 0` | M16N64 | 4 | 128 |
| schema v1, complete M8 groups above 16 | `O % 128 == 0` | M8N128 | 4 | 128 |

For B at least 512 and divisible by 32, `O % 128 == 0` selects layout-specific
device pre-expansion before this online-staging table. Pair2 selects the M64
tiles described above when `B % 64 == 0 && K % 512 == 0`; other shapes and
schema v1 retain M32N128. M64N64 groups two token tiles, and M64N128 groups
four, before advancing along O. Partial final groups are supported.
The grid is `{O / N_tile, B / M_tile, 1}`. When a TensorOps tile is
available, the old gate/up producer fusion is deliberately bypassed so both
Row4 linear operations can use the faster M5 path; exact SiLU and multiply
semantics are unchanged. MPP shapes allocate only the A8 activation and its F32
scale, not the portable path's unused half transpose.

The earlier choices came from real qkv, attention-output, gate/up, and down shape
sweeps. The pre-expanded path additionally swept all 32 combinations of
TM={32,64,128,256}, TN={32,64,128,256}, and SG={4,8}; M32N128/SG4 was the
common winner for the original scalar epilogue. The later cooperative-store
sweep selected M64 tiles for eligible Pair2 shapes. Every route is covered by
opt-in markers and bit-exact tests.

## Pair2 profiling follow-up, 2026-09-05

Starting at `fe2339efe0676d5d6c0aa31af22bd5e89d02dfb7`, the original W8A8-head
Pair2 8B model was measured on M5 Max with 40 GPU cores and 128 GB memory.
The resulting implementation and its tests are committed as `8f006558`.
Release, eight host threads, b2048/ub512, BF16 KV, FA enabled, full GPU offload,
mmap and warmup were kept fixed. Each workload used an A/B/B/A sequence with
five repetitions per run and at least 15 seconds between runs. These are
uninstrumented full-model means, not Shader Timeline estimates:

| workload | original tok/s | updated tok/s | change |
| --- | ---: | ---: | ---: |
| pp128 | 851.629 | 2959.236 | +247.48% |
| pp512 | 3568.206 | 3590.219 | +0.62% |
| tg128 | 140.855 | 143.090 | +1.59% |

The large pp128 gain comes from enabling exact online MPP staging for Pair2.
Single-stream Pair2 decode additionally uses packets of eight FP16 products
before widening to F32. Each packet and every intermediate integer sum is
bounded by `8 * 128 * 2 = 2048`, so this transformation is exact. Full-K sums
remain exact in F32 through K65536. M5 uses eight SIMDgroups and two K
partitions per O32 tile, then reuses the dead activation staging storage for
the final integer reduction. Dispatch geometry keeps the four-SIMDgroup
fallback valid when the optional M5 path is disabled. Four K partitions and
BK4096 were measured and rejected after regressions.

Hardware counter samples identified high last-level-cache utilization in
large-prefill MPP, low occupancy and unused F16 capacity in Pair2 decode, and
about 493 GB/s external reads in the W8 output head. Sparse Shader Timeline
samples covered only 12.44% of the selected pp512 window and 16.44% of the
decode window; their normalized shader shares are not full-model time shares.
Long prefill and single-stream decode gains remain modest.

Validation covered real projection shapes, the full W8 head, portable gates,
K65536, fusion boundaries, and 128- and 643-token inputs each followed by 128
greedy decode steps. All 39,199,488 prefill/decode logits matched the original
binary byte for byte. Raw traces, tests, binary hashes, benchmark JSON, commands and
the detailed report are in the local artifact directory
`~/row4-metal-profile-20260905/`.

## Pair2 coalesced expansion follow-up, 2026-09-10

Changing the Pair2 expansion thread order reduced its pp512-weighted GPU time
from 19.210 ms to 11.219 ms (41.60%). The complete A8/expand/MPP/epilogue chain
fell from 112.596 ms to 104.626 ms. These are warm operator microbenchmarks,
weighted by 36/36/35/35 batched qkv/o/gate_up/down projections, not an exclusive
full-graph profiler breakdown. Both candidate threadgroup sizes, 128 and 256,
produced byte-identical INT4 buffers and outputs for all four real projections;
256 was faster overall and is the production choice.

Full-model measurements against commit `fe79b5dc5a449922caa64a37451948f30c124318`:

| workload | original tok/s | coalesced tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 3705.970 | 3935.643 | +6.20% |
| pp2048 | 3251.274 | 3424.861 | +5.34% |
| pp128 | 3017.314 | 3017.659 | +0.01% |
| tg128 | 145.856 | 145.035 | -0.56% |

Each workload used serial A/B/B/A launches, five samples per launch, all ten
samples per version included in the mean, warmup, and at least 15 seconds
between runs. The guarded Row4 harness verified the model inventory and paths.
Only Pair2 device pre-expansion changes; pp128 and tg128 are unchanged-path
controls. The new prefill marker appends `coalesced-expand` to the existing
`M32N128 BK512 device-preexpand` or `BK128 device-preexpand` marker.

Hardware: Apple M5 Max, 18 CPU / 40 GPU cores, 128 GB, macOS 26.5.1, AC power,
automatic power mode (`powermode=0`), no Linux governor or fixed clocks.
Build: Apple clang 21.0.0, Release `-O3 -DNDEBUG`, `GGML_NATIVE=ON`,
`GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, MSL 4.0. Workload: Qwen3 8B
Pair2 schema v2 with its original W8A8 head, eight host threads, b2048/ub512,
BF16 KV, FA1, ngl99, mmap1, empty-history prefill and last-token logits.
Both binaries and dynamic libraries were pinned separately for the comparison.

There is no persistent expansion cache or added scratch. Both versions use a
314.77 MiB Metal compute buffer for pp512. Strict M5 and portable real-shape
Row4 tests, the full output-head matrix, and relevant Metal backend-op tests
passed. A new O1152/K768/B544 test covers non-power-of-two dimensions and a
partial expansion threadgroup. Full-model pp512, pp643, and pp2048 followed
by 128 greedy decode steps matched the original full-vocabulary logits and
generated text byte for byte: 58,799,232 float values in total.

Local raw evidence is under `/Users/1806-admin/row4-prefill-opt-20260910/`:
`full-model/*.json`, `*.meta`, `*.paths.log`, and `*.stderr.log` record every
launch; `full-model-summary.json` lists their names and means. The same directory
contains `prototypes/`, `quality/summary.json`, exact test logs, the source patch,
a binary SHA-256 manifest, and `REPORT.zh-CN.md`. Reproduction drivers are
`run-bench.py` and `quality/run-quality.py`. Measurements exclude initial model
loading and shader compilation.

## Pair2 cooperative M64 stores follow-up, 2026-09-10

Starting from the coalesced expansion version, the full-tile Pair2 prefill path
now uses static tensor extents, grouped token-tile traversal, and cooperative
INT32 stores of the exact F32 epilogue bits. The M64N64/SG4 path serves qkv,
attention output, and down; wide projections use M64N128/SG8. The selected
complete Row4 chain fell from 104.872 ms to 97.302 ms in the weighted operator
probe, and GEMM plus epilogue fell from 86.875 ms to 78.965 ms. No buffers were
added, and the pp512 Metal compute buffer remains 314.77 MiB.

Full-model A/B/B/A measurements against the coalesced expansion version:

| workload | coalesced baseline tok/s | M64 cooperative tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 3935.473 | 4091.295 | +3.96% |
| pp2048 | 3423.312 | 3543.686 | +3.52% |
| pp128 | 3017.320 | 3010.026 | -0.24% |
| tg128 | 144.882 | 147.351 | +1.70% |

The same Qwen3 8B Pair2/W8A8-head model and M5 Max configuration described in
the preceding section were used: eight host threads, b2048/ub512, BF16 KV,
FA1, ngl99, mmap1, warmup, five repetitions per launch, and at least 15 seconds
between serial launches. Each mean includes ten samples per version. Baseline
is `fe79b5dc` plus the first coalescing patch; binaries and dynamic libraries
were separately pinned. pp128 and tg128 remain unchanged-path controls.

Strict M5/portable real-shape tests and relevant backend-op tests passed.
Random codes distinguish output tiles in the new O128/K512/B512,
O1152/K1024/B576, and O16384/K512/B512 oracle cases. A 17-token input period
also exposes row-tile permutations. Full-vocabulary logits for pp512, pp643,
and pp2048, each followed by 128 greedy decode steps, match the coalesced
baseline byte for byte (58,799,232 F32 values); generated text is identical.
Runtime tests require both new `cooperative-store` path markers.

The experiment follows the tile, static-extents, walk-order and cooperative
store guidance in Apple's [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
Candidates that failed exact output comparisons were excluded. A separate
index-arithmetic simplification showed no stable chain improvement and was
not retained. Operator timings are hot-buffer service times, not exclusive
full-model phase costs.

Raw evidence, source and binary manifests, and the reproducible serial harness
are under `/Users/1806-admin/row4-prefill-opt2-20260910/`: `full-model/`,
`full-model-summary.json`, `prototypes/`, `quality/`, the test logs, and
`REPORT.zh-CN.md`. No persistent INT4 cache or ubatch change is part of this
optimization.

## Pair2 blocked expansion follow-up, 2026-09-11

The Pair2 pre-expansion kernel now uses O128 x K32 blocks, two aligned ushort
loads per thread, and four 32-bit stores. K blocks vary fastest in the grid.
This improves source locality while preserving contiguous output stores and
the existing numeric INT4 layout. The same M64/M32 GEMM paths consume it.
No persistent cache or additional allocation was introduced; the pp512 Metal
compute buffer remains 314.77 MiB.

Full-model A/B/B/A against the preceding M64 cooperative version:

| workload | M64 baseline tok/s | blocked expansion tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 4082.464 | 4242.469 | +3.92% |
| pp2048 | 3542.433 | 3658.217 | +3.27% |
| pp128 | 3006.463 | 3007.952 | +0.05% |
| tg128 | 147.869 | 145.396 | -1.67% |

The same Qwen3 8B Pair2/W8A8-head model and M5 Max were used: eight host
threads, b2048/ub512, BF16 KV, FA1, ngl99, mmap1, warmup, five samples per
launch, and at least 15 seconds between serial launches. Each mean includes
ten samples per version. No samples were removed. This measures warm eval
throughput, excluding model loading and initial shader compilation.

The tg128 decrease repeated in an additional A/B/B/A (-1.71%). Pooling all
20 samples per version gives 147.761 -> 145.260 tok/s (-1.69%).
Both builds select the same decode pipeline, whose source is unchanged; the
cause has not been isolated. This is a measured decode regression alongside
the prefill improvement, not evidence of unchanged decode throughput.

The weighted expansion microbenchmark fell from 11.350 to 5.763 ms, and
the complete A8/expansion/GEMM/epilogue chain from 97.066 to 90.423 ms.
These are hot-buffer service times, not exclusive full-graph costs or DRAM
bandwidth measurements. The production marker adds `blocked-O128-K32`.

Strict M5 and portable real-shape matrices, full-head tests and relevant
backend-op tests passed. Random O384/K768/B544 adds coverage of the M32 BK128
route. Four random preexpanded cases update weights after a poison run while
reusing the same graph and scratch, then compare against an independent v1
oracle. Full-vocabulary logits for fresh-context pp512, pp643 and pp2048,
each followed by 128 greedy decode steps, match the baseline byte for byte
(58,799,232 F32 values); generated text is identical.

Raw timings, pinned runtimes, source patches, path markers, tests, quality
outputs and the reproducible driver are under
`/Users/1806-admin/row4-prefill-opt3-20260911/`, especially `REPORT.zh-CN.md`,
`full-model-summary.json`, `tg-pooled-summary.json`, `phase-summary.json`,
`path-evidence.json`, `prototypes/steady.log` and `quality/summary.json`.
The conditional estimate from eliminating the remaining expansion is about
4455 tok/s (+5.01%), assuming every other cost stays fixed;
a persistent full-model INT4 copy would require another 3.473 GB and safe
weight lifetime/concurrency handling.

## Single-stream and continuous-batch decode

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

Single-stream decode stays on the compressed-code kernel. The sweep selected
`kernel_row4_w1a8_decode_o16_o4_staged_act`: four SIMDgroups/128 threads cover
one O16 tile while preserving the branchless basis/I32/BF16 implementation.
The K loop statically interleaves four BK128 blocks into four independent I32
accumulator banks, then merges the exact integer sums before the unchanged SIMD
reduction and BF16-RNE epilogue. This gives the M5 load/store machinery four
independent weight reads and dependency chains to overlap. Across the four
real projections the final implementation lowers summed operator latency by
6.43% versus O32. Full-model tg128 reaches 104.458 tok/s, 3.17% above the
previous O16 path and 7.03% above the initial M4-tuned path.
Single-stream W8A8 output continues through the INT32 O128 kernel.

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

Earlier schema-v1 continuous-batch experiments used M8N128 for 2-8 rows and M16N64/SG4 for 9-16 rows. Missing rows were quantized as zero and padded only in backend scratch; the epilogue did not write them to the graph tensor. A seven-candidate M16 tile sweep selected M16N64/SG4. Across qkv, attention output, gate/up, and down it took about 774 us per layer, versus 989 us for M16N128/SG4, a 21.7% reduction.

Those experiments used matching W8A8 M8N128 and M16N128 integer TensorOps paths. For O151936/K4096, M8 fell from 13.91 ms on the portable kernel to 2.87 ms, and M16 fell from 3.84 ms to 2.70 ms. All paths kept the original I32 scale and BF16-RNE epilogue. On the 16-slot Turbo4 server, those M16 kernels sustained about 21.4 decode steps/s, or 342.6 generated token/s in aggregate, and about 21.0 steps/s for 14 real rows padded to M16. The Pair2 follow-up below supersedes these small-batch dispatch choices.

## Pair2 batched-decode follow-up, 2026-09-05

The baseline for this follow-up is the already optimized Pair2 implementation
in `8f006558`, including packet8, split-K2 and online prefill. It is not the
original `fe2339ef` baseline. The retained batched-decode implementation and
its tests are committed as `aca5fb36`. The GGUF retains its original W8A8 head.

The W8 physical layout contains contiguous `{K128, O16}` tiles. The new head
kernel presents each tile directly as the transposed right MPP operand, with
element strides `{1, 128}`. This removes the previous O128 gather/transpose,
16 KiB threadgroup weight staging, and two barriers per BK128 iteration.
Cooperative INT32 accumulation spans the complete K reduction, followed by the
unchanged exact scale/BF16 epilogue. B4-8 pads to M8, B9-16 to M16, and B17-32
to M32. Allocation and dispatch share one selector; padding is never exposed
as output rows. B1-3 and unsupported shapes retain their existing routes.

For Row4, two or four token rows share one compressed-code load and codebook
decode inside an O32 LUT tile. Every row keeps independent packet8/F32
accumulators. The activation staging budget chooses B4 for K4096 and B2 for
K12288 on this machine. K4096 uses four SIMDgroups; longer K uses eight and an
exact two-way integer merge in the dead activation storage, with a 128-thread
fallback when the pipeline cannot support 256 threads. No persistent expanded
weight copy or global partial-sum buffer is added.

Production Pair2 selection for the four model projections is:

| actual B | QKV / attention output | gate/up | down |
| --- | --- | --- | --- |
| 1 | existing independent LUT | existing independent LUT | existing independent LUT |
| 2-3 | independent LUT | independent LUT | independent LUT |
| 4-7 | independent LUT | shared B4 LUT | shared B2 LUT, split-K2 |
| 8-11 | shared B4 LUT | shared B4 LUT | shared B2 LUT, split-K2 |
| 12-16 | shared B4 LUT | M16N64 MPP | shared B2 LUT, split-K2 |
| 17-31 | QKV: M32N128 MPP; output: shared B4 LUT | M32N128 MPP | shared B2 LUT, split-K2 |
| 32+ | existing prefill selector | existing prefill selector | existing prefill selector |

B2/B3 experiments failed to show a reliable full-model gain and were reverted.
The small MPP gates require K <= 4096, O >= 16384 for B12-16, and O >= 6144
for B17-31, plus complete output tiles. The B2/B4 LUT selector checks actual
threadgroup-memory capacity. Full-model measurements and boundary cases use
`llama-batched-bench` with independent prompts and one next-token row per
sequence per decode step, including every sequence's full logits.

Strict tests require markers for both shared LUT tiles and all three direct
W8 tiles. The real-shape oracle includes the B11/12, B16/17 and B31/32/33
dispatch boundaries. Fifteen full-model cases, including distinct per-sequence
inputs and 1024-token histories, compare 524,331,136 float32 logits byte for
byte with the preserved baseline. All match.

Reproduction scripts, serial A/B/B/A measurements, path logs, rejected
experiments and the Chinese report are retained in
`~/row4-metal-batch-20260905/`.

For independent pp128 prompts followed by 128 concurrent decode steps, the
retained kernels measured the following aggregate throughput. Each version
has six samples per B in A/B/B/A order, after one complete warmup sweep per
launch. The later selector-only B2/B3 rollback does not alter these B>=4
shader paths; its exact diff and a preserved measurement binary are archived.

| concurrent sequences | previous optimized Pair2 tok/s | retained kernels tok/s | change |
| ---: | ---: | ---: | ---: |
| 4 | 302.142 | 324.969 | +7.56% |
| 8 | 343.548 | 442.026 | +28.66% |
| 16 | 380.541 | 592.538 | +55.71% |
| 32 | 585.871 | 662.490 | +13.08% |

With 1024-token histories, B8 and B16 improve by 28.50% and 46.93%. These
are aggregate rates, not the per-sequence rate. The corresponding pp128 B16
step interval falls from 42.141 to 27.033 ms. Formal final-binary controls
measure pp512 at -0.04% and single-stream tg128 at -1.01%; this change does
not claim to accelerate either unchanged path.

Fixed one-second B16 counter windows show the W8 head's external reads rising
from 193.64 to 466.36 GB/s and GPU matrix-unit utilization from 4.66% to 12.66%,
while the instruction-throughput limiter falls from 38.01% to 13.74%. These
are GPU-global counters conditioned on sparse shader samples, not exclusive
per-dispatch counters or full-model time shares. All available counters,
normalized samples, window rules and full traces are archived with the report.

## Correctness evidence

`tests/test-row4.cpp` uses bit comparisons, not floating-point tolerances. The
final test matrix covers:

- B = 1, 2, 3, 4, 7, 8, 9, 11, 12, 15, 16, 17, 24, 31, 32, 33, 64, 96, 128,
  256, and 512;
- all four schema v1 online M5 tiles plus both layout-specific pre-expanded
  paths, with a required Pair2 B512 `device-preexpand` marker;
- both Pair2 M64 cooperative-store paths, random output-tile patterns, and
  a B576 partial group of token tiles;
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

The following measurements describe the original schema v1 implementation.
The dated Pair2 sections above contain its later prefill and decode results.

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

- Single-stream decode uses the tuned compressed O16 Row4 kernel and O128 W8A8 kernel. Continuous batches of 2-16 rows use padded M8/M16 TensorOps kernels.
- The portable precompiled metallib does not contain MSL 4 TensorOps kernels;
  runtime source compilation is required.
- Online staging is fixed at 16 KiB. BK256 would consume the full 32 KiB
  threadgroup-memory budget on this device and reduced occupancy in experiments.
- MPP cannot consume the Row4 codebook nibble directly, so exact online staging
  or a lossless temporary numeric-INT4 expansion is unavoidable with this API.
- The measured MPP and memory roofs are empirical results for this machine,
  operating system, driver, and compiler, not published chip peak values.


## RMSNorm/A8 fusion follow-up, 2026-09-12

QAT RMSNorm can now produce the following Row4 A8 activation in the same
kernel for contiguous K4096 inputs on M5, when B is at least 32 and divisible
by 32. The RMS node must have one adjacent Row4 consumer in the same graph
segment and a single contiguous weight row. The host checks the full Row4
scratch allocation and overlap with the RMS operands. Other shapes and graph
boundaries retain their independent operators; schema v1 and Pair2 are both
covered.

The kernel preserves the standalone 256-thread RMS reduction and every BF16
rounding boundary. It writes the complete RMS carrier and retains 16 values
per thread to form A8 without reading that carrier twice. Scale calculation,
half-away rounding, INT32 products, and output rounding remain unchanged.
For pp512 this removes 71 dispatches and about 1.191 GB of logical reads,
including cache traffic. It does not allocate a persistent expanded-weight
cache or change b2048/ub512.

The first prototype exposed a shared-memory race also present in the older
B1 fusion: an A8 maximum could overwrite the inverse RMS before another
SIMDgroup loaded it. The final B1 and prefill kernels reserve a separate
inverse-RMS slot, with 48 bytes of aligned threadgroup storage. A pp643
full-model comparison caught the race after the pp512 and operator tests had
passed; the failed candidate and its measurements are excluded from the
reported results.

Final validation covers full RMS carriers and linear outputs, independent
Row4 oracles, B1/16/31/32/33/64/128/256/512/544/2048, schema v1, partial graph
execution, repeated weights, noncontiguous input, extra consumers, requested
outputs and non-QAT gates. Strict M5, portable and real-shape backend tests
passed. For pp512, pp643 and pp2048, each followed by 128 greedy decode steps,
all 58,799,232 F32 logits and generated text match the starting runtime byte
for byte.

An isolated profiler uses one encoder per dispatch because dispatch-boundary
counter sampling is unavailable on this device. The complete pp512 graph
contains 1012 dispatches before fusion and 941 after it. K4096 RMS/A8 time in
this instrumented graph decreases from 5.523 to 3.824 ms. Encoder splitting
perturbs scheduling and adds overhead; these times are diagnostic, and all
reported throughput comes from the uninstrumented runtime.

The earlier decode regression was also investigated with old/new host and
shader combinations. It followed the shader module in whole-model tg128, but
did not reproduce in decode kernels running on identical buffers in one
process. Its underlying cause remains unresolved; the shared-memory fix is
not claimed to explain that performance regression. Detailed evidence,
including all rejected runs, is in `~/row4-prefill-opt4-20260912/`.

Final uninstrumented A/B/B/A means (five samples per launch, all ten samples
per version included, at least 15 seconds between launches):

| workload | starting tok/s | fused tok/s | change |
| --- | ---: | ---: | ---: |
| pp128 | 3011.574 | 3051.371 | +1.32% |
| pp512 | 4244.360 | 4290.640 | +1.09% |
| pp2048 | 3658.181 | 3705.905 | +1.30% |
| tg128 | 144.798 | 145.339 | +0.37% |

The model, eight host threads, b2048/ub512, BF16 KV, FA1, ngl99, mmap1 and
warmup are identical to the preceding round. The machine is the same M5 Max
with AC automatic power mode and Apple clang 21.0.0 Release build. Exact
commands, path evidence and raw artifact paths are in
`~/row4-prefill-opt4-20260912/benchmark-summary.json`.

## Prefill scheduling overlap follow-up, 2026-09-12

Activation preparation and lossless weight expansion are independent. For
preexpanded MPP shapes, their shared buffer barrier now comes after both
dispatches, immediately before MPP. RMS/A8 fusion checks the whole scratch
tail against its still-live operands before allowing this overlap. The
ordinary quantizer and expansion use disjoint regions of the linear output
allocation.

After a preexpanded MPP dispatch, a bounded lookahead searches the next 63
nodes within the same command buffer for a Pair2 expansion that can safely
run alongside it. Each encoding context holds at most one pending target.
The checks include every intervening node and source, full backing storage
for views, backend scratch, and possible writes to the future codes. The
expanded weights stay in their original graph allocation; the target's
pre-MPP barrier publishes them before use. No state survives a graph compute
or crosses a command-buffer boundary, so updated weights and partial graphs
retain their existing semantics. Serial encoders skip this lookahead.

On pp512, 34 gate/up expansions move earlier: one alongside qkv MPP and 33
alongside an earlier down MPP. Activation preparation loses 142 redundant
barriers. Dispatch count and weight traffic are unchanged. Metal compute
storage remains 314.77 MiB at b2048/ub512, with no additional GPU allocation.
The existing CPU encoding/GPU execution overlap is preserved.

A same-buffer microbenchmark uses a concurrent encoder and compares explicit
serial barriers against overlap. Down MPP plus an independent gate/up
expansion decreases from 762.235 to 695.190 microseconds (-8.80%). Dispatch
order matters: starting the large expansion first does not yield the same
benefit. Per-dispatch encoder splitting would disable the concurrency being
measured, so it is not used to claim this gain.

Validation adds B512/B544 two-linear graphs with an independent integer
oracle, live data deliberately placed in future INT4 scratch, serial and
split-graph gates, and updated codes on repeated execution. A separate RMS
case places its weight in future INT4 scratch and verifies that expansion
waits for the weight reads. Strict M5, portable and real-shape backend cases
pass. Full-model pp512/pp643/pp2048 plus 128 greedy decode steps compare
58,799,232 F32 logits and generated text byte for byte.

Final serial ABBA means (ten samples per version, b2048/ub512, t8,
BF16 KV, FA1, ngl99, mmap1, warmup, at least 15 seconds cooldown):

| workload | starting tok/s | overlap tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 4299.376 | 4453.458 | +3.58% |
| pp2048 | 3703.025 | 3805.436 | +2.77% |
| pp128 | 3050.731 | 3045.810 | -0.16% |
| tg128 | 145.128 | 144.682 | -0.31% |

Full-model ABBA results and raw evidence are recorded under
`/Users/1806-admin/row4-prefill-opt5-20260912/`, with `run-bench.py`,
`benchmark-summary.json`, `micro/`, `quality/` and `checks/`.

## SwiGLU/A8 preparation fusion follow-up, 2026-09-14

The K12288 QAT SwiGLU kernel can now prepare A8 and its complete row scale
directly for a sole adjacent Pair2 down projection. The existing M5 selector
must choose preexpanded weights (B >= 512, divisible by 32), and the pipeline
must support 1024 threads per threadgroup. Each thread retains three `ushort4`
values across the row maximum reduction. The two BF16 rounding boundaries,
exact scale division, half-away A8 rounding and subsequent INT32 MPP products
are unchanged.

A8 and scales occupy the otherwise-dead MUL allocation. The host checks this
compact range against the full gate/up backing allocation and the full down
allocation, including backend scratch. In the actual graph, down scratch can
reuse still-live gate/up data, so the producer barrier before down expansion
remains necessary. The INT4 weights retain their original down scratch, and
the existing gate/up expansion lookahead remains active. Observable MUL
outputs, extra consumers, split graphs, incompatible aliases, other shapes
and portable paths retain their existing execution.

For pp512, runtime markers confirm 35 SwiGLU/A8 fusions and the existing 34
gate/up lookaheads. This removes 35 independent quantization dispatches and
36 MiB of logical packed-BF16 writes/reads per fused layer, or 1.230 GiB over
the 35 layers. Logical traffic includes cache traffic; it is not a measured
DRAM saving. Metal compute storage remains 314.77 MiB at b2048/ub512.

The threadgroup size was selected using the complete FFN chain with shared
scratch and the existing next-gate/up lookahead. A 512-thread prototype reduced
isolated SwiGLU/A8 preparation from 146.944 to 101.597 microseconds, but reduced
full-model pp512 throughput by 0.345%. Copying A8 back to down scratch did not
resolve the complete-chain regression. The 1024-thread variant instead reduced
the complete FFN microbenchmark from 2009.076 to 1996.893 microseconds (-0.606%).
All compared outputs were byte exact. These experiments do not establish a
specific cache or register-allocation cause.

Final uninstrumented measurements combine ABBA and reverse BAAB, ten samples
per launch, all 40 samples per version for pp512/pp2048. The pp128/tg128
controls use one ABBA with five samples per launch and ten per version.
All launches are sequential with warmup and at least 15 seconds cooldown.

| workload | starting tok/s | fused tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 4458.265 | 4471.634 | +0.30% |
| pp2048 | 3813.561 | 3822.374 | +0.23% |
| pp128 | 3046.536 | 3051.611 | +0.17% |
| tg128 | 145.386 | 145.065 | -0.22% |

The two rounds separately give +0.378%/+0.222% for pp512 and
+0.169%/+0.293% for pp2048. These are small measured improvements. The new
branch is inactive for the short-prefill and decode controls; their single
ABBA does not establish a persistent gain or regression. The model is
`qwen3-row4-v2-pair2.gguf`, with t8, b2048/ub512, BF16 KV, FA1, ngl99, mmap1
and warmup. Hardware is an M5 Max (40 GPU cores, 18 CPU cores, 128 GB), macOS
26.5.1, AC automatic power mode with unfixed clocks. Apple clang 21.0.0 builds
Release with `-O3 -DNDEBUG`, native code, embedded Metal and Apple BLAS.
The starting revision is `fe79b5dc5a449922caa64a37451948f30c124318` plus the
preceding production changes captured in `baseline.patch`.

Strict M5 tests check every A8 integer against an independent CPU oracle,
exact scales and down outputs, zero/tiny rows, B128/B512/B544, requested
outputs, extra consumers, split graphs and aliased scratch. Portable Row4
and real-shape ROW4_LINEAR/W8A8_LINEAR backend tests also pass. Full-model
pp512/pp643/pp2048, each followed by 128 greedy decode steps, compare
58,799,232 F32 logits and generated text byte for byte against the starting
runtime. The prefill comparison covers the last prompt token's logits;
every decode step's logits are checked. Scoped formatting passes, with no
clang-tidy diagnostics on lines added or changed in this round.

An attention/o-projection overlap experiment also enabled all 36 attention
windows while retaining 34 gate/up lookaheads, but full-model pp512/pp2048
changes were only +0.059%/-0.059%; it is saved as an experiment and is not
enabled in production. The prior local preparation already overlaps A8
quantization and o-weight expansion. Moving expansion earlier therefore
cannot remove its entire isolated 10.577 microseconds: the measured residual
above quantization is about 3.313 microseconds per layer, roughly 0.119 ms
over 36 layers before additional contention. This is a diagnostic estimate,
not a full-model bound or prediction.

Source/binary snapshots, both final measurement orders, exact commands and
path markers are under `/Users/1806-admin/row4-prefill-opt6-20260914/`:
`fusion1024-summary.json`, `fusion1024-model/`, `fusion1024-confirm/`,
`fusion1024-quality/`, `micro/`, `checks/` and `REPORT.zh-CN.md`.

## Scratch, residual and token-pipeline trials, 2026-09-14

Prefill o/down projections now fuse an adjacent QAT COMPLEX_ADD into the
M64N64 MPP epilogue. Eligibility is limited to Pair2, O4096, K4096/K12288,
B >= 512 divisible by 64, and a sole unobserved projection consumer in the
same encoding range. Observable projections, extra consumers, reversed
operands, non-QAT adds, partial tiles and conflicting scratch aliases retain
the existing path. Exact in-place residual addition is supported. The host
protects the A8/scale/INT4 tail, weights, packed MUL activation and partially
overlapping residual storage before selecting this fusion.

The projection's BF16 boundary is retained before adding either complex
component. A thread-local volatile integer materializes its carrier bits:
without this boundary, the compiler can fold the low component's addition
of zero, changing signed zero, subnormal flushing and NaN canonicalization.
The ordinary prototype failed 917,504 of 2,097,152 edge-profile outputs;
the retained variant matches the original Metal COMPLEX_ADD_QAT byte for
byte. No device-memory intermediate is required for that boundary.

At pp512, runtime markers confirm 70 residual fusions, the existing 35
SwiGLU/A8 fusions and 34 gate/up expansion lookaheads. Eliminating one 8 MiB
projection write and read per residual saves 1.094 GiB of logical traffic
and 70 separate add dispatches per 512-token microbatch. This includes cache
traffic and is not a measured DRAM saving. Metal compute storage remains
314.77 MiB at b2048/ub512.

With complete FFN scratch reuse and the next-gate/up lookahead, the residual
microbenchmark decreases from 2064.108 to 2024.101 microseconds (-1.94%).
Full-model measurements combine ABBA and BAAB, ten samples per launch and
all 40 samples per version for the main prefill workloads. Controls use
one ABBA, five samples per launch and ten per version.

| workload | starting tok/s | residual-fused tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 4464.601 | 4533.122 | +1.53% |
| pp2048 | 3821.721 | 3868.039 | +1.21% |
| pp128 | 3049.842 | 3048.889 | -0.03% |
| tg128 | 144.809 | 145.098 | +0.20% |

The separate ABBA/BAAB changes are +1.510%/+1.560% for pp512 and
+1.329%/+1.095% for pp2048. The new branch is inactive in the controls;
their small differences do not establish persistent performance changes.
All launches are sequential, warmed up, with at least 15 seconds cooldown.
The model is qwen3-row4-v2-pair2.gguf, t8, b2048/ub512, BF16 KV, FA1,
ngl99 and mmap1. Hardware remains Apple M5 Max (40 GPU cores, 18 CPU cores,
128 GB), macOS 26.5.1, AC automatic power mode with unfixed clocks, and
Apple clang 21.0.0 Release (-O3 -DNDEBUG), native/embedded Metal/Apple BLAS.
The baseline is fe79b5dc5a449922caa64a37451948f30c124318 plus the preceding
production changes captured in this round's baseline.patch.

Before this fusion, two isolated down-INT4 scratch variants were tested.
A persistent private 24 MiB buffer made expansion overlap gate/up MPP,
but full-model ABBA changes were only -0.052%/+0.067% for pp512/pp2048.
Reusing the dead SiLU allocation avoided the extra memory and enabled
35 such overlaps while retaining 34 old lookaheads, but combined ABBA/BAAB
changes were -0.288%/-0.115%. Neither experiment established a gain;
both remain in external source/binary snapshots.

The token-pipeline experiment tries 64/128/256-row chunks, complete K12288
quantization rows and the original large-prefill MPP kernels. It includes
serial chunks, gate/up with preceding SwiGLU overlap, and a three-stage
pipeline adding earlier down chunks, in both dispatch orders. All outputs
are byte exact. After baseline drift in the initial sweep, paired ABBA and
BAAB complete-FFN measurements with residual fusion gave +17.05%, +3.34%
and +0.52% time for the 64/128/256-row gate/SwiGLU pipeline, respectively.
The 256-row three-stage variants took +16.22%/+13.02% time. These are
microbenchmark results. The closest candidate, a 256-row two-stage pipeline,
was also integrated into the runtime and tested against the residual-fused
version, with strict path assertions and full-model exact comparisons.
It enabled 35 token pipelines and preserved the other fusion/lookahead
counts and 314.77 MiB compute storage. ABBA plus BAAB, 40 samples per version,
gave:

| workload | residual-only tok/s | token-pipeline tok/s | change |
| --- | ---: | ---: | ---: |
| pp512 | 4526.164 | 4514.668 | -0.25% |
| pp2048 | 3865.758 | 3853.787 | -0.31% |

Both orders were slightly slower for both workloads. The token pipeline
remains an external experiment; only residual fusion is retained. More
independent dispatches did not produce an additional throughput gain for
these shapes. The experiments do not establish a specific cache, occupancy
or frequency cause.

Raw benchmarks, exact comparison logs, source/binary snapshots and scripts
are under /Users/1806-admin/row4-prefill-opt7-20260914/: stage1-private-summary.json,
stage1-pool-summary.json, stage2-summary.json, quality/, micro/, checks/,
and REPORT.zh-CN.md. Snapshots pin DYLD_LIBRARY_PATH, and quality logs
record the actual loaded Metal library. All samples, including the drifting
initial microbenchmark baseline, are retained.

Final Release, strict M5, portable and real-shape ROW4_LINEAR/W8A8_LINEAR
checks pass. Residual tests include BF16 edge values, B128/B512/B544,
observable outputs, extra consumers, reversed/non-QAT/split graphs,
scratch aliases, in-place addition and complete FFN graphs. Full-model
pp512/pp643/pp2048 plus 128 greedy decode steps compare 58,799,232 F32
logits and text byte for byte with the starting runtime (last prompt-token
logits and every decode step). All final runtime libraries and llama-bench
are byte-identical to the measured residual snapshot; only a test setup's
nested conditional was rewritten, followed by another strict M5 run.
Scoped git clang-format passes, with no clang-tidy diagnostics on lines
added or changed in the final implementation and tests. The initial
stage-3 sweep drift, all paired measurements and unretained source/binary
snapshots remain available for future reproduction.

## Output pruning, norm/RoPE, Flash3 and INT4 cache trials, 2026-09-14

Four follow-up directions were tested in order. The retained default changes
move last-layer output-row selection before the o projection, fuse eligible
Q/K RMSNorm with RoPE and K-cache storage, read BF16 norm weights directly,
and reuse BF16 K/V tiles across two query heads in long FlashAttention-3
prefill. The full Q/attention pruning and matrix tile/traversal experiments
were correct but did not improve full-model performance and are not retained.

Last-layer o pruning keeps all K/V rows and requested output semantics. It
reduces Metal compute storage from 314.77 to 306.77 MiB at b2048/ub512.
Its isolated ABBA/BAAB gains are small: +0.124%/+0.165% for pp512/pp2048.
Full Q pruning instead gives -0.982%/-0.502%, including the padded query
variant that preserves matrix reduction order.

The RMS128 fusion supports M5 QAT heads8/32 with at least 32 tokens and
compatible strides. It crosses checked metadata-only nodes, retains the
original reduction order and every BF16 boundary, and optionally writes
BF16 K directly through an exclusive SET_ROWS. Sole-use BF16 norm weights
avoid the intermediate F32 cast; observable casts and incompatible layouts
keep the original graph. Output/extra-use/split/alias gates remain in place.
An explicit frequency-present argument handles a frequency vector aliased
to X, while a first-dimension check rejects unsupported broadcast weights.
Both Metal and host address ranges protect the K write. Each B512 model
microbatch has 72 RMS/RoPE fusions, 72 direct BF16 weights and 36 K stores.
The isolated full-model gain is +2.562%/+2.351%.

The actual production attention is BF16 FlashAttention-3 with online
softmax. Its new GQA2 branch reuses each K/V 8x8 load while retaining Q8/C64,
K accumulation order and F32 softmax/P/O arithmetic. It uses four SIMD
groups and 17,408 bytes of shared memory. Selection requires M5, Q >= 512,
KV >= 1024, D128, a Q/KV head ratio of four and compatible ordinary mask,
sequence and stride settings, with no sinks/bias/softcap. The original GQA1
and other data-type paths retain their original computation. The initially
broader selector gave -0.083%/+1.215% for pp512/pp2048, so short cases retain
the existing path. Final pp2048/ub512 has 108 GQA2 dispatches; pp512 has none.

`GGML_METAL_ROW4_INT4_CACHE=1` enables an optional persistent lossless INT4
expansion cache for eligible Pair2 leaf codes in Metal WEIGHTS buffers.
It defaults off. Cold expansion completes before current graph command
buffers are enqueued; warm graphs skip repeated expansion and its lookahead.
Backend set/memset/clear, partial aliases, async writes and graph writes
invalidate the cache, including a write from a cache-disabled context.
Owner buffer release frees cache storage. Invalidated storage remains alive
for preceding commands until safely rebuilt. Each source weight buffer has
a 4 GiB cap, with device available-memory and allocation fallback checks.

This last-token-logits workload caches 141 matrices, adding 3,388,997,632
bytes (3.15625 GiB) beyond the unchanged compute allocation. All-output
requests can involve 144 matrices. Initial isolated warm throughput gains
are +2.736%/+2.611%; the 141 cold builds took about 86-107 ms in individual
diagnostic runs. Those cold samples include first pipeline compilation and
are not a repeated cold-start comparison. Repeated prefill can amortize this
cost; the cache is not enabled unconditionally.

Matrix reuse trials cover four M/N/SG geometries and group-M=1/2/4/8, all
four model shapes at B512/B2048, cached GEMM and expand-plus-GEMM, rotating
eight weight buffers and both measurement orders. Some microbenchmarks gain
2-7%, but the integrated candidate loses 0.426%/0.418% for pp512/pp2048 at
ub512 and is flat (-0.031%) at pp2048/ub2048. It is archived as `reuse-tuned`;
production retains M64N64/G2 and gate/up M64N128/G4.

Final measurements use pinned source/library snapshots with all numerical
fixes and rejected experiments removed. Main workloads use balanced order
ABCCBACABBAC (A baseline, B default, C cache), ten timed samples per launch,
40 samples per version, warmup and at least 15 seconds cooldown. All launches
are serial, with no concurrent GPU job, compilation or static analysis.
Fusion debug is confined to separate correctness/path-validation runs.

| workload | baseline tok/s | final default tok/s | final cached tok/s |
| --- | ---: | ---: | ---: |
| pp512-ub512 | 4523.570 | 4651.248（+2.82%） | 4796.717（+6.04%） |
| pp2048-ub512 | 3868.552 | 4009.613（+3.65%） | 4121.510（+6.54%） |

The baseline is this round's starting runtime, including prior optimizations,
not clean upstream HEAD. Controls use five samples per launch: pp128 uses
ABBA (10 per version). An initial tg128 ABBA showed a small decrease with
baseline drift, so a reverse BAAB was added; all 20 samples per version are
combined below. The individual rounds remain in final-tg-control-summary.json.
Both orders show a small decrease (-0.410%/-0.352%, combined -0.381%).
This measured decode cost is retained alongside the prefill gains; its cause
has not been isolated, so it is not attributed directly to device noise.

| workload | baseline tok/s | default tok/s | change |
| --- | ---: | ---: | ---: |
| pp128-ub512 | 3049.390 | 3134.202 | +2.78% |
| tg128-ub512 | 144.966 | 144.414 | -0.38% |

The cache-disabled ubatch trial at pp2048 gives 4002.509, 4175.190 and
4247.489 tok/s for ub512/1024/2048 (+0%, +4.31%, +6.12%), 40 samples each.
Metal compute storage is 306.77/613.52/1227.02 MiB; CPU compute storage is
8.01/16.03/32.05 MiB. Final ub2048 default/cache means are
4248.946/4297.767 tok/s (40 samples each). Ubatch stays an explicit
caller choice: ub2048 adds 920.25 MiB of GPU compute storage over ub512.
Neither these memory figures nor the optional 3.15625 GiB cache include
the model's other allocations.

Final full-model baseline/default and default/cache comparisons each check
58,799,232 F32 logits byte for byte: pp512/pp643/pp2048 last-token logits
plus 128 greedy decode steps per prompt, including identical generated text.
Both final configurations also pass full-logit comparisons across ub512,
ub1024 and ub2048. Selected/all-output and separate/unified multi-sequence
graphs pass across all three versions, with eight continuation steps.
Strict M5, portable, real-shape Row4/W8A8 and 56 BF16 FlashAttention backend
cases pass. New cases cover rounding edges, strides, tails, observables,
aliases, frequency/broadcast corner cases and 36 cache-update/lifetime steps.
Scoped git clang-format passes; four changed C++ sources have no clang-tidy
diagnostics on changed lines. Final runtime libraries match the measured
snapshots.

Reproduction uses qwen3-row4-v2-pair2.gguf, t8, b2048/ub512 unless specified,
BF16 KV, FA1, ngl99, mmap1 and warmup. Hardware remains M5 Max (40 GPU/18 CPU
cores, 128 GB), macOS 26.5.1, AC automatic mode with unfixed clocks, Apple
clang 21 Release (-O3 -DNDEBUG), native/embedded Metal/Apple BLAS enabled.
Use the existing Row4 benchmark harness with `BINARY=build-rel/bin/llama-bench`,
`THREADS=8 REPS=10 BATCH=2048 UBATCH=512 N_PROMPT=512` and the model path;
change N_PROMPT to 2048 for long prefill, UBATCH to 2048 for that explicit
configuration, and set GGML_METAL_ROW4_INT4_CACHE=1 to enable persistence.

All experiments, raw samples, exact commands, path counts, cold diagnostics,
source/binary snapshots and checks are under
`/Users/1806-admin/row4-prefill-opt8-20260914/`: `REPORT.zh-CN.md`,
`final-results-summary.json`, `final-comparison/`, `final-tg-confirm/`, `quality/`, `checks/`,
`micro/`, the individual stage `*-summary.json` files and `environment-final.json`.


## Round 9: compact BF16 handoff and batched cold cache (2026-09-14)

All six follow-up directions were implemented and tested sequentially. This
round retains dead RMS carrier elimination/direct BF16 norm weights, actual-mask
Flash3 bounds, compact BF16 attention output into whole-token A8, QKV epilogue
V-cache stores, reduced host preparation work and batched persistent INT4 builds.
A full-mask scan and selective persistent caching were tested and rejected.

RMS4096 fusion preserves its original reduction order and BF16 rounding, while
unobserved sole-use prefill outputs avoid F32 stores. Cast-elision checks include
the RMS input dependency. Flash3 bounds are derived from the actual F16 mask by
scanning backwards to the last non-minus-infinity key per query; contributing
C64 tiles retain their order. Noncausal, nonmonotonic and all-masked cases work.
The full-mask-scan prototype was slower at ub512 and is not retained. The retained
bound mainly benefits ub2048 (+0.54% isolated); ub512 is essentially neutral.

A sole, unobserved FA->RESHAPE->Row4 consumer in one command buffer can pass a
compact BF16 carrier directly to the existing full-K4096 A8 quantizer. Output,
extra-user, split and incompatible cases retain their original path. The prefill
residual epilogue remains fused. Isolated pp512/pp2048 gains are +0.69%/+0.38%.

QKV MPP writes BF16 V cache in its epilogue and elides the later SET_ROWS dispatch,
while preserving the full F32 QKV carrier. Early cache readers/writers, writes
through index aliases, aliases of the live A8/INT4 tail, incompatible layouts
and split command buffers fall back. Isolated gains are +0.26%/+0.21%.

Host preparation skips cache invalidation graph scans only before any cache has
existed on the device, so cache-disabled writer contexts still invalidate old
entries. Buffer-owned hash buckets replace linear cache lookup, and preexpand
lookahead searches start at the current node. Existing graph use_counts already
provide hashed use checks. A synthetic graph of 1024 nodes cache-disabled scan saves about
1.5 us, but full-model throughput is neutral; no throughput gain is claimed for
this host change.

Cold persistent INT4 preparation encodes builds in one command buffer and waits
once. Entries are published only after completion, with pending-batch identity
and write-generation checks. Cancellation, unsupported graphs and allocation
failure retain ordinary expansion fallback; invalidated allocations stay alive
for preceding work. Tests cover duplicate/conflicting batches, writes during
preparation, cancellation, partial eligibility and repeated buffer lifetimes.

Four fresh processes per version/workload, in ABBA+BAAB order, give:

| workload | serial cold prefill ms | batched cold prefill ms | change |
| --- | ---: | ---: | ---: |
| pp512/ub512 | 206.725 | 164.759 | -20.30% |
| pp2048/ub512 | 604.466 | 553.518 | -8.43% |

Model/context loading is outside this timing; first decode+synchronize and
initial operator preparation are inside. These are fresh device caches, not
filesystem-cold loads. Cache-build times are 91.630 -> 55.042 ms and 98.862 -> 55.303 ms.
Two subsequent warm prefills per process remain essentially unchanged. Every
full-logit output matches. These cold latency reductions are not warm TPS gains.

A selective candidate caches only matrices up to 32 MiB, leaving the 35 gate/up
matrices on safe lookahead. It reduces persistent storage from 141 matrices /
3.15625 GiB to 106 matrices / 1.515625 GiB and restores 34 lookahead expansions, but loses
2.44%/2.00% at pp512/pp2048 in ABBA+BAAB. It is rejected on the current 128 GB device.
The public setting remains `GGML_METAL_ROW4_INT4_CACHE=1` for the full optional
cache, default off; no extra cache policy is exposed.

Final pinned baseline-to-deliverable throughput, comparing the same cache mode
on both sides (b2048, t8, BF16 KV, FA1, ngl99, mmap1, warmup):

| configuration / workload | round-start tok/s | final tok/s | change |
| --- | ---: | ---: | ---: |
| default, ub512, pp512 | 4652.977 | 4680.163 | +0.58% |
| default, ub512, pp2048 | 3970.699 | 3993.711 | +0.58% |
| INT4 cache, ub512, pp512 | 4791.903 | 4815.189 | +0.49% |
| INT4 cache, ub512, pp2048 | 4050.726 | 4079.437 | +0.71% |
| default control, ub512, pp128 | 3131.314 | 3136.694 | +0.17% |
| default control, ub512, tg128 | 144.390 | 145.308 | +0.64% |
| default, ub2048, pp2048 | 4250.538 | 4316.972 | +1.56% |

Stage ablations use ABBA+BAAB and 40 samples/version. Final core comparisons start
with 20 samples/launch (40/version); default prefill receives a reverse comparison of 10 samples per launch
confirmation after startup drift, retaining every sample. Short/tg controls use
five samples/launch; ub2048 is reported separately. Exact sample counts, orders
and commands are in the summary JSONs. Runs are serial, without concurrent GPU
tests, compilation or static checks. Clocks remain unfixed in AC automatic mode.
Do not add isolated stage deltas or attribute a drifting run to a specific
hardware cause without separate evidence.

At pp512, markers show 71 RMS no-stores, 70 direct BF16 norm weights, 36 mask bounds,
35 compact attention handoffs and 36 V stores. Default lookahead 34, residual 70 and
SwiGLU 35 remain. Logical traffic avoided is 568 MiB of RMS writes, 140 MiB attention
writes, 280 MiB across two quantizer reads and 72 MiB V-copy reads. This 1060 MiB total
is not a measured DRAM traffic reduction. Compute buffers remain 306.77 MiB Metal
and 8.01 MiB CPU; optional persistent INT4 adds 3,388,997,632 bytes.

Final default versus round start and final cache versus default each compare
58,799,232 F32 logits byte-for-byte (pp512/643/2048 plus 128 greedy steps per prompt).
Both final modes pass cross-ub512/1024/2048 checks. Selected/all-output and
separate/unified multi-sequence graphs match across all versions with eight
continuation steps. Strict M5 and portable Row4 tests, 60 Row4/23 W8A8 backend cases
and 56 BF16 attention cases pass. Scoped git clang-format and changed-line
clang-tidy checks pass. Existing user files are unchanged; no commit is made.

Environment is M5 Max (40 GPU/18 CPU cores, 128 GB), macOS 26.5.1 (25F80), Apple clang 21,
Release -O3 -DNDEBUG, native/embedded Metal/Apple BLAS enabled. The round-start
baseline includes earlier dirty work on fe79b5dc5a449922caa64a37451948f30c124318,
not clean upstream. The historical 6963.448 tok/s integer-matrix reference is not
a full-model theoretical ceiling or GPU utilization metric.

Evidence and reproduction are under
`/Users/1806-admin/row4-prefill-opt9-20260914/`: `REPORT.zh-CN.md`,
`final-*-summary.json`, `final-path-proof.json`, `cold-abba-baab/`, `quality/`,
`checks/`, `environment-final.json`, `this-round.patch`, and the pinned
`final-default/`, `final-cache/` and `final-deliverable/` snapshots.
