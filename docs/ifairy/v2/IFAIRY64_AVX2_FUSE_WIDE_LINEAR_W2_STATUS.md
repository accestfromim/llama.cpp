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
- prepares one shared activation LUT for U0, U1, W0, and W1;
- evaluates the four weights as two pair QGEMM passes inside the same fused
  node;
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
ggml/src/ggml-cpu/ifairy-fuse-lut.cpp
ggml/src/ggml-cpu/ifairy-fuse-lut-qgemm.cpp
ggml/src/ggml-cpu/ifairy-fuse-lut-qgemm.h
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

## 4. LUT Evolution: Initial Path to Current Implementation

This section records the complete LUT optimization path used by the fused
IFAIRY64 widely-linear layer. It distinguishes the retained production design
from experiments that were measured and later reverted.

The starting point for this chronology is the non-fused x86 IFAIRY64 LUT16
path that existed when the fused-wide-linear work began. Older legacy ARM/3W
LUT layout experiments are documented separately and are not repeated here.

### 4.1 LUT Meaning and Invariants

The LUT is built from the activation, not from the weights. For every pair of
quantized complex activation values, a 4-bit packed weight pattern selects one
of 16 possible contributions.

The retained layout is:

```text
16 entries x 4 complex-product channels x int8 = 64 bytes per activation group
```

The four channels contain the partial real/imaginary contributions required by
complex multiplication. Packed weight tiles contain only the 4-bit LUT indexes
and per-row weight scales.

The invariants preserved through every optimization are:

- IFAIRY64 weights remain quantized in independent K64 blocks;
- activation values are temporarily quantized to INT8 in K64 blocks;
- activation and weight scales are restored before producing the final result;
- the LUT semantics remain equivalent to `weight * conj(input)`;
- U branches recover `U * x` by negating the imaginary activation scale rather
  than building a separate LUT for `conj(x)`.

### 4.2 Starting Point: Non-Fused LUT Matrix Multiplications

The initial widely-linear graph contained four independent matrix
multiplications followed by intermediate additions:

```text
mul_mat(U0, x_conj)  // mul_mat semantics produce U0 * conj(x_conj) = U0 * x
mul_mat(U1, x_conj)  // produces U1 * x
mul_mat(W0, x)       // produces W0 * conj(x)
mul_mat(W1, x)       // produces W1 * conj(x)
add + add + add + optional bias
```

Each `mul_mat` was scheduled as a separate graph node. The generic LUT path
could prepack each weight tensor into 16-output-row tiles, but activation
quantization, LUT preprocessing, thread scheduling, and graph barriers could
not be shared across the four nodes.

The packed IFAIRY64 weight tile retained by later versions is:

```text
ifairy64_lut_wtile_16:
    qs[16 K64 group-pairs][16 output rows]  // two 4-bit indexes per byte
    d_real[16]                              // fp16 weight scales
    d_imag[16]
```

Each tile is 320 bytes and is cached in `tensor->extra->packed_w`. The model
loader eagerly prepares these packed tensors when LUT execution is enabled, so
decode does not perform the weight transform on its first token.

### 4.3 Graph Fusion: One Four-Weight Node

`GGML_OP_IFAIRY_WIDE_LINEAR_W2` moved U0, U1, W0, W1, activation, and optional
bias into one graph node. This removed:

- three intermediate addition nodes;
- intermediate tensor materialization between the four matrix multiplications;
- graph-level scheduling and barriers for those removed nodes;
- repeated final output conversion and bias handling.

The first fused LUT prototypes still reused the existing per-matrix QGEMM
interfaces. They proved that all four weights could be owned by one node, but
did not yet eliminate all repeated LUT preprocessing and QGEMM overhead.

### 4.4 Dedicated Fused LUT QGEMM

The fused LUT implementation was separated from the generic LUT implementation:

```text
ifairy-fuse-lut.cpp          activation preparation, threading, output
ifairy-fuse-lut-qgemm.cpp    x86 pair QGEMM hot path
```

The dedicated QGEMM processes 16 output rows at a time. Two packed weight
branches are evaluated together:

```text
pair 1: U0 + U1, write temporary FP32 complex output
pair 2: W0 + W1, add to the same temporary output
```

Keeping a two-weight pair kernel was intentional. A direct four-weight AVX2
kernel increased live accumulator count, code size, and register pressure. It
was tested but regressed 8-thread decode, so the current implementation keeps
two pair passes inside the single fused graph node.

Within each K64 block, LUT INT8 values are accumulated into INT16 sums. F16C
converts fp16 weight scales to FP32, and FMA restores activation/weight scales
into FP32 output accumulators.

### 4.5 One Shared LUT for All Four Weights

The next change made the activation workspace explicitly shared by all four
weights:

```text
params->wdata:
    quantized activation blocks
    one LUT
    one activation-scale array
    temporary FP32 complex output
```

Only one LUT is constructed per activation column. It is consumed first by the
U pair and then by the W pair. No second `conj(x)` LUT is constructed:

```text
U0/U1: use shared LUT, negate imaginary activation scale
W0/W1: use shared LUT, use normal imaginary activation scale
```

This removed repeated LUT construction while preserving the original complex
semantics.

### 4.6 K64 Activation Quantization

The activation side was changed from legacy K256-oriented temporary
quantization to `block_ifairy64_q16`, matching the IFAIRY64 weight block size:

```text
block_ifairy64_q16:
    64 x INT8 real activation
    64 x INT8 imaginary activation
    fp16 real scale
    fp16 imaginary scale
```

Values are limited to `[-63, 63]`. A LUT entry combines at most two activation
values, so its result remains representable in signed INT8.

Matching activation and weight blocks at K64:

- removes old K256-to-K64 slicing and scale replication;
- gives every weight block the correct activation scale directly;
- makes block-level parallel LUT preparation possible;
- reduces unnecessary temporary data and loop bookkeeping.

For larger prefill batches, rows are divided among threads and the activation
quantizer uses AVX2. For decode and other cases where `N < thread_count`, K64
blocks of the same activation row are divided among threads.

### 4.7 One Internal Barrier

The original fused-LUT preparation still had avoidable synchronization around
activation quantization and LUT generation. The retained implementation lets
each thread completely prepare its assigned activation rows or K64 blocks:

```text
thread-local assigned work:
    quantize activation block
    immediately build that block's LUT and scales

one barrier:
    all LUT blocks are now ready

independent output-row QGEMM:
    each thread owns a range of 16-row output tiles
```

The fused LUT node therefore contains one internal barrier between LUT
preparation and QGEMM. The normal ggml graph executor still performs its
standard barrier between graph nodes.

### 4.8 Pair Kernel and Output-Row Tiling

Output rows are divided in 16-row tiles:

```text
tile0 = tiles_total * thread_index / thread_count
tile1 = tiles_total * (thread_index + 1) / thread_count
```

Each thread receives contiguous packed weight tiles and writes a disjoint
output-row range. This avoids atomics and cross-thread output reduction.

The final output stage adds optional bias and converts the temporary FP32
complex result to the bf16-pair container. Column-invariant index and address
calculations were hoisted outside the output-row loop; this cleanup is retained
although its isolated end-to-end benefit was below benchmark noise.

### 4.9 Shared Real/Imaginary Weight Index Decode

Before the current optimization, the pair kernel processed real and imaginary
LUT channels separately:

```text
decode packed weight indexes -> lookup real channel
decode the same indexes again -> lookup imaginary channel
```

The current kernel decodes each packed weight byte once:

```text
load and broadcast packed weight indexes
split low/high nibbles once
    -> lookup and accumulate real LUT channels
    -> lookup and accumulate imaginary LUT channels
```

This removes duplicate broadcasts, masks, shifts, and packed-weight loads.
Eight INT16 accumulator vectors remain live while both channels are processed.
On the Ryzen 9 7940H native build, the compiler uses extended YMM registers and
the hot path has only one temporary YMM save/restore pair.

This change produced the largest retained kernel-level improvement:

| Threads | Test | Before tok/s | Current tok/s | Change |
| ---: | --- | ---: | ---: | ---: |
| 4 | pp64 | 17.539364 | 21.928456 | +25.0% |
| 4 | tg64 | 8.228292 | 11.156050 | +35.6% |
| 8 | pp64 | 22.616312 | 27.533854 | +21.7% |
| 8 | tg64 | 6.355514 | 10.036237 | +57.9% |

### 4.10 Current End-to-End Fused LUT Flow

The retained implementation now executes as:

```text
model load:
    prepack U0/U1/W0/W1 into cached 16-row IFAIRY64 LUT tiles

each fused wide-linear node:
    parallel K64 activation quantization
    build one shared LUT and activation-scale array
    one internal barrier
    partition output rows into 16-row tiles
    pair QGEMM U0/U1 using shared LUT
    pair QGEMM W0/W1 using the same LUT and add to output
    add optional bias
    convert FP32 accumulation to bf16-pair output
```

The graph node is enabled by `LLAMA_FAIRY2I_FUSED_WIDE_LINEAR_W2=1`. Its fused
LUT implementation additionally requires `GGML_IFAIRY_LUT=1`, packed IFAIRY64
weights, and the LUT16 implementation. If those conditions are not satisfied,
CPU routing falls back to the no-LUT FairyFuse implementation of the same
graph node.

The current fusion boundaries are therefore:

| Level | Current state |
| --- | --- |
| Graph | Four matrix multiplications and additions represented by one node |
| Activation preparation | Quantized once and represented by one shared LUT |
| Threading | One internal preparation barrier, disjoint output-row tiles |
| QGEMM | Two pair passes, not one monolithic four-weight AVX2 loop |
| LUT channel decode | Real and imaginary channels share one index decode |

The main code ownership map is:

| Stage | Code entry |
| --- | --- |
| Graph creation | `src/llama-model.cpp`, `ggml_ifairy_wide_linear_w2()` |
| CPU routing/workspace sizing | `ggml/src/ggml-cpu/ggml-cpu.c` |
| Eager weight prepack | `ggml/src/ggml-ifairy-lut-transform.cpp` |
| Activation quantization/LUT preparation/threading | `ggml/src/ggml-cpu/ifairy-fuse-lut.cpp` |
| Pair QGEMM and shared channel decode | `ggml/src/ggml-cpu/ifairy-fuse-lut-qgemm.cpp` |
| Generic LUT reference/preprocess helpers | `ggml/src/ggml-ifairy-lut-qgemm.cpp` |

### 4.11 Evaluated and Reverted Experiments

The following experiments were useful for identifying constraints but are not
part of the current implementation:

- **Direct four-weight AVX2 accumulation:** reduced intermediate output traffic
  but increased register pressure and regressed 8-thread decode.
- **Block-interleaved four-weight accumulation after shared channel decode:**
  processed U and W pairs consecutively for every K64 block, reused the same
  LUT block, and wrote output once. The generated native hot loop had no YMM
  stack spills, but interleaving four independent weight streams regressed all
  measured shapes. The retained two-pass pair kernel provides better sequential
  weight streaming and more stable decode performance.
- **Prefill-only four-weight kernel:** did not provide a stable improvement.
- **Tile-first prefill loop order:** reduced LUT locality and regressed pp64.
- **Manual AVX2 interleaved output store:** did not beat compiler-generated
  output code.
- **Additional internal barriers:** removed after block-level activation
  preparation proved correct.

Historical performance logs were collected across different machine states, so
only same-session comparisons should be interpreted as precise deltas. The
milestone trend nevertheless shows the effect of removing graph overhead,
sharing LUT preparation, reducing synchronization, matching K64 activation
blocks, and finally sharing channel index decode:

| Historical stage | 4T pp64 | 4T tg64 | 8T pp64 | 8T tg64 |
| --- | ---: | ---: | ---: | ---: |
| Non-fused LUT16 | 0.721 | 0.725 | 1.306 | 0.785 |
| Early fused pair QGEMM | 9.592 | 4.661 | 18.545 | 4.253 |
| Single-LUT pair kernels | 14.308 | 6.122 | 18.517 | 5.007 |
| One-barrier block preparation | 15.084 | 7.699 | 19.236 | 6.484 |
| One-barrier K64, memory pressure cleared | 17.539 | 8.228 | 22.616 | 6.356 |
| Current shared channel index decode | 21.928 | 11.156 | 27.534 | 10.036 |

The reverted block-interleaved experiment compared with the current shared
channel index baseline as follows:

| Threads | Test | Current pair passes | Block-interleaved | Change |
| ---: | --- | ---: | ---: | ---: |
| 4 | pp64 | 21.928 | 21.769 | -0.7% |
| 4 | tg64 | 11.156 | 10.163 | -8.9% |
| 8 | pp64 | 27.534 | 25.782 | -6.4% |
| 8 | tg64 | 10.036 | 9.545 | -4.9% |


## 5. Historical No-LUT FairyFuse Versus Non-Fused LUT16

This comparison predates the optimized fused-LUT path described above. The
`AVX2 fused W2` rows use the no-LUT FairyFuse kernel; they should not be read as
the performance of the current shared-LUT implementation.

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

## 6. Current Fused LUT Validation

The shared-channel-index implementation described in Section 4.9 was compared
with the previous one-barrier fused LUT implementation:

Compared with the previous one-barrier fused LUT implementation:

| Threads | Test | Before tok/s | After tok/s | Change |
| ---: | --- | ---: | ---: | ---: |
| 4 | pp64 | 17.539364 | 21.928456 | +25.0% |
| 4 | tg64 | 8.228292 | 11.156050 | +35.6% |
| 8 | pp64 | 22.616312 | 27.533854 | +21.7% |
| 8 | tg64 | 6.355514 | 10.036237 | +57.9% |

Both `build-rel` and `build-rel-lut` pass the complete `test-ifairy` suite.
The native Ryzen 9 7940H build uses extended YMM registers, so the same kernel
still requires separate validation on an AVX2-only CPU with 16 YMM registers.


The current fused LUT path also passes a WikiText-2 perplexity spot check:

| Context | Chunks | Current PPL | Previous fused LUT PPL |
| ---: | ---: | ---: | ---: |
| 128 | 1 | 7.9112 +/- 3.03494 | 7.9112 +/- 3.03494 |
| 256 | 1 | 64.6641 +/- 28.46448 | 65.2566 +/- 28.84096 |

No NaN, invalid output, or quality regression was observed. The high absolute
PPL at context 256 is an existing property of this single-chunk test setup,
not a regression introduced by shared channel index decoding.

## 7. Next Work

- Validate the shared-channel-index kernel on an AVX2-only CPU with only 16 YMM
  registers.
- Profile the remaining pair-pass weight streaming, scale restoration, and K64
  activation quantization costs.
- Investigate generic non-fused LUT16 performance separately from the fused-LUT
  production path.
- Measure thread affinity, polling, and graph-barrier costs for 8-thread
  decode.
- Test the full-width AVX512 kernel on hardware with native 512-bit execution.
- Evaluate whether a lower-register-pressure four-weight or super-block kernel
  can reuse the shared LUT without regressing decode.
