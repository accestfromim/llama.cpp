# Row4 Prefix Sliding Metal experiment

## Scope and provenance

This experiment adds Prefix Sliding attention to the Row4 Metal runtime. It starts from `codex/row4-turboquant-mixed-kv@6b62d5f737b3` and uses these implementations as semantic references:

- `prefix-sliding@4350eb6a`
- `vllm/st29@22abbc6c`
- `flash-attention/st29@bd2d1519`

The implementation does not change GGUF and does not import training or reinforcement-learning code. It is limited to the Qwen3 Row4 model, full Metal placement, causal attention, Flash Attention, and non-unified KV caches. Prefix Sliding is disabled unless both configuration values are positive.

## Attention rule

Let `P` be the tokenized request prompt length, `W` the recent window including the current token, and `q` the absolute query position. The query can attend to:

```text
[0, P) OR [max(P, q - W + 1), q]
```

The prefix and recent regions participate in one online softmax. RoPE positions stay absolute. The cache does not shift or renumber positions.

The experiment default is:

```text
W = 8192
prefix_cap = 4096
```

The prefix boundary is request metadata, not an inferred cache position. Server and CLI callers set it after tokenizing the complete prompt. Prompts above the configured cap are rejected.

## Configuration

The common CLI accepts:

```text
--prefix-sliding-window N
--prefix-sliding-prefix-cap N
```

The matching environment variables are:

```text
LLAMA_ARG_PREFIX_SLIDING_WINDOW
LLAMA_ARG_PREFIX_SLIDING_PREFIX_CAP
```

The C API adds:

```c
bool llama_memory_seq_set_prefix(llama_memory_t mem, llama_seq_id seq_id, llama_pos prefix_end);
llama_pos llama_memory_seq_get_prefix(llama_memory_t mem, llama_seq_id seq_id);
```

Supported KV configurations are BF16/BF16 and Turbo4 K with Turbo3 V. K4/V3 remains compatible with `TURBO_K_MEAN_CENTER`, `TURBO_K_MEAN_WARMUP`, and `TURBO_KV_BOUNDARY_BF16_LAYERS`.

Example BF16 server:

```bash
build-prefix-metal/bin/llama-server \
    -m qwen3-row4-v2-pair2.gguf -ngl 99 -fa on \
    -ctk bf16 -ctv bf16 -c 1056768 -np 16 -b 2048 -ub 512 \
    --no-context-shift --prefix-sliding-window 8192 \
    --prefix-sliding-prefix-cap 4096
```

Example K4/V3 server:

```bash
TURBO_K_MEAN_CENTER=2 TURBO_K_MEAN_WARMUP=128 \
TURBO_KV_BOUNDARY_BF16_LAYERS=0 \
build-prefix-metal/bin/llama-server \
    -m qwen3-row4-v2-pair2.gguf -ngl 99 -fa on \
    -ctk turbo4 -ctv turbo3 -c 1056768 -np 16 -b 2048 -ub 512 \
    --no-context-shift --prefix-sliding-window 8192 \
    --prefix-sliding-prefix-cap 4096
```

The runtime rejects non-Row4 models, non-causal attention, CPU or mixed placement, other KV pairs, disabled Flash Attention, disabled KQV offload, unified KV, model-defined SWA, context shift, chunk relocation reuse, speculative decoding, and multimodal mode.

## Physical KV layout

Logical context length and physical KV capacity are separate. Each stream allocates:

```text
PAD(min(n_ctx_per_seq, prefix_cap + W - 1 + n_ubatch), cache_padding)
```

Slot search further limits the active physical range using the real `P`, so a short prompt does not scan the unused part of the cap allocation. Prefix cells are never overwritten. A new ubatch can overwrite only non-prefix cells older than the recent window of its earliest query.

The current Row4 model has 36 layers, 8 KV heads, and head dimension 128. Its per-cell storage is 144 KiB for BF16/BF16 and 32.625 KiB for K4/V3. With cap 4096, ubatch 512, and 256-cell padding, the allocations are:

| Window | Cells per stream | BF16 MiB per stream | K4/V3 MiB per stream | BF16 GiB for 16 | K4/V3 GiB for 16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4,864 | 684.00 | 154.97 | 10.688 | 2.421 |
| 512 | 5,120 | 720.00 | 163.12 | 11.250 | 2.549 |
| 1,024 | 5,632 | 792.00 | 179.44 | 12.375 | 2.804 |
| 2,048 | 6,656 | 936.00 | 212.06 | 14.625 | 3.313 |
| 4,096 | 8,704 | 1,224.00 | 277.31 | 19.125 | 4.333 |
| 8,192 | 12,800 | 1,800.00 | 407.81 | 28.125 | 6.372 |
| 16,384 | 20,992 | 2,952.00 | 668.81 | 46.125 | 10.450 |

## Cache lifecycle and state

Prefix metadata follows sequence clear, full copy, keep, remove, whole-state restore, and sequence-state restore. Position add and divide operations are unsupported while Prefix Sliding is enabled.

Session and sequence state file versions are 10 and 3. The KV payload has an internal magic and version and records window, cap, K/V types, each sequence prefix boundary, and all K-mean centering tensors. New code reads legacy session 9 and sequence 2 files only when Prefix Sliding is disabled. A Prefix state restore rejects mismatched window, cap, or K/V types.

The centering token count is independent of the physical ring index. This is required because physical cell zero can be reused after wrap and is not a request boundary. Full sequence removal resets the count for both `[-1, -1)` and `[0, end)` clear forms.

## Correctness validation

The Apple M5 Max Release Metal build completed these focused checks:

| Check | Result |
| --- | --- |
| Explicit Prefix mask, BF16 and K4/V3, decode and 32/512-token prefill | 6/6 passed, no selected `not supported` case |
| D128, 32 Q heads, 8 KV heads CPU-reference comparison | Passed within the existing `5e-4` NMSE budget |
| K-mean centering, I32/I64, 1/4 streams, physical wrap | 10 focused cases passed |
| Model lifecycle, 4 streams, chunked prompt, absolute position 511 | Passed for BF16 and K4/V3 |
| Prefix protection and bounded sequence-state size after saturation | Passed |
| Whole and sequence state roundtrip, full copy, remove, keep, clear | Passed |
| Current Prefix state load and legacy non-Prefix state load | Passed |
| Legacy state into Prefix context and mismatched Prefix state | Rejected |
| Short BF16 and K4/V3 sequences where `W` covers all history | Byte-identical generated output with Prefix disabled |

The existing Metal BF16 exact and K4/V3 fused attention kernels already reject fully masked blocks before loading K/V. Prefix Sliding therefore changes the host mask and cache lifecycle but does not add a second softmax, dense matrix, page table, or duplicate shader path.

## Quality at 64K

The quality check uses the v2 Pair2 LUT model, cap 4096, window 8192, a 4096-token prefix, absolute positions through 65,536, and 256 fixed suffix logits. BF16 Prefix Sliding is the logits reference. K4/V3 uses centering mode 2, warmup 128, and no BF16 boundary layers.

| Configuration | Mean KL | Top-1 | Top-5 overlap | Exact Top-5 | BF16 Top-1 in candidate Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16/BF16 | 0 | 100% | 100% | 100% | 100% |
| K4/V3 centered | 0.054845 | 86.328% | 87.266% | 48.047% | 98.828% |

The 16-chunk WikiText-2 check uses context 2048. Window 8192 covers the full evaluated context, so it isolates KV quantization from attention truncation:

| Configuration | PPL |
| --- | ---: |
| BF16/BF16 | 31.8656 +/- 0.81957 |
| K4/V3 centered | 31.8328 +/- 0.81334 |

## W8192 single-stream performance

The long-context microbenchmark keeps a 4096-token prefix, uses an 8192-token recent window, restores one prepared context for each sample, and reports five timed repetitions. `pp512` is prompt processing for the next 512 absolute positions and `tg128` is one-stream decode for 128 positions.

| Absolute context | BF16 pp512 tok/s | K4/V3 pp512 tok/s | K4/V3 delta | BF16 tg128 tok/s | K4/V3 tg128 tok/s | K4/V3 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3,691.55 | 2,606.46 | -29.4% | 146.72 | 110.00 | -25.0% |
| 1,024 | 2,395.47 | 2,490.88 | +4.0% | 142.16 | 101.10 | -28.9% |
| 8,192 | 1,200.66 | 1,056.31 | -12.0% | 104.22 | 81.36 | -21.9% |
| 32,768 | 925.74 | 787.50 | -14.9% | 90.90 | 73.87 | -18.7% |
| 65,536 | 852.45 | 783.23 | -8.1% | 79.96 | 73.48 | -8.1% |

The physical attention span stops growing once the 4096-token prefix and 8192-token recent window are full. The remaining depth trend comes from ring management, absolute-position graph work, and benchmark state preparation, not a scan of the evicted middle. K4/V3 approaches BF16 at 64K but remains slower in decode because its fused kernel still pays WHT and centroid decode costs.

## W8192 sustained multi-stream decode

The multi-stream benchmark allocates 21 independent streams, uses disjoint groups for the 1/4/16-stream cases, and removes only the generated suffix before each repetition so every sample starts at the same absolute position. Each concurrency shape receives 512 untimed decode steps before five timed `tg128` repetitions. This is a sustained-load measurement; unlike the single-stream table above, state restore does not create a cooling interval between samples.

| Absolute context | Streams | BF16 tok/s | K4/V3 tok/s | K4/V3 delta |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 145.56 | 104.39 | -28.29% |
| 0 | 4 | 258.30 | 213.04 | -17.52% |
| 0 | 16 | 372.22 | 333.91 | -10.29% |
| 1,024 | 1 | 137.43 | 101.40 | -26.22% |
| 1,024 | 4 | 231.63 | 199.54 | -13.85% |
| 1,024 | 16 | 311.69 | 307.59 | -1.32% |
| 8,192 | 1 | 86.79 | 81.07 | -6.59% |
| 8,192 | 4 | 141.72 | 141.33 | -0.28% |
| 8,192 | 16 | 175.07 | 190.09 | +8.58% |
| 32,768 | 1 | 74.99 | 73.59 | -1.86% |
| 32,768 | 4 | 115.57 | 120.46 | +4.23% |
| 32,768 | 16 | 136.17 | 155.64 | +14.29% |
| 65,536 | 1 | 74.87 | 72.95 | -2.56% |
| 65,536 | 4 | 114.67 | 120.00 | +4.65% |
| 65,536 | 16 | 135.97 | 155.46 | +14.34% |

The expected crossover is visible. K4/V3 is slower for short-context single-stream decode, reaches parity near 8K with four streams, and is 14.34% faster than BF16 at 64K with 16 streams. At 64K, 16 streams deliver 2.13x the K4/V3 single-stream throughput and 1.82x the BF16 single-stream throughput. Compression reduces the bandwidth term, while enough concurrent queries amortize WHT, centroid decode, and output-projection setup.

The following dense figures are historical measurements from the same model and code base before Prefix Sliding. They are included to make the long-context benefit visible, but they were not collected in the same benchmark invocation and are not a strict paired comparison.

| Absolute context | BF16 dense pp512 | BF16 Prefix pp512 | BF16 dense tg128 | BF16 Prefix tg128 | K4/V3 dense pp512 | K4/V3 Prefix pp512 | K4/V3 dense tg128 | K4/V3 Prefix tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3,673.23 | 3,691.55 | 124.52 | 146.72 | 3,181.90 | 2,606.46 | 103.97 | 110.00 |
| 1,024 | 3,058.35 | 2,395.47 | 109.08 | 142.16 | 1,982.36 | 2,490.88 | 65.62 | 101.10 |
| 8,192 | 1,328.11 | 1,200.66 | 95.86 | 104.22 | 771.80 | 1,056.31 | 18.12 | 81.36 |
| 32,768 | 303.30 | 925.74 | 48.95 | 90.90 | 223.73 | 787.50 | 5.22 | 73.87 |
| 65,536 | 132.48 | 852.45 | 29.00 | 79.96 | 111.33 | 783.23 | 2.68 | 73.48 |

At 64K, Prefix Sliding raises BF16 `pp512` by 543.5% and `tg128` by 175.7% relative to these dense measurements. The K4/V3 gains are 603.5% and 2639.4%, respectively.

## AIME2026 window sweep

The AIME2026 sweep uses greedy decoding, a deterministic seed per problem, 64K maximum generated context, 16 concurrent server slots, and cap 4096. Each completed configuration evaluates every problem once. A pass requires the answer in a `boxed` expression; none of the reported passes depends on the fallback last-integer parser.

| Window | BF16 passed | K4/V3 passed | BF16 generated tokens | K4/V3 generated tokens | BF16 KV MiB | K4/V3 KV MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 12/30 | 13/30 | 916,313 | 1,063,592 | 47,232.00 | 10,703.25 |
| 8,192 | 11/30 | 16/30 | 1,128,892 | 1,047,580 | 28,800.00 | 6,527.25 |
| 4,096 | 10/30 | 7/30 | 1,356,934 | 1,397,432 | 19,584.00 | 4,439.25 |
| 2,048 | 9/30 | 6/30 | 1,494,693 | 1,549,539 | 14,976.00 | 3,395.25 |

`P` marks a passed problem in the per-problem results:

| Problem | BF16 W16K | K4/V3 W16K | BF16 W8K | K4/V3 W8K | BF16 W4K | K4/V3 W4K | BF16 W2K | K4/V3 W2K |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | P | P | P | P | P | P | P | P |
| 2 | - | - | - | P | - | P | - | - |
| 3 | P | P | P | P | P | - | P | P |
| 4 | - | P | - | P | - | - | - | - |
| 5 | P | P | P | P | P | P | P | P |
| 6 | P | P | P | P | P | P | P | P |
| 7 | P | P | P | P | - | - | - | - |
| 8 | P | P | P | P | P | P | P | P |
| 9 | - | - | - | - | - | - | - | - |
| 10 | - | - | - | - | - | - | - | - |
| 11 | - | - | - | - | - | - | - | - |
| 12 | P | P | P | P | - | - | - | - |
| 13 | - | - | - | - | - | - | - | - |
| 14 | - | - | - | - | - | - | - | - |
| 15 | - | - | - | - | - | - | - | - |
| 16 | - | P | - | P | - | - | P | - |
| 17 | - | - | - | - | - | - | - | - |
| 18 | - | - | - | - | - | - | - | - |
| 19 | P | P | P | P | - | - | P | - |
| 20 | P | P | P | P | P | P | P | P |
| 21 | - | - | P | P | P | P | P | - |
| 22 | - | P | - | P | P | - | - | - |
| 23 | P | P | - | - | P | - | - | - |
| 24 | P | - | P | P | P | - | - | - |
| 25 | - | - | - | P | - | - | - | - |
| 26 | P | - | - | - | - | - | - | - |
| 27 | - | - | - | - | - | - | - | - |
| 28 | - | - | - | - | - | - | - | - |
| 29 | - | - | - | - | - | - | - | - |
| 30 | - | - | - | - | - | - | - | - |

The completed sweep shows a practical quality knee between 4K and 8K. W8K is the best observed operating point: K4/V3 reaches 16/30, uses 22.66% of the BF16 KV memory, and generates fewer tokens than its BF16 peer. W16K increases KV allocation by 64.0% relative to W8K but scores 13/30 with K4/V3 and 12/30 with BF16. It does not justify the extra memory on this single deterministic evaluation. W4K and W2K reduce memory further, but both formats lose passes and consume more generation tokens, so the smaller physical cache does not translate into a better end-to-end tradeoff.

K4/V3 exceeding BF16 at W8K is not evidence that quantization improves model accuracy. Greedy long rollouts amplify small logit changes into different trajectories. The 64K logits check above still measures nonzero quantization error. The AIME result shows that centered K4/V3 preserves task behavior well at W8K, not that it is intrinsically more accurate.

The BF16 W1K run was already in flight when the smaller-window sweep was stopped and completed at 3/30, with problems 1, 5, and 16 passing. K4/V3 W1K and both W512/W256 configurations were not run. This unpaired W1K result is only supporting evidence of a sharp quality collapse and is excluded from the comparison table.

## Build and tooling

The implementation was rebuilt in a fresh Release Metal directory. `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, `test-arg-parser`, `test-backend-ops`, and `test-prefix-sliding` all build successfully. Diff-only `clang-format` was run on the modified C/C++ hunks; formatting suggestions that would split log messages or replace the repository's constructor style were not retained. `clang-tidy` is not installed on this host.

The aggregate `all` target still reaches two pre-existing optional Row4 microbenchmark failures from the base branch: `ifairy-microbench` cannot find `ggml-ifairy-lut-impl.h`, and `ifairy-vecdot-microbench` has an unresolved symbol. Neither target contains Prefix Sliding code; all production and focused test targets listed above build.

Raw logs, state files, JSON, and benchmark helpers are under the untracked `tmp/row4-prefix-sliding/` directory.
