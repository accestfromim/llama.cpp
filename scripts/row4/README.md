# Qwen3 Row4 oracle

`row4_oracle.py` is an independent, pure-Torch oracle for the frozen Row4
W1A8-INT8 and lm_head W8A8 profile.  It does not import the unavailable
training-time `row4_qat` package, the FP8 reference kernels, or the GGUF
converter's quantization helpers.

The tool reads only selected complete `O16 x K` tiles from the real
checkpoint.  For layers 0 and 35 it covers every component boundary in fused
QKV and gate/up, the first and last tiles of O and down, and the first/last
tile containing a non-positive learned scale in every component where one
exists.  It also covers the first and last lm_head tiles.  Each golden contains
raw little-endian bits for:

- the latent checkpoint BF16 rows and saved signed BF16 Row4 scales;
- logical Row4 nibbles, canonical split8 packed bytes, and decoded INT8 rows;
- F32 activation input, its BF16 RNE round-trip, A8 codes, and F32 scale;
- exact INT32 accumulators, pre-BF16 F32 results, BF16 outputs, and F32
  carriers;
- lm_head W8 codes, tiled bytes, and offline F32 row scales.

Artifacts must live outside the repository.  They are never model assets and
must not be committed.

The dedicated converter resolves the existing output parent before writing
and refuses any writable directory owned by an untrusted uid, plus any
group- or world-writable ancestor without both a sticky bit and an owner of
root or the current euid.  This fail-closed rule prevents another user—or the
owner of an untrusted sticky ancestor—from swapping the private temporary
directory while `GGUFWriter` opens it by pathname; standard root-owned `/tmp`
remains supported.  Conversion writes inside a unique mode-0700 directory and
publishes the completed GGUF with an atomic, no-clobber hard link.  The final
GGUF also records the required `general.quantization_version=2` metadata and
`tokenizer.ggml.add_bos_token=true`.  The reference chat template emits
`{{ bos_token }}` as its first output expression; llama.cpp's common chat path
removes that rendered leading token when automatic BOS is enabled, then the
tokenizer prepends exactly one BOS.  This pairing is intentional and must not
be changed independently.

## Dependencies and self-test

Use the same Python environment as `convert_row4_qwen3.py` (PyTorch,
SafeTensors, and NumPy):

```sh
python3 scripts/row4/row4_oracle.py self-test
```

The self-test is synthetic and checks all 16 Row4 codes, tie/sign-zero rules,
the `80 91 a2 b3 c4 d5 e6 f7` split8 known answer, half-away rounding, A8,
the W8 zero-row rule, signed scale handling, and physical shapes.

## Capture actual checkpoint primitives

```sh
export ROW4_ORACLE_DIR=/absolute/external/row4-oracles
export ROW4_CHECKPOINT_DIR=/absolute/path/to/checkpoint-15860
python3 scripts/row4/row4_oracle.py capture \
  --input-ids 1,2 \
  --run-name checkpoint-primitives
```

`--checkpoint /absolute/path/to/checkpoint-15860` may be used instead of
`ROW4_CHECKPOINT_DIR`; there is no machine-specific built-in checkpoint path.

By default, input IDs key a deterministic integer-derived F32 carrier for each
projection shape.  This tests real checkpoint weights without requiring a
full model forward.  To evaluate carriers captured from the same runtime
input, pass a Torch dictionary with these `[T,K]` F32 tensors:

```text
layer0.qkv       layer0.o       layer0.gate_up       layer0.down
layer35.qkv      layer35.o      layer35.gate_up      layer35.down
lm_head
```

The number of rows `T` must equal the number of `--input-ids`:

```sh
python3 scripts/row4/row4_oracle.py capture \
  --input-ids '[151644, 9707]' \
  --activations /absolute/path/to/f32-carriers.pt \
  --run-name captured-carriers
```

Capture computes full SHA-256 hashes for config/tokenizer files and all four
checkpoint shards.  `--skip-shard-hashes` is only for quick local development;
its manifest explicitly records that the required full shard hashes are
absent.

## Verify the converted GGUF

`verify` performs the capture and additionally checks the fixed Row4 metadata,
type IDs, every selected canonical packed tile, signed BF16 scale bits, W8
lm_head bytes, and F32 lm_head scale bits:

```sh
python3 scripts/row4/row4_oracle.py verify \
  --gguf /absolute/path/to/qwen3-row4-int8.gguf \
  --run-name gguf-verified
```

`ROW4_GGUF=/absolute/path/to/qwen3-row4-int8.gguf` is the equivalent portable
default for `verify`.  Before any selected sample is compared, verification
requires the complete fixed metadata contract (including YaRN attention
pre-factor `1.0` and automatic BOS enabled), the exact 436 tensor
names/types/shapes, and exactly 2,739,236,352 tensor payload bytes.

The result directory name is reserved with an atomic no-clobber `mkdir`, and
`manifest.json` is moved into place last as the completion marker.  This is
portable across macOS and Linux, rejects concurrent creators and broken
symlinks, and never overwrites an existing output directory.

## Compare external logits

Full model execution is intentionally not reconstructed here: the checkpoint
does not include its `row4_qat` implementation.  Reference and runtime logits
can instead be exported as a tensor, or as a dictionary containing `logits`
and optional `target_ids`, in `.pt`/`.pth`; `.npy` is also accepted.

```sh
python3 scripts/row4/row4_oracle.py compare-logits \
  --reference /absolute/path/reference.pt \
  --candidate /absolute/path/runtime.pt \
  --output /absolute/path/metrics.json
```

The command enforces NMSE `<=1e-5`, max absolute error `<=5e-2`, identical
argmax, at least 9/10 top-10 overlap for every token, and—when targets are
available—mean NLL difference `<=5e-3` and relative PPL difference `<=0.5%`.

## Full checkpoint modeling capture

`full_model.py` is a separate full-forward entry point.  Before importing the
checkpoint's unmodified `modeling_qwen3_row4_int8.py`, it injects transparent
`row4_qat.Row4Int8Linear` and `row4_qat.Int8Linear` replacements.  Their state
dict schema matches the checkpoint and their forward methods use the same
BF16-RNE, half-away A8, exact INT32 accumulation, ordered scale multiplication,
and BF16 output contract as the primitive oracle.

The shim processes weights in aligned output-row chunks, so it does not create
a full F32 weight copy.  `--cache-decoded-row4` optionally retains the full
decoded INT8 Row4 weights between prefill and decode; this uses roughly another
8 GB but avoids decoding the projection weights three times.  W8 lm_head
quantization remains dynamic on every forward, as specified by the checkpoint
modeling comments.

Run the tiny shim test without loading the model:

```sh
python3 scripts/row4/full_model.py self-test
```

It compares both compatibility classes against `numeric.py` at BF16 bit level,
checks the expected state-dict keys, the lm_head all-zero row, and cached versus
uncached Row4 decoding.

The reference environment is intentionally strict.  It must match the
fixed 36-layer/4096-hidden/12288-intermediate topology, SiLU and all-full
attention semantics, untied/bias-free BF16 checkpoint schema, config and
installed `transformers==5.2.0`, its training
`attn_impl=flash_attention_3`, exact four-shard index/header inventory, and a
Hopper-class CUDA device.  Inspect it without loading tensor payloads:

```sh
python3 scripts/row4/full_model.py preflight \
  --checkpoint /absolute/path/to/checkpoint-15860 \
  --device cuda:0 \
  --attn-implementation flash_attention_3
```

In the original training/inference environment, make sure PyTorch with CUDA,
`transformers==5.2.0`, Accelerate, SafeTensors, and the same working FA3 backend
are installed.  Then run:

```sh
export ROW4_ORACLE_DIR=/absolute/external/row4-oracles
python3 scripts/row4/full_model.py capture \
  --checkpoint /absolute/path/to/checkpoint-15860 \
  --device cuda:0 \
  --attn-implementation flash_attention_3 \
  --input-ids 151643,785,4226,374 \
  --decode-ids 220,19 \
  --cache-decoded-row4 \
  --run-name fa3-prefill-decode-golden
```

This executes one fixed prefill and two fixed single-token decode calls through
the original model and KV cache.  It exports:

- BF16 and widened-F32 logits for all three phases;
- actual layer 0/35 QKV, O, gate/up, and down carriers plus the lm_head carrier;
- one `*_carriers.pt` per phase, directly accepted by
  `row4_oracle.py capture --activations`;
- scored logits and targets spanning all prefill predictions plus decode 1,
  along with mean NLL and PPL;
- argmax/top-10 IDs, environment/backend details, source/config/tokenizer
  hashes, and all four shard hashes.

`--allow-nonreference` permits diagnostic eager/CPU/MPS runs, but the manifest
sets `reference_capture=false` and lists every mismatch.  Such output must not
be presented as the original FA3 forward golden.

Likewise, `--skip-shard-hashes` is development-only and always forces
`reference_ready=false`/`reference_capture=false`; a reference manifest must
contain SHA-256 provenance for all four shards.

The local Apple M4 host is not a reference execution environment: it has 24 GB
unified memory versus a 16.38 GB BF16 checkpoint, no CUDA/Hopper FA3, and its
current Row4 virtual environment does not match Transformers 5.2 or include
Accelerate.  Therefore the full 16 GB forward is deliberately not run here;
only the lightweight shim self-test and read-only preflight are safe locally.
