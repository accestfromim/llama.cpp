# Fairy2i Migration Notes

This page records the compatibility policy for the current refactor.

## Branching And Commits

Work should happen on a dedicated branch. Each independent change should be
tested and committed before starting the next change.

Commit messages should follow the repository style:

```text
<module>: concise message
```

## GGUF Metadata

New converters write the normalized schema documented in
`docs/fairy2i/spec.md`:

```text
general.architecture = "fairy2i"
fairy2i.schema_version = 1
fairy2i.base_arch = "llama" | "qwen2"
fairy2i.quant.format = "fairy2i_tile64_v2"
fairy2i.quant.variant = "tile64_v2"
fairy2i.tokenizer.profile = "llama_bpe" | "qwen2"
```

Compatibility with earlier experimental files belongs in an explicit legacy
loader or migration tool. New Fairy2i readers must not silently treat missing
schema keys as normalized Fairy2i v1 files.

## Converter Entry Points

Preferred:

```bash
PYTHONPATH=gguf-py .venv/bin/python -m fairy2i.convert ...
```

Compatibility:

```bash
gguf-py/convert_fairy2i_llama.py
gguf-py/convert_fairy2i_qwen2.py
```

The compatibility scripts now call shared Fairy2i schema helpers. Llama no
longer imports Qwen2 converter internals for tile64_v2 packing.

## CPU Build Options

Legacy-only:

```text
GGML_LEGACY_IFAIRY_CPU
GGML_LEGACY_IFAIRY_CPU_LUT
GGML_LEGACY_IFAIRY_CPU_AVX512
GGML_LEGACY_IFAIRY_CPU_ARM_DOTPROD
```

New:

```text
GGML_FAIRY2I_CPU_LUT
GGML_FAIRY2I_CPU_AVX512
```

The old names now map only to legacy iFairy options:

```text
GGML_IFAIRY_LUT_CPU      -> GGML_LEGACY_IFAIRY_CPU_LUT
GGML_IFAIRY_FUSE_AVX512 -> GGML_LEGACY_IFAIRY_CPU_AVX512
```

They do not enable Fairy2i features. Legacy iFairy validation must include a
Fairy2i-off build so old vecdot, W2 fused, and LUT execution cannot silently
depend on `GGML_FAIRY2I*`.

Recommended validation commands:

```bash
cmake -B build-ifairy-direct \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF
cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j 4
ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct

cmake -B build-ifairy-legacy \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=ON
cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j 4
GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy

cmake -B build-rel-fairy2i \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=ON \
    -DGGML_LEGACY_IFAIRY_CPU=OFF
cmake --build build-rel-fairy2i --target test-fairy2i test-backend-ops -j 4
GGML_FAIRY2I_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
GGML_FAIRY2I_LUT=1 ./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
```

`ggml-base` no longer compiles Fairy2i or legacy iFairy LUT/QGEMM execution
sources. Those files are owned by the CPU backend feature gates; model loading
does not eagerly prepack Fairy2i LUT weights.

## OpenCL Scope

OpenCL is part of the full decoupling scope. New Fairy2i OpenCL routing uses
`GGML_OPENCL_FAIRY2I`; legacy iFairy routing keeps the older
`GGML_OPENCL_IFAIRY64` gate until the legacy backend is isolated.
