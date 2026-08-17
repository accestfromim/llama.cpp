---
name: upstream-pr-port
description: Port a selected upstream llama.cpp PR into the Fairy2i fork without importing unrelated upstream history.
---

# Port an upstream PR

Use this workflow when a change from `upstream/master` must be adapted to the fork after the divergence commit.

## Scope first

1. Record the upstream commit SHA, PR number, and changed paths.
2. Check whether the commit is a merge, revert, or depends on earlier commits.
3. Compare the upstream patch against the fork at the same semantic entry points; do not assume line-level applicability.
4. Keep one logical change per transplant. Do not mix upstream cleanup with Fairy2i behavior changes.

## Mapping rules

- `src/` owns model loading, GGUF metadata, tensor mapping, and graph construction.
- `ggml/src/ggml-cpu/` is hot-path code. Preserve scalar and non-ARM fallbacks.
- Fairy2i and legacy iFairy code has local routing and layout contracts documented by the nearest `AGENTS.md`.
- `gguf-py/` and `scripts/` own conversion, validation, and reference artifacts.
- `tests/` and `ci/` are part of the change contract, not optional follow-up work.

For every upstream file, classify the patch as one of:

- directly reusable: same symbol and same invariant;
- adaptable: same behavior but different local API, layout, or build flag;
- not applicable: upstream architecture or backend is absent from this fork.

Do not port a file only because its path matches. Port the invariant and then re-express it using the local interfaces.

## Dependency order

Prefer this order:

1. build/configuration and feature detection;
2. shared types, constants, and dispatch declarations;
3. scalar/reference implementation;
4. optimized ARM implementation;
5. loader/converter metadata and layout checks;
6. tests and independent oracles;
7. benchmark or documentation updates.

If a candidate depends on an earlier upstream PR, either port the dependency first or record the reason it is unnecessary in the local tree.

## Required review points

- Check `w * conj(x)` versus `w * x` for complex/iFairy paths.
- Check compile-time flags, runtime environment toggles, architecture guards, and shape gates independently.
- Check data layout, tile order, scale dtype, alignment, and tail behavior.
- Check output dtype and BF16 carrier packing at every boundary.
- Check that a new fast path has a generic/reference comparison test.
- Check that an unsupported platform falls back safely instead of failing to compile.

## Validation gates

Run the smallest gate that exercises the changed contract, then the full relevant matrix:

- CMake configure with the affected feature flags;
- targeted C++ test and `ctest -R` filter;
- `test-backend-ops` for changed ggml operators;
- `scripts/test-gguf-py.sh` for converter or GGUF changes;
- `scripts/ci-fairy2i-cpu.sh` for Fairy2i/legacy iFairy CPU changes;
- benchmark only after correctness passes, with one benchmark process at a time.

Record exact commands, compiler/architecture, relevant environment variables, and raw artifacts. A successful build without path-marker or oracle evidence is not sufficient for a performance transplant.

## Stop conditions

Stop and split the work when the PR introduces a new public API, new quantization type, new backend infrastructure, or a shared graph/sampler change not required by the selected behavior. Review the design before continuing.

Do not commit, push, create a PR, or write PR/reviewer text on the contributor's behalf.
