# Android ARM64 CPU Dispatch Plan

## Current State

- Public vecdot entry for iFairy weights is `ggml_vec_dot_ifairy_q16_K`.
  File: `ggml/src/ggml-cpu/quants.h`
- The CPU type table binds `GGML_TYPE_IFAIRY` to `ggml_vec_dot_ifairy_q16_K`.
  File: `ggml/src/ggml-cpu/ggml-cpu.c`
- Generic scalar fallback lives in `ggml_vec_dot_ifairy_q16_K_generic`.
  File: `ggml/src/ggml-cpu/quants.c`
- Current ARM-specific iFairy vecdot implementation lives in `ggml/src/ggml-cpu/arch/arm/quants.c`.
- Current ARM iFairy fast path is guarded by `#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)`.
  This means the existing ARM implementation is compile-time selected, not runtime dispatched.
- Existing ARM CPU feature helpers exposed to upper layers are:
  - `ggml_cpu_has_neon()`
  - `ggml_cpu_has_dotprod()`
  - `ggml_cpu_has_matmul_int8()`
  - `ggml_cpu_has_sve()`
  File: `ggml/include/ggml-cpu.h`, implementation currently in `ggml/src/ggml-cpu/ggml-cpu.c`
- There is already AArch64 feature probing infrastructure based on `getauxval(AT_HWCAP)` in `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`.
  That code is currently used for backend scoring / variant selection, not for the iFairy vecdot runtime dispatch entry.

## Current Implementations

### Public Entry

- `ggml_vec_dot_ifairy_q16_K`
  File: `ggml/src/ggml-cpu/arch/arm/quants.c`

### Scalar Fallback

- `ggml_vec_dot_ifairy_q16_K_generic`
  File: `ggml/src/ggml-cpu/quants.c`

### Existing ARM NEON-related Sources

- `ggml/src/ggml-cpu/arch/arm/quants.c`
- `ggml/src/ggml-cpu/arch/arm/repack.cpp`

### Existing dotprod Fast Path

- The current iFairy dotprod fast path is embedded directly in:
  - `ggml/src/ggml-cpu/arch/arm/quants.c`
- It uses `__ARM_FEATURE_DOTPROD` and inline `sdot` instructions.

## Android Build Wiring

- `examples/llama.android/llama/src/main/cpp/CMakeLists.txt` pulls the repo root with:
  - `add_subdirectory(../../../../../../ build-llama)`
- The Android Gradle module passes a few CMake arguments from:
  - `examples/llama.android/llama/build.gradle.kts`
- ARM backend sources are selected in:
  - `ggml/src/ggml-cpu/CMakeLists.txt`
- iFairy LUT support is enabled from ggml core CMake:
  - `ggml/CMakeLists.txt`
  - `ggml/src/CMakeLists.txt`

## Interfaces To Keep Stable

- Keep `ggml_vec_dot_ifairy_q16_K` as the public entry.
- Keep `ggml_get_type_traits_cpu(GGML_TYPE_IFAIRY)->vec_dot` unchanged.
- Keep Android JNI / `llama.android` app-facing APIs unchanged.
- Keep `ggml_vec_dot_ifairy_q16_K_generic` as the scalar fallback.

## Proposed Split

### Files To Keep As-Is Semantically

- `ggml/src/ggml-cpu/quants.c`
  Keep the scalar generic implementation as the correctness baseline.
- `ggml/src/ggml-cpu/ggml-cpu.c`
  Keep the type-traits entry points and public CPU feature API names.

### Files To Split / Refactor

- `ggml/src/ggml-cpu/arch/arm/quants.c`
  Remove the embedded iFairy dotprod-only public implementation and replace it with a dispatch entry.

### New Dispatch / Implementation Files

- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.h`
  Internal prototypes for the split implementations.
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`
  Public entry `ggml_vec_dot_ifairy_q16_K`, runtime dispatch, one-time logging, and ordinary NEON implementation.
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`
  dotprod-only implementation compiled with source-specific flags.

## Runtime Dispatch Design

- Dispatch priority:
  1. dotprod
  2. ordinary NEON
  3. scalar generic fallback
- Dispatch must happen once and cache the selected function pointer.
- CPU feature probing on Android/Linux AArch64 should use cached `getauxval(AT_HWCAP)` results through the existing `ggml_cpu_has_*` API.

## CMake Changes Required

- `ggml/src/ggml-cpu/CMakeLists.txt`
  - add `ggml-cpu/arch/arm/quants-ifairy.c`
  - add `ggml-cpu/arch/arm/quants-ifairy-dotprod.c`
  - apply source-specific compile options to the dotprod file only
- No global `-march=armv8.2-a+dotprod` on the whole ggml or Android native target
- `examples/llama.android` CMake should remain structurally unchanged and continue to build through the repo root

## Risks

- Android `arm64-v8a` guarantees AArch64 but not dotprod, so wrong flag scoping would break compatibility.
- The current `ggml_cpu_has_neon()` / `ggml_cpu_has_dotprod()` implementations are compile-time based and must be updated for runtime dispatch to be meaningful.
- The ordinary NEON implementation must preserve iFairy semantics exactly: `w * conj(x)`.
- Debug logging must not spam hot paths; it should only log once when the implementation is selected.
