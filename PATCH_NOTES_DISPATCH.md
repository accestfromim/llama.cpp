# Patch Notes: Android ARM64 CPU Dispatch

## Modified Files

- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-cpu/CMakeLists.txt`
- `ggml/src/ggml-cpu/arch/arm/quants.c`
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.h`
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`
- `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`
- `DISPATCH_PLAN.md`
- `PATCH_NOTES_DISPATCH.md`
- `ANDROID_DISPATCH_NOTES.md`

## What Changed Per File

### `ggml/src/ggml-cpu/ggml-cpu.c`

- Added cached AArch64 Linux/Android CPU feature probing using `getauxval(AT_HWCAP)`.
- Updated `ggml_cpu_has_neon()` to use cached runtime probing on AArch64 Linux/Android.
- Updated `ggml_cpu_has_dotprod()` to use cached runtime probing on AArch64 Linux/Android.

Why:
- The old logic was compile-time based, which is not sufficient for one `arm64-v8a` binary that must run on devices with and without dotprod.

### `ggml/src/ggml-cpu/CMakeLists.txt`

- Added new ARM sources:
  - `ggml-cpu/arch/arm/quants-ifairy.c`
  - `ggml-cpu/arch/arm/quants-ifairy-dotprod.c`
- Added source-specific compile options for the dotprod file on Android `arm64-v8a` only:
  - `-march=armv8.2-a+dotprod`

Why:
- dotprod must not be enabled globally for the whole CPU backend.
- The dedicated dotprod source must be compiled separately so the generic `arm64-v8a` target stays compatible.

### `ggml/src/ggml-cpu/arch/arm/quants.c`

- Removed the embedded public iFairy vecdot implementation from the generic ARM quant file.

Why:
- That implementation mixed public entry and dotprod-specific code in one source file.
- The dispatch design requires the dotprod implementation to live in a dedicated source file.

### `ggml/src/ggml-cpu/arch/arm/quants-ifairy.h`

- Added shared internal declarations for:
  - ordinary NEON implementation
  - dotprod implementation
  - dotprod availability query
  - cached env helper for the iFairy activation tensor fast path

Why:
- The dispatch layer and the dotprod source need a small shared internal contract.

### `ggml/src/ggml-cpu/arch/arm/quants-ifairy.c`

- Added the new public entry `ggml_vec_dot_ifairy_q16_K`.
- Added one-time runtime dispatch with priority:
  - dotprod
  - NEON
  - scalar generic
- Added one-time debug logging for the selected path.
- Added an ordinary NEON implementation that does not require dotprod.

Why:
- This is the main runtime dispatch layer for Android `arm64-v8a`.
- It preserves the public API while allowing one binary to contain multiple code paths.

### `ggml/src/ggml-cpu/arch/arm/quants-ifairy-dotprod.c`

- Added the dedicated dotprod-only iFairy vecdot implementation.
- Added `ggml_vec_dot_ifairy_q16_K_dotprod_available()` so dispatch can verify the fast path was actually compiled into the binary.

Why:
- dotprod code must be isolated in a source file that can receive source-specific compile flags.

## Android-Specific Changes

- Runtime feature detection now uses `getauxval(AT_HWCAP)` for AArch64 Linux/Android.
- The dotprod implementation is compiled from a dedicated source file only.
- Only that dedicated source gets `-march=armv8.2-a+dotprod`, and only under Android `arm64-v8a`.
- The `llama.android` project keeps building through the existing repo-root CMake integration.

## Runtime Dispatch Changes

- The public entry point remains `ggml_vec_dot_ifairy_q16_K`.
- Dispatch runs once via `pthread_once`.
- The selected function pointer is cached and reused.
- No CPU feature detection is performed inside the hot inner loops.

## Why This Patch Is Minimal

- No JNI/API changes
- No Android project split
- No model graph changes
- No changes to public type-traits wiring beyond keeping the existing entry
- No global dotprod enablement for the backend target

## Residual Risks

- The new ordinary NEON path needs real AArch64 validation to confirm both correctness and performance behavior.
- This patch intentionally scopes the source-specific dotprod compile flag to Android `arm64-v8a`; non-Android AArch64 builds continue to rely on their existing build configuration.
- `git clang-format` was not available in this environment, so formatting verification could not be run here.
