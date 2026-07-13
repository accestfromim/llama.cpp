#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run cmake -B build-cpu-clean \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=OFF
run cmake --build build-cpu-clean --target ggml-base ggml-cpu -j "${JOBS}"

run cmake -B build-rel-fairy2i-direct \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=OFF
run cmake --build build-rel-fairy2i-direct --target test-fairy2i test-fairy2i-loader -j "${JOBS}"
run ctest --test-dir build-rel-fairy2i-direct --output-on-failure -R fairy2i

run cmake -B build-rel-fairy2i \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_FAIRY2I=ON \
    -DGGML_FAIRY2I_CPU=ON \
    -DGGML_FAIRY2I_CPU_LUT=ON \
    -DGGML_LEGACY_IFAIRY_CPU=OFF
run cmake --build build-rel-fairy2i --target test-fairy2i test-fairy2i-loader test-backend-ops -j "${JOBS}"
run ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
run env GGML_FAIRY2I_TEST_REQUIRE_LUT=1 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
run env GGML_FAIRY2I_LUT=0 ctest --test-dir build-rel-fairy2i --output-on-failure -R fairy2i
run ./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2
run env GGML_FAIRY2I_LUT=0 ./build-rel-fairy2i/bin/test-backend-ops test -b CPU -o FAIRY2I_WIDE_LINEAR_W2

run cmake -B build-ifairy-direct \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=OFF
run cmake --build build-ifairy-direct --target test-legacy-ifairy-direct -j "${JOBS}"
run ctest --test-dir build-ifairy-direct --output-on-failure -R legacy-ifairy-direct

run cmake -B build-ifairy-legacy \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_FAIRY2I=OFF \
    -DGGML_LEGACY_IFAIRY_CPU=ON \
    -DGGML_LEGACY_IFAIRY_CPU_LUT=ON
run cmake --build build-ifairy-legacy --target test-legacy-ifairy test-legacy-ifairy-direct -j "${JOBS}"
run env GGML_IFAIRY_LUT=1 ctest --test-dir build-ifairy-legacy --output-on-failure -R legacy-ifairy
