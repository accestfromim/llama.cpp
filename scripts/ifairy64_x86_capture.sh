#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${TMP_DIR:-${ROOT}/tmp}"
BUILD_LUT="${BUILD_LUT:-${ROOT}/build-rel-lut}"

mkdir -p "${TMP_DIR}"

CAPTURE_LOG="${CAPTURE_LOG:-$(mktemp "${TMP_DIR}/ifairy64_x86_capture.$(date +%Y%m%dT%H%M%S).XXXXXX.log")}"
OUT_DIR="${OUT_DIR:-$(mktemp -d "${TMP_DIR}/ifairy64_x86_perf.$(date +%Y%m%dT%H%M%S).XXXXXX")}"

CASES="${CASES:-lut-merged}"
RUN_BENCH="${RUN_BENCH:-1}"
RUN_STAT="${RUN_STAT:-0}"
RUN_RECORD="${RUN_RECORD:-0}"
RUN_REPORT="${RUN_REPORT:-0}"
GEN_TOKENS="${GEN_TOKENS:-8}"
REPS="${REPS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
UBATCH_SIZE="${UBATCH_SIZE:-${BATCH_SIZE}}"
FLASH_ATTN="${FLASH_ATTN:-1}"
LUT_DEBUG="${LUT_DEBUG:-1}"
BUILD_FIRST="${BUILD_FIRST:-1}"
BUILD_TARGETS="${BUILD_TARGETS:-llama-bench}"
if [[ -z "${BUILD_JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        BUILD_JOBS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        BUILD_JOBS="$(sysctl -n hw.ncpu)"
    else
        BUILD_JOBS="4"
    fi
fi

export OUT_DIR CASES RUN_BENCH RUN_STAT RUN_RECORD RUN_REPORT GEN_TOKENS REPS BATCH_SIZE UBATCH_SIZE FLASH_ATTN LUT_DEBUG
export BUILD_LUT

run_capture() {
    echo "capture_log=${CAPTURE_LOG}"
    echo "out_dir=${OUT_DIR}"
    echo "root=${ROOT}"
    echo "started_at=$(date -Is)"
    echo "cases=${CASES}"
    echo "run_bench=${RUN_BENCH}"
    echo "run_stat=${RUN_STAT}"
    echo "run_record=${RUN_RECORD}"
    echo "run_report=${RUN_REPORT}"
    echo "gen_tokens=${GEN_TOKENS}"
    echo "reps=${REPS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "ubatch_size=${UBATCH_SIZE}"
    echo "flash_attn=${FLASH_ATTN}"
    echo "lut_debug=${LUT_DEBUG}"
    echo "build_first=${BUILD_FIRST}"
    echo "build_lut=${BUILD_LUT}"
    echo "build_targets=${BUILD_TARGETS}"
    echo "build_jobs=${BUILD_JOBS}"

    if [[ "${BUILD_FIRST}" == "1" ]]; then
        read -r -a build_targets <<< "${BUILD_TARGETS}"
        printf '$'
        printf ' %q' cmake --build "${BUILD_LUT}" --target "${build_targets[@]}" -j "${BUILD_JOBS}"
        printf '\n\n'
        cmake --build "${BUILD_LUT}" --target "${build_targets[@]}" -j "${BUILD_JOBS}"
        echo
    fi

    printf '$'
    printf ' %q' bash "${ROOT}/scripts/ifairy64_x86_perf.sh"
    printf '\n\n'

    bash "${ROOT}/scripts/ifairy64_x86_perf.sh"

    echo
    for summary in context.txt kernel_aggregate.tsv kernel_shape_aggregate.tsv path_aggregate.tsv; do
        if [[ -f "${OUT_DIR}/${summary}" ]]; then
            echo "## ${summary}"
            cat "${OUT_DIR}/${summary}"
            echo
        fi
    done

    logs=("${OUT_DIR}"/*_bench.log)
    if [[ -e "${logs[0]}" ]]; then
        echo "## route-summary"
        grep -H -E "decode-shared|decode-fused|fairy2i|tg8|t/s" "${logs[@]}" | tail -80 || true
        echo
    fi

    echo
    echo "finished_at=$(date -Is)"
}

set +e
(
    set -euo pipefail
    run_capture
) > "${CAPTURE_LOG}" 2>&1
status=$?
set -e

{
    echo "capture_log=${CAPTURE_LOG}"
    echo "out_dir=${OUT_DIR}"
    echo "status=${status}"
} >&2

exit "${status}"
