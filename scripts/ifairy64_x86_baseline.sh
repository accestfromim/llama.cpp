#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-${ROOT}/models/fairy2i_32b_ifairy64.gguf}"
HF_MODEL="${HF_MODEL:-/root/complex-llm-edge/models/raw/fairy32b-half}"
CONVERT="${CONVERT:-0}"
REQUIRE_ROOT_OUTPUTS="${REQUIRE_ROOT_OUTPUTS:-1}"
PYTHON="${PYTHON:-python3}"
LLAMA_CURL="${LLAMA_CURL:-OFF}"

BUILD_REL="${BUILD_REL:-${ROOT}/build-rel}"
BUILD_LUT="${BUILD_LUT:-${ROOT}/build-rel-lut}"
BUILD="${BUILD:-1}"
BUILD_TARGETS="${BUILD_TARGETS:-llama-cli llama-bench test-ifairy}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

THREADS="${THREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
UBATCH_SIZE="${UBATCH_SIZE:-${BATCH_SIZE}}"
FLASH_ATTN="${FLASH_ATTN:-1}"
PROMPT="${PROMPT:-hello}"
CLI_TOKENS="${CLI_TOKENS:-16}"
REPS="${REPS:-1}"
OUT_DIR="${OUT_DIR:-${ROOT}/tmp/ifairy64_x86_baseline.$(date +%Y%m%dT%H%M%S)}"

require_under_root() {
    local label="$1"
    local path="$2"
    local abs_path

    abs_path="$(realpath -m "${path}")"
    case "${abs_path}" in
        "${ROOT}" | "${ROOT}"/*)
            ;;
        *)
            echo "${label} must be under ROOT=${ROOT}: ${abs_path}" >&2
            echo "set REQUIRE_ROOT_OUTPUTS=0 to allow external output paths" >&2
            exit 1
            ;;
    esac
}

check_output_paths() {
    if [[ "${REQUIRE_ROOT_OUTPUTS}" != "1" ]]; then
        return
    fi

    require_under_root "MODEL" "${MODEL}"
    require_under_root "BUILD_REL" "${BUILD_REL}"
    require_under_root "BUILD_LUT" "${BUILD_LUT}"
    require_under_root "OUT_DIR" "${OUT_DIR}"
}

run_logged() {
    local name="$1"
    shift

    local log="${OUT_DIR}/${name}.log"
    {
        printf '## %s\n' "${name}"
        printf '$'
        printf ' %q' "$@"
        printf '\n\n'
        "$@"
    } 2>&1 | tee "${log}"
}

write_context() {
    {
        echo "root=${ROOT}"
        echo "model=${MODEL}"
        echo "hf_model=${HF_MODEL}"
        echo "python=${PYTHON}"
        "${PYTHON}" --version 2>/dev/null || true
        echo "build_rel=${BUILD_REL}"
        echo "build_lut=${BUILD_LUT}"
        echo "threads=${THREADS}"
        echo "batch_size=${BATCH_SIZE}"
        echo "ubatch_size=${UBATCH_SIZE}"
        echo "flash_attn=${FLASH_ATTN}"
        echo "repetitions=${REPS}"
        echo "out_dir=${OUT_DIR}"
        echo
        git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || true
        git -C "${ROOT}" status --short --branch 2>/dev/null || true
        echo
        uname -a || true
        echo
        lscpu 2>/dev/null || true
    } > "${OUT_DIR}/context.txt"
}

maybe_convert_model() {
    if [[ -f "${MODEL}" ]]; then
        return
    fi

    if [[ "${CONVERT}" != "1" ]]; then
        echo "missing MODEL=${MODEL}" >&2
        echo "set MODEL=/path/to/ifairy64.gguf, or run with CONVERT=1 HF_MODEL=/path/to/hf-model" >&2
        exit 1
    fi

    if [[ ! -d "${HF_MODEL}" ]]; then
        echo "missing HF_MODEL=${HF_MODEL}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${MODEL}")"
    run_logged convert_ifairy64 \
        "${PYTHON}" "${ROOT}/gguf-py/convert_fairy2i_qwen2.py" \
        "${HF_MODEL}" "${MODEL}" \
        --quant-variant tile64_v2 \
        --output-layer ifairy \
        --verbose
}

build_targets() {
    if [[ "${BUILD}" != "1" ]]; then
        return
    fi

    read -r -a targets <<< "${BUILD_TARGETS}"

    run_logged configure_rel \
        cmake -B "${BUILD_REL}" -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_LUT_CPU=OFF -DLLAMA_CURL="${LLAMA_CURL}"
    run_logged build_rel \
        cmake --build "${BUILD_REL}" -j "${JOBS}" --target "${targets[@]}"

    run_logged configure_lut \
        cmake -B "${BUILD_LUT}" -DCMAKE_BUILD_TYPE=Release -DGGML_IFAIRY_LUT_CPU=ON -DLLAMA_CURL="${LLAMA_CURL}"
    run_logged build_lut \
        cmake --build "${BUILD_LUT}" -j "${JOBS}" --target "${targets[@]}"
}

check_binaries() {
    local bin
    for bin in \
        "${BUILD_REL}/bin/test-ifairy" \
        "${BUILD_REL}/bin/llama-cli" \
        "${BUILD_REL}/bin/llama-bench" \
        "${BUILD_LUT}/bin/test-ifairy" \
        "${BUILD_LUT}/bin/llama-cli" \
        "${BUILD_LUT}/bin/llama-bench"; do
        if [[ ! -x "${bin}" ]]; then
            echo "missing executable: ${bin}" >&2
            exit 1
        fi
    done
}

run_baseline() {
    local -a bench_args=(
        -m "${MODEL}"
        -ngl 0
        --device none
        --threads "${THREADS}"
        -b "${BATCH_SIZE}"
        -ub "${UBATCH_SIZE}"
        -fa "${FLASH_ATTN}"
        --no-warmup
        -p 0
        -n 32
        -r "${REPS}"
        -o md
    )

    run_logged test_ifairy_rel_vecdot \
        "${BUILD_REL}/bin/test-ifairy" --ifairy64-vecdot-only
    run_logged test_ifairy_rel_lut_only \
        "${BUILD_REL}/bin/test-ifairy" --ifairy-lut-only
    run_logged test_ifairy_lut_vecdot \
        "${BUILD_LUT}/bin/test-ifairy" --ifairy64-vecdot-only
    run_logged test_ifairy_lut_lut_only \
        "${BUILD_LUT}/bin/test-ifairy" --ifairy-lut-only

    run_logged cli_rel \
        "${BUILD_REL}/bin/llama-cli" \
        -m "${MODEL}" --gpu-layers 0 -t "${THREADS}" \
        -fa "${FLASH_ATTN}" -p "${PROMPT}" -n "${CLI_TOKENS}" -no-cnv

    run_logged cli_lut \
        env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 \
        "${BUILD_LUT}/bin/llama-cli" \
        -m "${MODEL}" --gpu-layers 0 -t "${THREADS}" \
        -fa "${FLASH_ATTN}" -p "${PROMPT}" -n "${CLI_TOKENS}" -no-cnv

    run_logged bench_rel \
        "${BUILD_REL}/bin/llama-bench" "${bench_args[@]}"

    run_logged bench_lut \
        env GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL=lut16 \
        "${BUILD_LUT}/bin/llama-bench" "${bench_args[@]}"
}

check_output_paths
mkdir -p "${OUT_DIR}"
write_context
maybe_convert_model
build_targets
check_binaries
run_baseline

echo
echo "logs: ${OUT_DIR}"
