#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-${ROOT}/models/fairy2i_32b_ifairy64.gguf}"
BUILD_REL="${BUILD_REL:-${ROOT}/build-rel}"
BUILD_LUT="${BUILD_LUT:-${ROOT}/build-rel-lut}"
REQUIRE_ROOT_OUTPUTS="${REQUIRE_ROOT_OUTPUTS:-1}"

CASES="${CASES:-lut}"
LUT_IMPL="${LUT_IMPL:-lut16}"
LUT_DEBUG="${LUT_DEBUG:-0}"
THREADS="${THREADS:-4}"
REPS="${REPS:-1}"
GEN_TOKENS="${GEN_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
UBATCH_SIZE="${UBATCH_SIZE:-${BATCH_SIZE}}"
FLASH_ATTN="${FLASH_ATTN:-1}"
OUT_DIR="${OUT_DIR:-${ROOT}/tmp/ifairy64_x86_perf.$(date +%Y%m%dT%H%M%S)}"

PERF="${PERF:-perf}"
RUN_BENCH="${RUN_BENCH:-0}"
RUN_STAT="${RUN_STAT:-1}"
RUN_RECORD="${RUN_RECORD:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
ALLOW_PERF_FAILURE="${ALLOW_PERF_FAILURE:-1}"
PERF_FREQ="${PERF_FREQ:-999}"
PERF_CALL_GRAPH="${PERF_CALL_GRAPH:-dwarf}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,branches,branch-misses,cache-references,cache-misses}"
PERF_REPORT_SORT="${PERF_REPORT_SORT:-comm,dso,symbol}"

perf_requested() {
    [[ "${RUN_STAT}" == "1" || "${RUN_RECORD}" == "1" || "${RUN_REPORT}" == "1" ]]
}

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

check_inputs() {
    if [[ "${REQUIRE_ROOT_OUTPUTS}" == "1" ]]; then
        require_under_root "MODEL" "${MODEL}"
        require_under_root "BUILD_REL" "${BUILD_REL}"
        require_under_root "BUILD_LUT" "${BUILD_LUT}"
        require_under_root "OUT_DIR" "${OUT_DIR}"
    fi

    if [[ ! -f "${MODEL}" ]]; then
        echo "missing MODEL=${MODEL}" >&2
        exit 1
    fi

    if perf_requested && ! command -v "${PERF}" >/dev/null 2>&1; then
        echo "missing perf tool: ${PERF}" >&2
        exit 1
    fi
}

run_logged_impl() {
    local allow_fail="$1"
    shift

    local name="$1"
    shift

    local log="${OUT_DIR}/${name}.log"
    local status

    set +e
    {
        printf '## %s\n' "${name}"
        printf '$'
        printf ' %q' "$@"
        printf '\n\n'
        "$@"
    } 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "${status}" -ne 0 ]]; then
        if [[ "${allow_fail}" == "1" ]]; then
            echo "allowed failure (${status}): ${log}" >&2
            return 0
        fi
        return "${status}"
    fi
}

run_logged() {
    run_logged_impl 0 "$@"
}

run_logged_allow_fail() {
    run_logged_impl 1 "$@"
}

write_context() {
    {
        echo "root=${ROOT}"
        echo "model=${MODEL}"
        echo "build_rel=${BUILD_REL}"
        echo "build_lut=${BUILD_LUT}"
        echo "cases=${CASES}"
        echo "lut_impl=${LUT_IMPL}"
        echo "lut_debug=${LUT_DEBUG}"
        echo "threads=${THREADS}"
        echo "batch_size=${BATCH_SIZE}"
        echo "ubatch_size=${UBATCH_SIZE}"
        echo "flash_attn=${FLASH_ATTN}"
        echo "repetitions=${REPS}"
        echo "gen_tokens=${GEN_TOKENS}"
        echo "run_bench=${RUN_BENCH}"
        echo "run_stat=${RUN_STAT}"
        echo "run_record=${RUN_RECORD}"
        echo "run_report=${RUN_REPORT}"
        echo "allow_perf_failure=${ALLOW_PERF_FAILURE}"
        echo "perf_events=${PERF_EVENTS}"
        echo "perf_freq=${PERF_FREQ}"
        echo "perf_call_graph=${PERF_CALL_GRAPH}"
        echo "out_dir=${OUT_DIR}"
        echo
        git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || true
        git -C "${ROOT}" status --short --branch 2>/dev/null || true
        echo
        if command -v "${PERF}" >/dev/null 2>&1; then
            "${PERF}" --version || true
        else
            echo "perf tool not found: ${PERF}"
        fi
        echo
        uname -a || true
        echo
        lscpu 2>/dev/null || true
    } > "${OUT_DIR}/context.txt"
}

write_hotspot_notes() {
    cat > "${OUT_DIR}/hotspot_notes.md" <<EOF
# IFAIRY64 x86 Hotspot Notes

Current question:
- Is time mainly in preprocess, qgemm, activation quantize, transform/prepack, or dispatch/framework overhead?

Map hot symbols into these buckets:
- transform/prepack: ggml_ifairy_lut_transform_tensor, ggml_ifairy_lut_transform_*
- activation quantize: quantize_row_ifairy_q16_tensor, quantize_row_ifairy_q16_lut_c
- preprocess: ggml_ifairy64_lut_preprocess_ex_lut16, ggml_ifairy64_lut_preprocess_lut16
- qgemm: ggml_ifairy64_lut_qgemm_lut16, ggml_ifairy64_lut_qgemm_fused_lut16
- dispatch/framework: ggml_compute_forward_mul_mat, ggml_graph_compute, threadpool/scheduler/backend glue

Classification checkpoint:
- Baseline log path:
- Hotspot evidence:
- High-level bucket: frontend / bad speculation / backend / distributed overhead
- Backend subtype if applicable: memory-bound / compute-bound
- Kernel-level cause:
- Next missing evidence:
EOF
}

case_bin=""
case_env=()

prepare_case() {
    local name="$1"

    case_bin=""
    case_env=()

    case "${name}" in
        rel)
            case_bin="${BUILD_REL}/bin/llama-bench"
            ;;
        rel-merged | merged-rel)
            case_bin="${BUILD_REL}/bin/llama-bench"
            case_env=(LLAMA_FAIRY2I_MERGED_OUTPUT=1)
            ;;
        lut)
            case_bin="${BUILD_LUT}/bin/llama-bench"
            case_env=(GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL="${LUT_IMPL}")
            ;;
        lut-merged | merged-lut)
            case_bin="${BUILD_LUT}/bin/llama-bench"
            case_env=(LLAMA_FAIRY2I_MERGED_OUTPUT=1 GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_IMPL="${LUT_IMPL}")
            ;;
        *)
            echo "invalid case: ${name} (expected: rel, rel-merged, lut, or lut-merged)" >&2
            exit 1
            ;;
    esac

    if [[ "${LUT_DEBUG}" == "1" && "${name}" == *lut* ]]; then
        case_env+=(GGML_IFAIRY_LUT_DEBUG=1)
    fi

    if [[ ! -x "${case_bin}" ]]; then
        echo "missing executable: ${case_bin}" >&2
        exit 1
    fi
}

run_bench() {
    local name="$1"

    if [[ "${RUN_BENCH}" != "1" ]]; then
        return
    fi

    run_logged "${name}_bench" \
        env "${case_env[@]}" "${case_bin}" "${bench_args[@]}"
}

run_perf_stat() {
    local name="$1"
    local stat_file="${OUT_DIR}/${name}.perf-stat.txt"

    if [[ "${RUN_STAT}" != "1" ]]; then
        return
    fi

    local runner=run_logged
    if [[ "${ALLOW_PERF_FAILURE}" == "1" ]]; then
        runner=run_logged_allow_fail
    fi

    "${runner}" "${name}_perf_stat" \
        "${PERF}" stat \
        -e "${PERF_EVENTS}" \
        -o "${stat_file}" \
        -- \
        env "${case_env[@]}" "${case_bin}" "${bench_args[@]}"
}

run_perf_record() {
    local name="$1"
    local data_file="${OUT_DIR}/${name}.perf.data"

    if [[ "${RUN_RECORD}" != "1" ]]; then
        return
    fi

    local runner=run_logged
    if [[ "${ALLOW_PERF_FAILURE}" == "1" ]]; then
        runner=run_logged_allow_fail
    fi

    "${runner}" "${name}_perf_record" \
        "${PERF}" record \
        -F "${PERF_FREQ}" \
        --call-graph "${PERF_CALL_GRAPH}" \
        -o "${data_file}" \
        -- \
        env "${case_env[@]}" "${case_bin}" "${bench_args[@]}"
}

run_perf_report() {
    local name="$1"
    local data_file="${OUT_DIR}/${name}.perf.data"
    local report_file="${OUT_DIR}/${name}.perf-report.txt"

    if [[ "${RUN_REPORT}" != "1" || "${RUN_RECORD}" != "1" ]]; then
        return
    fi
    if [[ ! -f "${data_file}" ]]; then
        echo "missing perf data: ${data_file}" >&2
        exit 1
    fi

    {
        printf '$'
        printf ' %q' "${PERF}" report --stdio -i "${data_file}" --sort "${PERF_REPORT_SORT}"
        printf '\n\n'
        "${PERF}" report --stdio -i "${data_file}" --sort "${PERF_REPORT_SORT}"
    } > "${report_file}" 2>&1
    echo "perf report: ${report_file}"
}

write_debug_aggregates() {
    local -a logs=("${OUT_DIR}"/*.log)
    if [[ ! -e "${logs[0]}" ]]; then
        return
    fi

    awk '
        function val(key,    i, parts) {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ "^" key "=") {
                    split($i, parts, "=")
                    return parts[2]
                }
            }
            return ""
        }
        $1 == "ifairy_lut_kernel:" {
            lg = FILENAME
            sub(".*/", "", lg)
            sub("\\.log$", "", lg)
            type = val("type")
            key = lg "\t" type
            count[key] += 1
            scratch[key] += val("scratch")
            prep[key] += val("preprocess")
            qgemm[key] += val("qgemm")
            total[key] += val("total")
        }
        END {
            print "log\ttype\tcount\tscratch_ms\tpreprocess_ms\tqgemm_ms\ttotal_ms"
            for (key in count) {
                printf "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n", key, count[key], scratch[key], prep[key], qgemm[key], total[key]
            }
        }
    ' "${logs[@]}" > "${OUT_DIR}/kernel_aggregate.tsv"

    awk '
        function val(key,    i, parts) {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ "^" key "=") {
                    split($i, parts, "=")
                    return parts[2]
                }
            }
            return ""
        }
        $1 == "ifairy_lut_kernel:" {
            lg = FILENAME
            sub(".*/", "", lg)
            sub("\\.log$", "", lg)
            type = val("type")
            key = lg "\t" type "\t" val("m") "\t" val("k") "\t" val("n")
            count[key] += 1
            scratch[key] += val("scratch")
            prep[key] += val("preprocess")
            qgemm[key] += val("qgemm")
            total[key] += val("total")
        }
        END {
            print "log\ttype\tm\tk\tn\tcount\tscratch_ms\tpreprocess_ms\tqgemm_ms\ttotal_ms"
            for (key in count) {
                printf "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n", key, count[key], scratch[key], prep[key], qgemm[key], total[key]
            }
        }
    ' "${logs[@]}" > "${OUT_DIR}/kernel_shape_aggregate.tsv"

    awk '
        function val(key,    i, parts) {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ "^" key "=") {
                    split($i, parts, "=")
                    return parts[2]
                }
            }
            return ""
        }
        $1 == "ifairy_lut:" {
            lg = FILENAME
            sub(".*/", "", lg)
            sub("\\.log$", "", lg)
            path = val("path")
            key = lg "\t" path
            count[key] += 1
            quant[key] += val("quant")
            prep[key] += val("prep")
            gemm[key] += val("gemm")
        }
        END {
            print "log\tpath\tcount\tquant_ms\tprep_ms\tgemm_ms\ttotal_ms"
            for (key in count) {
                printf "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n", key, count[key], quant[key], prep[key], gemm[key], quant[key] + prep[key] + gemm[key]
            }
        }
    ' "${logs[@]}" > "${OUT_DIR}/path_aggregate.tsv"

    echo "debug aggregates:"
    echo "  ${OUT_DIR}/kernel_aggregate.tsv"
    echo "  ${OUT_DIR}/kernel_shape_aggregate.tsv"
    echo "  ${OUT_DIR}/path_aggregate.tsv"
}

bench_args=(
    -m "${MODEL}"
    -ngl 0
    --device none
    --threads "${THREADS}"
    -b "${BATCH_SIZE}"
    -ub "${UBATCH_SIZE}"
    -fa "${FLASH_ATTN}"
    --no-warmup
    -p 0
    -n "${GEN_TOKENS}"
    -r "${REPS}"
    -o md
)

check_inputs
mkdir -p "${OUT_DIR}"
write_context
write_hotspot_notes

for name in ${CASES}; do
    prepare_case "${name}"
    run_bench "${name}"
    run_perf_stat "${name}"
    run_perf_record "${name}"
    run_perf_report "${name}"
done

write_debug_aggregates

echo
echo "logs: ${OUT_DIR}"
