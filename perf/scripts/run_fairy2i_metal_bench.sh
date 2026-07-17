#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: run_fairy2i_metal_bench.sh pp|tg MODEL TAG

Environment:
  BINARY           llama-bench binary (default: build-rel-metal/bin/llama-bench)
  RESULTS_DIR      result directory (default: /tmp/fairy2i-w1-metal-saturation/results)
  REPS             repetitions (default: 5)
  THREADS          host threads (default: 8)
  BATCH            logical batch size (default: 2048)
  UBATCH           physical batch size (default: 512)
  N_GPU_LAYERS     GPU layers (default: 99)
  FLASH_ATTN       Flash Attention 0/1 (default: 1)
  MMAP             mmap 0/1 (default: 1)
  XCTRACE_TEMPLATE optional xctrace template name/path; records PREFIX.trace
  COOLDOWN_SECONDS minimum interval after the preceding test (default/minimum: 60)
  COOLDOWN_STATE   shared cooldown state directory

The script refuses concurrent runs and records the end timestamp even when the
benchmark fails. All invocations sharing COOLDOWN_STATE therefore start at
least COOLDOWN_SECONDS after the preceding invocation finished.
EOF
}

if [[ $# -ne 3 || ( $1 != pp && $1 != tg ) ]]; then
    usage
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=$1
model=$2
tag=$3

binary=${BINARY:-"$repo_root/build-rel-metal/bin/llama-bench"}
results_dir=${RESULTS_DIR:-/tmp/fairy2i-w1-metal-saturation/results}
repetitions=${REPS:-5}
threads=${THREADS:-8}
batch=${BATCH:-2048}
ubatch=${UBATCH:-512}
n_gpu_layers=${N_GPU_LAYERS:-99}
flash_attn=${FLASH_ATTN:-1}
use_mmap=${MMAP:-1}
profile_template=${XCTRACE_TEMPLATE:-}
cooldown_seconds=${COOLDOWN_SECONDS:-60}
state_dir=${COOLDOWN_STATE:-/tmp/fairy2i-metal-bench-cooldown}

if (( cooldown_seconds < 60 )); then
    echo "COOLDOWN_SECONDS must be at least 60" >&2
    exit 2
fi
if [[ ! -x $binary ]]; then
    echo "llama-bench binary is not executable: $binary" >&2
    exit 2
fi
if [[ ! -f $model ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

mkdir -p "$results_dir" "$state_dir"
lock_dir="$state_dir/active.lock"
last_end_file="$state_dir/last-end-epoch"

if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "another guarded benchmark is active: $lock_dir" >&2
    exit 3
fi
printf '%s\n' "$$" > "$lock_dir/pid"

prefix=
finish() {
    status=$?
    trap - EXIT INT TERM
    end_epoch=$(date +%s)
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\n' "$end_epoch" > "$last_end_file"
    if [[ -n $prefix ]]; then
        {
            printf 'end_epoch=%s\n' "$end_epoch"
            printf 'end_utc=%s\n' "$end_iso"
            printf 'exit_status=%s\n' "$status"
        } >> "$prefix.meta"
    fi
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir"
    exit "$status"
}
trap finish EXIT INT TERM

if [[ -f $last_end_file ]]; then
    last_end=$(<"$last_end_file")
    now=$(date +%s)
    elapsed=$((now - last_end))
    if (( elapsed < cooldown_seconds )); then
        remaining=$((cooldown_seconds - elapsed))
        echo "cooldown: waiting ${remaining}s before the next benchmark" >&2
        sleep "$remaining"
    fi
fi

workload=(--n-prompt 512 --n-gen 0)
if [[ $mode == tg ]]; then
    workload=(--n-prompt 0 --n-gen 128)
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
prefix="$results_dir/$tag.$stamp"
command=(
    "$binary"
    --model "$model"
    "${workload[@]}"
    --repetitions "$repetitions"
    --threads "$threads"
    --batch-size "$batch"
    --ubatch-size "$ubatch"
    --n-gpu-layers "$n_gpu_layers"
    --flash-attn "$flash_attn"
    --mmap "$use_mmap"
    --output json
)

start_epoch=$(date +%s)
start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'tag=%s\n' "$tag"
    printf 'mode=%s\n' "$mode"
    printf 'model=%s\n' "$model"
    printf 'binary=%s\n' "$binary"
    printf 'repetitions=%s\n' "$repetitions"
    printf 'threads=%s\n' "$threads"
    printf 'batch=%s\n' "$batch"
    printf 'ubatch=%s\n' "$ubatch"
    printf 'n_gpu_layers=%s\n' "$n_gpu_layers"
    printf 'flash_attn=%s\n' "$flash_attn"
    printf 'mmap=%s\n' "$use_mmap"
    printf 'xctrace_template=%s\n' "$profile_template"
    printf 'cooldown_seconds=%s\n' "$cooldown_seconds"
    printf 'start_epoch=%s\n' "$start_epoch"
    printf 'start_utc=%s\n' "$start_iso"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
    env | LC_ALL=C sort | grep -E '^(GGML_|LLAMA_)' || true
} > "$prefix.meta"

echo "benchmark: mode=$mode repetitions=$repetitions tag=$tag" >&2
echo "artifacts: $prefix.{json,stderr.log,meta}" >&2
if [[ -n $profile_template ]]; then
    echo "profile: template=$profile_template trace=$prefix.trace" >&2
    xcrun xctrace record \
        --template "$profile_template" \
        --output "$prefix.trace" \
        --no-prompt \
        --target-stdout "$prefix.json" \
        --launch -- "${command[@]}" \
        > >(tee "$prefix.xctrace.log") \
        2> >(tee "$prefix.stderr.log" >&2)
else
    "${command[@]}" > >(tee "$prefix.json") 2> >(tee "$prefix.stderr.log" >&2)
fi
