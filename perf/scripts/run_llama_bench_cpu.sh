#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ( $1 != pp && $1 != tg ) ]]; then
    echo "usage: MODEL=/path/to/model.gguf $0 pp|tg" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=$1
model=${MODEL:?set MODEL to the GGUF under test}
binary=${BINARY:-"$repo_root/build-rel-fairy2i/bin/llama-bench"}
threads=${THREADS:-8}
repetitions=${REPS:-3}
results_dir=${RESULTS_DIR:-/tmp/fairy2i-bundle-validation/results}
tag=${TAG:-"$(basename "$model" .gguf).$mode"}

mkdir -p "$results_dir"
log="$results_dir/$tag.log"

workload=(--n-prompt 0 --n-gen 128)
if [[ $mode == pp ]]; then
    workload=(--n-prompt 512 --n-gen 0)
fi

command=(
    "$binary"
    --model "$model"
    "${workload[@]}"
    --threads "$threads"
    --repetitions "$repetitions"
    --n-gpu-layers 0
    --device none
    --verbose
)

echo "model=$model mode=$mode threads=$threads repetitions=$repetitions warmup=1 log=$log"
GGML_FAIRY2I_LUT=${GGML_FAIRY2I_LUT:-1} \
GGML_FAIRY2I_LUT_IMPL=${GGML_FAIRY2I_LUT_IMPL:-lut16} \
GGML_FAIRY2I_CPU_DEBUG=${GGML_FAIRY2I_CPU_DEBUG:-1} \
    "${command[@]}" 2>&1 | tee "$log"
