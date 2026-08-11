#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "usage: $0 cpu|metal pp|tg MODEL TAG" \
        "" \
        "Environment:" \
        "  BINARY               llama-bench binary (backend-specific build by default)" \
        "  RESULTS_DIR          artifact directory (default: /tmp/row4-bench/results)" \
        "  THREADS              host threads (default: 8)" \
        "  REPS                 repetitions (CPU: 3, Metal: 5)" \
        "  BATCH / UBATCH       batch sizes (defaults: 2048 / 512)" \
        "  N_PROMPT / N_GEN     pp/tg sizes (defaults: 512 / 128)" \
        "  PYTHON               Python with numpy + local gguf support (default: python3)" \
        "  ROW4_FORCE_PATH      optional CPU path: scalar, dotprod, or i8mm" \
        "  BASELINE_TPS         optional baseline mean tok/s" \
        "  MAX_REGRESSION_PCT   allowed regression (default: 3)" \
        "  REQUIRE_PATH_MARKERS require expected markers (default: 1)" \
        "  COOLDOWN_SECONDS     Metal cooldown (default/minimum: 15)" \
        "  COOLDOWN_STATE       shared lock/cooldown directory" >&2
}

if [[ $# -ne 4 || ( $1 != cpu && $1 != metal ) || ( $2 != pp && $2 != tg ) ]]; then
    usage
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
backend=$1
mode=$2
model=$3
tag=$4
run_note=${RUN_NOTE:-}

if [[ ! -f $model ]]; then
    printf 'model does not exist: %s\n' "$model" >&2
    exit 2
fi

if [[ $backend == cpu ]]; then
    default_binary="$repo_root/build-rel/bin/llama-bench"
    default_reps=3
    n_gpu_layers=0
    cooldown_default=0
else
    default_binary="$repo_root/build-rel-metal/bin/llama-bench"
    default_reps=5
    n_gpu_layers=99
    cooldown_default=15
fi

binary=${BINARY:-$default_binary}
results_dir=${RESULTS_DIR:-/tmp/row4-bench/results}
threads=${THREADS:-8}
repetitions=${REPS:-$default_reps}
batch=${BATCH:-2048}
ubatch=${UBATCH:-512}
n_prompt=${N_PROMPT:-512}
n_gen=${N_GEN:-128}
force_path=${ROW4_FORCE_PATH:-}
python_bin=${PYTHON:-python3}
baseline_tps=${BASELINE_TPS:-}
max_regression_pct=${MAX_REGRESSION_PCT:-3}
require_markers=${REQUIRE_PATH_MARKERS:-1}
cooldown_seconds=${COOLDOWN_SECONDS:-$cooldown_default}
state_dir=${COOLDOWN_STATE:-/tmp/row4-bench-state}

if [[ ! -x $binary ]]; then
    printf 'llama-bench binary is not executable: %s\n' "$binary" >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable was not found: %s\n' "$python_bin" >&2
    exit 2
fi
binary_dir=$(cd "$(dirname "$binary")" && pwd)
build_dir=$(cd "$binary_dir/.." && pwd)
cmake_cache="$build_dir/CMakeCache.txt"
if [[ $backend == metal && $cooldown_seconds -lt 15 ]]; then
    printf 'Metal COOLDOWN_SECONDS must be at least 15\n' >&2
    exit 2
fi
if [[ $backend == metal && -n $force_path ]]; then
    printf 'ROW4_FORCE_PATH is CPU-only\n' >&2
    exit 2
fi
if [[ -n $force_path && $force_path != scalar && $force_path != dotprod && $force_path != i8mm ]]; then
    printf 'invalid ROW4_FORCE_PATH: %s\n' "$force_path" >&2
    exit 2
fi

mkdir -p "$results_dir" "$state_dir"
lock_dir="$state_dir/active.lock"
last_end_file="$state_dir/last-end-epoch"

if ! mkdir "$lock_dir" 2>/dev/null; then
    printf 'another guarded benchmark is active: %s\n' "$lock_dir" >&2
    exit 3
fi
printf '%s\n' "$$" > "$lock_dir/pid"

prefix=
finish() {
    status=$?
    trap - EXIT INT TERM
    end_epoch=$(date +%s)
    end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\n' "$end_epoch" > "$last_end_file"
    if [[ -n $prefix ]]; then
        {
            printf 'end_epoch=%s\n' "$end_epoch"
            printf 'end_utc=%s\n' "$end_utc"
            printf 'exit_status=%s\n' "$status"
        } >> "$prefix.meta"
    fi
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir"
    exit "$status"
}
trap finish EXIT INT TERM

# The lock prevents this harness from racing itself. Also reject a benchmark
# started outside the harness; concurrent llama-bench measurements are invalid.
if pgrep -x llama-bench >/dev/null 2>&1; then
    printf 'an existing llama-bench process is running; refusing a concurrent measurement\n' >&2
    exit 3
fi

if [[ -f $last_end_file && $cooldown_seconds -gt 0 ]]; then
    last_end=$(<"$last_end_file")
    now=$(date +%s)
    elapsed=$((now - last_end))
    if (( elapsed < cooldown_seconds )); then
        remaining=$((cooldown_seconds - elapsed))
        printf 'cooldown: waiting %ss before the next benchmark\n' "$remaining" >&2
        sleep "$remaining"
    fi
fi

effective_n_prompt=$n_prompt
effective_n_gen=0
workload=(--n-prompt "$effective_n_prompt" --n-gen "$effective_n_gen")
if [[ $mode == tg ]]; then
    effective_n_prompt=0
    effective_n_gen=$n_gen
    workload=(--n-prompt "$effective_n_prompt" --n-gen "$effective_n_gen")
fi

command=(
    "$binary"
    --model "$model"
    "${workload[@]}"
    --threads "$threads"
    --repetitions "$repetitions"
    --batch-size "$batch"
    --ubatch-size "$ubatch"
    --cache-type-k bf16
    --cache-type-v bf16
    --n-gpu-layers "$n_gpu_layers"
    --mmap 1
    --verbose
    --output json
)
if [[ $backend == cpu ]]; then
    command+=(--device none)
else
    command+=(--flash-attn 1)
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
prefix="$results_dir/$tag.$backend.$mode.$stamp"
start_epoch=$(date +%s)
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    printf 'tag=%s\n' "$tag"
    printf 'note=%s\n' "$run_note"
    printf 'backend=%s\n' "$backend"
    printf 'mode=%s\n' "$mode"
    printf 'model=%s\n' "$model"
    printf 'binary=%s\n' "$binary"
    printf 'build_dir=%s\n' "$build_dir"
    printf 'python=%s\n' "$python_bin"
    printf 'threads=%s\n' "$threads"
    printf 'repetitions=%s\n' "$repetitions"
    printf 'batch=%s\n' "$batch"
    printf 'ubatch=%s\n' "$ubatch"
    printf 'cache_type_k=bf16\n'
    printf 'cache_type_v=bf16\n'
    printf 'n_prompt=%s\n' "$effective_n_prompt"
    printf 'n_gen=%s\n' "$effective_n_gen"
    printf 'n_gpu_layers=%s\n' "$n_gpu_layers"
    printf 'warmup=1\n'
    printf 'force_path=%s\n' "$force_path"
    printf 'baseline_tps=%s\n' "$baseline_tps"
    printf 'max_regression_pct=%s\n' "$max_regression_pct"
    printf 'start_epoch=%s\n' "$start_epoch"
    printf 'start_utc=%s\n' "$start_utc"
    printf 'git_commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=no) ]]; then
        printf 'git_dirty=1\n'
    else
        printf 'git_dirty=0\n'
    fi
    printf 'git_diff_sha256=%s\n' "$(git -C "$repo_root" diff --binary HEAD | shasum -a 256 | awk '{ print $1 }')"
    printf 'model_sha256=%s\n' "$(shasum -a 256 "$model" | awk '{ print $1 }')"
    printf 'system=%s\n' "$(uname -a)"
    printf 'cpu=%s\n' "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    printf 'clang=%s\n' "$(clang --version 2>/dev/null | sed -n '1p')"
    sw_vers 2>/dev/null | sed 's/^/sw_vers: /' || true
    for key in CMAKE_BUILD_TYPE GGML_NATIVE GGML_METAL; do
        value=
        if [[ -f $cmake_cache ]]; then
            value=$(sed -n "s/^${key}:[^=]*=//p" "$cmake_cache" | sed -n '1p')
        fi
        printf 'cmake.%s=%s\n' "$key" "$value"
    done
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
    env | LC_ALL=C sort | grep -E '^(GGML_|LLAMA_|ROW4_)' || true
} > "$prefix.meta"

printf 'benchmark: backend=%s mode=%s reps=%s tag=%s\n' "$backend" "$mode" "$repetitions" "$tag" >&2
printf 'artifacts: %s.{json,stderr.log,host_paths.log,paths.log,meta}\n' "$prefix" >&2

# Validate the complete reference-model tensor inventory and strict Row4
# metadata before timing. Runtime markers below prove each logical shape ran.
PYTHONPATH="$repo_root/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - "$model" <<'PY' | tee "$prefix.host_paths.log"
import re
import sys

from gguf import GGMLQuantizationType, GGUFReader

reader = GGUFReader(sys.argv[1])
tensors = {tensor.name: tensor for tensor in reader.tensors}
if len(tensors) != 436:
    raise SystemExit(f"expected 436 tensors, found {len(tensors)}")
if reader.alignment != 64:
    raise SystemExit(f"expected GGUF alignment 64, found {reader.alignment}")
if any(tensor.data_offset % 64 for tensor in reader.tensors):
    raise SystemExit("found tensor payload not aligned to 64 bytes")


def field_value(key: str):
    field = reader.fields.get(key)
    if field is None or len(field.data) != 1:
        raise SystemExit(f"missing or malformed GGUF metadata: {key}")
    part = field.parts[field.data[0]]
    if field.types[0].name == "STRING":
        return bytes(part).decode("utf-8")
    return part.reshape(-1)[0].item()


expected_metadata = {
    "general.architecture": "qwen3",
    "general.file_type": 43,
    "general.quantization_version": 2,
    "row4.schema_version": 1,
    "row4.weight_layout": "m16k128_split8_v1",
    "row4.codebook": "uv_axis_v1",
    "row4.numeric_profile": "bf16_a8_away_i32_bf16_v1",
    "row4.qkv_order": "q_k_v",
    "row4.ffn_order": "gate_up",
    "row4.lm_head_layout": "s8_m16k128_rowmajor_v1",
}
for key, expected in expected_metadata.items():
    actual = field_value(key)
    if actual != expected:
        raise SystemExit(f"invalid {key}: expected {expected!r}, found {actual!r}")

row4_specs = (
    ("qkv", re.compile(r"^blk\.(\d+)\.attn_qkv\.row4\.codes$"), (64, 4, 32, 384), 6144, 4096),
    ("o", re.compile(r"^blk\.(\d+)\.attn_output\.row4\.codes$"), (64, 4, 32, 256), 4096, 4096),
    ("gate_up", re.compile(r"^blk\.(\d+)\.ffn_gate_up\.row4\.codes$"), (64, 4, 32, 1536), 24576, 4096),
    ("down", re.compile(r"^blk\.(\d+)\.ffn_down\.row4\.codes$"), (64, 4, 96, 256), 4096, 12288),
)

for label, pattern, physical, logical_o, logical_k in row4_specs:
    matches = sorted(
        (tensor for name, tensor in tensors.items() if pattern.fullmatch(name)),
        key=lambda tensor: int(pattern.fullmatch(tensor.name).group(1)),
    )
    if len(matches) != 36:
        raise SystemExit(f"expected 36 {label} Row4 code tensors, found {len(matches)}")
    layer_ids = [int(pattern.fullmatch(tensor.name).group(1)) for tensor in matches]
    if layer_ids != list(range(36)):
        raise SystemExit(f"invalid {label} layer ids: {layer_ids}")
    invalid = [
        tensor.name
        for tensor in matches
        if tensor.tensor_type != GGMLQuantizationType.ROW4_CODES or tuple(map(int, tensor.shape)) != physical
    ]
    if invalid:
        raise SystemExit(f"invalid {label} Row4 physical tensor(s): {', '.join(invalid[:4])}")
    invalid_scales = []
    for tensor in matches:
        scale_name = tensor.name.removesuffix(".codes") + ".scales"
        scale = tensors.get(scale_name)
        if (
            scale is None
            or scale.tensor_type != GGMLQuantizationType.BF16
            or tuple(map(int, scale.shape)) != (logical_o,)
        ):
            invalid_scales.append(scale_name)
    if invalid_scales:
        raise SystemExit(f"invalid {label} Row4 scale tensor(s): {', '.join(invalid_scales[:4])}")
    print(
        f"row4_host: op=row4 tensor={label} count=36 O={logical_o} K={logical_k} "
        f"physical={'x'.join(map(str, physical))}"
    )

lm_head = tensors.get("output.w8.codes")
lm_scale = tensors.get("output.w8.scales")
lm_physical = (128, 16, 32, 9496)
if (
    lm_head is None
    or lm_head.tensor_type != GGMLQuantizationType.I8
    or tuple(map(int, lm_head.shape)) != lm_physical
):
    raise SystemExit("invalid or missing output.w8.codes canonical tensor")
if (
    lm_scale is None
    or lm_scale.tensor_type != GGMLQuantizationType.F32
    or tuple(map(int, lm_scale.shape)) != (151936,)
):
    raise SystemExit("invalid or missing output.w8.scales tensor")
print("row4_host: op=w8a8 tensor=lm_head count=1 O=151936 K=4096 physical=128x16x32x9496")
PY

env_args=(GGML_ROW4_CPU_DEBUG=1)
if [[ -n $force_path ]]; then
    env_args+=("GGML_ROW4_TEST_FORCE_PATH=$force_path")
fi
set +e
env "${env_args[@]}" "${command[@]}" > "$prefix.json" 2> "$prefix.stderr.log"
bench_status=$?
set -e
cat "$prefix.json"
cat "$prefix.stderr.log" >&2
if (( bench_status != 0 )); then
    exit "$bench_status"
fi

runtime_marker_regex='row4_cpu: op=(row4|w8a8)|ROW(4 Metal W1A8|8 Metal W8A8 lm_head) path:|kernel_row4_w1a8_gate_up_swiglu_qat_packed_bf16_prefill'
cp "$prefix.host_paths.log" "$prefix.paths.log"
grep -E "$runtime_marker_regex" "$prefix.stderr.log" >> "$prefix.paths.log" || true
if [[ $require_markers != 0 ]]; then
    required_host_markers=(
        'op=row4 tensor=qkv count=36 O=6144 K=4096'
        'op=row4 tensor=o count=36 O=4096 K=4096'
        'op=row4 tensor=gate_up count=36 O=24576 K=4096'
        'op=row4 tensor=down count=36 O=4096 K=12288'
        'op=w8a8 tensor=lm_head count=1 O=151936 K=4096'
    )
    for marker in "${required_host_markers[@]}"; do
        if ! grep -Fq "$marker" "$prefix.host_paths.log"; then
            printf 'missing required Row4 host marker: %s\n' "$marker" >&2
            exit 4
        fi
    done

    if [[ $backend == cpu ]]; then
        row4_cpu_path=dotprod
        row4_cpu_batch=1
        if [[ $mode == pp ]]; then
            row4_cpu_path=i8mm
            row4_cpu_batch='[0-9]+'
        fi
        w8_cpu_path=dotprod
        if [[ -n $force_path ]]; then
            row4_cpu_path=$force_path
            w8_cpu_path=$force_path
        fi
        row4_cpu_panel=0
        row4_cpu_aqpack=bf16_rne_a8_away_v1
        if [[ $mode == pp && $row4_cpu_path == i8mm ]]; then
            row4_cpu_panel=1
            row4_cpu_aqpack=bf16_rne_a8_away_pairk8_v1
        fi
        row4_cpu_prefix="row4_cpu: op=row4 path=$row4_cpu_path layout=m16k128_split8_v1 B=$row4_cpu_batch"
        w8_cpu_prefix="row4_cpu: op=w8a8 path=$w8_cpu_path layout=s8_m16k128_rowmajor_v1 B=1"
        required_runtime_markers=(
            "$row4_cpu_prefix O=6144 K=4096 nth=$threads aqpack=$row4_cpu_aqpack panel=$row4_cpu_panel prepack=0"
            "$row4_cpu_prefix O=4096 K=4096 nth=$threads aqpack=$row4_cpu_aqpack panel=$row4_cpu_panel prepack=0"
            "$row4_cpu_prefix O=24576 K=4096 nth=$threads aqpack=$row4_cpu_aqpack panel=$row4_cpu_panel prepack=0"
            "$row4_cpu_prefix O=4096 K=12288 nth=$threads aqpack=$row4_cpu_aqpack panel=$row4_cpu_panel prepack=0"
            "$w8_cpu_prefix O=151936 K=4096 nth=$threads aqpack=bf16_rne_a8_away_v1 panel=0 prepack=0"
        )
    else
        row4_dispatch=decode
        row4_act_rows=1
        if [[ $mode == pp ]]; then
            row4_dispatch=prefill
            row4_act_rows='[0-9]+'
        fi
        row4_metal_prefix="ROW4 Metal W1A8 path: $row4_dispatch layout=m16k128_split8_v1 act_rows=$row4_act_rows"
        w8_metal_prefix='ROW8 Metal W8A8 lm_head path: decode layout=s8_m16k128_rowmajor_v1 act_rows=1'
        gate_up_metal_marker="$row4_metal_prefix O=24576 K=4096 "
        if [[ $mode == pp ]]; then
            gate_up_metal_marker="($gate_up_metal_marker|kernel_row4_w1a8_gate_up_swiglu_qat_packed_bf16_prefill)"
        fi
        required_runtime_markers=(
            "$row4_metal_prefix O=6144 K=4096 "
            "$row4_metal_prefix O=4096 K=4096 "
            "$gate_up_metal_marker"
            "$row4_metal_prefix O=4096 K=12288 "
            "$w8_metal_prefix O=151936 K=4096 "
        )
    fi
    for marker in "${required_runtime_markers[@]}"; do
        if ! grep -Eq "$marker" "$prefix.stderr.log"; then
            printf 'missing required Row4 runtime marker: %s\n' "$marker" >&2
            exit 4
        fi
    done
fi

if [[ -n $baseline_tps ]]; then
    measured_tps=$("$python_bin" - "$prefix.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
rows = payload if isinstance(payload, list) else [payload]
values = [float(row["avg_ts"]) for row in rows if "avg_ts" in row]
if not values:
    raise SystemExit("llama-bench JSON contains no avg_ts")
print(sum(values) / len(values))
PY
)
    "$python_bin" - "$measured_tps" "$baseline_tps" "$max_regression_pct" <<'PY'
import sys

measured, baseline, allowed = map(float, sys.argv[1:])
floor = baseline * (1.0 - allowed / 100.0)
print(f"measured_tps={measured:.6f} baseline_tps={baseline:.6f} regression_floor={floor:.6f}")
if measured < floor:
    raise SystemExit(5)
PY
fi
