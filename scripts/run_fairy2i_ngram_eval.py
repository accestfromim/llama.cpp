#!/usr/bin/env python3
"""Run reproducible Fairy2i N-Gram speculative-decoding benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_FILES = {
    "summarization": Path("spec_bench/summarization.jsonl"),
    "rag": Path("spec_bench/rag.jsonl"),
    "hagrid": Path("hagrid/hagrid_questions.jsonl"),
    "triviaqa": Path("triviaqa/trivia_questions.jsonl"),
}
MODES = (
    "none",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
)
BACKENDS = (
    "cpu",
    "blas",
    "cuda",
    "hip",
    "musa",
    "metal",
    "vulkan",
    "sycl",
    "cann",
    "opencl",
    "webgpu",
    "zdnn",
    "rpc",
)

DECODE_RE = re.compile(
    r"^decoded\s+(\d+)\s+tokens\s+in\s+([0-9.]+)\s+seconds,"
    r"\s+speed:\s+([0-9.]+)\s+t/s\s*$",
    re.MULTILINE,
)
ENCODE_RE = re.compile(
    r"^encoded\s+(\d+)\s+tokens\s+in\s+([0-9.]+)\s+seconds,"
    r"\s+speed:\s+([0-9.]+)\s+t/s\s*$",
    re.MULTILINE,
)
INTEGER_METRICS = {
    name: re.compile(rf"^{name}\s*=\s*(\d+)\s*$", re.MULTILINE)
    for name in ("n_draft", "n_predict", "n_drafted", "n_accept")
}


@dataclass(frozen=True)
class Sample:
    task: str
    row: int
    sample_id: str
    prompt: str

    @property
    def key(self) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.sample_id)
        return f"{self.row:04d}-{safe_id[:64]}"


@dataclass(frozen=True)
class Backend:
    name: str
    env: dict[str, str]
    runner_args: tuple[str, ...]


def csv_choice(value: str, valid: tuple[str, ...], label: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(valid))
    if not result:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}: {', '.join(unknown)}"
        )
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"{label} contains duplicates")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable Fairy2i N-Gram benchmarks and generate "
            "JSON/Markdown summaries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="ngram-simple",
        help="one decoding mode to run",
    )
    parser.add_argument(
        "--tasks",
        default=list(TASK_FILES),
        type=lambda value: csv_choice(
            value, tuple(TASK_FILES), "task"
        ),
        help="comma-separated tasks",
    )
    parser.add_argument(
        "--samples-per-task",
        "--limit-per-task",
        dest="samples_per_task",
        type=int,
        default=20,
        help="number of samples from each selected task; 0 runs all samples",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Fairy2i GGUF model file",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        required=True,
        help="backend-specific llama-speculative-simple executable",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="dataset root containing spec_bench/, hagrid/, and triviaqa/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: results/fairy2i-ngram/<backend>/<mode>",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--max-new-tokens", type=int, default=512)
    generation.add_argument("--ctx-size", type=int, default=4096)
    generation.add_argument("--batch-size", type=int, default=4096)
    generation.add_argument("--ubatch-size", type=int, default=512)
    generation.add_argument("--threads", type=int, default=8)
    generation.add_argument("--threads-batch", type=int, default=8)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument(
        "--raw-completion",
        action="store_true",
        help="disable the GGUF chat template (not recommended for quality)",
    )
    generation.add_argument("--no-warmup", action="store_true")

    speculative = parser.add_argument_group("speculative decoding")
    speculative.add_argument("--draft-max", type=int, default=16)
    speculative.add_argument("--draft-min", type=int, default=0)
    speculative.add_argument("--ngram-n", type=int, default=12)
    speculative.add_argument("--ngram-m", type=int, default=48)
    speculative.add_argument("--ngram-min-hits", type=int, default=1)
    speculative.add_argument("--ngram-mod-match", type=int, default=24)
    speculative.add_argument("--ngram-mod-min", type=int, default=0)
    speculative.add_argument("--ngram-mod-max", type=int, default=16)

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--device",
        help=(
            "exact accelerator device name from the runner's --list-devices; "
            "required for accelerator backend profiles"
        ),
    )
    execution.add_argument("--timeout", type=int, default=1800)
    execution.add_argument("--fail-fast", action="store_true")
    execution.add_argument("--dry-run", action="store_true")
    execution.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild summaries without launching the runner",
    )
    execution.add_argument(
        "--extra-runner-arg",
        action="append",
        default=[],
        help="append a runner argument; use = before values beginning with -",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_identity(path: Path, content_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def backend_config(name: str, device: str | None) -> Backend:
    if name in {"cpu", "blas"}:
        if device:
            raise ValueError(
                f"--device cannot be used with the {name} backend"
            )
        environment = (
            {
                "GGML_FAIRY2I_LUT": "1",
                "GGML_FAIRY2I_LUT_IMPL": "lut16",
            }
            if name == "cpu"
            else {}
        )
        return Backend(name, environment, ("-dev", "none", "-ngl", "0"))

    if not device:
        raise ValueError(
            f"--device is required for the {name} backend; copy the exact "
            "name from RUNNER --list-devices"
        )
    environment = {}
    if name == "opencl":
        environment = {
            "GGML_OPENCL_FAIRY2I": "1",
            "GGML_OPENCL_FAIRY2I_TILE64_MUL_MAT_IMPL": "auto",
        }
    runner_args = ["-dev", device, "-ngl", "999"]
    return Backend(name, environment, tuple(runner_args))


def validate(args: argparse.Namespace) -> None:
    positive = {
        "--max-new-tokens": args.max_new_tokens,
        "--ctx-size": args.ctx_size,
        "--batch-size": args.batch_size,
        "--ubatch-size": args.ubatch_size,
        "--threads": args.threads,
        "--threads-batch": args.threads_batch,
        "--timeout": args.timeout,
        "--draft-max": args.draft_max,
        "--ngram-n": args.ngram_n,
        "--ngram-m": args.ngram_m,
        "--ngram-min-hits": args.ngram_min_hits,
        "--ngram-mod-match": args.ngram_mod_match,
        "--ngram-mod-max": args.ngram_mod_max,
    }
    for option, value in positive.items():
        if value <= 0:
            raise ValueError(f"{option} must be positive")
    if args.samples_per_task < 0:
        raise ValueError("--samples-per-task must be non-negative")
    if not 0 <= args.draft_min <= args.draft_max:
        raise ValueError("--draft-min must be between 0 and --draft-max")
    if not 0 <= args.ngram_mod_min <= args.ngram_mod_max:
        raise ValueError(
            "--ngram-mod-min must be between 0 and --ngram-mod-max"
        )
    if args.ubatch_size > args.batch_size:
        raise ValueError("--ubatch-size cannot exceed --batch-size")
    if args.batch_size > args.ctx_size:
        raise ValueError("--batch-size cannot exceed --ctx-size")


def load_samples(
    data_root: Path, tasks: list[str], limit: int
) -> tuple[list[Sample], list[dict[str, Any]]]:
    samples: list[Sample] = []
    datasets: list[dict[str, Any]] = []
    for task in tasks:
        path = data_root / TASK_FILES[task]
        if not path.is_file():
            raise ValueError(f"dataset not found: {path}")
        count = 0
        with path.open(encoding="utf-8") as dataset:
            for row, line in enumerate(dataset):
                if limit and count >= limit:
                    break
                if not line.strip():
                    continue
                value = json.loads(line)
                turns = value.get("turns")
                if not isinstance(turns, list) or not turns:
                    raise ValueError(f"{path}:{row + 1}: missing turns[0]")
                sample_id = value.get(
                    "question_id", value.get("id", row)
                )
                samples.append(
                    Sample(task, row, str(sample_id), str(turns[0]))
                )
                count += 1
        datasets.append(
            {
                "task": task,
                "path": str(path),
                "selected": count,
                "identity": file_identity(path, content_hash=True),
            }
        )
    return samples, datasets


def build_command(
    args: argparse.Namespace,
    backend: Backend,
    mode: str,
    prompt_path: Path,
) -> list[str]:
    command = [
        str(args.runner),
        "-m",
        str(args.model),
        "--offline",
        "--log-colors",
        "off",
        "--spec-type",
        mode,
        "--draft-max",
        str(args.draft_max),
        "--draft-min",
        str(args.draft_min),
        "--sampling-seq",
        "k",
        "--top-k",
        "1",
        "--temp",
        "0",
        "--seed",
        str(args.seed),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
        "-n",
        str(args.max_new_tokens),
        "--no-conversation" if args.raw_completion else "-cnv",
        *backend.runner_args,
    ]
    if args.no_warmup:
        command.append("--no-warmup")
    if mode == "ngram-simple":
        prefix = "--spec-ngram-simple"
    elif mode == "ngram-map-k":
        prefix = "--spec-ngram-map-k"
    elif mode == "ngram-map-k4v":
        prefix = "--spec-ngram-map-k4v"
    else:
        prefix = ""
    if prefix:
        command.extend(
            [
                f"{prefix}-size-n",
                str(args.ngram_n),
                f"{prefix}-size-m",
                str(args.ngram_m),
                f"{prefix}-min-hits",
                str(args.ngram_min_hits),
            ]
        )
    elif mode == "ngram-mod":
        command.extend(
            [
                "--spec-ngram-mod-n-match",
                str(args.ngram_mod_match),
                "--spec-ngram-mod-n-min",
                str(args.ngram_mod_min),
                "--spec-ngram-mod-n-max",
                str(args.ngram_mod_max),
            ]
        )
    command.extend(args.extra_runner_arg)
    command.extend(["-f", str(prompt_path)])
    return command


def parse_metrics(stderr: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    decoded = list(DECODE_RE.finditer(stderr))
    encoded = list(ENCODE_RE.finditer(stderr))
    if not decoded:
        errors.append("missing decoded timing")
    else:
        match = decoded[-1]
        metrics.update(
            decoded_tokens=int(match[1]),
            decode_seconds=float(match[2]),
            decode_tps_reported=float(match[3]),
        )
    if encoded:
        match = encoded[-1]
        metrics.update(
            encoded_tokens=int(match[1]),
            encode_seconds=float(match[2]),
            encode_tps_reported=float(match[3]),
        )
    for name, regex in INTEGER_METRICS.items():
        matches = list(regex.finditer(stderr))
        if matches:
            metrics[name] = int(matches[-1][1])
        else:
            errors.append(f"missing {name}")
    required = {
        "decoded_tokens",
        "decode_seconds",
        "n_predict",
        "n_drafted",
        "n_accept",
    }
    if required <= metrics.keys():
        if metrics["decoded_tokens"] != metrics["n_predict"]:
            errors.append("decoded_tokens differs from n_predict")
        if metrics["n_accept"] > metrics["n_drafted"]:
            errors.append("n_accept exceeds n_drafted")
        if metrics["decode_seconds"] <= 0:
            errors.append("decode_seconds is not positive")
        metrics["acceptance_pct"] = (
            100.0 * metrics["n_accept"] / metrics["n_drafted"]
            if metrics["n_drafted"]
            else 0.0
        )
    return metrics, errors


def record_path(
    output_dir: Path, sample: Sample, mode: str
) -> Path:
    return output_dir / "records" / sample.task / mode / f"{sample.key}.json"


def valid_record(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("fingerprint") == fingerprint
        and value.get("status") == "ok"
    ):
        return value
    return None


def run_one(
    args: argparse.Namespace,
    backend: Backend,
    output_dir: Path,
    fingerprint: str,
    sample: Sample,
    mode: str,
) -> dict[str, Any]:
    log_dir = output_dir / "logs" / sample.task / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{sample.key}.stdout.txt"
    stderr_path = log_dir / f"{sample.key}.stderr.log"
    environment = os.environ.copy()
    environment.update(backend.env)
    environment.update(
        {"LC_ALL": "C", "NO_COLOR": "1", "LLAMA_LOG_COLORS": "off"}
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=".prompt-",
        suffix=".txt",
        delete=False,
    ) as prompt:
        prompt.write(sample.prompt)
        prompt_path = Path(prompt.name)

    command = build_command(args, backend, mode, prompt_path)
    start = time.monotonic()
    timed_out = False
    returncode: int | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    finally:
        wall_seconds = time.monotonic() - start
        prompt_path.unlink(missing_ok=True)

    stderr_text = stderr_path.read_text(
        encoding="utf-8", errors="replace"
    )
    metrics, errors = parse_metrics(stderr_text)
    if timed_out:
        errors.insert(0, f"timeout after {args.timeout}s")
    if returncode != 0:
        errors.insert(0, f"runner exit code {returncode}")
    stdout_payload = stdout_path.read_bytes()
    record: dict[str, Any] = {
        "fingerprint": fingerprint,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "backend": backend.name,
        "mode": mode,
        "task": sample.task,
        "row": sample.row,
        "sample_id": sample.sample_id,
        "prompt_sha256": sha256_bytes(sample.prompt.encode()),
        "stdout_sha256": sha256_bytes(stdout_payload),
        "stdout_path": str(stdout_path.relative_to(output_dir)),
        "stderr_path": str(stderr_path.relative_to(output_dir)),
        "wall_seconds": wall_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "finished_at": utc_now(),
        "command": [
            "<PROMPT_FILE>" if part == str(prompt_path) else part
            for part in command
        ],
        **metrics,
    }
    if metrics.get("n_predict") is not None and wall_seconds > 0:
        record["process_e2e_tps"] = (
            metrics["n_predict"] / wall_seconds
        )
    return record


def load_records(
    output_dir: Path, fingerprint: str
) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output_dir / "records").rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("fingerprint") == fingerprint:
            records.append(value)
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["status"] == "ok"]
    tokens = sum(record["n_predict"] for record in successful)
    seconds = sum(record["decode_seconds"] for record in successful)
    drafted = sum(record["n_drafted"] for record in successful)
    accepted = sum(record["n_accept"] for record in successful)
    sample_tps = [
        record["n_predict"] / record["decode_seconds"]
        for record in successful
    ]
    return {
        "records": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "tokens": tokens,
        "decode_seconds": seconds,
        "decode_tps": tokens / seconds if seconds else None,
        "sample_tps_median": (
            statistics.median(sample_tps) if sample_tps else None
        ),
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_pct": 100.0 * accepted / drafted if drafted else 0.0,
        "draft_hit_samples": sum(
            record["n_drafted"] > 0 for record in successful
        ),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_summary(
    output_dir: Path,
    fingerprint: str,
    config: dict[str, Any],
    expected: int,
) -> dict[str, Any]:
    records = load_records(output_dir, fingerprint)
    by_mode = {
        mode: aggregate(
            [record for record in records if record["mode"] == mode]
        )
        for mode in config["modes"]
    }
    summary = {
        "generated_at": utc_now(),
        "fingerprint": fingerprint,
        "expected": expected,
        "recorded": len(records),
        "successful": sum(
            record["status"] == "ok" for record in records
        ),
        "config": config,
        "by_mode": by_mode,
    }
    atomic_json(output_dir / "summary.json", summary)

    lines = [
        "# Fairy2i N-Gram benchmark",
        "",
        f"- Backend: `{config['backend']['name']}`",
        f"- Fingerprint: `{fingerprint}`",
        f"- Progress: {summary['successful']}/{expected} successful",
        "",
        "| Mode | Success | Tokens | Decode TPS | Drafted | Accepted | Acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in by_mode.items():
        lines.append(
            f"| {mode} | {values['successful']} | {values['tokens']} | "
            f"{fmt(values['decode_tps'])} | {values['drafted']} | "
            f"{values['accepted']} | {fmt(values['acceptance_pct'])}% |"
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def make_config(
    args: argparse.Namespace,
    backend: Backend,
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": 6,
        "backend": asdict(backend),
        "modes": args.modes,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "runner": file_identity(args.runner),
        "model": file_identity(args.model),
        "datasets": datasets,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "threads_batch": args.threads_batch,
            "seed": args.seed,
            "chat_template": not args.raw_completion,
            "warmup": not args.no_warmup,
        },
        "speculative": {
            "draft_max": args.draft_max,
            "draft_min": args.draft_min,
            "ngram_n": args.ngram_n,
            "ngram_m": args.ngram_m,
            "ngram_min_hits": args.ngram_min_hits,
            "ngram_mod_match": args.ngram_mod_match,
            "ngram_mod_min": args.ngram_mod_min,
            "ngram_mod_max": args.ngram_mod_max,
        },
        "extra_runner_args": args.extra_runner_arg,
    }


def ensure_manifest(
    output_dir: Path, config: dict[str, Any]
) -> str:
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    fingerprint = sha256_bytes(canonical)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{output_dir} contains a different configuration; "
                "choose another --output-dir"
            )
    else:
        atomic_json(
            manifest_path,
            {
                "created_at": utc_now(),
                "fingerprint": fingerprint,
                "config": config,
            },
        )
    return fingerprint


def main() -> int:
    args = parse_args()
    args.modes = [args.mode]
    try:
        validate(args)
        args.model = args.model.expanduser().resolve()
        args.runner = args.runner.expanduser().resolve()
        args.data_root = args.data_root.expanduser().resolve()
        if not args.model.is_file():
            raise ValueError(f"model not found: {args.model}")
        if not args.runner.is_file():
            raise ValueError(f"runner not found: {args.runner}")
        samples, datasets = load_samples(
            args.data_root, args.tasks, args.samples_per_task
        )
        backend = backend_config(args.backend, args.device)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else REPO_ROOT
            / "results"
            / "fairy2i-ngram"
            / args.backend
            / args.mode
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        config = make_config(args, backend, datasets)
        fingerprint = ensure_manifest(output_dir, config)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    expected = len(samples) * len(args.modes)
    print(
        f"backend={backend.name} samples={len(samples)} "
        f"modes={','.join(args.modes)} runs={expected}"
    )
    print(f"output={output_dir}")
    print(f"fingerprint={fingerprint}")

    if args.summary_only:
        write_summary(output_dir, fingerprint, config, expected)
        print(f"summary={output_dir / 'summary.md'}")
        return 0

    failures = 0
    position = 0
    try:
        for sample in samples:
            for mode in args.modes:
                position += 1
                destination = record_path(output_dir, sample, mode)
                label = (
                    f"{sample.task}[{sample.row}] id={sample.sample_id} "
                    f"mode={mode}"
                )
                if valid_record(destination, fingerprint):
                    print(f"[{position:03d}/{expected:03d}] SKIP {label}")
                    continue
                if args.dry_run:
                    with tempfile.NamedTemporaryFile() as prompt:
                        command = build_command(
                            args, backend, mode, Path(prompt.name)
                        )
                    printable = [
                        "<PROMPT_FILE>"
                        if part == prompt.name
                        else part
                        for part in command
                    ]
                    print(
                        f"[{position:03d}/{expected:03d}] DRY  {label}"
                    )
                    print("  " + " ".join(printable))
                    continue
                print(
                    f"[{position:03d}/{expected:03d}] RUN  {label}",
                    flush=True,
                )
                record = run_one(
                    args,
                    backend,
                    output_dir,
                    fingerprint,
                    sample,
                    mode,
                )
                atomic_json(destination, record)
                write_summary(output_dir, fingerprint, config, expected)
                if record["status"] == "ok":
                    print(
                        f"  OK {record['n_predict']} tok "
                        f"{record['decode_tps_reported']:.3f} t/s "
                        f"draft={record['n_drafted']} "
                        f"accept={record['n_accept']} "
                        f"({record['acceptance_pct']:.2f}%)"
                    )
                else:
                    failures += 1
                    print(
                        "  FAIL " + "; ".join(record["errors"]),
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        break
            if failures and args.fail_fast:
                break
    except KeyboardInterrupt:
        print(
            "\ninterrupted; rerun the same command to resume",
            file=sys.stderr,
        )
        write_summary(output_dir, fingerprint, config, expected)
        return 130

    summary = write_summary(output_dir, fingerprint, config, expected)
    print(
        f"summary={output_dir / 'summary.md'} "
        f"successful={summary['successful']}/{expected}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
