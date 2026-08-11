#!/usr/bin/env python3
"""Capture and verify Qwen3 Row4/W8A8 primitive goldens.

The checkpoint intentionally does not ship its training-time ``row4_qat``
package.  This tool therefore implements the frozen inference profile with
ordinary CPU Torch operations and reads only selected O16 rows from each
safetensors shard.  It is independent from the converter implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit(
        "row4_oracle.py requires PyTorch; install the repository conversion "
        "requirements before running it"
    ) from exc

if __package__:
    from .numeric import (
        K_TILE,
        M_TILE,
        decode_row4_codes,
        finish_linear,
        int32_accumulate,
        pack_row4_m16k128,
        pack_w8_m16k128,
        quantize_activation,
        quantize_row4_codes,
        quantize_w8_rows,
        round_half_away_from_zero,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from numeric import (  # type: ignore[no-redef]
        K_TILE,
        M_TILE,
        decode_row4_codes,
        finish_linear,
        int32_accumulate,
        pack_row4_m16k128,
        pack_w8_m16k128,
        quantize_activation,
        quantize_row4_codes,
        quantize_w8_rows,
        round_half_away_from_zero,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    Path(value) if (value := os.environ.get("ROW4_CHECKPOINT_DIR")) else None
)
DEFAULT_GGUF = Path(value) if (value := os.environ.get("ROW4_GGUF")) else None
NUMERIC_PROFILE = "bf16_a8_away_i32_bf16_v1"
ROW4_LAYOUT = "m16k128_split8_v1"
W8_LAYOUT = "s8_m16k128_rowmajor_v1"
REFERENCE_DIMENSIONS = (36, 4096, 12288, 32, 8, 128, 151936)
REFERENCE_TRANSFORMERS_VERSION = "5.2.0"
REFERENCE_CHECKPOINT_TENSOR_COUNT = 651
REFERENCE_CHECKPOINT_PARAMETERS = 8_192_136_192
REFERENCE_CHECKPOINT_BYTES = 16_384_272_384
REFERENCE_GGUF_TENSOR_COUNT = 436
REFERENCE_GGUF_PAYLOAD_BYTES = 2_739_236_352
REFERENCE_SHARDS = tuple(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5))
REFERENCE_SHARD_TENSOR_COUNTS = (113, 232, 229, 77)
REFERENCE_YARN_FACTOR = 4.0
REFERENCE_YARN_EFFECTIVE_MSCALE = 1.0 + 0.1 * math.log(REFERENCE_YARN_FACTOR)
REFERENCE_YARN_RAW_ATTN_FACTOR = 1.0 / REFERENCE_YARN_EFFECTIVE_MSCALE
REFERENCE_YARN_GGUF_PRE_FACTOR = 1.0


@dataclass(frozen=True)
class Component:
    label: str
    checkpoint_stem: str
    out_features: int
    fused_offset: int


@dataclass(frozen=True)
class Bundle:
    label: str
    gguf_base: str
    in_features: int
    components: tuple[Component, ...]

    @property
    def out_features(self) -> int:
        return sum(component.out_features for component in self.components)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config field {key} must be a positive integer, got {value!r}")
    return value


def _matches_exact_value(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return value is expected
    if isinstance(expected, int):
        return isinstance(value, int) and not isinstance(value, bool) and value == expected
    if isinstance(expected, float):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) == expected
        )
    return value == expected


def reference_config_issues(config: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    exact_values: tuple[tuple[str, Any], ...] = (
        ("model_type", "qwen3"),
        ("architectures", ["Qwen3ForCausalLM"]),
        ("num_hidden_layers", REFERENCE_DIMENSIONS[0]),
        ("hidden_size", REFERENCE_DIMENSIONS[1]),
        ("intermediate_size", REFERENCE_DIMENSIONS[2]),
        ("num_attention_heads", REFERENCE_DIMENSIONS[3]),
        ("num_key_value_heads", REFERENCE_DIMENSIONS[4]),
        ("head_dim", REFERENCE_DIMENSIONS[5]),
        ("vocab_size", REFERENCE_DIMENSIONS[6]),
        ("hidden_act", "silu"),
        ("tie_word_embeddings", False),
        ("attention_bias", False),
        ("attention_dropout", 0.0),
        ("dtype", "bfloat16"),
        ("max_position_embeddings", 131072),
        ("max_window_layers", 36),
        ("sliding_window", None),
        ("use_sliding_window", False),
        ("use_cache", False),
        ("rms_norm_eps", 1.0e-6),
        ("bos_token_id", 151643),
        ("eos_token_id", 151645),
        ("pad_token_id", 151645),
        ("transformers_version", REFERENCE_TRANSFORMERS_VERSION),
    )
    for key, expected in exact_values:
        actual = config.get(key)
        if not _matches_exact_value(actual, expected):
            issues.append(f"config {key} must be {expected!r}, got {actual!r}")

    expected_layer_types = ["full_attention"] * REFERENCE_DIMENSIONS[0]
    if config.get("layer_types") != expected_layer_types:
        issues.append("config layer_types must be ['full_attention'] * 36")

    rope = config.get("rope_parameters")
    if not isinstance(rope, Mapping):
        issues.append("config rope_parameters must be an object")
    else:
        expected_rope = {
            "rope_type": "yarn",
            "factor": REFERENCE_YARN_FACTOR,
            "original_max_position_embeddings": 32768,
            "rope_theta": 1_000_000,
            "attn_factor": REFERENCE_YARN_RAW_ATTN_FACTOR,
        }
        actual_rope_keys = set(rope)
        expected_rope_keys = set(expected_rope)
        if actual_rope_keys != expected_rope_keys:
            issues.append(
                "config rope_parameters key set differs: "
                f"missing={sorted(expected_rope_keys - actual_rope_keys)}, "
                f"unexpected={sorted(actual_rope_keys - expected_rope_keys)}"
            )
        for key, expected in expected_rope.items():
            actual = rope.get(key)
            if not _matches_exact_value(actual, expected):
                issues.append(
                    f"config rope_parameters.{key} must be {expected!r}, got {actual!r}"
                )
    expected_auto_map = {
        "AutoModel": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
    }
    if config.get("auto_map") != expected_auto_map:
        issues.append("config auto_map does not match the Row4 v1 modeling source")
    return issues


def require_reference_config(config: Mapping[str, Any]) -> None:
    issues = reference_config_issues(config)
    if issues:
        raise ValueError("checkpoint is not the fixed Row4 v1 profile: " + "; ".join(issues))


def expected_checkpoint_tensor_shapes() -> dict[str, tuple[int, ...]]:
    layers, hidden, intermediate, heads, kv_heads, head_dim, vocab = REFERENCE_DIMENSIONS
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    projection_shapes = {
        "self_attn.q_proj": (q_dim, hidden),
        "self_attn.k_proj": (kv_dim, hidden),
        "self_attn.v_proj": (kv_dim, hidden),
        "self_attn.o_proj": (hidden, q_dim),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.self_attn.q_norm.weight"] = (head_dim,)
        shapes[f"{prefix}.self_attn.k_norm.weight"] = (head_dim,)
        for suffix, shape in projection_shapes.items():
            weight = f"{prefix}.{suffix}.weight"
            shapes[weight] = shape
            shapes[f"{prefix}.{suffix}.weight_scale"] = (shape[0],)
    return shapes


def build_bundles(config: Mapping[str, Any], layer: int) -> tuple[Bundle, ...]:
    hidden = _positive_int(config, "hidden_size")
    intermediate = _positive_int(config, "intermediate_size")
    heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    if q_out != hidden:
        raise ValueError(f"this Row4 profile requires q_out==hidden, got {q_out}!={hidden}")

    prefix = f"model.layers.{layer}"
    return (
        Bundle(
            "qkv",
            f"blk.{layer}.attn_qkv",
            hidden,
            (
                Component("q", f"{prefix}.self_attn.q_proj", q_out, 0),
                Component("k", f"{prefix}.self_attn.k_proj", kv_out, q_out),
                Component("v", f"{prefix}.self_attn.v_proj", kv_out, q_out + kv_out),
            ),
        ),
        Bundle(
            "o",
            f"blk.{layer}.attn_output",
            q_out,
            (Component("o", f"{prefix}.self_attn.o_proj", hidden, 0),),
        ),
        Bundle(
            "gate_up",
            f"blk.{layer}.ffn_gate_up",
            hidden,
            (
                Component("gate", f"{prefix}.mlp.gate_proj", intermediate, 0),
                Component("up", f"{prefix}.mlp.up_proj", intermediate, intermediate),
            ),
        ),
        Bundle(
            "down",
            f"blk.{layer}.ffn_down",
            intermediate,
            (Component("down", f"{prefix}.mlp.down_proj", hidden, 0),),
        ),
    )


class CheckpointReader:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint.resolve()
        self.index = _json_object(self.checkpoint / "model.safetensors.index.json")
        weight_map = self.index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise ValueError("checkpoint index has an invalid weight_map")
        self.weight_map: dict[str, str] = dict(weight_map)
        self.shards = tuple(sorted(set(self.weight_map.values())))
        if len(self.shards) != 4:
            raise ValueError(f"Row4 checkpoint requires four shards, got {self.shards}")
        disk_shards = {
            path.name
            for path in self.checkpoint.glob("model*.safetensors")
            if path.is_file()
        }
        if disk_shards != set(self.shards):
            raise ValueError(
                "checkpoint shard set does not match the index: "
                f"missing={sorted(set(self.shards) - disk_shards)}, "
                f"unexpected={sorted(disk_shards - set(self.shards))}"
            )
        for shard in self.shards:
            if not (self.checkpoint / shard).is_file():
                raise FileNotFoundError(f"checkpoint shard is missing: {self.checkpoint / shard}")

        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "capture requires safetensors; install the repository conversion requirements"
            ) from exc
        self.safe_open = safe_open
        self.tensor_headers: dict[str, tuple[str, str, tuple[int, ...]]] = {}
        shard_counts: list[int] = []
        for shard in self.shards:
            count = 0
            with self.safe_open(
                str(self.checkpoint / shard),
                framework="pt",
                device="cpu",
            ) as file:
                for name in file.keys():
                    if name in self.tensor_headers:
                        raise ValueError(f"duplicate tensor across checkpoint shards: {name}")
                    tensor_slice = file.get_slice(name)
                    self.tensor_headers[name] = (
                        shard,
                        str(tensor_slice.get_dtype()),
                        tuple(int(dim) for dim in tensor_slice.get_shape()),
                    )
                    count += 1
            shard_counts.append(count)
        self.shard_tensor_counts = tuple(shard_counts)

        indexed = set(self.weight_map)
        discovered = set(self.tensor_headers)
        wrong_shards = sorted(
            name
            for name in indexed & discovered
            if self.weight_map[name] != self.tensor_headers[name][0]
        )
        if indexed != discovered or wrong_shards:
            raise ValueError(
                "checkpoint index/header mismatch: "
                f"missing={sorted(indexed - discovered)[:8]}, "
                f"unexpected={sorted(discovered - indexed)[:8]}, "
                f"wrong_shard={wrong_shards[:8]}"
            )

        metadata = self.index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint index metadata must be an object")
        self.metadata: dict[str, Any] = dict(metadata)

    def _shard(self, tensor_name: str) -> Path:
        try:
            return self.checkpoint / self.weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(f"checkpoint tensor is missing: {tensor_name}") from exc

    def read_rows(self, tensor_name: str, start: int, end: int) -> torch.Tensor:
        if not 0 <= start < end:
            raise ValueError(f"invalid row slice {start}:{end} for {tensor_name}")
        with self.safe_open(str(self._shard(tensor_name)), framework="pt", device="cpu") as file:
            tensor_slice = file.get_slice(tensor_name)
            if str(tensor_slice.get_dtype()) != "BF16":
                raise TypeError(f"{tensor_name} must be checkpoint BF16")
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if len(shape) != 2 or end > shape[0]:
                raise ValueError(f"invalid row slice {start}:{end} for {tensor_name}={shape}")
            tensor = tensor_slice[start:end, :]
        return tensor.contiguous()

    def read_vector(self, tensor_name: str, start: int, end: int) -> torch.Tensor:
        with self.safe_open(str(self._shard(tensor_name)), framework="pt", device="cpu") as file:
            tensor_slice = file.get_slice(tensor_name)
            if str(tensor_slice.get_dtype()) != "BF16":
                raise TypeError(f"{tensor_name} must be checkpoint BF16")
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if len(shape) != 1 or not 0 <= start < end <= shape[0]:
                raise ValueError(f"invalid vector slice {start}:{end} for {tensor_name}={shape}")
            tensor = tensor_slice[start:end]
        return tensor.contiguous()


def reference_checkpoint_issues(checkpoint: CheckpointReader) -> list[str]:
    issues: list[str] = []
    if checkpoint.shards != REFERENCE_SHARDS:
        issues.append(f"checkpoint shards must be {REFERENCE_SHARDS}, got {checkpoint.shards}")
    if checkpoint.shard_tensor_counts != REFERENCE_SHARD_TENSOR_COUNTS:
        issues.append(
            "checkpoint shard tensor counts must be "
            f"{REFERENCE_SHARD_TENSOR_COUNTS}, got {checkpoint.shard_tensor_counts}"
        )
    if len(checkpoint.weight_map) != REFERENCE_CHECKPOINT_TENSOR_COUNT:
        issues.append(
            f"checkpoint index must map {REFERENCE_CHECKPOINT_TENSOR_COUNT} tensors, "
            f"got {len(checkpoint.weight_map)}"
        )

    expected = expected_checkpoint_tensor_shapes()
    actual_names = set(checkpoint.tensor_headers)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        issues.append(
            f"checkpoint tensor inventory differs: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    common = expected_names & actual_names
    non_bf16 = sorted(
        name for name in common if checkpoint.tensor_headers[name][1] != "BF16"
    )
    if non_bf16:
        issues.append(f"checkpoint tensors must all be BF16: {non_bf16[:8]}")
    bad_shapes = sorted(
        name for name in common if checkpoint.tensor_headers[name][2] != expected[name]
    )
    if bad_shapes:
        details = [
            f"{name}: expected {expected[name]}, got {checkpoint.tensor_headers[name][2]}"
            for name in bad_shapes[:8]
        ]
        issues.append("checkpoint tensor shapes differ: " + "; ".join(details))

    if not missing and not unexpected:
        parameters = sum(math.prod(checkpoint.tensor_headers[name][2]) for name in expected)
        if parameters != REFERENCE_CHECKPOINT_PARAMETERS:
            issues.append(
                f"checkpoint parameter count must be {REFERENCE_CHECKPOINT_PARAMETERS}, "
                f"got {parameters}"
            )
    expected_metadata = {
        "total_parameters": REFERENCE_CHECKPOINT_PARAMETERS,
        "total_size": REFERENCE_CHECKPOINT_BYTES,
    }
    for key, expected_value in expected_metadata.items():
        actual = checkpoint.metadata.get(key)
        if actual != expected_value:
            issues.append(
                f"checkpoint index metadata {key} must be {expected_value}, got {actual!r}"
            )
    return issues


def require_reference_checkpoint(checkpoint: CheckpointReader) -> None:
    issues = reference_checkpoint_issues(checkpoint)
    if issues:
        raise ValueError("checkpoint schema is not Row4 v1: " + "; ".join(issues))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
    }
    if hash_contents:
        record["sha256"] = sha256_file(path)
    return record


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("Row4 oracle raw artifacts currently require a little-endian host")
    contiguous = tensor.detach().cpu().contiguous()
    return bytes(contiguous.view(torch.uint8).reshape(-1).tolist())


def dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int8: "i8",
        torch.uint8: "u8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported oracle artifact dtype: {dtype}") from exc


class ArtifactWriter:
    def __init__(self, root: Path):
        self.root = root
        self.records: dict[str, dict[str, Any]] = {}

    def tensor(self, relative_stem: str, tensor: torch.Tensor) -> str:
        dtype = dtype_name(tensor.dtype)
        relative = f"{relative_stem}.{dtype}.bin"
        if relative in self.records:
            raise ValueError(f"duplicate artifact path: {relative}")
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tensor_bytes(tensor)
        path.write_bytes(payload)
        self.records[relative] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "byte_order": "little",
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        return relative


class ActivationSource:
    def __init__(self, path: Path | None, input_ids: Sequence[int]):
        self.input_ids = tuple(input_ids)
        self.path = path.resolve() if path is not None else None
        self.payload: Mapping[str, Any] | None = None
        if self.path is not None:
            loaded = torch.load(self.path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, Mapping):
                raise TypeError("--activations must contain a dict of F32 tensors")
            self.payload = loaded

    def get(self, key: str, in_features: int) -> tuple[torch.Tensor, str]:
        if self.payload is not None:
            if key not in self.payload:
                raise KeyError(f"activation artifact is missing key {key!r}")
            tensor = self.payload[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"activation {key!r} is not a tensor")
            tensor = tensor.detach().cpu().contiguous()
            if tensor.dtype != torch.float32 or tensor.ndim != 2:
                raise TypeError(f"activation {key!r} must be a [T,K] F32 tensor")
            if tensor.shape != (len(self.input_ids), in_features):
                raise ValueError(
                    f"activation {key!r} has shape {tuple(tensor.shape)}, expected "
                    f"({len(self.input_ids)}, {in_features})"
                )
            return tensor, "external_f32_carrier"

        seed_material = key.encode("utf-8") + b"\0" + json.dumps(self.input_ids).encode("ascii")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % 65521
        count = len(self.input_ids) * in_features
        index = torch.arange(count, dtype=torch.int64)
        integers = torch.remainder(index * 48271 + seed, 65521) - 32760
        tensor = (integers.to(torch.float32) / 4096.0).reshape(len(self.input_ids), in_features)
        # Preserve explicit signed zero cases in every probe without relying on
        # a random-number-generator implementation.
        if in_features >= 2:
            tensor[:, 0] = 0.0
            tensor[:, 1] = torch.tensor(-0.0, dtype=torch.float32)
        return tensor.contiguous(), "deterministic_integer_probe_v1"


def expected_gguf_tensor_inventory() -> dict[str, tuple[int, tuple[int, ...], int]]:
    """Return name -> (GGML type id, GGUF ``ne`` shape, payload bytes)."""

    layers, hidden, intermediate, _heads, kv_heads, head_dim, vocab = REFERENCE_DIMENSIONS
    kv_dim = kv_heads * head_dim
    inventory: dict[str, tuple[int, tuple[int, ...], int]] = {
        "token_embd.weight": (30, (hidden, vocab), vocab * hidden * 2),
        "output_norm.weight": (30, (hidden,), hidden * 2),
        "output.w8.codes": (
            24,
            (K_TILE, M_TILE, hidden // K_TILE, vocab // M_TILE),
            vocab * hidden,
        ),
        "output.w8.scales": (0, (vocab,), vocab * 4),
    }

    def add_row4(base: str, out_features: int, in_features: int) -> None:
        inventory[f"{base}.row4.codes"] = (
            47,
            (64, 4, in_features // K_TILE, out_features // M_TILE),
            out_features * in_features // 8,
        )
        inventory[f"{base}.row4.scales"] = (
            30,
            (out_features,),
            out_features * 2,
        )

    for layer in range(layers):
        inventory[f"blk.{layer}.attn_norm.weight"] = (30, (hidden,), hidden * 2)
        inventory[f"blk.{layer}.attn_q_norm.weight"] = (30, (head_dim,), head_dim * 2)
        inventory[f"blk.{layer}.attn_k_norm.weight"] = (30, (head_dim,), head_dim * 2)
        add_row4(f"blk.{layer}.attn_qkv", hidden + 2 * kv_dim, hidden)
        add_row4(f"blk.{layer}.attn_output", hidden, hidden)
        inventory[f"blk.{layer}.ffn_norm.weight"] = (30, (hidden,), hidden * 2)
        add_row4(f"blk.{layer}.ffn_gate_up", 2 * intermediate, hidden)
        add_row4(f"blk.{layer}.ffn_down", hidden, intermediate)
    return inventory


def validate_gguf_tensor_inventory(tensors: Mapping[str, Any], tensor_count: int) -> None:
    expected = expected_gguf_tensor_inventory()
    if tensor_count != len(tensors):
        raise ValueError("GGUF contains duplicate tensor names")
    if len(tensors) != REFERENCE_GGUF_TENSOR_COUNT or len(expected) != REFERENCE_GGUF_TENSOR_COUNT:
        raise ValueError(
            f"GGUF tensor count must be {REFERENCE_GGUF_TENSOR_COUNT}, got {len(tensors)}"
        )
    actual_names = set(tensors)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"GGUF tensor inventory differs: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    total_payload = 0
    for name, (expected_type, expected_shape, expected_bytes) in expected.items():
        tensor = tensors[name]
        actual_type = int(tensor.tensor_type)
        actual_shape = tuple(int(value) for value in tensor.shape.tolist())
        actual_bytes = int(tensor.n_bytes)
        if actual_type != expected_type:
            raise TypeError(
                f"GGUF tensor {name} has type {actual_type}, expected {expected_type}"
            )
        if actual_shape != expected_shape:
            raise ValueError(
                f"GGUF tensor {name} has ne={actual_shape}, expected {expected_shape}"
            )
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"GGUF tensor {name} has {actual_bytes} payload bytes, expected {expected_bytes}"
            )
        total_payload += actual_bytes
    if total_payload != REFERENCE_GGUF_PAYLOAD_BYTES:
        raise ValueError(
            f"GGUF tensor payload must be {REFERENCE_GGUF_PAYLOAD_BYTES}, got {total_payload}"
        )


class GGUFVerifier:
    EXPECTED_METADATA = {
        "general.architecture": "qwen3",
        "general.file_type": 43,
        "general.alignment": 64,
        "general.quantization_version": 2,
        "qwen3.context_length": 131072,
        "qwen3.embedding_length": 4096,
        "qwen3.block_count": 36,
        "qwen3.feed_forward_length": 12288,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.attention.value_length": 128,
        "qwen3.rope.dimension_count": 128,
        "qwen3.rope.freq_base": 1_000_000.0,
        "qwen3.rope.scaling.type": "yarn",
        "qwen3.rope.scaling.factor": REFERENCE_YARN_FACTOR,
        "qwen3.rope.scaling.original_context_length": 32768,
        # llama.cpp applies YaRN's default mscale after multiplying this
        # pre-factor.  The checkpoint's inverse ``attn_factor`` key is ignored
        # by Transformers 5.2.0 and must not be copied here.
        "qwen3.rope.scaling.attn_factor": REFERENCE_YARN_GGUF_PRE_FACTOR,
        "qwen3.vocab_size": 151936,
        "tokenizer.ggml.add_bos_token": True,
        "row4.schema_version": 1,
        "row4.weight_layout": ROW4_LAYOUT,
        "row4.codebook": "uv_axis_v1",
        "row4.numeric_profile": NUMERIC_PROFILE,
        "row4.qkv_order": "q_k_v",
        "row4.ffn_order": "gate_up",
        "row4.lm_head_layout": W8_LAYOUT,
    }

    def __init__(self, gguf_path: Path):
        gguf_python = REPO_ROOT / "gguf-py"
        sys.path.insert(0, str(gguf_python))
        try:
            import numpy as np
            from gguf import GGMLQuantizationType, GGUFReader
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GGUF verification requires numpy and the in-tree gguf-py package"
            ) from exc
        self.np = np
        self.GGMLQuantizationType = GGMLQuantizationType
        self.path = gguf_path.resolve()
        self.reader = GGUFReader(self.path, "r")
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.checks: list[dict[str, Any]] = []
        for key, expected in self.EXPECTED_METADATA.items():
            field = self.reader.get_field(key)
            if field is None:
                raise ValueError(f"GGUF is missing required metadata {key!r}")
            actual = field.contents()
            if actual != expected:
                raise ValueError(f"GGUF metadata {key!r}: expected {expected!r}, got {actual!r}")
            self.checks.append({"kind": "metadata", "name": key, "value": actual})
        rms_field = self.reader.get_field("qwen3.attention.layer_norm_rms_epsilon")
        if rms_field is None:
            raise ValueError("GGUF is missing required metadata 'qwen3.attention.layer_norm_rms_epsilon'")
        rms_epsilon = float(rms_field.contents())
        if not math.isclose(rms_epsilon, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "GGUF qwen3.attention.layer_norm_rms_epsilon: "
                f"expected 1e-06, got {rms_epsilon!r}"
            )
        self.checks.append(
            {
                "kind": "metadata",
                "name": "qwen3.attention.layer_norm_rms_epsilon",
                "value": rms_epsilon,
            }
        )
        validate_gguf_tensor_inventory(self.tensors, len(self.reader.tensors))
        self.checks.append(
            {
                "kind": "strict_tensor_inventory",
                "tensor_count": REFERENCE_GGUF_TENSOR_COUNT,
                "payload_bytes": REFERENCE_GGUF_PAYLOAD_BYTES,
            }
        )

    def _tensor(self, name: str, expected_type: int):
        try:
            tensor = self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor is missing: {name}") from exc
        if int(tensor.tensor_type) != expected_type:
            raise TypeError(
                f"GGUF tensor {name} has type {tensor.tensor_type}, expected enum {expected_type}"
            )
        return tensor

    def check_row4_bundle_shape(self, base: str, out_features: int, in_features: int) -> None:
        codes_name = f"{base}.row4.codes"
        codes = self._tensor(codes_name, 47)
        expected_data_shape = (out_features // M_TILE, in_features // K_TILE, 4, 64)
        expected_ne = [64, 4, in_features // K_TILE, out_features // M_TILE]
        if tuple(codes.data.shape) != expected_data_shape or codes.shape.tolist() != expected_ne:
            raise ValueError(
                f"GGUF Row4 physical shape differs for {codes_name}: "
                f"data={tuple(codes.data.shape)}, ne={codes.shape.tolist()}, "
                f"expected data={expected_data_shape}, ne={expected_ne}"
            )
        scales_name = f"{base}.row4.scales"
        scales = self._tensor(scales_name, 30)
        if scales.shape.tolist() != [out_features] or scales.n_bytes != out_features * 2:
            raise ValueError(
                f"GGUF Row4 scale shape differs for {scales_name}: "
                f"ne={scales.shape.tolist()}, bytes={scales.n_bytes}"
            )
        self.checks.append(
            {
                "kind": "row4_bundle_shape",
                "name": base,
                "logical_shape": [out_features, in_features],
                "physical_shape": list(expected_data_shape),
            }
        )

    def check_w8_shape(self, out_features: int, in_features: int) -> None:
        codes = self._tensor("output.w8.codes", 24)
        expected_data_shape = (out_features // M_TILE, in_features // K_TILE, M_TILE, K_TILE)
        expected_ne = [K_TILE, M_TILE, in_features // K_TILE, out_features // M_TILE]
        if tuple(codes.data.shape) != expected_data_shape or codes.shape.tolist() != expected_ne:
            raise ValueError(
                "GGUF W8 physical shape differs: "
                f"data={tuple(codes.data.shape)}, ne={codes.shape.tolist()}, "
                f"expected data={expected_data_shape}, ne={expected_ne}"
            )
        scales = self._tensor("output.w8.scales", 0)
        if scales.shape.tolist() != [out_features] or scales.n_bytes != out_features * 4:
            raise ValueError(
                "GGUF W8 scale shape differs: "
                f"ne={scales.shape.tolist()}, bytes={scales.n_bytes}"
            )
        self.checks.append(
            {
                "kind": "w8_bundle_shape",
                "name": "output.w8",
                "logical_shape": [out_features, in_features],
                "physical_shape": list(expected_data_shape),
            }
        )

    def check_row4_tile(self, base: str, output_tile: int, expected: bytes) -> None:
        name = f"{base}.row4.codes"
        tensor = self._tensor(name, 47)
        if tensor.data.ndim != 4 or output_tile >= tensor.data.shape[0]:
            raise ValueError(f"unexpected physical Row4 tensor shape for {name}: {tensor.data.shape}")
        actual = tensor.data[output_tile].tobytes(order="C")
        if actual != expected:
            mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
            raise ValueError(
                f"GGUF Row4 bytes differ for {name} output tile {output_tile} at tile byte {mismatch}: "
                f"actual=0x{actual[mismatch]:02x}, expected=0x{expected[mismatch]:02x}"
            )
        self.checks.append(
            {"kind": "row4_codes", "name": name, "output_tile": output_tile, "nbytes": len(expected)}
        )

    def check_bf16_scales(self, base: str, row_start: int, expected: bytes) -> None:
        name = f"{base}.row4.scales"
        tensor = self._tensor(name, 30)
        payload = tensor.data.reshape(-1).tobytes(order="C")
        actual = payload[row_start * 2 : row_start * 2 + len(expected)]
        if actual != expected:
            raise ValueError(f"GGUF signed BF16 scale bits differ for {name} rows {row_start}:{row_start + 16}")
        self.checks.append(
            {"kind": "row4_scales", "name": name, "row_start": row_start, "nbytes": len(expected)}
        )

    def check_w8_tile(self, output_tile: int, expected: bytes) -> None:
        name = "output.w8.codes"
        tensor = self._tensor(name, 24)
        if tensor.data.ndim != 4 or output_tile >= tensor.data.shape[0]:
            raise ValueError(f"unexpected physical W8 tensor shape for {name}: {tensor.data.shape}")
        actual = tensor.data[output_tile].tobytes(order="C")
        if actual != expected:
            mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
            raise ValueError(
                f"GGUF W8 bytes differ for output tile {output_tile} at tile byte {mismatch}: "
                f"actual=0x{actual[mismatch]:02x}, expected=0x{expected[mismatch]:02x}"
            )
        self.checks.append(
            {"kind": "w8_codes", "name": name, "output_tile": output_tile, "nbytes": len(expected)}
        )

    def check_f32_scales(self, row_start: int, expected: bytes) -> None:
        name = "output.w8.scales"
        tensor = self._tensor(name, 0)
        payload = tensor.data.reshape(-1).tobytes(order="C")
        actual = payload[row_start * 4 : row_start * 4 + len(expected)]
        if actual != expected:
            raise ValueError(f"GGUF F32 scale bits differ for {name} rows {row_start}:{row_start + 16}")
        self.checks.append(
            {"kind": "w8_scales", "name": name, "row_start": row_start, "nbytes": len(expected)}
        )


def selected_output_tiles(bundle: Bundle, checkpoint: CheckpointReader) -> list[int]:
    tiles: set[int] = set()
    for component in bundle.components:
        if component.fused_offset % M_TILE or component.out_features % M_TILE:
            raise ValueError(f"unaligned component in {bundle.label}: {component}")
        tiles.add(component.fused_offset // M_TILE)
        tiles.add((component.fused_offset + component.out_features - M_TILE) // M_TILE)
        # Learned Row4 scales are signed.  Include the first and last O16 tile
        # containing a non-positive saved value so a real checkpoint capture
        # exercises that contract, rather than relying only on a synthetic
        # signed-scale test.
        scale_name = f"{component.checkpoint_stem}.weight_scale"
        component_scales = checkpoint.read_vector(scale_name, 0, component.out_features)
        nonpositive = torch.nonzero(component_scales <= 0, as_tuple=False).reshape(-1)
        if nonpositive.numel() != 0:
            for local_row in (int(nonpositive[0]), int(nonpositive[-1])):
                tiles.add((component.fused_offset + (local_row // M_TILE) * M_TILE) // M_TILE)
    return sorted(tiles)


def component_for_tile(bundle: Bundle, output_tile: int) -> tuple[Component, int]:
    fused_row = output_tile * M_TILE
    for component in bundle.components:
        if component.fused_offset <= fused_row < component.fused_offset + component.out_features:
            local_row = fused_row - component.fused_offset
            if local_row + M_TILE > component.out_features:
                raise ValueError("selected tile crosses a fused component boundary")
            return component, local_row
    raise ValueError(f"output tile {output_tile} is outside bundle {bundle.label}")


def write_activation(
    writer: ArtifactWriter,
    stem: str,
    carrier: torch.Tensor,
) -> tuple[Any, dict[str, str]]:
    activation = quantize_activation(carrier)
    paths = {
        "input_carrier": writer.tensor(f"{stem}/activation_input", carrier),
        "bf16": writer.tensor(f"{stem}/activation_bf16", activation.bf16),
        "rounded_carrier": writer.tensor(f"{stem}/activation_rounded_carrier", activation.carrier_f32),
        "codes": writer.tensor(f"{stem}/activation_codes", activation.codes_i8),
        "scales": writer.tensor(f"{stem}/activation_scales", activation.scales_f32),
    }
    return activation, paths


def capture_bundle(
    checkpoint: CheckpointReader,
    bundle: Bundle,
    layer: int,
    activation_source: ActivationSource,
    writer: ArtifactWriter,
    verifier: GGUFVerifier | None,
) -> dict[str, Any]:
    activation_key = f"layer{layer}.{bundle.label}"
    carrier, source_kind = activation_source.get(activation_key, bundle.in_features)
    base_stem = f"layer_{layer:02d}/{bundle.label}"
    activation, activation_paths = write_activation(writer, base_stem, carrier)
    tiles: list[dict[str, Any]] = []
    if verifier is not None:
        verifier.check_row4_bundle_shape(bundle.gguf_base, bundle.out_features, bundle.in_features)

    for output_tile in selected_output_tiles(bundle, checkpoint):
        component, local_row = component_for_tile(bundle, output_tile)
        weight_name = f"{component.checkpoint_stem}.weight"
        scale_name = f"{component.checkpoint_stem}.weight_scale"
        weight = checkpoint.read_rows(weight_name, local_row, local_row + M_TILE)
        scales = checkpoint.read_vector(scale_name, local_row, local_row + M_TILE)
        if weight.shape != (M_TILE, bundle.in_features):
            raise ValueError(f"unexpected checkpoint shape for selected {weight_name}: {tuple(weight.shape)}")
        if not bool(torch.isfinite(scales.to(torch.float32)).all()):
            raise ValueError(f"saved scale contains a non-finite value: {scale_name}")

        logical_codes = quantize_row4_codes(weight)
        decoded = decode_row4_codes(logical_codes)
        packed = pack_row4_m16k128(logical_codes)[0]
        accumulator = int32_accumulate(
            activation.codes_i8.reshape(-1, bundle.in_features),
            decoded,
        )
        result = finish_linear(accumulator, activation.scales_f32, scales.to(torch.float32))
        fused_row = output_tile * M_TILE
        tile_stem = f"{base_stem}/tile_{output_tile:06d}_{component.label}"
        paths = {
            "latent_weight": writer.tensor(f"{tile_stem}/latent_weight", weight),
            "saved_scale": writer.tensor(f"{tile_stem}/saved_scale", scales),
            "logical_codes": writer.tensor(f"{tile_stem}/logical_codes", logical_codes),
            "packed_codes": writer.tensor(f"{tile_stem}/packed_codes", packed),
            "decoded_weight": writer.tensor(f"{tile_stem}/decoded_weight", decoded),
            "accumulator": writer.tensor(f"{tile_stem}/accumulator", result.accumulator_i32),
            "scaled_before_bf16": writer.tensor(f"{tile_stem}/scaled_before_bf16", result.scaled_f32),
            "output_bf16": writer.tensor(f"{tile_stem}/output_bf16", result.output_bf16),
            "output_carrier": writer.tensor(f"{tile_stem}/output_carrier", result.carrier_f32),
        }
        if verifier is not None:
            verifier.check_row4_tile(bundle.gguf_base, output_tile, tensor_bytes(packed))
            verifier.check_bf16_scales(bundle.gguf_base, fused_row, tensor_bytes(scales))
        tiles.append(
            {
                "output_tile": output_tile,
                "fused_rows": [fused_row, fused_row + M_TILE],
                "component": component.label,
                "component_rows": [local_row, local_row + M_TILE],
                "checkpoint_weight": weight_name,
                "checkpoint_scale": scale_name,
                "saved_scale_nonpositive_count": int(torch.count_nonzero(scales <= 0)),
                "artifacts": paths,
                "gguf_verified": verifier is not None,
            }
        )

    return {
        "label": bundle.label,
        "gguf_base": bundle.gguf_base,
        "logical_shape": [bundle.out_features, bundle.in_features],
        "activation_key": activation_key,
        "activation_source": source_kind,
        "activation_artifacts": activation_paths,
        "tiles": tiles,
    }


def capture_lm_head(
    checkpoint: CheckpointReader,
    config: Mapping[str, Any],
    activation_source: ActivationSource,
    writer: ArtifactWriter,
    verifier: GGUFVerifier | None,
) -> dict[str, Any]:
    hidden = _positive_int(config, "hidden_size")
    vocab = _positive_int(config, "vocab_size")
    if hidden % K_TILE or vocab % M_TILE:
        raise ValueError(f"unaligned lm_head shape O={vocab}, K={hidden}")
    carrier, source_kind = activation_source.get("lm_head", hidden)
    activation, activation_paths = write_activation(writer, "lm_head", carrier)
    output_tiles = sorted({0, vocab // M_TILE - 1})
    tiles: list[dict[str, Any]] = []
    if verifier is not None:
        verifier.check_w8_shape(vocab, hidden)

    for output_tile in output_tiles:
        row_start = output_tile * M_TILE
        weight = checkpoint.read_rows("lm_head.weight", row_start, row_start + M_TILE)
        weight_codes, row_scales = quantize_w8_rows(weight)
        packed = pack_w8_m16k128(weight_codes)[0]
        accumulator = int32_accumulate(
            activation.codes_i8.reshape(-1, hidden),
            weight_codes,
        )
        result = finish_linear(accumulator, activation.scales_f32, row_scales)
        tile_stem = f"lm_head/tile_{output_tile:06d}"
        paths = {
            "latent_weight": writer.tensor(f"{tile_stem}/latent_weight", weight),
            "weight_codes": writer.tensor(f"{tile_stem}/weight_codes", weight_codes),
            "packed_codes": writer.tensor(f"{tile_stem}/packed_codes", packed),
            "row_scales": writer.tensor(f"{tile_stem}/row_scales", row_scales),
            "accumulator": writer.tensor(f"{tile_stem}/accumulator", result.accumulator_i32),
            "scaled_before_bf16": writer.tensor(f"{tile_stem}/scaled_before_bf16", result.scaled_f32),
            "output_bf16": writer.tensor(f"{tile_stem}/output_bf16", result.output_bf16),
            "output_carrier": writer.tensor(f"{tile_stem}/output_carrier", result.carrier_f32),
        }
        if verifier is not None:
            verifier.check_w8_tile(output_tile, tensor_bytes(packed))
            verifier.check_f32_scales(row_start, tensor_bytes(row_scales))
        tiles.append(
            {
                "output_tile": output_tile,
                "rows": [row_start, row_start + M_TILE],
                "artifacts": paths,
                "gguf_verified": verifier is not None,
            }
        )
    return {
        "logical_shape": [vocab, hidden],
        "activation_key": "lm_head",
        "activation_source": source_kind,
        "activation_artifacts": activation_paths,
        "tiles": tiles,
    }


def parse_int_list(value: str) -> list[int]:
    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON integer list")
        values = parsed
    else:
        values = [part.strip() for part in stripped.split(",") if part.strip()]
    try:
        result = [int(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if not result:
        raise ValueError("integer list must not be empty")
    return result


def _ensure_external_output(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"oracle artifacts must stay outside the repository; choose ROW4_ORACLE_DIR outside {REPO_ROOT}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _publish_directory_no_clobber(temporary: Path, final_dir: Path) -> None:
    """Reserve ``final_dir`` atomically and publish ``manifest.json`` last.

    POSIX has no single portable Python operation that renames a non-empty
    directory with no-replace semantics on both macOS and Linux.  An exclusive
    ``mkdir`` reserves the public name (including against broken symlinks),
    then same-filesystem renames populate it.  Consumers must treat the final
    manifest as the completion marker.
    """

    if temporary.parent != final_dir.parent:
        raise ValueError("temporary and final oracle directories must share a parent")
    manifest = temporary / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"oracle temporary directory has no manifest: {manifest}")

    reserved = False
    try:
        final_dir.mkdir(mode=0o700)
        reserved = True
        for entry in sorted(temporary.iterdir(), key=lambda path: path.name):
            if entry.name == "manifest.json":
                continue
            os.rename(entry, final_dir / entry.name)
        os.rename(manifest, final_dir / manifest.name)
        temporary.rmdir()
    except BaseException:
        if reserved:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def capture_command(args: argparse.Namespace, *, require_gguf: bool) -> int:
    if args.checkpoint is None:
        raise ValueError("set ROW4_CHECKPOINT_DIR or pass --checkpoint")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = _json_object(checkpoint_path / "config.json")
    require_reference_config(config)
    layers = parse_int_list(args.layers)
    expected_layers = _positive_int(config, "num_hidden_layers")
    if layers != [0, expected_layers - 1]:
        raise ValueError(f"v1 golden layer selection is fixed to [0, {expected_layers - 1}]")
    input_ids = parse_int_list(args.input_ids)
    vocab = _positive_int(config, "vocab_size")
    if any(token < 0 or token >= vocab for token in input_ids):
        raise ValueError(f"input ids must be in [0, {vocab})")

    output_root_arg = args.output_dir or os.environ.get("ROW4_ORACLE_DIR")
    if not output_root_arg:
        raise ValueError("set ROW4_ORACLE_DIR or pass --output-dir")
    output_root = _ensure_external_output(Path(output_root_arg))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_name = f"qwen3-row4-oracle-{timestamp}"
    run_name = args.run_name or default_name
    if not run_name or run_name in (".", "..") or "/" in run_name:
        raise ValueError(f"invalid --run-name: {run_name!r}")
    final_dir = output_root / run_name
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing oracle directory: {final_dir}")

    gguf_path: Path | None = args.gguf.expanduser().resolve() if args.gguf is not None else None
    if require_gguf and gguf_path is None:
        raise ValueError("verify requires --gguf")
    if gguf_path is not None and not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    checkpoint = CheckpointReader(checkpoint_path)
    require_reference_checkpoint(checkpoint)
    activation_source = ActivationSource(args.activations, input_ids)
    verifier = GGUFVerifier(gguf_path) if gguf_path is not None else None
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_name}.tmp-", dir=output_root))
    writer = ArtifactWriter(temp_dir)
    try:
        captures: list[dict[str, Any]] = []
        for layer in layers:
            layer_bundles = [
                capture_bundle(checkpoint, bundle, layer, activation_source, writer, verifier)
                for bundle in build_bundles(config, layer)
            ]
            captures.append({"layer": layer, "bundles": layer_bundles})
        lm_head = capture_lm_head(checkpoint, config, activation_source, writer, verifier)

        metadata_files = (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        checkpoint_metadata = {
            name: file_record(checkpoint_path / name)
            for name in metadata_files
            if (checkpoint_path / name).is_file()
        }
        shard_records = {
            shard: file_record(
                checkpoint_path / shard,
                hash_contents=not args.skip_shard_hashes,
            )
            for shard in checkpoint.shards
        }
        manifest: dict[str, Any] = {
            "schema": "qwen3_row4_oracle_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "numeric_profile": NUMERIC_PROFILE,
            "row4_layout": ROW4_LAYOUT,
            "lm_head_layout": W8_LAYOUT,
            "git_revision": _git_revision(),
            "torch_version": torch.__version__,
            "checkpoint": {
                "path": str(checkpoint_path),
                "metadata_files": checkpoint_metadata,
                "shards": shard_records,
                "full_shard_hashes_recorded": not args.skip_shard_hashes,
            },
            "gguf": None
            if gguf_path is None
            else file_record(gguf_path, hash_contents=args.hash_gguf),
            "input_ids": input_ids,
            "activation_artifact": None
            if args.activations is None
            else file_record(args.activations.resolve()),
            "layers": captures,
            "lm_head": lm_head,
            "gguf_checks": [] if verifier is None else verifier.checks,
            "artifacts": writer.records,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory_no_clobber(temp_dir, final_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps({"oracle_dir": str(final_dir), "gguf_checks": 0 if verifier is None else len(verifier.checks)}))
    return 0


def _torch_load(path: Path) -> Any:
    if path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix == ".npy":
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("loading .npy logits requires numpy") from exc
        return torch.from_numpy(np.load(path, allow_pickle=False))
    raise ValueError(f"unsupported logits artifact {path}; use .pt/.pth or .npy")


def load_logits(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = _torch_load(path)
    targets = None
    if isinstance(payload, Mapping):
        logits = payload.get("logits")
        candidate_targets = payload.get("target_ids")
        if candidate_targets is not None:
            if not isinstance(candidate_targets, torch.Tensor):
                candidate_targets = torch.tensor(candidate_targets, dtype=torch.int64)
            targets = candidate_targets.detach().cpu().to(torch.int64)
    else:
        logits = payload
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits artifact {path} does not contain a tensor")
    if logits.ndim < 2 or not logits.dtype.is_floating_point:
        raise TypeError(f"logits in {path} must be a floating tensor with a vocabulary dimension")
    return logits.detach().cpu().to(torch.float32), targets


def compare_logits_command(args: argparse.Namespace) -> int:
    reference, embedded_targets = load_logits(args.reference)
    candidate, candidate_targets = load_logits(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"logit shapes differ: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    if candidate_targets is not None and embedded_targets is not None and not torch.equal(
        candidate_targets.reshape(-1), embedded_targets.reshape(-1)
    ):
        raise ValueError("reference and candidate artifacts contain different target_ids")
    targets = embedded_targets if embedded_targets is not None else candidate_targets
    if args.targets is not None:
        loaded_targets = _torch_load(args.targets)
        if isinstance(loaded_targets, Mapping):
            loaded_targets = loaded_targets.get("target_ids")
        if not isinstance(loaded_targets, torch.Tensor):
            raise TypeError("--targets must contain a tensor")
        targets = loaded_targets.detach().cpu().to(torch.int64)
    if args.target_ids is not None:
        targets = torch.tensor(parse_int_list(args.target_ids), dtype=torch.int64)

    ref = reference.reshape(-1, reference.shape[-1]).to(torch.float64)
    got = candidate.reshape_as(reference).reshape(-1, reference.shape[-1]).to(torch.float64)
    error = got - ref
    reference_energy = torch.mean(ref * ref).item()
    error_energy = torch.mean(error * error).item()
    nmse = error_energy / reference_energy if reference_energy != 0 else (0.0 if error_energy == 0 else math.inf)
    max_abs = torch.max(torch.abs(error)).item()
    ref_argmax = torch.argmax(ref, dim=-1)
    got_argmax = torch.argmax(got, dim=-1)
    argmax_fraction = torch.mean((ref_argmax == got_argmax).to(torch.float64)).item()
    top_count = min(10, ref.shape[-1])
    ref_top = torch.topk(ref, top_count, dim=-1).indices
    got_top = torch.topk(got, top_count, dim=-1).indices
    overlaps = (ref_top[:, :, None] == got_top[:, None, :]).any(dim=-1).sum(dim=-1).to(torch.float64)
    overlap_fraction = overlaps / top_count
    metrics: dict[str, Any] = {
        "shape": list(reference.shape),
        "nmse": nmse,
        "max_abs": max_abs,
        "argmax_fraction": argmax_fraction,
        "argmax_all_equal": bool(torch.equal(ref_argmax, got_argmax)),
        "top10_overlap_mean": torch.mean(overlap_fraction).item(),
        "top10_overlap_min": torch.min(overlap_fraction).item(),
    }

    passed = (
        nmse <= 1.0e-5
        and max_abs <= 5.0e-2
        and metrics["argmax_all_equal"]
        and metrics["top10_overlap_min"] >= 0.9
    )
    if targets is not None:
        targets_flat = targets.reshape(-1)
        if targets_flat.numel() != ref.shape[0]:
            raise ValueError(
                f"target count {targets_flat.numel()} does not match flattened token count {ref.shape[0]}"
            )
        if bool(((targets_flat < 0) | (targets_flat >= ref.shape[-1])).any()):
            raise ValueError("target id is outside the logits vocabulary")
        rows = torch.arange(ref.shape[0])
        ref_nll = torch.mean(torch.logsumexp(ref, dim=-1) - ref[rows, targets_flat]).item()
        got_nll = torch.mean(torch.logsumexp(got, dim=-1) - got[rows, targets_flat]).item()
        ref_ppl = math.exp(ref_nll)
        got_ppl = math.exp(got_nll)
        ppl_relative = abs(got_ppl - ref_ppl) / ref_ppl
        metrics.update(
            {
                "reference_mean_nll": ref_nll,
                "candidate_mean_nll": got_nll,
                "mean_nll_abs_diff": abs(got_nll - ref_nll),
                "reference_ppl": ref_ppl,
                "candidate_ppl": got_ppl,
                "ppl_relative_diff": ppl_relative,
            }
        )
        passed = passed and abs(got_nll - ref_nll) <= 5.0e-3 and ppl_relative <= 5.0e-3
    metrics["passed"] = passed
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed or args.no_enforce else 1


def self_test_command(_args: argparse.Namespace) -> int:
    logical = torch.arange(16, dtype=torch.uint8).reshape(16, 1)
    decoded = decode_row4_codes(logical)
    reencoded = quantize_row4_codes(decoded.to(torch.bfloat16))
    if not torch.equal(logical, reencoded):
        raise AssertionError("16-code encode/decode round-trip failed")

    zero_weight = torch.zeros((4, 1), dtype=torch.bfloat16)
    if quantize_row4_codes(zero_weight).item() != 10:
        raise AssertionError("tie/sign-zero rule failed")

    pack_input = torch.zeros((4, K_TILE), dtype=torch.uint8)
    pack_input[0, :16] = torch.arange(16, dtype=torch.uint8)
    packed = pack_row4_m16k128(pack_input)
    expected = bytes.fromhex("80 91 a2 b3 c4 d5 e6 f7")
    if tensor_bytes(packed[0, 0, 0, :8]) != expected:
        raise AssertionError("M16K128 split8 known-answer test failed")

    rounded = round_half_away_from_zero(
        torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=torch.float32)
    )
    expected_rounded = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if not torch.equal(rounded, expected_rounded):
        raise AssertionError("half-away-from-zero test failed")

    activation = quantize_activation(
        torch.tensor([[-127.0, -126.5, 0.0, 126.5, 127.0]], dtype=torch.float32)
    )
    expected_codes = torch.tensor([[-127, -127, 0, 127, 127]], dtype=torch.int8)
    if activation.scales_f32.item() != 1.0 or not torch.equal(activation.codes_i8, expected_codes):
        raise AssertionError("activation A8 known-answer test failed")

    w8_weight = torch.zeros((M_TILE, K_TILE), dtype=torch.bfloat16)
    w8_codes, w8_scales = quantize_w8_rows(w8_weight)
    if bool(w8_codes.any()) or bool(w8_scales.any()):
        raise AssertionError("W8 all-zero-row rule failed")
    if pack_w8_m16k128(w8_codes).shape != (1, 1, M_TILE, K_TILE):
        raise AssertionError("W8 M16K128 layout shape failed")

    accumulator = torch.tensor([[1, -1]], dtype=torch.int32)
    result = finish_linear(
        accumulator,
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([-2.0, -0.0], dtype=torch.float32),
    )
    expected_output = torch.tensor([[-1.0, 0.0]], dtype=torch.bfloat16)
    if not torch.equal(result.output_bf16, expected_output):
        raise AssertionError("signed-scale/output-BF16 test failed")

    print("row4 oracle synthetic self-test: PASS")
    return 0


def add_capture_arguments(parser: argparse.ArgumentParser, *, verify: bool) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint path; defaults to ROW4_CHECKPOINT_DIR",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=DEFAULT_GGUF if verify else None,
        help="GGUF path; verify defaults to ROW4_GGUF",
    )
    parser.add_argument("--output-dir", type=Path, help="external golden root; defaults to ROW4_ORACLE_DIR")
    parser.add_argument("--run-name", help="new directory name below the golden root")
    parser.add_argument("--layers", default="0,35", help="fixed v1 selection; must be 0,35")
    parser.add_argument("--input-ids", default="1,2", help="comma-separated or JSON token ids")
    parser.add_argument(
        "--activations",
        type=Path,
        help="optional torch dict: layer0.qkv/o/gate_up/down, layer35.*, lm_head -> [T,K] F32",
    )
    parser.add_argument(
        "--skip-shard-hashes",
        action="store_true",
        help="development-only: record shard sizes but omit required full SHA-256 hashes",
    )
    parser.add_argument("--hash-gguf", action="store_true", help="also SHA-256 the complete GGUF")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_test = subparsers.add_parser("self-test", help="run small numeric/layout known-answer tests")
    self_test.set_defaults(function=self_test_command)

    capture = subparsers.add_parser("capture", help="capture checkpoint primitive goldens")
    add_capture_arguments(capture, verify=False)
    capture.set_defaults(function=lambda args: capture_command(args, require_gguf=False))

    verify = subparsers.add_parser("verify", help="capture goldens and require bit-exact GGUF tiles")
    add_capture_arguments(verify, verify=True)
    verify.set_defaults(function=lambda args: capture_command(args, require_gguf=True))

    compare = subparsers.add_parser("compare-logits", help="compare exported reference/runtime logits")
    compare.add_argument("--reference", type=Path, required=True, help="reference .pt/.pth/.npy")
    compare.add_argument("--candidate", type=Path, required=True, help="runtime .pt/.pth/.npy")
    compare.add_argument("--targets", type=Path, help="optional target-id .pt/.pth/.npy")
    compare.add_argument("--target-ids", help="optional comma-separated or JSON target ids")
    compare.add_argument("--output", type=Path, help="write metrics JSON")
    compare.add_argument("--no-enforce", action="store_true", help="report thresholds without failing")
    compare.set_defaults(function=compare_logits_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
