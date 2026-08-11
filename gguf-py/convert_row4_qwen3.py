#!/usr/bin/env python3

"""Convert the strict Qwen3 Row4 W1A8-INT8 checkpoint profile to GGUF."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import gguf
from row4.checkpoint import CheckpointManifest, TensorReader, load_manifest
from row4.quant import pack_row4_m16k128, pack_w8_m16k128, quantize_row4_codes, quantize_w8_rows
from row4.spec import write_metadata


LINEAR_SHAPES = (
    ("self_attn.q_proj", "q"),
    ("self_attn.k_proj", "k"),
    ("self_attn.v_proj", "v"),
    ("self_attn.o_proj", "o"),
    ("mlp.gate_proj", "gate"),
    ("mlp.up_proj", "up"),
    ("mlp.down_proj", "down"),
)
REFERENCE_DIMENSIONS = (36, 4096, 12288, 32, 8, 128, 151936)
REFERENCE_TRANSFORMERS_VERSION = "5.2.0"
REFERENCE_CHECKPOINT_TENSOR_COUNT = 651
REFERENCE_CHECKPOINT_PARAMETERS = 8_192_136_192
REFERENCE_CHECKPOINT_BYTES = 16_384_272_384
REFERENCE_SHARDS = tuple(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5))
REFERENCE_SHARD_TENSOR_COUNTS = (113, 232, 229, 77)
REFERENCE_ROW4_CODE_BYTES = 868_220_928
REFERENCE_ROW4_SCALE_BYTES = 2_801_664
REFERENCE_LM_HEAD_CODE_BYTES = 622_329_856
REFERENCE_LM_HEAD_SCALE_BYTES = 607_744
REFERENCE_PAYLOAD_BYTES = 2_739_236_352
REFERENCE_SCALE_VALUES = 1_400_832
REFERENCE_NONPOSITIVE_SCALES = 15_340
EXPECTED_TENSOR_COUNT = 436
OUTPUT_ESTIMATE_OVERHEAD = 64 * 1024 * 1024
REFERENCE_YARN_FACTOR = 4.0
REFERENCE_YARN_EFFECTIVE_MSCALE = 1.0 + 0.1 * math.log(REFERENCE_YARN_FACTOR)
REFERENCE_YARN_RAW_ATTN_FACTOR = 1.0 / REFERENCE_YARN_EFFECTIVE_MSCALE
REFERENCE_YARN_GGUF_PRE_FACTOR = 1.0
REFERENCE_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION = "{{ bos_token }}"


@dataclass(frozen=True)
class Qwen3Dimensions:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def reference_tuple(self) -> tuple[int, ...]:
        return (
            self.layers,
            self.hidden,
            self.intermediate,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.vocab,
        )


@dataclass(frozen=True)
class PreflightReport:
    config: dict[str, object]
    manifest: CheckpointManifest
    dimensions: Qwen3Dimensions
    scale_values: int
    nonpositive_scales: int
    tensor_payload_bytes: int
    available_output_bytes: int


@dataclass(frozen=True)
class TensorWriteEntry:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int
    raw_dtype: gguf.GGMLQuantizationType
    chunks: Callable[[], Iterator[np.ndarray]]


@dataclass
class PrivateTemporaryOutput:
    directory: Path
    output: Path
    directory_fd: int
    directory_device: int
    directory_inode: int


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen3 config field {key!r} must be a positive integer, got {value!r}")
    return value


def get_dimensions(config: Mapping[str, object]) -> Qwen3Dimensions:
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected model_type='qwen3', got {config.get('model_type')!r}")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or architectures != ["Qwen3ForCausalLM"]:
        raise ValueError(f"expected architectures=['Qwen3ForCausalLM'], got {architectures!r}")
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("Row4 v1 requires tie_word_embeddings=false and a dedicated INT8 lm_head")

    dimensions = Qwen3Dimensions(
        layers=_positive_int(config, "num_hidden_layers"),
        hidden=_positive_int(config, "hidden_size"),
        intermediate=_positive_int(config, "intermediate_size"),
        heads=_positive_int(config, "num_attention_heads"),
        kv_heads=_positive_int(config, "num_key_value_heads"),
        head_dim=_positive_int(config, "head_dim"),
        vocab=_positive_int(config, "vocab_size"),
    )
    if dimensions.q_dim != dimensions.hidden:
        raise ValueError(
            f"Qwen3 q dimension {dimensions.q_dim} must equal hidden_size {dimensions.hidden}"
        )
    if dimensions.kv_heads > dimensions.heads or dimensions.heads % dimensions.kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    return dimensions


def _weight_and_scale_names(layer: int, suffix: str) -> tuple[str, str]:
    base = f"model.layers.{layer}.{suffix}"
    return f"{base}.weight", f"{base}.weight_scale"


def expected_tensor_shapes(dimensions: Qwen3Dimensions) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (dimensions.vocab, dimensions.hidden),
        "model.norm.weight": (dimensions.hidden,),
        "lm_head.weight": (dimensions.vocab, dimensions.hidden),
    }
    projection_shapes = {
        "self_attn.q_proj": (dimensions.q_dim, dimensions.hidden),
        "self_attn.k_proj": (dimensions.kv_dim, dimensions.hidden),
        "self_attn.v_proj": (dimensions.kv_dim, dimensions.hidden),
        "self_attn.o_proj": (dimensions.hidden, dimensions.q_dim),
        "mlp.gate_proj": (dimensions.intermediate, dimensions.hidden),
        "mlp.up_proj": (dimensions.intermediate, dimensions.hidden),
        "mlp.down_proj": (dimensions.hidden, dimensions.intermediate),
    }
    for layer in range(dimensions.layers):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (dimensions.hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (dimensions.hidden,)
        shapes[f"{prefix}.self_attn.q_norm.weight"] = (dimensions.head_dim,)
        shapes[f"{prefix}.self_attn.k_norm.weight"] = (dimensions.head_dim,)
        for suffix, shape in projection_shapes.items():
            weight, scale = _weight_and_scale_names(layer, suffix)
            shapes[weight] = shape
            shapes[scale] = (shape[0],)
    return shapes


def _matches_exact_value(value: object, expected: object) -> bool:
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


def _require_exact_value(config: Mapping[str, object], key: str, expected: object) -> None:
    value = config.get(key)
    if not _matches_exact_value(value, expected):
        raise ValueError(f"Row4 v1 requires config {key}={expected!r}, got {value!r}")


def validate_reference_config_profile(config: Mapping[str, object]) -> Qwen3Dimensions:
    """Validate the one checkpoint profile accepted by the deployment CLI.

    Dimension-generic helpers remain available for tiny plan/layout tests, but
    a real conversion must never silently reinterpret a merely compatible
    Qwen3 checkpoint as the frozen Row4 v1 reference model.
    """

    dimensions = get_dimensions(config)
    if dimensions.reference_tuple != REFERENCE_DIMENSIONS:
        raise ValueError(
            "Row4 v1 requires reference dimensions "
            f"{REFERENCE_DIMENSIONS}, got {dimensions.reference_tuple}"
        )

    exact_values: tuple[tuple[str, object], ...] = (
        ("hidden_act", "silu"),
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
        _require_exact_value(config, key, expected)

    layer_types = config.get("layer_types")
    expected_layer_types = ["full_attention"] * dimensions.layers
    if layer_types != expected_layer_types:
        raise ValueError(
            "Row4 v1 requires layer_types=['full_attention'] * 36, "
            f"got {layer_types!r}"
        )

    rope = config.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("Row4 v1 checkpoint requires rope_parameters")
    expected_rope = {
        "rope_type": "yarn",
        "factor": REFERENCE_YARN_FACTOR,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1_000_000,
        # Transformers 5.2.0 ignores ``attn_factor`` (it recognizes only
        # ``attention_factor``), so this inverse value does not cancel YaRN's
        # default effective mscale.  GGUF must store a pre-factor of 1 instead.
        "attn_factor": REFERENCE_YARN_RAW_ATTN_FACTOR,
    }
    actual_rope_keys = set(rope)
    expected_rope_keys = set(expected_rope)
    if actual_rope_keys != expected_rope_keys:
        raise ValueError(
            "Row4 v1 requires the exact rope_parameters key set: "
            f"missing={sorted(expected_rope_keys - actual_rope_keys)}, "
            f"unexpected={sorted(actual_rope_keys - expected_rope_keys)}"
        )
    for key, expected in expected_rope.items():
        if not _matches_exact_value(rope.get(key), expected):
            raise ValueError(
                f"Row4 v1 requires rope_parameters.{key}={expected!r}, "
                f"got {rope.get(key)!r}"
            )
    expected_auto_map = {
        "AutoModel": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen3_row4_int8.Qwen3ForCausalLM",
    }
    if config.get("auto_map") != expected_auto_map:
        raise ValueError(
            "Row4 v1 requires the reference specialized auto_map, got "
            f"{config.get('auto_map')!r}"
        )
    return dimensions


def validate_checkpoint_schema(
    manifest: CheckpointManifest,
    dimensions: Qwen3Dimensions,
) -> None:
    if manifest.shards != REFERENCE_SHARDS:
        raise ValueError(
            f"Row4 v1 requires shards {REFERENCE_SHARDS}, got {manifest.shards}"
        )
    shard_counts = tuple(
        sum(info.shard == shard for info in manifest.tensors.values())
        for shard in REFERENCE_SHARDS
    )
    if shard_counts != REFERENCE_SHARD_TENSOR_COUNTS:
        raise ValueError(
            "Row4 v1 checkpoint shard tensor counts differ: "
            f"expected {REFERENCE_SHARD_TENSOR_COUNTS}, got {shard_counts}"
        )

    expected = expected_tensor_shapes(dimensions)
    actual_names = set(manifest.tensors)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "invalid Qwen3 Row4 checkpoint tensor set: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if len(actual_names) != REFERENCE_CHECKPOINT_TENSOR_COUNT:
        raise ValueError(
            f"Qwen3 Row4 checkpoint tensor count mismatch: got {len(actual_names)}, "
            f"expected {REFERENCE_CHECKPOINT_TENSOR_COUNT}"
        )

    non_bf16 = sorted(name for name, info in manifest.tensors.items() if info.dtype != "BF16")
    if non_bf16:
        raise ValueError(f"all Row4 checkpoint tensors must be BF16: {non_bf16[:8]}")
    bad_shapes = sorted(
        name for name, shape in expected.items() if manifest.tensors[name].shape != shape
    )
    if bad_shapes:
        details = [
            f"{name}: expected {expected[name]}, got {manifest.tensors[name].shape}"
            for name in bad_shapes[:8]
        ]
        raise ValueError("Qwen3 Row4 tensor shape mismatch: " + "; ".join(details))

    aligned = ["lm_head.weight"]
    aligned.extend(
        _weight_and_scale_names(layer, suffix)[0]
        for layer in range(dimensions.layers)
        for suffix, _ in LINEAR_SHAPES
    )
    bad_alignment = [
        f"{name}={manifest.tensors[name].shape}"
        for name in aligned
        if any(dim % 128 != 0 for dim in manifest.tensors[name].shape)
    ]
    if bad_alignment:
        raise ValueError("Row4 v1 has no unaligned fallback; offending tensors: " + ", ".join(bad_alignment[:8]))

    parameter_count = sum(info.parameter_count for info in manifest.tensors.values())
    if parameter_count != REFERENCE_CHECKPOINT_PARAMETERS:
        raise ValueError(
            "Row4 v1 checkpoint parameter count mismatch: "
            f"got {parameter_count}, expected {REFERENCE_CHECKPOINT_PARAMETERS}"
        )
    expected_index_metadata = {
        "total_parameters": REFERENCE_CHECKPOINT_PARAMETERS,
        "total_size": REFERENCE_CHECKPOINT_BYTES,
    }
    for key, expected_value in expected_index_metadata.items():
        if manifest.metadata.get(key) != expected_value:
            raise ValueError(
                f"Row4 v1 checkpoint index metadata {key} must be {expected_value}, "
                f"got {manifest.metadata.get(key)!r}"
            )


def bf16_payload(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be checkpoint BF16, got {tensor.dtype}")
    payload = tensor.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    if np.any((payload & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError(f"{name} contains a non-finite BF16 value")
    return np.ascontiguousarray(payload)


def scan_saved_scales(
    reader: TensorReader,
    dimensions: Qwen3Dimensions,
) -> tuple[int, int]:
    scale_values = 0
    nonpositive = 0
    for layer in range(dimensions.layers):
        for suffix, _ in LINEAR_SHAPES:
            scale_name = _weight_and_scale_names(layer, suffix)[1]
            scale = reader.get(scale_name)
            bf16_payload(scale, scale_name)
            scale_values += scale.numel()
            nonpositive += int(torch.count_nonzero(scale <= 0).item())
    return scale_values, nonpositive


def _row4_storage_bytes(shape: Sequence[int]) -> int:
    return math.prod(shape) // 8


def expected_tensor_payload_bytes(dimensions: Qwen3Dimensions) -> int:
    shapes = expected_tensor_shapes(dimensions)
    total = math.prod(shapes["model.embed_tokens.weight"]) * 2
    total += math.prod(shapes["model.norm.weight"]) * 2
    total += math.prod(shapes["lm_head.weight"])
    total += dimensions.vocab * 4
    for layer in range(dimensions.layers):
        total += (2 * dimensions.hidden + 2 * dimensions.head_dim) * 2
        for suffix, _ in LINEAR_SHAPES:
            weight, scale = _weight_and_scale_names(layer, suffix)
            total += _row4_storage_bytes(shapes[weight])
            total += math.prod(shapes[scale]) * 2
    return total


def _load_tokenizer_files(
    model_dir: Path,
    vocab_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not tokenizer_config_path.is_file():
        raise FileNotFoundError("Row4 conversion requires tokenizer.json and tokenizer_config.json")
    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tokenizer JSON: {exc}") from exc
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("tokenizer JSON roots must be objects")

    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json is missing object model")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json is missing model.vocab")
    base_ids: dict[int, str] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not token:
            raise ValueError(f"tokenizer base token must be a non-empty string, got {token!r}")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(f"tokenizer base id for {token!r} must be an integer, got {token_id!r}")
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"tokenizer base id for {token!r} must be in [0, {vocab_size}), got {token_id}"
            )
        previous = base_ids.setdefault(token_id, token)
        if previous != token:
            raise ValueError(
                f"tokenizer base id {token_id} is assigned to both {previous!r} and {token!r}"
            )

    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer.json added_tokens must be a list")
    added_ids: dict[int, str] = {}
    added_contents: set[str] = set()
    for index, entry in enumerate(added_tokens):
        if not isinstance(entry, dict):
            raise ValueError(f"tokenizer added_tokens[{index}] must be an object")
        token_id = entry.get("id")
        content = entry.get("content")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(
                f"tokenizer added token id at index {index} must be an integer, got {token_id!r}"
            )
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"tokenizer added token id at index {index} must be in [0, {vocab_size}), "
                f"got {token_id}"
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"tokenizer added token content at index {index} must be a non-empty string"
            )
        if token_id in base_ids:
            raise ValueError(
                f"tokenizer id {token_id} collides between base token {base_ids[token_id]!r} "
                f"and added token {content!r}"
            )
        previous = added_ids.setdefault(token_id, content)
        if previous != content:
            raise ValueError(
                f"tokenizer added id {token_id} is assigned to both {previous!r} and {content!r}"
            )
        if content in added_contents:
            raise ValueError(f"tokenizer added token content is duplicated: {content!r}")
        if content in vocab:
            raise ValueError(f"tokenizer added token {content!r} collides with the base vocabulary")
        added_contents.add(content)
    return dict(tokenizer), dict(tokenizer_config)


def _validate_tokenizer_files(model_dir: Path, vocab_size: int) -> None:
    _, tokenizer_config = _load_tokenizer_files(model_dir, vocab_size)
    if tokenizer_config.get("bos_token") != REFERENCE_BOS_TOKEN:
        raise ValueError(
            "Row4 v1 tokenizer_config.json bos_token must be "
            f"{REFERENCE_BOS_TOKEN!r}, got {tokenizer_config.get('bos_token')!r}"
        )
    if "add_bos_token" in tokenizer_config:
        raise ValueError(
            "Row4 v1 tokenizer_config.json must omit add_bos_token; "
            "the GGUF contract supplies tokenizer.ggml.add_bos_token=true"
        )

    chat_template_path = model_dir / "chat_template.jinja"
    if not chat_template_path.is_file():
        raise FileNotFoundError("Row4 conversion requires chat_template.jinja")
    chat_template = chat_template_path.read_text(encoding="utf-8")
    bos_expressions = chat_template.count(REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION)
    if bos_expressions != 1:
        raise ValueError(
            "Row4 v1 chat_template.jinja must emit exactly one "
            f"{REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION!r}, found {bos_expressions}"
        )
    if chat_template.find("{{") != chat_template.find(REFERENCE_CHAT_TEMPLATE_BOS_EXPRESSION):
        raise ValueError(
            "Row4 v1 chat_template.jinja must emit bos_token as its first output expression"
        )


def run_preflight(model_dir: Path, output_file: Path | None) -> PreflightReport:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"required config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")

    dimensions = validate_reference_config_profile(config)
    manifest = load_manifest(model_dir, expected_shards=4)
    validate_checkpoint_schema(manifest, dimensions)
    _validate_tokenizer_files(model_dir, dimensions.vocab)
    reader = TensorReader(model_dir, manifest)
    scale_values, nonpositive = scan_saved_scales(reader, dimensions)
    payload_bytes = expected_tensor_payload_bytes(dimensions)

    if scale_values != REFERENCE_SCALE_VALUES or nonpositive != REFERENCE_NONPOSITIVE_SCALES:
        raise ValueError(
            "reference checkpoint scale statistics mismatch: "
            f"values={scale_values}, nonpositive={nonpositive}"
        )
    if payload_bytes != REFERENCE_PAYLOAD_BYTES:
        raise AssertionError(
            f"internal Row4 reference payload mismatch: {payload_bytes} != {REFERENCE_PAYLOAD_BYTES}"
        )

    output_parent = output_file.parent if output_file is not None else model_dir
    if not output_parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output_parent}")
    available = shutil.disk_usage(output_parent).free
    if output_file is not None and available < payload_bytes + OUTPUT_ESTIMATE_OVERHEAD:
        raise ValueError(
            "insufficient disk space for Row4 GGUF: "
            f"need at least {payload_bytes + OUTPUT_ESTIMATE_OVERHEAD:,}, available {available:,}"
        )
    return PreflightReport(
        config=dict(config),
        manifest=manifest,
        dimensions=dimensions,
        scale_values=scale_values,
        nonpositive_scales=nonpositive,
        tensor_payload_bytes=payload_bytes,
        available_output_bytes=available,
    )


def _entry(
    name: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    raw_dtype: gguf.GGMLQuantizationType,
    chunks: Callable[[], Iterator[np.ndarray]],
) -> TensorWriteEntry:
    numpy_dtype = np.dtype(dtype)
    return TensorWriteEntry(
        name=name,
        shape=shape,
        dtype=numpy_dtype,
        nbytes=math.prod(shape) * numpy_dtype.itemsize,
        raw_dtype=raw_dtype,
        chunks=chunks,
    )


def _iter_bf16_tensor(reader: TensorReader, name: str) -> Iterator[np.ndarray]:
    yield bf16_payload(reader.get(name), name)


def _iter_bf16_rows(
    reader: TensorReader,
    name: str,
    *,
    rows_per_chunk: int = 256,
) -> Iterator[np.ndarray]:
    rows, _ = reader.shape(name)
    with reader.open_2d(name) as read_rows:
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            yield bf16_payload(read_rows(start, end), name)


class Row4BundleWriter:
    def __init__(
        self,
        reader: TensorReader,
        weight_names: Sequence[str],
        scale_names: Sequence[str],
    ):
        self.reader = reader
        self.weight_names = tuple(weight_names)
        self.scale_names = tuple(scale_names)

    def iter_codes(self) -> Iterator[np.ndarray]:
        for name in self.weight_names:
            out_features, _ = self.reader.shape(name)
            with self.reader.open_2d(name) as read_rows:
                for start in range(0, out_features, 128):
                    chunk = read_rows(start, min(start + 128, out_features))
                    weight = chunk.to(torch.float32).cpu().numpy()
                    try:
                        packed = pack_row4_m16k128(quantize_row4_codes(weight))
                    except ValueError as exc:
                        raise ValueError(
                            f"failed to encode {name} rows "
                            f"{start}:{start + chunk.shape[0]}: {exc}"
                        ) from exc
                    yield packed

    def iter_scales(self) -> Iterator[np.ndarray]:
        for name in self.scale_names:
            yield bf16_payload(self.reader.get(name), name)


class W8BundleWriter:
    def __init__(self, reader: TensorReader, name: str):
        self.reader = reader
        self.name = name
        self._scales: np.ndarray | None = None

    def iter_codes(self) -> Iterator[np.ndarray]:
        if self._scales is not None:
            raise RuntimeError("W8 lm_head codes may be streamed only once")
        out_features, _ = self.reader.shape(self.name)
        scales = np.empty(out_features, dtype=np.float32)
        with self.reader.open_2d(self.name) as read_rows:
            for start in range(0, out_features, 128):
                end = min(start + 128, out_features)
                chunk = read_rows(start, end)
                weight = chunk.to(torch.float32).cpu().numpy()
                try:
                    quantized, chunk_scales = quantize_w8_rows(weight)
                except ValueError as exc:
                    raise ValueError(f"failed to quantize {self.name} rows {start}:{end}: {exc}") from exc
                scales[start:end] = chunk_scales
                yield pack_w8_m16k128(quantized)
        self._scales = scales

    def iter_scales(self) -> Iterator[np.ndarray]:
        if self._scales is None:
            raise RuntimeError("W8 lm_head codes must be written before scales")
        scales = self._scales
        self._scales = None
        yield scales


def _add_bf16(
    plan: list[TensorWriteEntry],
    reader: TensorReader,
    gguf_name: str,
    checkpoint_name: str,
) -> None:
    plan.append(
        _entry(
            gguf_name,
            reader.shape(checkpoint_name),
            np.uint16,
            gguf.GGMLQuantizationType.BF16,
            lambda checkpoint_name=checkpoint_name: _iter_bf16_tensor(reader, checkpoint_name),
        )
    )


def _add_row4_bundle(
    plan: list[TensorWriteEntry],
    reader: TensorReader,
    gguf_base: str,
    weight_names: Sequence[str],
    scale_names: Sequence[str],
) -> None:
    input_dims = {reader.shape(name)[1] for name in weight_names}
    if len(input_dims) != 1:
        raise ValueError(f"{gguf_base} fused inputs do not share K: {sorted(input_dims)}")
    in_features = input_dims.pop()
    out_features = sum(reader.shape(name)[0] for name in weight_names)
    bundle = Row4BundleWriter(reader, weight_names, scale_names)
    plan.append(
        _entry(
            f"{gguf_base}.row4.codes",
            (out_features // 16, in_features // 128, 4, 64),
            np.uint8,
            gguf.GGMLQuantizationType.ROW4_CODES,
            bundle.iter_codes,
        )
    )
    plan.append(
        _entry(
            f"{gguf_base}.row4.scales",
            (out_features,),
            np.uint16,
            gguf.GGMLQuantizationType.BF16,
            bundle.iter_scales,
        )
    )


def build_tensor_plan(reader: TensorReader, dimensions: Qwen3Dimensions) -> list[TensorWriteEntry]:
    plan: list[TensorWriteEntry] = [
        _entry(
            "token_embd.weight",
            (dimensions.vocab, dimensions.hidden),
            np.uint16,
            gguf.GGMLQuantizationType.BF16,
            lambda: _iter_bf16_rows(reader, "model.embed_tokens.weight"),
        )
    ]
    for layer in range(dimensions.layers):
        prefix = f"model.layers.{layer}"
        _add_bf16(plan, reader, f"blk.{layer}.attn_norm.weight", f"{prefix}.input_layernorm.weight")
        _add_bf16(plan, reader, f"blk.{layer}.attn_q_norm.weight", f"{prefix}.self_attn.q_norm.weight")
        _add_bf16(plan, reader, f"blk.{layer}.attn_k_norm.weight", f"{prefix}.self_attn.k_norm.weight")

        qkv = [
            _weight_and_scale_names(layer, f"self_attn.{projection}_proj")
            for projection in ("q", "k", "v")
        ]
        _add_row4_bundle(
            plan,
            reader,
            f"blk.{layer}.attn_qkv",
            [pair[0] for pair in qkv],
            [pair[1] for pair in qkv],
        )

        output = _weight_and_scale_names(layer, "self_attn.o_proj")
        _add_row4_bundle(plan, reader, f"blk.{layer}.attn_output", [output[0]], [output[1]])
        _add_bf16(plan, reader, f"blk.{layer}.ffn_norm.weight", f"{prefix}.post_attention_layernorm.weight")

        gate_up = [
            _weight_and_scale_names(layer, f"mlp.{projection}_proj")
            for projection in ("gate", "up")
        ]
        _add_row4_bundle(
            plan,
            reader,
            f"blk.{layer}.ffn_gate_up",
            [pair[0] for pair in gate_up],
            [pair[1] for pair in gate_up],
        )
        down = _weight_and_scale_names(layer, "mlp.down_proj")
        _add_row4_bundle(plan, reader, f"blk.{layer}.ffn_down", [down[0]], [down[1]])

    _add_bf16(plan, reader, "output_norm.weight", "model.norm.weight")
    w8 = W8BundleWriter(reader, "lm_head.weight")
    plan.append(
        _entry(
            "output.w8.codes",
            (dimensions.vocab // 16, dimensions.hidden // 128, 16, 128),
            np.int8,
            gguf.GGMLQuantizationType.I8,
            w8.iter_codes,
        )
    )
    plan.append(
        _entry(
            "output.w8.scales",
            (dimensions.vocab,),
            np.float32,
            gguf.GGMLQuantizationType.F32,
            w8.iter_scales,
        )
    )
    if len(plan) != 4 + dimensions.layers * 12:
        raise AssertionError(f"internal Row4 tensor plan count mismatch: {len(plan)}")
    return plan


def _looks_special(token: str) -> bool:
    return (
        token in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")
        or (token.startswith("<|") and token.endswith("|>"))
        or (token.startswith("<｜") and token.endswith("｜>"))
        or (token.startswith("<unused") and token.endswith(">"))
    )


def add_qwen2_vocab(model_dir: Path, config: Mapping[str, object], writer: gguf.GGUFWriter) -> None:
    vocab_size = _positive_int(config, "vocab_size")
    tokenizer, tokenizer_config = _load_tokenizer_files(model_dir, vocab_size)
    model = tokenizer["model"]
    assert isinstance(model, dict)
    vocab = model["vocab"]
    assert isinstance(vocab, dict)
    reverse_vocab = {token_id: token for token, token_id in vocab.items()}
    added_by_id: dict[int, dict[str, object]] = {}
    added_tokens = tokenizer.get("added_tokens", [])
    assert isinstance(added_tokens, list)
    for entry in added_tokens:
        assert isinstance(entry, dict)
        token_id = entry["id"]
        content = entry["content"]
        assert isinstance(token_id, int) and not isinstance(token_id, bool)
        assert isinstance(content, str)
        reverse_vocab[token_id] = content
        added_by_id[token_id] = entry

    tokens: list[str] = []
    token_types: list[int] = []
    for token_id in range(vocab_size):
        token = reverse_vocab.get(token_id)
        if token is None:
            tokens.append(f"[PAD{token_id}]")
            token_types.append(gguf.TokenType.UNUSED)
        elif token_id in added_by_id:
            tokens.append(token)
            is_special = bool(added_by_id[token_id].get("special", False)) or _looks_special(token)
            token_types.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            tokens.append(token)
            token_types.append(gguf.TokenType.NORMAL)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    gguf.SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)

    if "add_prefix_space" in tokenizer_config:
        writer.add_add_space_prefix(bool(tokenizer_config["add_prefix_space"]))


def register_tensor_plan(writer: gguf.GGUFWriter, plan: Sequence[TensorWriteEntry]) -> None:
    for entry in plan:
        writer.add_tensor_info(
            entry.name,
            entry.shape,
            entry.dtype,
            entry.nbytes,
            raw_dtype=entry.raw_dtype,
        )


def write_tensor_plan(
    writer: gguf.GGUFWriter,
    plan: Sequence[TensorWriteEntry],
    *,
    verbose: bool,
) -> None:
    writer.write_ti_data_to_file()
    for index, entry in enumerate(plan, start=1):
        if verbose:
            print(f"writing tensor {index}/{len(plan)}: {entry.name}")
        writer.write_tensor_data_stream(entry.chunks())


def add_model_metadata(
    writer: gguf.GGUFWriter,
    config: Mapping[str, object],
    dimensions: Qwen3Dimensions,
) -> None:
    writer.add_custom_alignment(64)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_name(str(config.get("_name_or_path", "Qwen3-Row4-INT8")))
    writer.add_context_length(131072)
    writer.add_embedding_length(dimensions.hidden)
    writer.add_block_count(dimensions.layers)
    writer.add_feed_forward_length(dimensions.intermediate)
    writer.add_head_count(dimensions.heads)
    writer.add_head_count_kv(dimensions.kv_heads)
    writer.add_key_length(dimensions.head_dim)
    writer.add_value_length(dimensions.head_dim)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_dimension_count(dimensions.head_dim)
    writer.add_rope_freq_base(1_000_000.0)
    writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
    writer.add_rope_scaling_factor(REFERENCE_YARN_FACTOR)
    writer.add_rope_scaling_orig_ctx_len(32768)
    writer.add_rope_scaling_attn_factors(REFERENCE_YARN_GGUF_PRE_FACTOR)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_ROW4)
    writer.add_vocab_size(dimensions.vocab)
    writer.add_add_bos_token(True)
    write_metadata(writer)


def print_preflight(report: PreflightReport) -> None:
    print(
        "Qwen3 Row4 preflight passed: "
        f"shards={len(report.manifest.shards)} tensors={len(report.manifest.tensors)} "
        f"layers={report.dimensions.layers}"
    )
    print(
        f"saved_scales={report.scale_values:,} nonpositive={report.nonpositive_scales:,} "
        f"GGUF_tensors={EXPECTED_TENSOR_COUNT}"
    )
    print(
        f"tensor_payload={report.tensor_payload_bytes:,} bytes "
        f"available={report.available_output_bytes:,} bytes"
    )


def _resolve_safe_output_path(output_file: Path) -> Path:
    """Resolve the output parent and reject writable non-sticky ancestors.

    The GGUF writer opens its destination by pathname.  A different user must
    therefore be unable to rename the private directory between its creation
    and that open.  Sticky shared directories (for example ``/tmp``) retain
    the required owner-only rename rule only when owned by root or the current
    process.
    """

    if not output_file.name:
        raise ValueError(f"output path must name a file, got {output_file}")
    resolved_parent = output_file.parent.resolve(strict=True)
    trusted_owner_uids = {0, os.geteuid()}
    for ancestor in (resolved_parent, *resolved_parent.parents):
        ancestor_stat = os.stat(ancestor, follow_symlinks=False)
        if not stat.S_ISDIR(ancestor_stat.st_mode):
            raise NotADirectoryError(f"output ancestor is not a directory: {ancestor}")
        mode = ancestor_stat.st_mode
        trusted_owner = ancestor_stat.st_uid in trusted_owner_uids
        if mode & stat.S_IWUSR and not trusted_owner:
            raise PermissionError(
                "unsafe output ancestor is writable by an untrusted owner: "
                f"{ancestor} (uid {ancestor_stat.st_uid}, mode {stat.S_IMODE(mode):#05o})"
            )
        writable_by_others = mode & (stat.S_IWGRP | stat.S_IWOTH)
        if writable_by_others and not (mode & stat.S_ISVTX and trusted_owner):
            raise PermissionError(
                "unsafe output ancestor is group/world-writable without a trusted sticky owner: "
                f"{ancestor} (uid {ancestor_stat.st_uid}, mode {stat.S_IMODE(mode):#05o})"
            )
    return resolved_parent / output_file.name


def _create_private_temporary(output_file: Path) -> PrivateTemporaryOutput:
    """Create a unique mode-0700 directory beside the final output."""

    output_file = _resolve_safe_output_path(output_file)
    raw_directory = tempfile.mkdtemp(
        prefix=f".{output_file.name}.tmp-",
        dir=output_file.parent,
    )
    directory = Path(raw_directory)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except BaseException:
        directory.rmdir()
        raise
    directory_stat = os.fstat(directory_fd)
    return PrivateTemporaryOutput(
        directory=directory,
        output=directory / output_file.name,
        directory_fd=directory_fd,
        directory_device=directory_stat.st_dev,
        directory_inode=directory_stat.st_ino,
    )


def _cleanup_private_temporary(temporary: PrivateTemporaryOutput) -> None:
    """Best-effort cleanup without ever unlinking the public output name."""

    if temporary.directory_fd >= 0:
        try:
            os.unlink(temporary.output.name, dir_fd=temporary.directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            # The completed public hard link, if any, is intentionally left
            # untouched.  Cleanup failure must not delete a path that another
            # process may have replaced after publication.
            pass
        finally:
            os.close(temporary.directory_fd)
            temporary.directory_fd = -1

    try:
        current = temporary.directory.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != (
        temporary.directory_device,
        temporary.directory_inode,
    ):
        return
    try:
        temporary.directory.rmdir()
    except OSError:
        pass


def _atomic_publish_no_clobber(temporary: Path, output_file: Path) -> None:
    """Atomically publish a completed file without replacing any target.

    A hard link is an atomic no-clobber operation on both macOS and Linux.
    The temporary lives beside the target, so both names are necessarily on
    the same filesystem.  A concurrent regular file, directory, or symlink at
    ``output_file`` makes ``os.link`` fail with ``FileExistsError``.
    """

    os.link(temporary, output_file, follow_symlinks=False)


def write_conversion_output(
    model_dir: Path,
    output_file: Path,
    report: PreflightReport,
    plan: Sequence[TensorWriteEntry],
    *,
    verbose: bool,
) -> None:
    output_file = _resolve_safe_output_path(output_file)
    if output_file.exists() or output_file.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_file}")
    writer: gguf.GGUFWriter | None = None
    temporary: PrivateTemporaryOutput | None = None
    try:
        writer = gguf.GGUFWriter(None, arch="qwen3")
        add_model_metadata(writer, report.config, report.dimensions)
        add_qwen2_vocab(model_dir, report.config, writer)
        register_tensor_plan(writer, plan)
        temporary = _create_private_temporary(output_file)
        writer.write_header_to_file(temporary.output)
        writer.write_kv_data_to_file()
        write_tensor_plan(writer, plan, verbose=verbose)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        raise
    else:
        assert writer is not None
        writer.close()
        writer = None
        assert temporary is not None
        _atomic_publish_no_clobber(temporary.output, output_file)
    finally:
        if temporary is not None:
            _cleanup_private_temporary(temporary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3 Row4 W1A8-INT8 weights to strict GGUF v1")
    parser.add_argument("model_dir", type=Path, help="Qwen3 Row4 checkpoint directory")
    parser.add_argument("output_file", type=Path, nargs="?", help="destination GGUF")
    parser.add_argument("--dry-run", action="store_true", help="validate without writing")
    parser.add_argument("--verbose", action="store_true", help="print per-tensor write progress")
    args = parser.parse_args(argv)

    if args.output_file is None and not args.dry_run:
        raise ValueError("output_file is required unless --dry-run is used")
    output_file: Path | None = args.output_file
    if output_file is not None:
        output_file = _resolve_safe_output_path(output_file)
        if output_file.exists() or output_file.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {output_file}")

    report = run_preflight(args.model_dir, output_file)
    print_preflight(report)
    if args.dry_run:
        return

    assert output_file is not None
    reader = TensorReader(args.model_dir, report.manifest)
    plan = build_tensor_plan(reader, report.dimensions)
    plan_bytes = sum(entry.nbytes for entry in plan)
    if plan_bytes != report.tensor_payload_bytes:
        raise AssertionError(
            f"Row4 tensor plan payload mismatch: plan={plan_bytes}, preflight={report.tensor_payload_bytes}"
        )
    row4_codes = sum(entry.nbytes for entry in plan if entry.raw_dtype == gguf.GGMLQuantizationType.ROW4_CODES)
    row4_scales = sum(
        entry.nbytes for entry in plan if entry.name.endswith(".row4.scales")
    )
    if (row4_codes, row4_scales, plan[-2].nbytes, plan[-1].nbytes) != (
        REFERENCE_ROW4_CODE_BYTES,
        REFERENCE_ROW4_SCALE_BYTES,
        REFERENCE_LM_HEAD_CODE_BYTES,
        REFERENCE_LM_HEAD_SCALE_BYTES,
    ):
        raise AssertionError("internal Row4 reference tensor byte statistics mismatch")

    write_conversion_output(
        args.model_dir,
        output_file,
        report,
        plan,
        verbose=args.verbose,
    )
    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
