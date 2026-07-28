#!/usr/bin/env python3

"""
Convert a Qwen2-based Fairy2i Hugging Face checkpoint to GGUF.

This script is intentionally separate from convert_fairy2i.py:
- tokenizer export follows the Qwen2/GPT-2-style path used by convert_hf_to_gguf.py
- RoPE base is read from config["rope_parameters"]["rope_theta"] when present
- required Q/K/V attention biases are exported and validated from safetensors headers
- the default Bundle v1 layout stores one W2 scale group per complex M64xK64 tile
  while retaining tile64_v2 as the quantization algorithm metadata
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import struct
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open

import gguf
from fairy2i.quant.tile64_v2 import (
    FAIRY2I_TILE64,
    iter_quantize_linear_to_fairy2i_bundle_v1_m64,
    quantize_linear_to_fairy2i_tile64_v2_stages,
)
from fairy2i.spec import (
    WEIGHT_LAYOUT_BUNDLE_V1,
    WEIGHT_LAYOUT_TILE64_V2,
    Fairy2IMetadata,
    write_metadata,
)
from fairy2i.tokenizer.chat_template import (
    FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN,
    FAIRY2I_DEEPSEEK_CHAT_TOKENS,
    FAIRY2I_DEEPSEEK_EOS_TOKEN,
    load_fairy2i_chat_template,
    normalize_fairy2i_chat_template_value,
)


QWEN2_PRETOKENIZER_HASHES = {
    # ref: convert_hf_to_gguf.py get_vocab_base_pre()
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c",
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea",
}

LINEAR_SPECS = (
    ("self_attn.q_proj.weight", "attn_q", "q"),
    ("self_attn.k_proj.weight", "attn_k", "k"),
    ("self_attn.v_proj.weight", "attn_v", None),
    ("self_attn.o_proj.weight", "attn_output", None),
    ("mlp.gate_proj.weight", "ffn_gate", None),
    ("mlp.up_proj.weight", "ffn_up", None),
    ("mlp.down_proj.weight", "ffn_down", None),
)

QKV_BIAS_SPECS = (
    ("self_attn.q_proj.bias", "attn_q.bias"),
    ("self_attn.k_proj.bias", "attn_k.bias"),
    ("self_attn.v_proj.bias", "attn_v.bias"),
)

SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

GGUF_ESTIMATE_OVERHEAD_BYTES = 64 * 1024 * 1024
SAFETENSORS_MAX_HEADER_BYTES = 100_000_000


@dataclass(frozen=True)
class SafetensorsTensorInfo:
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.parameter_count * SAFETENSORS_DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class SafetensorsManifest:
    weight_map: dict[str, str]
    tensors: dict[str, SafetensorsTensorInfo]
    shards: tuple[str, ...]
    index_metadata: dict[str, object]
    parameter_count: int
    tensor_bytes: int


@dataclass(frozen=True)
class Qwen2Dimensions:
    hidden_real: int
    hidden_complex: int
    ff_real: int
    ff_complex: int
    n_layer: int
    n_head: int
    n_head_kv: int
    head_dim: int
    q_dim: int
    kv_dim: int
    vocab_size: int


@dataclass(frozen=True)
class Qwen2TokenizerInfo:
    chat_template: str | list[dict[str, str]]
    token_count: int


@dataclass(frozen=True)
class Qwen2PreflightReport:
    manifest: SafetensorsManifest
    dimensions: Qwen2Dimensions
    tokenizer: Qwen2TokenizerInfo
    qat_linear_count: int
    qkv_bias_count: int
    estimated_output_bytes: int
    available_output_bytes: int


def _preview(items: list[str], *, limit: int = 8) -> str:
    preview = ", ".join(items[:limit])
    if len(items) > limit:
        preview += f", ... ({len(items)} total)"
    return preview


def _read_safetensors_header(path: Path) -> dict[str, SafetensorsTensorInfo]:
    file_size = path.stat().st_size
    with path.open("rb") as file:
        header_size_bytes = file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors file {path}: missing 8-byte header length")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size == 0 or header_size > file_size - 8 or header_size > SAFETENSORS_MAX_HEADER_BYTES:
            raise ValueError(
                f"invalid safetensors header length in {path}: {header_size} for {file_size}-byte file"
            )
        header_bytes = file.read(header_size)

    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header in {path}: expected an object")

    payload_size = file_size - 8 - header_size
    tensors: dict[str, SafetensorsTensorInfo] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or not all(
                isinstance(metadata_name, str) and isinstance(metadata_value, str)
                for metadata_name, metadata_value in entry.items()
            ):
                raise ValueError(f"invalid __metadata__ entry in {path}")
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError(f"invalid tensor header entry in {path}: {name!r}")

        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype for {name} in {path}: {dtype!r}")
        if not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape):
            raise ValueError(f"invalid safetensors shape for {name} in {path}: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int and offset >= 0 for offset in offsets)
        ):
            raise ValueError(f"invalid safetensors data_offsets for {name} in {path}: {offsets!r}")

        start, end = offsets
        expected_nbytes = math.prod(shape) * SAFETENSORS_DTYPE_BYTES[dtype]
        if end < start or end - start != expected_nbytes:
            raise ValueError(
                f"safetensors byte range mismatch for {name} in {path}: "
                f"offsets={offsets}, shape={shape}, dtype={dtype}, expected_bytes={expected_nbytes}"
            )
        if end > payload_size:
            raise ValueError(
                f"safetensors byte range for {name} exceeds payload in {path}: {end} > {payload_size}"
            )

        tensors[name] = SafetensorsTensorInfo(
            shard=path.name,
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(start, end),
        )
        intervals.append((start, end, name))

    expected_start = 0
    for start, end, name in sorted(intervals):
        if start != expected_start:
            raise ValueError(
                f"non-contiguous safetensors payload in {path}: tensor {name} starts at {start}, "
                f"expected {expected_start}"
            )
        expected_start = end
    if expected_start != payload_size:
        raise ValueError(
            f"safetensors payload size mismatch in {path}: tensors cover {expected_start} bytes, "
            f"payload has {payload_size}"
        )

    return tensors


def _load_safetensors_index(model_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"required sharded checkpoint index not found: {index_file}")

    try:
        index = json.loads(index_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint index JSON in {index_file}: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"invalid checkpoint index in {index_file}: expected an object")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_file}")
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid metadata in {index_file}")
    return dict(weight_map), dict(metadata)


def load_safetensors_manifest(model_dir: Path, *, expected_shard_count: int = 2) -> SafetensorsManifest:
    weight_map, index_metadata = _load_safetensors_index(model_dir)
    shards = tuple(sorted(set(weight_map.values())))
    if len(shards) != expected_shard_count:
        raise ValueError(
            f"expected exactly {expected_shard_count} safetensors shards, "
            f"index references {len(shards)}: {list(shards)}"
        )

    disk_shards = {path.name for path in model_dir.glob("*.safetensors") if path.is_file()}
    referenced_shards = set(shards)
    missing_shards = sorted(referenced_shards - disk_shards)
    unexpected_shards = sorted(disk_shards - referenced_shards)
    if missing_shards or unexpected_shards:
        details = []
        if missing_shards:
            details.append(f"missing={missing_shards}")
        if unexpected_shards:
            details.append(f"unexpected={unexpected_shards}")
        raise ValueError("safetensors shard set does not match the index: " + ", ".join(details))

    tensors: dict[str, SafetensorsTensorInfo] = {}
    duplicates: list[str] = []
    for shard in shards:
        for name, info in _read_safetensors_header(model_dir / shard).items():
            if name in tensors:
                duplicates.append(name)
            tensors[name] = info
    if duplicates:
        raise ValueError(f"duplicate tensors across safetensors shards: {_preview(sorted(duplicates))}")

    missing_from_headers = sorted(set(weight_map) - set(tensors))
    missing_from_index = sorted(set(tensors) - set(weight_map))
    wrong_shards = sorted(
        name for name, info in tensors.items() if weight_map.get(name) is not None and weight_map[name] != info.shard
    )
    if missing_from_headers or missing_from_index or wrong_shards:
        details = []
        if missing_from_headers:
            details.append(f"missing from headers: {_preview(missing_from_headers)}")
        if missing_from_index:
            details.append(f"missing from index: {_preview(missing_from_index)}")
        if wrong_shards:
            details.append(f"wrong shard mapping: {_preview(wrong_shards)}")
        raise ValueError("checkpoint index/header mismatch: " + "; ".join(details))

    parameter_count = sum(info.parameter_count for info in tensors.values())
    tensor_bytes = sum(info.nbytes for info in tensors.values())
    declared_size = index_metadata.get("total_size")
    if declared_size is not None and (not isinstance(declared_size, int) or declared_size != tensor_bytes):
        raise ValueError(
            f"checkpoint index total_size mismatch: metadata={declared_size!r}, "
            f"computed_from_headers={tensor_bytes}"
        )

    return SafetensorsManifest(
        weight_map=weight_map,
        tensors=tensors,
        shards=shards,
        index_metadata=index_metadata,
        parameter_count=parameter_count,
        tensor_bytes=tensor_bytes,
    )


def _positive_config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen2 config field {key!r} must be a positive integer, got {value!r}")
    return value


def _positive_config_float(config: Mapping[str, object], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Qwen2 config field {key!r} must be a positive finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Qwen2 config field {key!r} must be a positive finite number, got {value!r}")
    return result


def get_qwen2_dimensions(config: Mapping[str, object]) -> Qwen2Dimensions:
    if config.get("model_type") != "qwen2":
        raise ValueError(f"expected model_type=qwen2, got {config.get('model_type')!r}")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen2ForCausalLM" not in architectures:
        raise ValueError(f"expected Qwen2ForCausalLM architecture, got {architectures!r}")

    hidden_real = _positive_config_int(config, "hidden_size")
    ff_real = _positive_config_int(config, "intermediate_size")
    n_layer = _positive_config_int(config, "num_hidden_layers")
    n_head = _positive_config_int(config, "num_attention_heads")
    n_head_kv = _positive_config_int(config, "num_key_value_heads")
    vocab_size = _positive_config_int(config, "vocab_size")

    if hidden_real % 2 != 0:
        raise ValueError(f"hidden_size must be even for complex weights, got {hidden_real}")
    if ff_real % 2 != 0:
        raise ValueError(f"intermediate_size must be even for complex weights, got {ff_real}")
    if n_head_kv > n_head or n_head % n_head_kv != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads for Qwen2 GQA: "
            f"num_attention_heads={n_head}, num_key_value_heads={n_head_kv}"
        )
    if hidden_real % n_head != 0 and "head_dim" not in config:
        raise ValueError(
            f"hidden_size={hidden_real} is not divisible by num_attention_heads={n_head} "
            "and config has no head_dim"
        )

    head_dim_value = config.get("head_dim", hidden_real // n_head)
    if not isinstance(head_dim_value, int) or isinstance(head_dim_value, bool) or head_dim_value <= 0:
        raise ValueError(f"Qwen2 config field 'head_dim' must be a positive integer, got {head_dim_value!r}")
    head_dim = head_dim_value
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    if q_dim != hidden_real:
        raise ValueError(
            f"Qwen2 q projection dimension must equal hidden_size for this checkpoint: "
            f"num_attention_heads * head_dim = {q_dim}, hidden_size = {hidden_real}"
        )

    return Qwen2Dimensions(
        hidden_real=hidden_real,
        hidden_complex=hidden_real // 2,
        ff_real=ff_real,
        ff_complex=ff_real // 2,
        n_layer=n_layer,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        q_dim=q_dim,
        kv_dim=kv_dim,
        vocab_size=vocab_size,
    )


def expected_qwen2_tensor_shapes(dimensions: Qwen2Dimensions) -> dict[str, tuple[int, ...]]:
    hidden_real = dimensions.hidden_real
    ff_real = dimensions.ff_real
    q_dim = dimensions.q_dim
    kv_dim = dimensions.kv_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (dimensions.vocab_size, hidden_real),
        "model.norm.weight": (hidden_real,),
        "lm_head.weight": (dimensions.vocab_size, hidden_real),
    }
    linear_shapes = {
        "self_attn.q_proj.weight": (q_dim, hidden_real),
        "self_attn.k_proj.weight": (kv_dim, hidden_real),
        "self_attn.v_proj.weight": (kv_dim, hidden_real),
        "self_attn.o_proj.weight": (hidden_real, q_dim),
        "mlp.gate_proj.weight": (ff_real, hidden_real),
        "mlp.up_proj.weight": (ff_real, hidden_real),
        "mlp.down_proj.weight": (hidden_real, ff_real),
    }
    bias_shapes = {
        "self_attn.q_proj.bias": (q_dim,),
        "self_attn.k_proj.bias": (kv_dim,),
        "self_attn.v_proj.bias": (kv_dim,),
    }

    for il in range(dimensions.n_layer):
        prefix = f"model.layers.{il}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden_real,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden_real,)
        shapes.update({f"{prefix}.{suffix}": shape for suffix, shape in linear_shapes.items()})
        shapes.update({f"{prefix}.{suffix}": shape for suffix, shape in bias_shapes.items()})
    return shapes


def qwen2_qat_linear_keys(dimensions: Qwen2Dimensions) -> list[str]:
    return [
        f"model.layers.{il}.{suffix}"
        for il in range(dimensions.n_layer)
        for suffix, _, _ in LINEAR_SPECS
    ]


def qwen2_qkv_bias_keys(dimensions: Qwen2Dimensions) -> list[str]:
    return [
        f"model.layers.{il}.{suffix}"
        for il in range(dimensions.n_layer)
        for suffix, _ in QKV_BIAS_SPECS
    ]


def validate_qwen2_checkpoint(
    config: Mapping[str, object],
    manifest: SafetensorsManifest,
) -> tuple[Qwen2Dimensions, int, int]:
    dimensions = get_qwen2_dimensions(config)
    expected_shapes = expected_qwen2_tensor_shapes(dimensions)
    actual_names = set(manifest.tensors)
    expected_names = set(expected_shapes)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required tensors: {_preview(missing)}")
        if unexpected:
            details.append(f"unexpected tensors: {_preview(unexpected)}")
        raise ValueError("invalid Qwen2 Fairy2i checkpoint tensor set: " + "; ".join(details))

    non_bf16 = sorted(name for name, info in manifest.tensors.items() if info.dtype != "BF16")
    if non_bf16:
        details = [f"{name}={manifest.tensors[name].dtype}" for name in non_bf16]
        raise ValueError(f"all checkpoint tensors must be BF16: {_preview(details)}")

    bad_shapes = sorted(
        name for name, shape in expected_shapes.items() if manifest.tensors[name].shape != shape
    )
    if bad_shapes:
        details = [
            f"{name}: expected {expected_shapes[name]}, got {manifest.tensors[name].shape}"
            for name in bad_shapes
        ]
        raise ValueError(f"Qwen2 checkpoint tensor shape mismatch: {_preview(details)}")

    linear_keys = qwen2_qat_linear_keys(dimensions)
    bad_alignment = []
    for name in linear_keys + ["lm_head.weight"]:
        shape = manifest.tensors[name].shape
        if len(shape) != 2 or shape[0] % 2 != 0 or shape[1] % 2 != 0:
            bad_alignment.append(f"{name}={shape} (dimensions must be even)")
            continue
        complex_shape = (shape[0] // 2, shape[1] // 2)
        if complex_shape[0] % FAIRY2I_TILE64 != 0 or complex_shape[1] % FAIRY2I_TILE64 != 0:
            bad_alignment.append(f"{name}={shape} (complex shape {complex_shape} is not M64/K64 aligned)")
    if bad_alignment:
        raise ValueError(f"Qwen2 Fairy2i weights must use complete M64xK64 tiles: {_preview(bad_alignment)}")

    bias_keys = qwen2_qkv_bias_keys(dimensions)
    expected_tensor_count = 3 + dimensions.n_layer * (
        2 + len(LINEAR_SPECS) + len(QKV_BIAS_SPECS)
    )
    if len(manifest.tensors) != expected_tensor_count:
        raise ValueError(
            f"Qwen2 checkpoint tensor count mismatch: expected {expected_tensor_count}, "
            f"got {len(manifest.tensors)}"
        )
    expected_linear_count = dimensions.n_layer * len(LINEAR_SPECS)
    if len(linear_keys) != expected_linear_count:
        raise AssertionError("internal Qwen2 linear count mismatch")
    expected_bias_count = dimensions.n_layer * len(QKV_BIAS_SPECS)
    if len(bias_keys) != expected_bias_count:
        raise AssertionError("internal Qwen2 bias count mismatch")

    return dimensions, expected_linear_count, expected_bias_count


def _chat_template_strings(chat_template: str | list[dict[str, str]]) -> list[str]:
    if isinstance(chat_template, str):
        if not chat_template.strip():
            raise ValueError("Qwen2 chat template must not be empty")
        return [chat_template]
    if not chat_template:
        raise ValueError("Qwen2 chat template choices must not be empty")

    names: set[str] = set()
    templates: list[str] = []
    for choice in chat_template:
        name = choice.get("name")
        template = choice.get("template")
        if (
            not isinstance(name, str)
            or not name
            or not all(character.isascii() and (character.isalnum() or character == "_") for character in name)
        ):
            raise ValueError(
                "all Qwen2 chat template choices must have a non-empty ASCII alphanumeric/underscore string 'name'"
            )
        if name in names:
            raise ValueError(f"duplicate Qwen2 chat template choice name: {name}")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("all Qwen2 chat template choices must contain a non-empty string 'template'")
        names.add(name)
        templates.append(template)
    if "default" not in names:
        raise ValueError("Qwen2 chat template choices must include a 'default' template")
    return templates


def validate_qwen2_tokenizer(
    model_dir: Path,
    config: Mapping[str, object],
) -> Qwen2TokenizerInfo:
    tokenizer_json_file = model_dir / "tokenizer.json"
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    if not tokenizer_json_file.is_file():
        raise FileNotFoundError(f"required Qwen2 tokenizer file not found: {tokenizer_json_file}")
    if not tokenizer_config_file.is_file():
        raise FileNotFoundError(f"required Qwen2 tokenizer config not found: {tokenizer_config_file}")

    try:
        tokenizer_json = json.loads(tokenizer_json_file.read_text(encoding="utf-8"))
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Qwen2 tokenizer JSON: {exc}") from exc
    if not isinstance(tokenizer_json, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("Qwen2 tokenizer.json and tokenizer_config.json must contain JSON objects")

    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not all(
        isinstance(token, str) and isinstance(token_id, int) and not isinstance(token_id, bool)
        for token, token_id in vocab.items()
    ):
        raise ValueError(f"invalid vocab in {tokenizer_json_file}")
    added_tokens = tokenizer_json.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError(f"invalid added_tokens in {tokenizer_json_file}")

    id_to_token: dict[int, str] = {}
    for token, token_id in vocab.items():
        if token_id < 0:
            raise ValueError(f"negative token id for {token!r}: {token_id}")
        if token_id in id_to_token and id_to_token[token_id] != token:
            raise ValueError(f"duplicate tokenizer id {token_id}: {id_to_token[token_id]!r}, {token!r}")
        id_to_token[token_id] = token
    for entry in added_tokens:
        if not isinstance(entry, dict):
            raise ValueError("Qwen2 added_tokens entries must be objects")
        token = entry.get("content")
        token_id = entry.get("id")
        if not isinstance(token, str) or not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"invalid Qwen2 added token entry: {entry!r}")
        if token_id in id_to_token and id_to_token[token_id] != token:
            raise ValueError(f"duplicate tokenizer id {token_id}: {id_to_token[token_id]!r}, {token!r}")
        id_to_token[token_id] = token

    vocab_size = _positive_config_int(config, "vocab_size")
    out_of_range = sorted(token_id for token_id in id_to_token if token_id >= vocab_size)
    if out_of_range:
        raise ValueError(
            f"tokenizer ids must be below vocab_size={vocab_size}: "
            f"{_preview([str(token_id) for token_id in out_of_range])}"
        )

    for config_key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = config.get(config_key)
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id not in id_to_token:
            raise ValueError(f"{config_key}={token_id!r} is not present in the tokenizer")
    missing_chat_tokens = [token for token in FAIRY2I_DEEPSEEK_CHAT_TOKENS if token not in id_to_token.values()]
    if missing_chat_tokens:
        raise ValueError(f"Qwen2 tokenizer is missing Fairy2i chat tokens: {missing_chat_tokens}")

    chat_template = load_fairy2i_chat_template(model_dir, tokenizer_config)
    if chat_template is None:
        raise ValueError("Qwen2 Fairy2i checkpoint is missing a chat template")
    chat_template = normalize_fairy2i_chat_template_value(chat_template)
    for template in _chat_template_strings(chat_template):
        generation_marker = template.rfind("add_generation_prompt")
        assistant_marker = template.rfind(FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN)
        eos_marker = template.rfind(FAIRY2I_DEEPSEEK_EOS_TOKEN)
        if generation_marker < 0 or assistant_marker < generation_marker or eos_marker > assistant_marker:
            raise ValueError(
                "Qwen2 Fairy2i chat template must end generation prompts with "
                f"{FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN}"
            )

    return Qwen2TokenizerInfo(chat_template=chat_template, token_count=len(id_to_token))


def _wide_linear_storage_bytes(shape: tuple[int, ...], weight_layout: str) -> int:
    out_complex = shape[0] // 2
    in_complex = shape[1] // 2
    tile_count = out_complex // FAIRY2I_TILE64 * (in_complex // FAIRY2I_TILE64)
    if weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        return tile_count * (4096 + 16)
    if weight_layout == WEIGHT_LAYOUT_TILE64_V2:
        return tile_count * 5120
    raise ValueError(f"unsupported Fairy2i weight layout: {weight_layout}")


def estimate_qwen2_gguf_bytes(
    dimensions: Qwen2Dimensions,
    *,
    output_layer: str,
    weight_layout: str,
) -> int:
    shapes = expected_qwen2_tensor_shapes(dimensions)
    tensor_bytes = math.prod(shapes["model.embed_tokens.weight"]) * 2
    tensor_bytes += math.prod(shapes["model.norm.weight"]) * 4
    for il in range(dimensions.n_layer):
        tensor_bytes += dimensions.hidden_real * 4 * 2
        for suffix, _, _ in LINEAR_SPECS:
            tensor_bytes += _wide_linear_storage_bytes(shapes[f"model.layers.{il}.{suffix}"], weight_layout)
        for suffix, _ in QKV_BIAS_SPECS:
            tensor_bytes += math.prod(shapes[f"model.layers.{il}.{suffix}"]) * 4

    output_shape = shapes["lm_head.weight"]
    if output_layer in ("wide-linear", "both"):
        tensor_bytes += _wide_linear_storage_bytes(output_shape, weight_layout)
    if output_layer in ("dense", "both"):
        tensor_bytes += math.prod(output_shape) * 2
    return tensor_bytes + GGUF_ESTIMATE_OVERHEAD_BYTES


def run_qwen2_preflight(
    model_dir: Path,
    *,
    output_file: Path | None,
    output_layer: str,
    weight_layout: str,
) -> Qwen2PreflightReport:
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"required Qwen2 config not found: {config_file}")
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Qwen2 config JSON in {config_file}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"invalid Qwen2 config in {config_file}: expected an object")

    manifest = load_safetensors_manifest(model_dir)
    dimensions, linear_count, bias_count = validate_qwen2_checkpoint(config, manifest)
    tokenizer = validate_qwen2_tokenizer(model_dir, config)
    _positive_config_int(config, "max_position_embeddings")
    _positive_config_float(config, "rms_norm_eps")
    rope_theta = get_rope_theta(dict(config))
    if not math.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError(f"Qwen2 RoPE theta must be a positive finite number, got {rope_theta!r}")
    estimated_output_bytes = estimate_qwen2_gguf_bytes(
        dimensions,
        output_layer=output_layer,
        weight_layout=weight_layout,
    )
    output_parent = output_file.parent if output_file is not None else model_dir
    if not output_parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output_parent}")
    available_output_bytes = shutil.disk_usage(output_parent).free
    if output_file is not None and available_output_bytes < estimated_output_bytes:
        raise ValueError(
            f"insufficient disk space for estimated GGUF output: need {estimated_output_bytes} bytes, "
            f"available {available_output_bytes} bytes under {output_parent}"
        )

    return Qwen2PreflightReport(
        manifest=manifest,
        dimensions=dimensions,
        tokenizer=tokenizer,
        qat_linear_count=linear_count,
        qkv_bias_count=bias_count,
        estimated_output_bytes=estimated_output_bytes,
        available_output_bytes=available_output_bytes,
    )


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def print_qwen2_preflight(report: Qwen2PreflightReport) -> None:
    manifest = report.manifest
    dimensions = report.dimensions
    index_parameters = manifest.index_metadata.get("total_parameters")
    print(
        "Fairy2i Qwen2 preflight passed: "
        f"shards={len(manifest.shards)} tensors={len(manifest.tensors)} BF16={len(manifest.tensors)} "
        f"qat_linears={report.qat_linear_count} qkv_biases={report.qkv_bias_count}"
    )
    print(
        f"parameters={manifest.parameter_count:,} tensor_bytes={manifest.tensor_bytes:,} "
        f"({_format_bytes(manifest.tensor_bytes)}), computed from safetensors headers"
    )
    if index_parameters is not None:
        print(f"index metadata total_parameters={index_parameters!r} ignored")
    print(
        f"hidden_complex={dimensions.hidden_complex} ff_complex={dimensions.ff_complex} "
        f"estimated_output={_format_bytes(report.estimated_output_bytes)} "
        f"available={_format_bytes(report.available_output_bytes)}"
    )


def token_looks_special(token: str | bytes) -> bool:
    if isinstance(token, bytes):
        token_text = token.decode("utf-8")
    else:
        token_text = token

    seems_special = token_text in (
        "<pad>",
        "<mask>",
        "<2mass>",
        "[@BOS@]",
    )
    seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
    seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))
    seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))
    return seems_special


def get_qwen2_tokenizer_pre(model_dir: Path) -> str:
    chktxt = (
        "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n"
        "🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 "
        "3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= "
        "нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, "
        "'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"
    )

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        chktok = tokenizer.encode(chktxt).ids
        chkhsh = sha256(str(chktok).encode()).hexdigest()
        if chkhsh in QWEN2_PRETOKENIZER_HASHES:
            return "qwen2"

        print(
            f"warning: unrecognized Qwen2 tokenizer pre hash {chkhsh}, falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"warning: failed to evaluate Qwen2 tokenizer pre-tokenizer via tokenizers ({exc}), "
            "falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )

    return "qwen2"


def set_vocab_qwen2(
    model_dir: Path,
    config: dict,
    writer: gguf.GGUFWriter,
    tokenizer_info: Qwen2TokenizerInfo | None = None,
) -> None:
    tokenizer_json_file = model_dir / "tokenizer.json"
    if not tokenizer_json_file.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

    tokenizer_json = json.loads(tokenizer_json_file.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"invalid vocab in {tokenizer_json_file}")
    assert max(vocab.values()) < vocab_size

    tokpre = get_qwen2_tokenizer_pre(model_dir)
    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
    added_tokens = tokenizer_json.get("added_tokens", [])

    added_vocab: dict[str, int] = {}
    added_tokens_decoder: dict[int, dict] = {}
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if not isinstance(item, dict):
                continue
            token = item.get("content")
            token_id = item.get("id")
            if not isinstance(token, str) or not isinstance(token_id, int):
                continue
            added_vocab[token] = token_id
            added_tokens_decoder[token_id] = item
            reverse_vocab[token_id] = token

    tokens: list[str] = []
    toktypes: list[int] = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        token = reverse_vocab[i]
        if token in added_vocab:
            decoder_entry = added_tokens_decoder.get(i)
            is_special = bool((decoder_entry or {}).get("special", False)) or token_looks_special(token)
            toktypes.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

        tokens.append(token)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(tokpre)
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, load_merges=True)
    if tokenizer_info is not None:
        special_vocab.chat_template = tokenizer_info.chat_template
    elif special_vocab.chat_template is not None:
        special_vocab.chat_template = normalize_fairy2i_chat_template_value(special_vocab.chat_template)
    special_vocab.add_to_gguf(writer)

    tokenizer_config_file = model_dir / "tokenizer_config.json"
    if tokenizer_config_file.is_file():
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))
        if "add_prefix_space" in tokenizer_config:
            writer.add_add_space_prefix(tokenizer_config["add_prefix_space"])


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_file}")
        return {k: v for k, v in weight_map.items()}

    model_files = sorted(model_dir.glob("*.safetensors"))
    if len(model_files) != 1:
        raise ValueError("no shard index and cannot infer a single safetensors file")

    filename = model_files[0].name
    with safe_open(str(model_files[0]), framework="pt", device="cpu") as f:
        return {key: filename for key in f.keys()}


class TensorReader:
    def __init__(
        self,
        model_dir: Path,
        weight_map: Dict[str, str],
        tensor_info: Mapping[str, SafetensorsTensorInfo] | None = None,
    ):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self.tensor_info = tensor_info

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def shape(self, key: str) -> tuple[int, ...]:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        if self.tensor_info is not None:
            return self.tensor_info[key].shape
        filename = self.weight_map[key]
        with safe_open(str(self.model_dir / filename), framework="pt", device="cpu") as file:
            return tuple(file.get_slice(key).get_shape())

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        filename = self.weight_map[key]
        path = self.model_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    @contextmanager
    def open_tensor_source(
        self,
        key: str,
    ) -> Iterator[Callable[[slice, slice], torch.Tensor]]:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        filename = self.weight_map[key]
        path = self.model_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as file:
            tensor_slice = file.get_slice(key)

            def source(row_slice: slice, col_slice: slice) -> torch.Tensor:
                return tensor_slice[row_slice, col_slice]

            yield source


def pack_token_embedding(embed: torch.Tensor, hidden_complex: int) -> np.ndarray:
    real = embed[:, :hidden_complex].to(torch.float32)
    imag = embed[:, hidden_complex:].to(torch.float32)

    real_bits = real.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    imag_bits = imag.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)

    packed = ((imag_bits << 16) | (real_bits & 0xFFFF)).to(torch.int32).view(torch.float32)
    return packed.cpu().numpy()


def get_rope_theta(config: dict) -> float:
    rope_params = config.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    return 10000.0


@dataclass(frozen=True)
class TensorWriteEntry:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int
    raw_dtype: gguf.GGMLQuantizationType | None
    chunks: Callable[[], Iterator[np.ndarray]]


def _tensor_write_entry(
    name: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    chunks: Callable[[], Iterator[np.ndarray]],
    *,
    raw_dtype: gguf.GGMLQuantizationType | None = None,
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


def _iter_tensor_f32(reader: TensorReader, key: str) -> Iterator[np.ndarray]:
    tensor = reader.get(key)
    data = tensor.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    yield np.ascontiguousarray(data)


def _iter_tensor_rows(
    reader: TensorReader,
    key: str,
    *,
    dtype: torch.dtype,
    rows_per_chunk: int = 1024,
) -> Iterator[np.ndarray]:
    shape = reader.shape(key)
    if len(shape) != 2:
        raise ValueError(f"row streaming requires a 2D tensor, got {key} with shape {shape}")
    with reader.open_tensor_source(key) as source:
        for row_start in range(0, shape[0], rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, shape[0])
            chunk = source(slice(row_start, row_end), slice(0, shape[1]))
            yield np.ascontiguousarray(chunk.to(dtype).cpu().numpy())


def _iter_packed_embedding_rows(
    reader: TensorReader,
    key: str,
    hidden_complex: int,
    *,
    rows_per_chunk: int = 1024,
) -> Iterator[np.ndarray]:
    shape = reader.shape(key)
    if shape != (shape[0], hidden_complex * 2):
        raise ValueError(
            f"token embedding shape mismatch while streaming: expected second dimension "
            f"{hidden_complex * 2}, got {shape}"
        )
    with reader.open_tensor_source(key) as source:
        for row_start in range(0, shape[0], rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, shape[0])
            chunk = source(slice(row_start, row_end), slice(0, shape[1]))
            yield np.ascontiguousarray(pack_token_embedding(chunk, hidden_complex))


class BundleM64TensorWriter:
    def __init__(
        self,
        reader: TensorReader,
        key: str,
        out_target: int,
        in_target: int,
    ):
        self.reader = reader
        self.key = key
        self.out_target = out_target
        self.in_target = in_target
        self._scale_chunks: list[np.ndarray] | None = None

    def iter_codes(self) -> Iterator[np.ndarray]:
        if self._scale_chunks is not None:
            raise RuntimeError(f"Bundle scale chunks for {self.key} were not consumed")
        scale_chunks: list[np.ndarray] = []
        with self.reader.open_tensor_source(self.key) as source:
            for codes, scales in iter_quantize_linear_to_fairy2i_bundle_v1_m64(
                source,
                self.out_target,
                self.in_target,
                weight_shape=self.reader.shape(self.key),
            ):
                scale_chunks.append(scales)
                yield codes
        self._scale_chunks = scale_chunks

    def iter_scales(self) -> Iterator[np.ndarray]:
        if self._scale_chunks is None:
            raise RuntimeError(f"Bundle codes for {self.key} must be written before scales")
        scale_chunks = self._scale_chunks
        self._scale_chunks = None
        yield from scale_chunks


class Tile64V2TensorWriter:
    def __init__(
        self,
        reader: TensorReader,
        key: str,
        out_target: int,
        in_target: int,
    ):
        self.reader = reader
        self.key = key
        self.out_target = out_target
        self.in_target = in_target
        self._stages: dict[str, np.ndarray] | None = None

    def iter_stage(self, stage_name: str) -> Iterator[np.ndarray]:
        if self._stages is None:
            weight = self.reader.get(self.key)
            self._stages = quantize_linear_to_fairy2i_tile64_v2_stages(
                weight,
                self.out_target,
                self.in_target,
            )
            del weight
        if stage_name not in self._stages:
            raise RuntimeError(f"tile64_v2 stage {stage_name} for {self.key} was already consumed")
        yield self._stages.pop(stage_name)
        if not self._stages:
            self._stages = None
            gc.collect()


def _add_wide_linear_plan(
    plan: list[TensorWriteEntry],
    reader: TensorReader,
    *,
    hf_key: str,
    gguf_base: str,
    out_target: int,
    in_target: int,
    weight_layout: str,
) -> None:
    tile_count = out_target // FAIRY2I_TILE64 * (in_target // FAIRY2I_TILE64)
    if weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        bundle_writer = BundleM64TensorWriter(reader, hf_key, out_target, in_target)
        plan.append(
            _tensor_write_entry(
                f"{gguf_base}.bundle.codes",
                (tile_count, FAIRY2I_TILE64, 4, 16),
                np.uint8,
                bundle_writer.iter_codes,
                raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES,
            )
        )
        plan.append(
            _tensor_write_entry(
                f"{gguf_base}.bundle.scales",
                (tile_count, 4, 2),
                np.float16,
                bundle_writer.iter_scales,
            )
        )
        return

    if weight_layout != WEIGHT_LAYOUT_TILE64_V2:
        raise ValueError(f"unsupported Fairy2i weight layout: {weight_layout}")
    tile_writer = Tile64V2TensorWriter(reader, hf_key, out_target, in_target)
    packed_shape = (out_target, in_target // FAIRY2I_TILE64 * 20)
    for stage_name in ("U.s0", "U.s1", "W.s0", "W.s1"):
        plan.append(
            _tensor_write_entry(
                f"{gguf_base}.{stage_name}",
                packed_shape,
                np.uint8,
                lambda stage_name=stage_name: tile_writer.iter_stage(stage_name),
                raw_dtype=gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2,
            )
        )


def build_qwen2_tensor_write_plan(
    reader: TensorReader,
    dimensions: Qwen2Dimensions,
    *,
    output_layer: str,
    weight_layout: str,
) -> list[TensorWriteEntry]:
    plan: list[TensorWriteEntry] = []
    plan.append(
        _tensor_write_entry(
            "token_embd",
            (dimensions.vocab_size, dimensions.hidden_complex),
            np.float32,
            lambda: _iter_packed_embedding_rows(
                reader,
                "model.embed_tokens.weight",
                dimensions.hidden_complex,
            ),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
    )
    plan.append(
        _tensor_write_entry(
            "output_norm",
            (dimensions.hidden_real,),
            np.float32,
            lambda: _iter_tensor_f32(reader, "model.norm.weight"),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
    )

    output_shape = reader.shape("lm_head.weight")
    output_out_c = output_shape[0] // 2
    output_in_c = output_shape[1] // 2
    if output_layer in ("wide-linear", "both"):
        _add_wide_linear_plan(
            plan,
            reader,
            hf_key="lm_head.weight",
            gguf_base="output",
            out_target=output_out_c,
            in_target=output_in_c,
            weight_layout=weight_layout,
        )
    if output_layer in ("dense", "both"):
        plan.append(
            _tensor_write_entry(
                "output",
                output_shape,
                np.float16,
                lambda: _iter_tensor_rows(reader, "lm_head.weight", dtype=torch.float16),
                raw_dtype=gguf.GGMLQuantizationType.F16,
            )
        )

    for il in range(dimensions.n_layer):
        layer_prefix = f"model.layers.{il}"
        plan.append(
            _tensor_write_entry(
                f"blk.{il}.attn_norm",
                (dimensions.hidden_real,),
                np.float32,
                lambda il=il: _iter_tensor_f32(
                    reader,
                    f"model.layers.{il}.input_layernorm.weight",
                ),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )
        )
        plan.append(
            _tensor_write_entry(
                f"blk.{il}.ffn_norm",
                (dimensions.hidden_real,),
                np.float32,
                lambda il=il: _iter_tensor_f32(
                    reader,
                    f"model.layers.{il}.post_attention_layernorm.weight",
                ),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )
        )

        for hf_suffix, gguf_base, _ in LINEAR_SPECS:
            hf_key = f"{layer_prefix}.{hf_suffix}"
            weight_shape = reader.shape(hf_key)
            _add_wide_linear_plan(
                plan,
                reader,
                hf_key=hf_key,
                gguf_base=f"blk.{il}.{gguf_base}",
                out_target=weight_shape[0] // 2,
                in_target=weight_shape[1] // 2,
                weight_layout=weight_layout,
            )

        for hf_suffix, gguf_name in QKV_BIAS_SPECS:
            hf_key = f"{layer_prefix}.{hf_suffix}"
            plan.append(
                _tensor_write_entry(
                    f"blk.{il}.{gguf_name}",
                    reader.shape(hf_key),
                    np.float32,
                    lambda hf_key=hf_key: _iter_tensor_f32(reader, hf_key),
                    raw_dtype=gguf.GGMLQuantizationType.F32,
                )
            )

    return plan


def register_qwen2_tensor_write_plan(
    writer: gguf.GGUFWriter,
    plan: list[TensorWriteEntry],
) -> None:
    for entry in plan:
        writer.add_tensor_info(
            entry.name,
            entry.shape,
            entry.dtype,
            entry.nbytes,
            raw_dtype=entry.raw_dtype,
        )


def write_qwen2_tensor_write_plan(
    writer: gguf.GGUFWriter,
    plan: list[TensorWriteEntry],
    *,
    verbose: bool,
) -> None:
    writer.write_ti_data_to_file()
    for index, entry in enumerate(plan, start=1):
        if verbose:
            print(f"writing tensor {index}/{len(plan)}: {entry.name}")
        writer.write_tensor_data_stream(entry.chunks())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2-based Fairy2i Hugging Face weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to Qwen2-based Fairy2i model directory")
    parser.add_argument("output_file", type=Path, nargs="?", help="Output GGUF file path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint headers, tokenizer, shapes, and disk space without writing GGUF",
    )
    parser.add_argument(
        "--residual-steps",
        type=int,
        default=2,
        help="Residual quantization steps (only 2 is supported)",
    )
    parser.add_argument(
        "--output-layer",
        choices=["wide-linear", "dense", "both"],
        default="wide-linear",
        help="Output projection storage: wide-linear (default), dense, or both (for A/B debugging)",
    )
    parser.add_argument(
        "--qk-permute",
        action="store_true",
        help="Unsupported compatibility flag; Qwen2 Fairy2i weights must retain their stored Q/K row order",
    )
    parser.add_argument(
        "--no-attn-bias",
        action="store_true",
        help="Unsupported compatibility flag; Qwen2 Fairy2i requires all Q/K/V bias tensors",
    )
    parser.add_argument(
        "--quant-variant",
        choices=["tile64_v2"],
        default="tile64_v2",
        help="Quantization/export variant. tile64_v2 matches the training-side QAT kernel.",
    )
    parser.add_argument(
        "--weight-layout",
        choices=[WEIGHT_LAYOUT_BUNDLE_V1],
        default=WEIGHT_LAYOUT_BUNDLE_V1,
        help="Fairy2i weight storage layout (Qwen2 supports Bundle v1 only)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args(argv)

    if args.residual_steps != 2:
        raise ValueError("only --residual-steps 2 is currently supported")
    if args.weight_layout != WEIGHT_LAYOUT_BUNDLE_V1:
        raise ValueError("the Qwen2 Fairy2i converter only supports --weight-layout bundle_v1")
    if args.qk_permute:
        raise ValueError(
            "--qk-permute is forbidden for the Qwen2 Fairy2i checkpoint; "
            "Q/K weights must retain their training-time row order"
        )
    if args.no_attn_bias:
        raise ValueError(
            "--no-attn-bias is forbidden for the Qwen2 Fairy2i checkpoint; "
            "all Q/K/V biases are required"
        )
    if args.output_file is None and not args.dry_run:
        raise ValueError("output_file is required unless --dry-run is set")
    if not args.dry_run and args.output_file is not None and (
        args.output_file.exists() or args.output_file.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_file}")

    model_dir: Path = args.model_dir
    report = run_qwen2_preflight(
        model_dir,
        output_file=args.output_file,
        output_layer=args.output_layer,
        weight_layout=args.weight_layout,
    )
    print_qwen2_preflight(report)
    if args.dry_run:
        return

    output_file = args.output_file
    assert output_file is not None
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dimensions = report.dimensions
    rope_theta = get_rope_theta(config)
    if args.verbose:
        print(f"rope_theta={rope_theta}")
        print(f"output_layer={args.output_layer}, weight_layout={args.weight_layout}")

    reader = TensorReader(
        model_dir,
        report.manifest.weight_map,
        report.manifest.tensors,
    )
    tensor_plan = build_qwen2_tensor_write_plan(
        reader,
        dimensions,
        output_layer=args.output_layer,
        weight_layout=args.weight_layout,
    )

    temporary_output = output_file.with_name(f".{output_file.name}.tmp")
    if temporary_output.exists() or temporary_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing temporary output: {temporary_output}")

    writer = gguf.GGUFWriter(str(temporary_output), arch="fairy2i")
    try:
        writer.add_custom_alignment(64)
        writer.add_name(config.get("_name_or_path", "Fairy2i-Qwen2"))
        writer.add_context_length(int(config["max_position_embeddings"]))
        writer.add_embedding_length(dimensions.hidden_complex)
        writer.add_block_count(dimensions.n_layer)
        writer.add_feed_forward_length(dimensions.ff_complex)
        writer.add_head_count(dimensions.n_head)
        writer.add_head_count_kv(dimensions.n_head_kv)
        writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
        writer.add_rope_freq_base(rope_theta)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_FAIRY2I_BUNDLE_V1)
        writer.add_vocab_size(int(config["vocab_size"]))
        write_metadata(
            writer,
            Fairy2IMetadata(
                base_arch="qwen2",
                base_model_type=config.get("model_type"),
                base_architecture=(config.get("architectures") or [None])[0],
                attn_layout="qwen2_real",
                tokenizer_profile="qwen2",
                quant_variant=args.quant_variant,
                residual_steps=args.residual_steps,
                weight_layout=args.weight_layout,
            ),
        )

        set_vocab_qwen2(model_dir, config, writer, report.tokenizer)
        register_qwen2_tensor_write_plan(writer, tensor_plan)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        write_qwen2_tensor_write_plan(writer, tensor_plan, verbose=args.verbose)
        writer.close()
        if output_file.exists() or output_file.is_symlink():
            raise FileExistsError(f"refusing to overwrite output created during conversion: {output_file}")
        os.replace(temporary_output, output_file)
    except BaseException:
        writer.close()
        temporary_output.unlink(missing_ok=True)
        raise

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
