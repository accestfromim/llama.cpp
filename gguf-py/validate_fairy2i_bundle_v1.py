#!/usr/bin/env python3

"""Validate a Fairy2i bundle_v1 GGUF and its tensor mapping.

Schema2 is compared bit-exactly against a tile64_v2 GGUF after canonical code
reordering. Schema3 uses a BF16-compute/F32-reduction contract with serialized
BF16 scale payloads, so its bundle codes/scales are validated structurally and
hashed while common tensors and the linear tensor mapping are still checked
against the supplied GGUF when one is available. ``--schema3-structural-only``
validates a schema3 bundle without a tile64_v2 reference.
"""

from __future__ import annotations

import argparse
import hashlib
import numbers
from pathlib import Path

import numpy as np

import gguf


BUNDLE_SUFFIX = ".bundle.codes"
BRANCH_SUFFIXES_BY_ORDER = {
    "U0,W0": (".U.s0", ".W.s0"),
    "U0,U1,W0,W1": (".U.s0", ".U.s1", ".W.s0", ".W.s1"),
}
QWEN2_LAYER_LINEAR_SUFFIXES = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)
QWEN2_LAYER_COMMON_SUFFIXES = (
    "attn_norm",
    "ffn_norm",
    "attn_q.bias",
    "attn_k.bias",
    "attn_v.bias",
)


def v2_bases_for_bundle(base: str) -> tuple[str, ...]:
    if base.endswith(".attn_qkv"):
        prefix = base[: -len("attn_qkv")]
        return (prefix + "attn_q", prefix + "attn_k", prefix + "attn_v")
    return (base,)


def field_value(reader: gguf.GGUFReader, name: str) -> object:
    field = reader.fields.get(name)
    if field is None:
        raise ValueError(f"missing GGUF field: {name}")
    return field.contents()


def tensor_map(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    return {tensor.name: tensor for tensor in reader.tensors}


def uint32_field_value(reader: gguf.GGUFReader, name: str) -> int:
    field = reader.fields.get(name)
    if field is None:
        raise ValueError(f"missing GGUF field: {name}")
    types = getattr(field, "types", None)
    if types is None or list(types) != [gguf.GGUFValueType.UINT32]:
        actual_types = (
            "missing"
            if types is None
            else ", ".join(getattr(value, "name", repr(value)) for value in types)
        )
        raise ValueError(
            f"{name}: expected scalar GGUF UINT32, got {actual_types}"
        )
    value = field.contents()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name}: expected scalar GGUF UINT32, got {type(value).__name__}"
        )
    if not 0 <= int(value) <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name}: GGUF UINT32 value is out of range: {value}")
    return int(value)


def validate_schema3_reference_linear_dims(
    base: str,
    logical_k: int,
    logical_m: int,
    schema3_linear_dims: dict[str, tuple[int, int]],
) -> None:
    expected = schema3_linear_dims[base]
    actual = (logical_k, logical_m)
    if actual != expected:
        raise ValueError(
            "schema3 reference linear dimensions mismatch: "
            f"{base} expected K={expected[0]}, M={expected[1]}, "
            f"got K={logical_k}, M={logical_m}"
        )


def qwen2_schema3_tensor_sets(n_layer: int) -> tuple[set[str], set[str]]:
    if n_layer <= 0:
        raise ValueError(f"schema3 exact profile requires a positive block count, got {n_layer}")
    bundle_bases = {"output"}
    common_names = {"token_embd", "output_norm"}
    for il in range(n_layer):
        bundle_bases.update(
            f"blk.{il}.{suffix}" for suffix in QWEN2_LAYER_LINEAR_SUFFIXES
        )
        common_names.update(
            f"blk.{il}.{suffix}" for suffix in QWEN2_LAYER_COMMON_SUFFIXES
        )
    return bundle_bases, common_names


def qwen2_schema3_tensor_contract(
    reader: gguf.GGUFReader,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, ...]]]:
    """Derive every exact-profile tensor shape from required model metadata."""

    dimension_keys = (
        "fairy2i.block_count",
        "fairy2i.embedding_length",
        "fairy2i.feed_forward_length",
        "fairy2i.attention.head_count",
        "fairy2i.attention.head_count_kv",
        "fairy2i.vocab_size",
    )
    dimensions = {name: uint32_field_value(reader, name) for name in dimension_keys}
    invalid = {name: value for name, value in dimensions.items() if value <= 0}
    if invalid:
        raise ValueError(f"schema3 exact dimensions must be positive: {invalid}")

    n_layer = dimensions["fairy2i.block_count"]
    n_embd = dimensions["fairy2i.embedding_length"]
    n_ff = dimensions["fairy2i.feed_forward_length"]
    n_head = dimensions["fairy2i.attention.head_count"]
    n_head_kv = dimensions["fairy2i.attention.head_count_kv"]
    n_vocab = dimensions["fairy2i.vocab_size"]

    if n_head % n_head_kv != 0:
        raise ValueError(
            "schema3 exact GQA dimensions are inconsistent: "
            f"head_count={n_head} is not divisible by head_count_kv={n_head_kv}"
        )
    if (2 * n_embd) % n_head != 0:
        raise ValueError(
            "schema3 exact attention dimensions are inconsistent: "
            f"2*embedding_length={2 * n_embd} is not divisible by "
            f"head_count={n_head}"
        )
    n_head_dim = (2 * n_embd) // n_head
    if "fairy2i.rope.dimension_count" in reader.fields:
        n_rot = uint32_field_value(reader, "fairy2i.rope.dimension_count")
        if n_rot != n_head_dim:
            raise ValueError(
                "schema3 exact attention dimensions are inconsistent: "
                f"derived head dimension={n_head_dim}, "
                f"rope.dimension_count={n_rot}"
            )
    if n_vocab % 2 != 0:
        raise ValueError(f"schema3 exact vocab size must be even, got {n_vocab}")

    n_embd_gqa = n_head_kv * n_head_dim
    if n_embd_gqa % 2 != 0:
        raise ValueError(
            "schema3 exact KV carrier width must be even: "
            f"head_count_kv={n_head_kv}, head_dim={n_head_dim}"
        )

    linear_dims: dict[str, tuple[int, int]] = {
        "output": (n_embd, n_vocab // 2),
    }
    common_shapes: dict[str, tuple[int, ...]] = {
        "token_embd": (n_embd, n_vocab),
        "output_norm": (2 * n_embd,),
    }
    for il in range(n_layer):
        prefix = f"blk.{il}."
        linear_dims.update(
            {
                prefix + "attn_q": (n_embd, n_embd),
                prefix + "attn_k": (n_embd, n_embd_gqa // 2),
                prefix + "attn_v": (n_embd, n_embd_gqa // 2),
                prefix + "attn_output": (n_embd, n_embd),
                prefix + "ffn_gate": (n_embd, n_ff),
                prefix + "ffn_up": (n_embd, n_ff),
                prefix + "ffn_down": (n_ff, n_embd),
            }
        )
        common_shapes.update(
            {
                prefix + "attn_norm": (2 * n_embd,),
                prefix + "ffn_norm": (2 * n_embd,),
                prefix + "attn_q.bias": (2 * n_embd,),
                prefix + "attn_k.bias": (n_embd_gqa,),
                prefix + "attn_v.bias": (n_embd_gqa,),
            }
        )

    invalid_linear_dims = {
        name: dims
        for name, dims in linear_dims.items()
        if dims[0] % 64 != 0 or dims[1] % 64 != 0
    }
    if invalid_linear_dims:
        name, dims = next(iter(invalid_linear_dims.items()))
        raise ValueError(
            "schema3 exact W2 logical dimensions must be multiples of 64: "
            f"{name} has K={dims[0]}, M={dims[1]}"
        )

    return linear_dims, common_shapes


def validate_schema3_common_finite(
    common_tensors: dict[str, gguf.ReaderTensor],
) -> None:
    chunk_elements = 1 << 20
    for name, tensor in common_tensors.items():
        flat = np.asarray(tensor.data).reshape(-1)
        for start in range(0, flat.size, chunk_elements):
            chunk = flat[start : start + chunk_elements]
            if name == "token_embd":
                bits = chunk.view(np.uint32)
                invalid = ((bits & 0x7F80) == 0x7F80) | (
                    ((bits >> 16) & 0x7F80) == 0x7F80
                )
            else:
                invalid = ~np.isfinite(chunk)
            if np.any(invalid):
                local_index = int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    "non-finite schema3 exact common tensor value: "
                    f"{name} at flat index {start + local_index}"
                )
            if name != "token_embd":
                non_bf16 = (chunk.view(np.uint32) & np.uint32(0xFFFF)) != 0
                if np.any(non_bf16):
                    local_index = int(np.flatnonzero(non_bf16)[0])
                    raise ValueError(
                        "schema3 exact common tensor is not a BF16-widened "
                        f"F32 value: {name} at flat index {start + local_index}; "
                        "reconvert schema3 from the original BF16 checkpoint"
                    )


def decode_v2_codes(stage: np.ndarray) -> np.ndarray:
    """Decode one M64 strip from V2 residue-lane bytes to [M64, KB, K64]."""

    packed = stage[..., :16]
    decoded = np.empty((*packed.shape[:2], 64), dtype=np.uint8)
    for part in range(4):
        decoded[..., part * 16 : (part + 1) * 16] = (packed >> (2 * part)) & 0x03
    return decoded


def v2_strip_to_bundle_codes(stage: np.ndarray) -> np.ndarray:
    """Return one M64 strip in [KB, 64 slots, 16 row lanes] order."""

    decoded = decode_v2_codes(stage)
    consecutive = decoded.reshape(64, decoded.shape[1], 16, 4)
    packed = (
        consecutive[..., 0]
        | (consecutive[..., 1] << 2)
        | (consecutive[..., 2] << 4)
        | (consecutive[..., 3] << 6)
    ).astype(np.uint8)
    return np.ascontiguousarray(
        packed.reshape(4, 16, decoded.shape[1], 16).transpose(2, 0, 3, 1).reshape(-1, 64, 16)
    )


def v2_strip_scales(stage: np.ndarray, base: str, branch: str, mb: int) -> np.ndarray:
    """Validate M64 scale replication and return [KB, real/imag] float16."""

    real_bytes = np.asarray(stage[..., 16:18])
    imag_bytes = np.asarray(stage[..., 18:20])
    if not np.all(real_bytes == real_bytes[0:1]) or not np.all(imag_bytes == imag_bytes[0:1]):
        raise ValueError(f"{base}{branch}: scale differs within M64 tile {mb}")

    real = np.ascontiguousarray(real_bytes[0]).view(np.float16).reshape(-1)
    imag = np.ascontiguousarray(imag_bytes[0]).view(np.float16).reshape(-1)
    return np.stack((real, imag), axis=-1)


def validate_common_tensors(
    v2_tensors: dict[str, gguf.ReaderTensor],
    bundle_tensors: dict[str, gguf.ReaderTensor],
    branch_suffixes: tuple[str, ...],
) -> tuple[int, int]:
    v2_layout_names = {
        name for name in v2_tensors if name.endswith(branch_suffixes)
    }
    bundle_layout_names = {
        name
        for name in bundle_tensors
        if name.endswith(".bundle.codes") or name.endswith(".bundle.scales")
    }
    v2_common_names = set(v2_tensors) - v2_layout_names
    bundle_common_names = set(bundle_tensors) - bundle_layout_names
    if v2_common_names != bundle_common_names:
        missing = sorted(v2_common_names - bundle_common_names)
        extra = sorted(bundle_common_names - v2_common_names)
        raise ValueError(
            "common tensor sets differ: "
            f"missing from bundle={missing[:4]}, extra in bundle={extra[:4]}"
        )

    common_names = sorted(v2_common_names)
    common_bytes = 0
    for name in common_names:
        v2_tensor = v2_tensors[name]
        bundle_tensor = bundle_tensors[name]
        if v2_tensor.tensor_type != bundle_tensor.tensor_type or not np.array_equal(
            v2_tensor.shape, bundle_tensor.shape
        ):
            raise ValueError(f"common tensor metadata mismatch: {name}")
        if not np.array_equal(v2_tensor.data, bundle_tensor.data):
            raise ValueError(f"common tensor data mismatch: {name}")
        common_bytes += v2_tensor.n_bytes
    return len(common_names), common_bytes


def validate_layout(v2_path: Path | None, bundle_path: Path) -> dict[str, object]:
    v2_reader = gguf.GGUFReader(v2_path, "r") if v2_path is not None else None
    bundle_reader = gguf.GGUFReader(bundle_path, "r")
    v2_tensors = tensor_map(v2_reader) if v2_reader is not None else {}
    bundle_tensors = tensor_map(bundle_reader)

    branch_order = str(field_value(bundle_reader, "fairy2i.weight.branch_order"))
    branch_suffixes = BRANCH_SUFFIXES_BY_ORDER.get(branch_order)
    if branch_suffixes is None:
        raise ValueError(f"unsupported bundle branch order: {branch_order}")
    branch_count = len(branch_suffixes)
    base_arch = str(field_value(bundle_reader, "fairy2i.base_arch"))

    schema_version = uint32_field_value(bundle_reader, "fairy2i.schema_version")
    if schema_version == 2:
        has_numeric_profile = "fairy2i.quant.numeric_profile" in bundle_reader.fields
        has_scale_dtype = "fairy2i.weight.scale_dtype" in bundle_reader.fields
        if base_arch == "qwen2" and branch_order == "U0,U1,W0,W1":
            if not has_numeric_profile or not has_scale_dtype:
                raise ValueError(
                    "Qwen2 schema2 W2 bundle requires "
                    "legacy_f16_v1/f16 numeric metadata"
                )
        if has_numeric_profile != has_scale_dtype:
            raise ValueError("schema2 numeric profile metadata must be declared as a pair")
        if has_numeric_profile and (
            field_value(bundle_reader, "fairy2i.quant.numeric_profile")
            != "legacy_f16_v1"
            or field_value(bundle_reader, "fairy2i.weight.scale_dtype") != "f16"
        ):
            raise ValueError("schema2 bundle has an invalid legacy numeric contract")
        scale_tensor_type = gguf.GGMLQuantizationType.F16
        comparison_mode = "legacy_bit_exact"
    elif schema_version == 3:
        if base_arch != "qwen2":
            raise ValueError("schema3 exact profile requires base_arch=qwen2")
        if branch_order != "U0,U1,W0,W1":
            raise ValueError("schema3 exact profile requires the W2 branch order")
        if (
            field_value(bundle_reader, "fairy2i.quant.numeric_profile")
            != "script_f32reduce_bf16scale_v1"
        ):
            raise ValueError("fairy2i.quant.numeric_profile: schema3 bundle has an invalid numeric profile")
        if field_value(bundle_reader, "fairy2i.weight.scale_dtype") != "bf16":
            raise ValueError("fairy2i.weight.scale_dtype: schema3 bundle has an invalid scale dtype")
        scale_tensor_type = gguf.GGMLQuantizationType.BF16
        comparison_mode = (
            "exact_bundle_structural_with_reference"
            if v2_reader is not None
            else "exact_bundle_structural_only"
        )
    else:
        raise ValueError(f"unsupported bundle schema version: {schema_version}")
    if v2_reader is None and schema_version != 3:
        raise ValueError("schema2 validation requires a tile64_v2 reference GGUF")

    expected_fields: dict[str, object] = {
        "general.architecture": "fairy2i",
        "general.file_type": 42,
        "general.alignment": 64,
        "fairy2i.schema_version": schema_version,
        "fairy2i.weight.layout": "bundle_m64k64_v1",
        "fairy2i.weight.scale_scope": "m64_k64",
        "fairy2i.weight.code_order": "m16_q4_branch_lane",
        "fairy2i.weight.branch_order": branch_order,
        "fairy2i.weight.m_block": 64,
        "fairy2i.weight.k_block": 64,
        "fairy2i.weight.m_subtile": 16,
    }
    if schema_version == 3:
        expected_fields.update(
            {
                "fairy2i.quant.format": "fairy2i_tile64_v2",
                "fairy2i.quant.variant": "tile64_v2",
                "fairy2i.quant.residual_steps": 2,
                "fairy2i.quant.codebook": "{+/-1,+/-i}",
                "fairy2i.quant.tile_size": 64,
                "fairy2i.quant.scale_stat": "dominant_mean_abs",
                "fairy2i.attn.layout": "qwen2_real",
                "fairy2i.tokenizer.profile": "qwen2",
            }
        )
    for name, expected in expected_fields.items():
        actual = (
            uint32_field_value(bundle_reader, name)
            if schema_version == 3 and isinstance(expected, int)
            else field_value(bundle_reader, name)
        )
        if actual != expected:
            raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")

    unaligned = [tensor.name for tensor in bundle_reader.tensors if tensor.data_offset % 64 != 0]
    if unaligned:
        raise ValueError(f"bundle tensor offsets are not 64-byte aligned: {unaligned[:4]}")

    bundle_bases = sorted(name[: -len(BUNDLE_SUFFIX)] for name in bundle_tensors if name.endswith(BUNDLE_SUFFIX))
    if not bundle_bases:
        raise ValueError("bundle GGUF has no bundle code tensors")
    expected_bundle_scales = {base + ".bundle.scales" for base in bundle_bases}
    actual_bundle_scales = {name for name in bundle_tensors if name.endswith(".bundle.scales")}
    if actual_bundle_scales != expected_bundle_scales:
        raise ValueError("bundle code and scale tensor sets differ")
    schema3_linear_dims: dict[str, tuple[int, int]] = {}
    if schema_version == 3:
        expected_bases, expected_common_names = qwen2_schema3_tensor_sets(
            uint32_field_value(bundle_reader, "fairy2i.block_count")
        )
        schema3_linear_dims, expected_common_shapes = qwen2_schema3_tensor_contract(
            bundle_reader
        )
        actual_bases = set(bundle_bases)
        if actual_bases != expected_bases:
            missing = sorted(expected_bases - actual_bases)
            extra = sorted(actual_bases - expected_bases)
            raise ValueError(
                "schema3 exact linear tensor sets differ: "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        bundle_layout_names = {
            name
            for name in bundle_tensors
            if name.endswith(".bundle.codes") or name.endswith(".bundle.scales")
        }
        actual_common_names = set(bundle_tensors) - bundle_layout_names
        if actual_common_names != expected_common_names:
            missing = sorted(expected_common_names - actual_common_names)
            extra = sorted(actual_common_names - expected_common_names)
            raise ValueError(
                "schema3 exact common tensor sets differ: "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        invalid_common_types = sorted(
            name
            for name in actual_common_names
            if bundle_tensors[name].tensor_type != gguf.GGMLQuantizationType.F32
        )
        if invalid_common_types:
            raise ValueError(
                "schema3 exact common tensors must use F32 carriers: "
                f"{invalid_common_types[:4]}"
            )
        invalid_common_shapes = sorted(
            (
                name,
                tuple(int(value) for value in bundle_tensors[name].shape),
                expected_common_shapes[name],
            )
            for name in actual_common_names
            if tuple(int(value) for value in bundle_tensors[name].shape)
            != expected_common_shapes[name]
        )
        if invalid_common_shapes:
            name, actual, expected = invalid_common_shapes[0]
            raise ValueError(
                "invalid schema3 exact common tensor shape: "
                f"{name} expected {expected}, got {actual}"
            )
        validate_schema3_common_finite(
            {name: bundle_tensors[name] for name in actual_common_names}
        )

    canonical_hash = hashlib.sha256()
    v2_weight_bytes = 0
    bundle_weight_bytes = 0
    for base in bundle_bases:
        codes = bundle_tensors[base + ".bundle.codes"]
        scales = bundle_tensors.get(base + ".bundle.scales")
        if scales is None:
            raise ValueError(f"missing bundle scales: {base}")
        if codes.tensor_type != gguf.GGMLQuantizationType.FAIRY2I_BUNDLE_CODES:
            raise ValueError(f"invalid bundle code type: {base}")
        if scales.tensor_type != scale_tensor_type:
            raise ValueError(f"invalid bundle scale type: {base}")
        if scale_tensor_type == gguf.GGMLQuantizationType.BF16:
            scale_storage = np.asarray(scales.data)
            scale_bits = (
                np.ascontiguousarray(scale_storage).view(np.uint16)
                if scale_storage.dtype == np.uint8
                else np.asarray(scale_storage, dtype=np.uint16)
            )
            if np.any((scale_bits & np.uint16(0x7F80)) == np.uint16(0x7F80)):
                raise ValueError(f"non-finite bundle scales: {base}")
            if np.any((scale_bits & np.uint16(0x8000)) != 0):
                raise ValueError(f"negative bundle scales: {base}")
        else:
            if not np.all(np.isfinite(scales.data)):
                raise ValueError(f"non-finite bundle scales: {base}")
            if np.any(np.signbit(scales.data)):
                raise ValueError(f"negative bundle scales: {base}")

        code_shape = tuple(int(value) for value in codes.shape)
        scale_shape = tuple(int(value) for value in scales.shape)
        if schema_version == 3:
            logical_k, logical_m = schema3_linear_dims[base]
            expected_physical_tiles = (logical_m // 64) * (logical_k // 64)
            expected_code_shape = (16, branch_count, 64, expected_physical_tiles)
            expected_scale_shape = (2, branch_count, expected_physical_tiles)
            if code_shape != expected_code_shape:
                raise ValueError(
                    f"invalid bundle code shape: {base} expected "
                    f"{expected_code_shape}, got {code_shape}"
                )
            if scale_shape != expected_scale_shape:
                raise ValueError(
                    f"invalid bundle scale shape: {base} expected "
                    f"{expected_scale_shape}, got {scale_shape}"
                )
            canonical_hash.update(base.encode("utf-8"))
            canonical_hash.update(np.asarray(codes.data).tobytes(order="C"))
            canonical_hash.update(np.asarray(scales.data).tobytes(order="C"))

        if v2_reader is None:
            bundle_weight_bytes += codes.n_bytes + scales.n_bytes
            continue

        components: list[tuple[str, list[gguf.ReaderTensor], int, int]] = []
        logical_k = 0
        for source_base in v2_bases_for_bundle(base):
            branch_tensors = []
            for suffix in branch_suffixes:
                tensor = v2_tensors.get(source_base + suffix)
                if tensor is None or tensor.tensor_type != gguf.GGMLQuantizationType.FAIRY2I_TILE64_V2:
                    raise ValueError(f"missing V2 branch: {source_base}{suffix}")
                branch_tensors.append(tensor)

            component_k, component_m = (int(value) for value in branch_tensors[0].shape)
            if component_m % 64 != 0 or component_k % 64 != 0:
                raise ValueError(
                    f"invalid V2 logical shape for {source_base}: M={component_m}, K={component_k}"
                )
            if schema_version == 3:
                validate_schema3_reference_linear_dims(
                    base,
                    component_k,
                    component_m,
                    schema3_linear_dims,
                )
            if any(
                tuple(int(value) for value in tensor.shape) != (component_k, component_m)
                for tensor in branch_tensors
            ):
                raise ValueError(f"V2 branch shape mismatch: {source_base}")
            if logical_k and component_k != logical_k:
                raise ValueError(f"merged bundle K mismatch: {base} has {logical_k} and {component_k}")
            logical_k = component_k
            components.append((source_base, branch_tensors, component_m, component_k))

        k_blocks = logical_k // 64
        physical_tiles = sum((component_m // 64) * k_blocks for _, _, component_m, _ in components)
        if tuple(int(value) for value in codes.shape) != (16, branch_count, 64, physical_tiles):
            raise ValueError(f"invalid bundle code shape: {base} {tuple(codes.shape)}")
        if tuple(int(value) for value in scales.shape) != (2, branch_count, physical_tiles):
            raise ValueError(f"invalid bundle scale shape: {base} {tuple(scales.shape)}")

        if schema_version == 2:
            canonical_hash.update(base.encode("utf-8"))
        tile_offset = 0
        for source_base, branch_tensors, component_m, _ in components:
            for mb in range(component_m // 64):
                tile_slice = slice(tile_offset, tile_offset + k_blocks)
                for branch, (suffix, v2_tensor) in enumerate(zip(branch_suffixes, branch_tensors)):
                    stage = np.asarray(v2_tensor.data).reshape(component_m, k_blocks, 20)[
                        mb * 64 : (mb + 1) * 64
                    ]
                    actual_codes = np.asarray(codes.data[tile_slice, :, branch, :])
                    actual_scales = np.asarray(scales.data[tile_slice, branch, :])
                    if schema_version == 2:
                        expected_codes = v2_strip_to_bundle_codes(stage)
                        if not np.array_equal(expected_codes, actual_codes):
                            raise ValueError(
                                f"canonical code mismatch: {source_base}{suffix} M64 tile {mb}"
                            )

                        expected_scales = v2_strip_scales(stage, source_base, suffix, mb)
                        if not np.array_equal(
                            expected_scales.view(np.uint16),
                            actual_scales.view(np.uint16),
                        ):
                            raise ValueError(
                                f"canonical scale mismatch: {source_base}{suffix} M64 tile {mb}"
                            )

                    if schema_version == 2:
                        canonical_hash.update(actual_codes.tobytes(order="C"))
                        canonical_hash.update(actual_scales.tobytes(order="C"))
                tile_offset += k_blocks

        v2_weight_bytes += sum(tensor.n_bytes for _, tensors, _, _ in components for tensor in tensors)
        bundle_weight_bytes += codes.n_bytes + scales.n_bytes

    if v2_reader is not None:
        expected_v2_layout = {
            source_base + suffix
            for base in bundle_bases
            for source_base in v2_bases_for_bundle(base)
            for suffix in branch_suffixes
        }
        actual_v2_layout = {name for name in v2_tensors if name.endswith(branch_suffixes)}
        if actual_v2_layout != expected_v2_layout:
            raise ValueError("V2 and bundle linear tensor sets differ")

        common_count, common_bytes = validate_common_tensors(v2_tensors, bundle_tensors, branch_suffixes)
    else:
        stale_layout = sorted(name for name in bundle_tensors if name.endswith(branch_suffixes))
        if stale_layout:
            raise ValueError(f"schema3 bundle contains legacy tile64_v2 tensors: {stale_layout[:4]}")
        bundle_layout_names = {
            name
            for name in bundle_tensors
            if name.endswith(".bundle.codes") or name.endswith(".bundle.scales")
        }
        common_names = set(bundle_tensors) - bundle_layout_names
        common_count = len(common_names)
        common_bytes = sum(bundle_tensors[name].n_bytes for name in common_names)

    return {
        "schema_version": schema_version,
        "comparison_mode": comparison_mode,
        "branch_order": branch_order,
        "linear_count": len(bundle_bases),
        "common_tensor_count": common_count,
        "tensor_count": len(bundle_reader.tensors),
        "common_bytes": common_bytes,
        "v2_weight_bytes": v2_weight_bytes,
        "bundle_weight_bytes": bundle_weight_bytes,
        "v2_file_bytes": v2_path.stat().st_size if v2_path is not None else None,
        "bundle_file_bytes": bundle_path.stat().st_size,
        "canonical_sha256": canonical_hash.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema3-structural-only",
        action="store_true",
        help="validate one schema3 bundle without a tile64_v2 reference",
    )
    parser.add_argument(
        "gguf",
        type=Path,
        nargs="+",
        help="TILE64_V2_GGUF BUNDLE_V1_GGUF, or one BUNDLE_V1_GGUF with --schema3-structural-only",
    )
    args = parser.parse_args()

    if args.schema3_structural_only:
        if len(args.gguf) != 1:
            parser.error("--schema3-structural-only accepts exactly one bundle GGUF")
        v2_path = None
        bundle_path = args.gguf[0]
    else:
        if len(args.gguf) != 2:
            parser.error("validation requires TILE64_V2_GGUF and BUNDLE_V1_GGUF")
        v2_path, bundle_path = args.gguf

    result = validate_layout(v2_path, bundle_path)
    print("Fairy2i bundle_v1 validation passed")
    for name, value in result.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
