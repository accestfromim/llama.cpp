#!/usr/bin/env python3

"""Validate a Fairy2i bundle_v1 GGUF against a tile64_v2 GGUF.

The comparison is bit-exact after converting each tile64_v2 branch to the
canonical bundle byte order. W1 and W2 branch sets are supported, and
dense/common tensors are compared directly.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import gguf


BUNDLE_SUFFIX = ".bundle.codes"
BRANCH_SUFFIXES_BY_ORDER = {
    "U0,W0": (".U.s0", ".W.s0"),
    "U0,U1,W0,W1": (".U.s0", ".U.s1", ".W.s0", ".W.s1"),
}


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
    layout_names = {
        name
        for name in v2_tensors
        if name.endswith(branch_suffixes)
    } | {
        name
        for name in bundle_tensors
        if name.endswith(".bundle.codes") or name.endswith(".bundle.scales")
    }
    common_names = sorted((set(v2_tensors) & set(bundle_tensors)) - layout_names)
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


def validate_layout(v2_path: Path, bundle_path: Path) -> dict[str, object]:
    v2_reader = gguf.GGUFReader(v2_path, "r")
    bundle_reader = gguf.GGUFReader(bundle_path, "r")
    v2_tensors = tensor_map(v2_reader)
    bundle_tensors = tensor_map(bundle_reader)

    branch_order = str(field_value(bundle_reader, "fairy2i.weight.branch_order"))
    branch_suffixes = BRANCH_SUFFIXES_BY_ORDER.get(branch_order)
    if branch_suffixes is None:
        raise ValueError(f"unsupported bundle branch order: {branch_order}")
    branch_count = len(branch_suffixes)

    expected_fields = {
        "general.file_type": 42,
        "general.alignment": 64,
        "fairy2i.schema_version": 2,
        "fairy2i.weight.layout": "bundle_m64k64_v1",
        "fairy2i.weight.scale_scope": "m64_k64",
        "fairy2i.weight.code_order": "m16_q4_branch_lane",
        "fairy2i.weight.branch_order": branch_order,
        "fairy2i.weight.m_block": 64,
        "fairy2i.weight.k_block": 64,
        "fairy2i.weight.m_subtile": 16,
    }
    for name, expected in expected_fields.items():
        actual = field_value(bundle_reader, name)
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
        if scales.tensor_type != gguf.GGMLQuantizationType.F16:
            raise ValueError(f"invalid bundle scale type: {base}")

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

        canonical_hash.update(base.encode("utf-8"))
        tile_offset = 0
        for source_base, branch_tensors, component_m, _ in components:
            for mb in range(component_m // 64):
                tile_slice = slice(tile_offset, tile_offset + k_blocks)
                for branch, (suffix, v2_tensor) in enumerate(zip(branch_suffixes, branch_tensors)):
                    stage = np.asarray(v2_tensor.data).reshape(component_m, k_blocks, 20)[
                        mb * 64 : (mb + 1) * 64
                    ]
                    expected_codes = v2_strip_to_bundle_codes(stage)
                    actual_codes = np.asarray(codes.data[tile_slice, :, branch, :])
                    if not np.array_equal(expected_codes, actual_codes):
                        raise ValueError(f"canonical code mismatch: {source_base}{suffix} M64 tile {mb}")

                    expected_scales = v2_strip_scales(stage, source_base, suffix, mb)
                    actual_scales = np.asarray(scales.data[tile_slice, branch, :])
                    if not np.array_equal(expected_scales.view(np.uint16), actual_scales.view(np.uint16)):
                        raise ValueError(f"canonical scale mismatch: {source_base}{suffix} M64 tile {mb}")

                    canonical_hash.update(expected_codes)
                    canonical_hash.update(expected_scales.view(np.uint8))
                tile_offset += k_blocks

        v2_weight_bytes += sum(tensor.n_bytes for _, tensors, _, _ in components for tensor in tensors)
        bundle_weight_bytes += codes.n_bytes + scales.n_bytes

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
    return {
        "branch_order": branch_order,
        "linear_count": len(bundle_bases),
        "common_tensor_count": common_count,
        "tensor_count": len(bundle_reader.tensors),
        "common_bytes": common_bytes,
        "v2_weight_bytes": v2_weight_bytes,
        "bundle_weight_bytes": bundle_weight_bytes,
        "v2_file_bytes": v2_path.stat().st_size,
        "bundle_file_bytes": bundle_path.stat().st_size,
        "canonical_sha256": canonical_hash.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile64_v2_gguf", type=Path)
    parser.add_argument("bundle_v1_gguf", type=Path)
    args = parser.parse_args()

    result = validate_layout(args.tile64_v2_gguf, args.bundle_v1_gguf)
    print("Fairy2i bundle_v1 validation passed")
    for name, value in result.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
