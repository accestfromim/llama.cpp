from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fairy2i.quant.tile64_v2 import (
    Fairy2IBranch,
    pack_fairy2i_bundle_v1,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale_branch_data,
    unpack_fairy2i_bundle_v1,
)
from fairy2i.spec import (
    BUNDLE_CODE_ORDER,
    BUNDLE_CODE_ORDER_M16_JOINT,
    BUNDLE_CODE_ORDER_M8_BITPLANE,
    BUNDLE_CODE_ORDER_M8_JOINT,
    BUNDLE_CODE_ORDER_NATIVE_BRANCH,
    BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16,
    BUNDLE_CODE_ORDER_NATIVE_JOINT,
    BUNDLE_CODE_ORDER_ROW_JOINT,
    BUNDLE_CODE_ORDERS,
)


def _bundle_to_logical_q4(codes: np.ndarray) -> np.ndarray:
    tiles, slots, branches, lanes = codes.shape
    if slots != 64 or lanes != 16:
        raise ValueError(f"invalid canonical bundle shape: {codes.shape}")
    return codes.reshape(tiles, 4, 16, branches, 16).transpose(0, 1, 4, 2, 3).reshape(tiles, 64, 16, branches)


def _logical_q4_to_bundle(logical: np.ndarray) -> np.ndarray:
    tiles, rows, q4_groups, branches = logical.shape
    if rows != 64 or q4_groups != 16:
        raise ValueError(f"invalid logical q4 shape: {logical.shape}")
    return (
        logical.reshape(tiles, 4, 16, 16, branches)
        .transpose(0, 1, 3, 4, 2)
        .reshape(tiles, 64, branches, 16)
    )


def repack_fairy2i_bundle_codes(codes: np.ndarray, code_order: str) -> np.ndarray:
    """Reorder canonical bundle_v1 bytes without changing any 2-bit code."""

    if code_order not in BUNDLE_CODE_ORDERS:
        raise ValueError(f"unsupported Fairy2i Metal code order: {code_order}")
    if codes.dtype != np.uint8 or codes.ndim != 4 or codes.shape[1] != 64 or codes.shape[3] != 16:
        raise ValueError(f"invalid bundle code tensor: shape={codes.shape}, dtype={codes.dtype}")
    if code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16:
        raise ValueError("inline16 packing requires scales; use pack_fairy2i_metal_layout")
    if codes.shape[2] != 2 and code_order != BUNDLE_CODE_ORDER:
        raise ValueError(f"experimental Metal layouts currently require W1 branches=2, got {codes.shape[2]}")
    if code_order == BUNDLE_CODE_ORDER:
        return np.ascontiguousarray(codes)

    shape = codes.shape
    tiles = shape[0]
    branches = shape[2]
    logical = _bundle_to_logical_q4(codes)

    if code_order == BUNDLE_CODE_ORDER_M16_JOINT:
        packed = codes.reshape(tiles, 4, 16, branches, 16).transpose(0, 1, 2, 4, 3)
    elif code_order == BUNDLE_CODE_ORDER_M8_JOINT:
        packed = logical.reshape(tiles, 8, 8, 16, branches).transpose(0, 1, 3, 2, 4)
    elif code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH:
        packed = logical.reshape(tiles, 2, 4, 8, 4, 4, branches).transpose(0, 1, 4, 2, 5, 6, 3)
    elif code_order == BUNDLE_CODE_ORDER_NATIVE_JOINT:
        packed = logical.reshape(tiles, 2, 4, 8, 4, 4, branches).transpose(0, 1, 4, 2, 5, 3, 6)
    elif code_order == BUNDLE_CODE_ORDER_ROW_JOINT:
        packed = logical
    elif code_order == BUNDLE_CODE_ORDER_M8_BITPLANE:
        by_m8 = logical.reshape(tiles, 8, 8, 16, branches).transpose(0, 1, 3, 4, 2)
        parts = (by_m8[..., None] >> (2 * np.arange(4, dtype=np.uint8))) & np.uint8(3)
        bit_index = np.arange(8, dtype=np.uint32)[:, None] + 8 * np.arange(4, dtype=np.uint32)[None, :]
        bit_weight = np.left_shift(np.uint32(1), bit_index)
        sign = np.sum((parts & 1).astype(np.uint32) * bit_weight, axis=(-2, -1), dtype=np.uint64).astype("<u4")
        axis = np.sum(((parts >> 1) & 1).astype(np.uint32) * bit_weight, axis=(-2, -1), dtype=np.uint64).astype("<u4")
        packed = np.stack((sign, axis), axis=-1).view(np.uint8)
    else:
        raise AssertionError(code_order)

    return np.ascontiguousarray(packed.reshape(shape))


def canonicalize_fairy2i_metal_codes(codes: np.ndarray, code_order: str) -> np.ndarray:
    """Return code bytes in canonical bundle_v1 [tile][m16_q4][branch][lane16] order."""

    if code_order not in BUNDLE_CODE_ORDERS:
        raise ValueError(f"unsupported Fairy2i Metal code order: {code_order}")
    if code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16:
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != 2064:
            raise ValueError(f"invalid inline16 bundle code tensor: shape={codes.shape}, dtype={codes.dtype}")
        payload = np.ascontiguousarray(codes[:, 16:]).reshape(codes.shape[0], 64, 2, 16)
        return canonicalize_fairy2i_metal_codes(payload, BUNDLE_CODE_ORDER_NATIVE_BRANCH)
    if codes.dtype != np.uint8 or codes.ndim != 4 or codes.shape[1] != 64 or codes.shape[3] != 16:
        raise ValueError(f"invalid bundle code tensor: shape={codes.shape}, dtype={codes.dtype}")
    if code_order == BUNDLE_CODE_ORDER:
        return np.ascontiguousarray(codes)

    shape = codes.shape
    tiles, _, branches, _ = shape
    if branches != 2:
        raise ValueError(f"experimental Metal layouts currently require W1 branches=2, got {branches}")
    raw = np.ascontiguousarray(codes).reshape(-1)

    if code_order == BUNDLE_CODE_ORDER_M16_JOINT:
        canonical = raw.reshape(tiles, 4, 16, 16, branches).transpose(0, 1, 2, 4, 3).reshape(shape)
        return np.ascontiguousarray(canonical)
    if code_order == BUNDLE_CODE_ORDER_M8_JOINT:
        logical = raw.reshape(tiles, 8, 16, 8, branches).transpose(0, 1, 3, 2, 4).reshape(tiles, 64, 16, branches)
    elif code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH:
        logical = (
            raw.reshape(tiles, 2, 4, 4, 4, branches, 8)
            .transpose(0, 1, 3, 6, 2, 4, 5)
            .reshape(tiles, 64, 16, branches)
        )
    elif code_order == BUNDLE_CODE_ORDER_NATIVE_JOINT:
        logical = (
            raw.reshape(tiles, 2, 4, 4, 4, 8, branches)
            .transpose(0, 1, 3, 5, 2, 4, 6)
            .reshape(tiles, 64, 16, branches)
        )
    elif code_order == BUNDLE_CODE_ORDER_ROW_JOINT:
        logical = raw.reshape(tiles, 64, 16, branches)
    elif code_order == BUNDLE_CODE_ORDER_M8_BITPLANE:
        planes = raw.view("<u4").reshape(tiles, 8, 16, branches, 2)
        bit_index = np.arange(8, dtype=np.uint32)[:, None] + 8 * np.arange(4, dtype=np.uint32)[None, :]
        sign = (planes[..., 0, None, None] >> bit_index) & 1
        axis = (planes[..., 1, None, None] >> bit_index) & 1
        symbols = sign | (axis << 1)
        packed = np.sum(symbols << (2 * np.arange(4, dtype=np.uint32)), axis=-1, dtype=np.uint64).astype(np.uint8)
        logical = packed.transpose(0, 1, 4, 2, 3).reshape(tiles, 64, 16, branches)
    else:
        raise AssertionError(code_order)

    return _logical_q4_to_bundle(logical)


def pack_fairy2i_metal_layout(
    branches: Mapping[str, Fairy2IBranch], branch_order: Sequence[str], code_order: str
) -> tuple[np.ndarray, np.ndarray]:
    codes, scales = pack_fairy2i_bundle_v1(branches, branch_order)
    if code_order != BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16:
        return repack_fairy2i_bundle_codes(codes, code_order), scales

    payload = repack_fairy2i_bundle_codes(codes, BUNDLE_CODE_ORDER_NATIVE_BRANCH).reshape(codes.shape[0], -1)
    header = np.zeros((codes.shape[0], 16), dtype=np.uint8)
    scale_bytes = scales.view(np.uint8).reshape(codes.shape[0], -1)
    header[:, : scale_bytes.shape[1]] = scale_bytes
    inline_codes = np.ascontiguousarray(np.concatenate((header, payload), axis=1))
    dummy_scales = np.zeros((1,), dtype=np.float16)
    return inline_codes, dummy_scales


def extract_fairy2i_metal_scales(codes: np.ndarray, scales: np.ndarray, code_order: str) -> np.ndarray:
    if code_order != BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16:
        return scales
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != 2064:
        raise ValueError(f"invalid inline16 bundle code tensor: shape={codes.shape}, dtype={codes.dtype}")
    return np.ascontiguousarray(codes[:, :8]).view(np.float16).reshape(codes.shape[0], 2, 2)


def unpack_fairy2i_metal_layout(
    codes: np.ndarray,
    scales: np.ndarray,
    rows: int,
    cols: int,
    branch_order: Sequence[str],
    code_order: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    canonical = canonicalize_fairy2i_metal_codes(codes, code_order)
    canonical_scales = extract_fairy2i_metal_scales(codes, scales, code_order)
    return unpack_fairy2i_bundle_v1(canonical, canonical_scales, rows, cols, branch_order)


def quantize_linear_to_fairy2i_metal_layout_w1_learned_scale(
    weight: Any,
    quant_scale: Any,
    out_target: int,
    in_target: int,
    code_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    branches = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale_branch_data(
        weight, quant_scale, out_target, in_target
    )
    return pack_fairy2i_metal_layout(branches, ("U.s0", "W.s0"), code_order)
