from __future__ import annotations

import numpy as np
import pytest

from fairy2i.quant.tile64_v2 import TILE64, pack_fairy2i_bundle_v1
from fairy2i.quant.tile64_v3_metal import (
    canonicalize_fairy2i_metal_codes,
    extract_fairy2i_metal_scales,
    pack_fairy2i_metal_layout,
    unpack_fairy2i_metal_layout,
)
from fairy2i.spec import BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16, BUNDLE_CODE_ORDERS


def make_branch(seed: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(rows, dtype=np.uint16)[:, None]
    col = np.arange(cols, dtype=np.uint16)[None, :]
    symbol = ((3 * row + 5 * col + seed) & 3).astype(np.uint8)
    real = np.zeros((rows, cols), dtype=np.float32)
    imag = np.zeros_like(real)
    real[symbol == 0] = -1.0
    real[symbol == 1] = 1.0
    imag[symbol == 2] = -1.0
    imag[symbol == 3] = 1.0
    scale_shape = (rows // TILE64, cols // TILE64)
    scale_real = (np.arange(np.prod(scale_shape), dtype=np.float32).reshape(scale_shape) + 1.25 + seed)
    scale_imag = (np.arange(np.prod(scale_shape), dtype=np.float32).reshape(scale_shape) + 2.5 + seed)
    return real, imag, scale_real, scale_imag


@pytest.mark.parametrize("code_order", BUNDLE_CODE_ORDERS)
@pytest.mark.parametrize("shape", [(64, 64), (128, 192)])
def test_metal_layout_round_trip(code_order: str, shape: tuple[int, int]) -> None:
    rows, cols = shape
    branch_order = ("U.s0", "W.s0")
    branches = {name: make_branch(seed, rows, cols) for seed, name in enumerate(branch_order)}
    canonical_codes, canonical_scales = pack_fairy2i_bundle_v1(branches, branch_order)

    codes, scales = pack_fairy2i_metal_layout(branches, branch_order, code_order)

    if code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16:
        assert codes.shape == (canonical_codes.shape[0], 2064)
        assert codes.nbytes == canonical_codes.nbytes + canonical_codes.shape[0] * 16
        assert scales.shape == (1,)
    else:
        assert codes.shape == canonical_codes.shape
        assert codes.nbytes == canonical_codes.nbytes
        np.testing.assert_array_equal(scales, canonical_scales)
    assert codes.flags.c_contiguous
    np.testing.assert_array_equal(canonicalize_fairy2i_metal_codes(codes, code_order), canonical_codes)
    np.testing.assert_array_equal(extract_fairy2i_metal_scales(codes, scales, code_order), canonical_scales)

    unpacked = unpack_fairy2i_metal_layout(codes, scales, rows, cols, branch_order, code_order)
    canonical_unpacked = unpack_fairy2i_metal_layout(
        canonical_codes, canonical_scales, rows, cols, branch_order, BUNDLE_CODE_ORDERS[0]
    )
    for name in branch_order:
        for actual, expected in zip(unpacked[name], canonical_unpacked[name]):
            np.testing.assert_array_equal(actual, expected)


def test_bitplane_exercises_all_symbols() -> None:
    branch_order = ("U.s0", "W.s0")
    branches = {name: make_branch(seed, TILE64, TILE64) for seed, name in enumerate(branch_order)}
    canonical_codes, _ = pack_fairy2i_bundle_v1(branches, branch_order)
    codes, _ = pack_fairy2i_metal_layout(branches, branch_order, BUNDLE_CODE_ORDERS[-2])

    assert not np.array_equal(codes, canonical_codes)
    np.testing.assert_array_equal(
        canonicalize_fairy2i_metal_codes(codes, BUNDLE_CODE_ORDERS[-2]), canonical_codes
    )
