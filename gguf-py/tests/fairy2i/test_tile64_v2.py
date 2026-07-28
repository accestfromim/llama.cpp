from __future__ import annotations

import numpy as np
import pytest

import fairy2i.quant.tile64_v2 as tile64_v2
from fairy2i.quant.tile64_v2 import (
    TILE64,
    encode_stage_codes,
    iter_quantize_linear_to_fairy2i_bundle_v1_m64,
    merge_fairy2i_bundle_v1_m,
    pack_fairy2i_bundle_v1,
    pack_fairy2i_tile64_v2_stage,
    quantize_linear_to_fairy2i_bundle_v1_m64,
    quantize_linear_to_fairy2i_tile64_v2_branch_data,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale,
    quantize_matrix_tile64_v2,
    quantize_tile64_once,
    unpack_fairy2i_bundle_v1,
)


def make_bundle_branch(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(TILE64, dtype=np.uint16)[:, None]
    col = np.arange(TILE64, dtype=np.uint16)[None, :]
    expected = ((row + col + seed) & 3).astype(np.uint8)
    real = np.zeros((TILE64, TILE64), dtype=np.float32)
    imag = np.zeros_like(real)
    real[expected == 0] = -1.0
    real[expected == 1] = 1.0
    imag[expected == 2] = -1.0
    imag[expected == 3] = 1.0
    scale_real = np.array([[1.25 + seed]], dtype=np.float32)
    scale_imag = np.array([[2.5 + seed]], dtype=np.float32)
    return real, imag, scale_real, scale_imag, expected


def make_wide_linear_weight(
    u_real: np.ndarray,
    u_imag: np.ndarray,
    w_real: np.ndarray,
    w_imag: np.ndarray,
) -> np.ndarray:
    a11 = u_real + w_real
    a12 = -u_imag + w_imag
    a21 = u_imag + w_imag
    a22 = u_real - w_real
    return np.block([[a11, a12], [a21, a22]]).astype(np.float32)


def test_encode_stage_codes_packs_four_codes_per_byte() -> None:
    stage_real = np.zeros((1, TILE64), dtype=np.float32)
    stage_imag = np.zeros((1, TILE64), dtype=np.float32)

    stage_real[0, 0] = -1.0
    stage_real[0, 16] = 1.0
    stage_imag[0, 32] = -1.0
    stage_imag[0, 48] = 1.0

    packed = encode_stage_codes(stage_real, stage_imag)

    assert packed.shape == (1, 16)
    assert packed[0, 0] == 0b11100100


def test_pack_fairy2i_tile64_v2_stage_size() -> None:
    stage_real = np.ones((TILE64, TILE64), dtype=np.float32)
    stage_imag = np.zeros((TILE64, TILE64), dtype=np.float32)
    scale_real = np.ones((1, 1), dtype=np.float32)
    scale_imag = np.zeros((1, 1), dtype=np.float32)

    packed = pack_fairy2i_tile64_v2_stage(stage_real, stage_imag, scale_real, scale_imag)

    assert packed.shape == (TILE64, 20)
    assert packed.dtype == np.uint8


@pytest.mark.parametrize("branch_order", [("U.s0", "W.s0"), ("U.s0", "U.s1", "W.s0", "W.s1")])
def test_bundle_v1_pack_round_trip(branch_order: tuple[str, ...]) -> None:
    branches = {}
    expected = {}
    for seed, name in enumerate(branch_order):
        real, imag, scale_real, scale_imag, branch_codes = make_bundle_branch(seed)
        branches[name] = (real, imag, scale_real, scale_imag)
        expected[name] = (branch_codes, scale_real.astype(np.float16), scale_imag.astype(np.float16))

    codes, scales = pack_fairy2i_bundle_v1(branches, branch_order)
    assert codes.shape == (1, 64, len(branch_order), 16)
    assert scales.shape == (1, len(branch_order), 2)
    assert codes.nbytes == len(branch_order) * 1024
    assert scales.nbytes == len(branch_order) * 4
    assert codes[0, 0, 0, 0] == np.uint8(0b11100100)

    decoded = unpack_fairy2i_bundle_v1(codes, scales, TILE64, TILE64, branch_order)
    for name in branch_order:
        np.testing.assert_array_equal(decoded[name][0], expected[name][0])
        np.testing.assert_array_equal(decoded[name][1], expected[name][1])
        np.testing.assert_array_equal(decoded[name][2], expected[name][2])


def test_bundle_v1_slot_order_is_m16_then_q4() -> None:
    branch = make_bundle_branch(0)
    codes, _ = pack_fairy2i_bundle_v1(
        {"U.s0": branch[:4], "W.s0": make_bundle_branch(1)[:4]},
        ("U.s0", "W.s0"),
    )

    # slot = q4 + 16*m16: rows 0..15 are in slots 0..15, rows 16..31 in 16..31.
    assert codes[0, 0, 0, 0] == np.uint8(0b11100100)
    assert codes[0, 1, 0, 0] == np.uint8(0b11100100)
    assert codes[0, 16, 0, 0] == np.uint8(0b11100100)


def test_bundle_v1_merge_m_preserves_component_tile_order() -> None:
    branch_order = ("U.s0", "W.s0")
    first_branches = {name: make_bundle_branch(seed)[:4] for seed, name in enumerate(branch_order)}
    second_branches = {name: make_bundle_branch(seed + 2)[:4] for seed, name in enumerate(branch_order)}
    first = pack_fairy2i_bundle_v1(first_branches, branch_order)
    second = pack_fairy2i_bundle_v1(second_branches, branch_order)

    codes, scales = merge_fairy2i_bundle_v1_m((first, second))

    assert codes.shape == (2, 64, 2, 16)
    assert scales.shape == (2, 2, 2)
    decoded = unpack_fairy2i_bundle_v1(codes, scales, 2 * TILE64, TILE64, branch_order)
    first_decoded = unpack_fairy2i_bundle_v1(*first, TILE64, TILE64, branch_order)
    second_decoded = unpack_fairy2i_bundle_v1(*second, TILE64, TILE64, branch_order)
    for name in branch_order:
        for component in range(3):
            expected = np.concatenate((first_decoded[name][component], second_decoded[name][component]), axis=0)
            np.testing.assert_array_equal(decoded[name][component], expected)


@pytest.mark.parametrize("dtype_name", ("float32", "bfloat16"))
def test_bundle_v1_m64_streaming_matches_nonstreaming_bytes(dtype_name: str) -> None:
    torch = pytest.importorskip("torch")

    generator = torch.Generator().manual_seed(20260728)
    # The logical complex matrix is 65x71 and therefore exercises both M/K
    # padding and multiple physical tiles at the 128x128 target.
    weight = torch.randn((130, 142), generator=generator, dtype=getattr(torch, dtype_name))
    branches = quantize_linear_to_fairy2i_tile64_v2_branch_data(weight, 128, 128)
    expected_codes, expected_scales = pack_fairy2i_bundle_v1(
        branches,
        ("U.s0", "U.s1", "W.s0", "W.s1"),
    )

    actual_codes, actual_scales = quantize_linear_to_fairy2i_bundle_v1_m64(weight, 128, 128)

    np.testing.assert_array_equal(actual_codes, expected_codes)
    np.testing.assert_array_equal(actual_scales, expected_scales)
    assert actual_codes.shape == (4, 64, 4, 16)
    assert actual_scales.shape == (4, 4, 2)

    decoded = unpack_fairy2i_bundle_v1(
        actual_codes,
        actual_scales,
        128,
        128,
        ("U.s0", "U.s1", "W.s0", "W.s1"),
    )
    for codes, scale_real, scale_imag in decoded.values():
        assert codes.shape == (128, 128)
        assert scale_real.shape == (2, 2)
        assert scale_imag.shape == (2, 2)


def test_bundle_v1_m64_callback_reads_only_complete_source_strips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    generator = torch.Generator().manual_seed(19)
    weight = torch.randn((256, 256), generator=generator, dtype=torch.float32)
    branches = quantize_linear_to_fairy2i_tile64_v2_branch_data(weight, 128, 128)
    expected = pack_fairy2i_bundle_v1(branches, ("U.s0", "U.s1", "W.s0", "W.s1"))
    calls: list[tuple[slice, slice]] = []

    def source(rows: slice, cols: slice) -> "torch.Tensor":
        calls.append((rows, cols))
        assert rows.stop - rows.start <= TILE64
        assert cols.stop - cols.start == 128
        return weight[rows, cols]

    def reject_legacy_split(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the M64 path must not construct whole-matrix U/W components")

    monkeypatch.setattr(tile64_v2, "split_wide_linear_components", reject_legacy_split)
    strips = list(
        iter_quantize_linear_to_fairy2i_bundle_v1_m64(
            source,
            128,
            128,
            weight_shape=(256, 256),
        )
    )
    actual = merge_fairy2i_bundle_v1_m(strips)

    assert len(strips) == 2
    assert all(codes.shape == (2, 64, 4, 16) for codes, _ in strips)
    assert all(scales.shape == (2, 4, 2) for _, scales in strips)
    # Four quadrants are read once per M64 strip, not once per K64 tile.
    assert len(calls) == 8
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_bundle_v1_m64_uses_one_scale_pair_for_all_64_rows() -> None:
    torch = pytest.importorskip("torch")

    row_magnitudes = np.arange(1, TILE64 + 1, dtype=np.float32)[:, None]
    u_real = np.broadcast_to(row_magnitudes, (TILE64, TILE64)).copy()
    zeros = np.zeros_like(u_real)
    weight = torch.from_numpy(make_wide_linear_weight(u_real, zeros, zeros, zeros))

    codes, scales = quantize_linear_to_fairy2i_bundle_v1_m64(weight, TILE64, TILE64)
    decoded = unpack_fairy2i_bundle_v1(
        codes,
        scales,
        TILE64,
        TILE64,
        ("U.s0", "U.s1", "W.s0", "W.s1"),
    )

    assert scales.shape == (1, 4, 2)
    assert scales[0, 0, 0] == np.float16(32.5)
    assert scales[0, 0, 1] == np.float16(0.0)
    assert decoded["U.s0"][0].shape == (TILE64, TILE64)
    assert decoded["U.s0"][1].shape == (1, 1)
    assert decoded["U.s0"][2].shape == (1, 1)


def test_quantize_tile64_once_ties_zeros_and_signs() -> None:
    tile_real = np.array([[1.0, -1.0, 0.0, 2.0, -3.0]], dtype=np.float32)
    tile_imag = np.array([[1.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    quant_real, quant_imag, scale_real, scale_imag = quantize_tile64_once(
        tile_real,
        tile_imag,
    )

    # Equal magnitudes, including an exact zero, belong to the imaginary
    # category. Signs are retained independently within the shared category.
    assert scale_real == pytest.approx(2.5)
    assert scale_imag == pytest.approx(2.0 / 3.0)
    np.testing.assert_array_equal(
        quant_real,
        np.array([[0.0, 0.0, 0.0, 2.5, -2.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        quant_imag,
        np.array([[2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]], dtype=np.float32),
    )

    zero_real, zero_imag, zero_scale_real, zero_scale_imag = quantize_tile64_once(
        np.zeros((TILE64, TILE64), dtype=np.float32),
        np.zeros((TILE64, TILE64), dtype=np.float32),
    )
    assert zero_scale_real == 0.0
    assert zero_scale_imag == 0.0
    assert not np.any(zero_real)
    assert not np.any(zero_imag)


def test_quantize_matrix_tile64_requires_divisible_dims() -> None:
    real = np.zeros((TILE64, TILE64 + 1), dtype=np.float32)
    imag = np.zeros_like(real)

    with pytest.raises(ValueError, match="requires dims divisible"):
        quantize_matrix_tile64_v2(real, imag)


def test_w1_learned_scale_packs_only_stage0_with_scale_channels() -> None:
    torch = pytest.importorskip("torch")

    u_real = np.ones((TILE64, TILE64), dtype=np.float32)
    u_imag = np.full((TILE64, TILE64), 0.25, dtype=np.float32)
    w_real = np.full((TILE64, TILE64), 0.5, dtype=np.float32)
    w_imag = np.full((TILE64, TILE64), -2.0, dtype=np.float32)

    a11 = u_real + w_real
    a12 = -u_imag + w_imag
    a21 = u_imag + w_imag
    a22 = u_real - w_real
    weight = torch.from_numpy(np.block([[a11, a12], [a21, a22]]))
    scale = torch.tensor([[[1.25]], [[2.5]], [[3.75]], [[4.5]]], dtype=torch.float32)

    packed = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale(weight, scale, TILE64, TILE64)

    assert set(packed) == {"U.s0", "W.s0"}
    u_stage = packed["U.s0"].reshape(TILE64, 20)
    w_stage = packed["W.s0"].reshape(TILE64, 20)

    assert np.frombuffer(u_stage[0, 16:18].tobytes(), dtype=np.float16)[0] == np.float16(1.25)
    assert np.frombuffer(u_stage[0, 18:20].tobytes(), dtype=np.float16)[0] == np.float16(2.5)
    assert np.frombuffer(w_stage[0, 16:18].tobytes(), dtype=np.float16)[0] == np.float16(3.75)
    assert np.frombuffer(w_stage[0, 18:20].tobytes(), dtype=np.float16)[0] == np.float16(4.5)


def test_w1_learned_scale_rejects_bad_scale_shape() -> None:
    torch = pytest.importorskip("torch")

    weight = torch.zeros((TILE64 * 2, TILE64 * 2), dtype=torch.float32)
    bad_scale = torch.zeros((4, 1, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="learned scale shape mismatch"):
        quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale(weight, bad_scale, TILE64, TILE64)
