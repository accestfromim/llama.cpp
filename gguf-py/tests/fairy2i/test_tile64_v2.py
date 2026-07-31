from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fairy2i.quant.tile64_v2 as tile64_v2
from fairy2i.quant.tile64_v2 import (
    NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
    TILE64,
    canonical_quantize_complex_tile64_v2_script_bf16_f32,
    canonical_quantize_wide_linear_tile64_script_bf16_f32,
    encode_stage_codes,
    iter_quantize_linear_to_fairy2i_bundle_v1_m64,
    merge_fairy2i_bundle_v1_m,
    pack_fairy2i_bundle_v1,
    pack_fairy2i_tile64_v2_stage,
    pairwise_sum_float32,
    quantize_linear_to_fairy2i_bundle_v1_m64,
    quantize_linear_to_fairy2i_tile64_v2_branch_data,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale,
    quantize_complex_tile64_v2_script_bf16_f32,
    quantize_matrix_tile64_v2,
    quantize_tile64_once,
    reconstruct_complex_tile64_v2_script_bf16_f32,
    reconstruct_wide_linear_tile64_script_bf16_f32,
    round_float32_to_bf16,
    round_float32_to_bf16_bits,
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


def test_bundle_v1_branch_packer_rejects_exact_profile_without_mask_codes() -> None:
    branch_order = ("U.s0", "U.s1", "W.s0", "W.s1")
    branches = {
        name: make_bundle_branch(seed)[:4]
        for seed, name in enumerate(branch_order)
    }

    with pytest.raises(ValueError, match="cannot recover exact-profile sign masks"):
        pack_fairy2i_bundle_v1(
            branches,
            branch_order,
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
        )


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


def test_script_bf16_rounding_is_rne_and_preserves_signed_zero() -> None:
    input_bits = np.array(
        [
            0x00000000,
            0x80000000,
            0x00008000,
            0x00008001,
            0x3F808000,
            0x3F818000,
            0x7F7F0000,
            0x7F7F8000,
            0xFF7F8000,
        ],
        dtype=np.uint32,
    )
    expected_bits = np.array(
        [
            0x00000000,
            0x80000000,
            0x00000000,
            0x00010000,
            0x3F800000,
            0x3F820000,
            0x7F7F0000,
            0x7F800000,
            0xFF800000,
        ],
        dtype=np.uint32,
    )

    actual = round_float32_to_bf16(input_bits.view(np.float32))

    np.testing.assert_array_equal(actual.view(np.uint32), expected_bits)
    np.testing.assert_array_equal(
        round_float32_to_bf16_bits(input_bits.view(np.float32)),
        (expected_bits >> np.uint32(16)).astype(np.uint16),
    )


def test_script_f32_reduction_is_balanced_pairwise_not_sequential() -> None:
    values = np.zeros(TILE64 * TILE64, dtype=np.float32)
    values[:4] = np.asarray([1e20, 1.0, -1e20, 1.0], dtype=np.float32)

    sequential = np.float32(0.0)
    for value in values:
        sequential = np.float32(sequential + value)

    assert pairwise_sum_float32(values).view(np.uint32) == np.float32(0.0).view(
        np.uint32
    )
    assert sequential.view(np.uint32) == np.float32(1.0).view(np.uint32)


def _script_profile_fixture() -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(TILE64, dtype=np.int32)[:, None]
    col = np.arange(TILE64, dtype=np.int32)[None, :]
    real = (((row * 17 + col * 13) % 257) - 128).astype(np.float32) / np.float32(32.0)
    imag = (((row * 29 + col * 7 + 11) % 263) - 131).astype(np.float32) / np.float32(32.0)
    return real, imag


def _script_profile_wide_fixture(
    mb_count: int,
    kb_count: int,
) -> np.ndarray:
    out_target = mb_count * TILE64
    in_target = kb_count * TILE64
    components = [
        np.empty((out_target, in_target), dtype=np.float32)
        for _ in range(4)
    ]
    fixture_real, fixture_imag = _script_profile_fixture()
    for mb in range(mb_count):
        rows = slice(mb * TILE64, (mb + 1) * TILE64)
        for kb in range(kb_count):
            cols = slice(kb * TILE64, (kb + 1) * TILE64)
            tile_seed = mb * kb_count + kb
            components[0][rows, cols] = np.roll(
                fixture_real,
                shift=(3 * tile_seed, 5 * tile_seed),
                axis=(0, 1),
            )
            components[1][rows, cols] = np.roll(
                fixture_imag,
                shift=(7 * tile_seed, 2 * tile_seed),
                axis=(0, 1),
            )
            components[2][rows, cols] = np.roll(
                fixture_real,
                shift=(7 + 2 * tile_seed, 11 + tile_seed),
                axis=(0, 1),
            )
            components[3][rows, cols] = np.roll(
                fixture_imag,
                shift=(13 + tile_seed, 17 + 3 * tile_seed),
                axis=(0, 1),
            )
    return np.block(
        [
            [components[0], components[1]],
            [components[2], components[3]],
        ]
    )


def _pack_scalar_oracle_stage_codes(codes: np.ndarray) -> np.ndarray:
    """Pack scalar-oracle codes without using the production bundle helpers."""

    assert codes.shape == (TILE64, TILE64)
    assert codes.dtype == np.uint8
    q4 = codes.reshape(TILE64, TILE64 // 4, 4)
    packed = (
        q4[..., 0]
        | (q4[..., 1] << np.uint8(2))
        | (q4[..., 2] << np.uint8(4))
        | (q4[..., 3] << np.uint8(6))
    ).astype(np.uint8)
    return np.ascontiguousarray(
        packed.reshape(TILE64 // 16, 16, TILE64 // 4)
        .transpose(0, 2, 1)
        .reshape(TILE64, 16)
    )


def _canonical_bundle_from_wide_checkpoint(
    checkpoint_weight: np.ndarray,
    out_target: int,
    in_target: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a whole bundle tile-by-tile from the independent scalar oracle."""

    assert checkpoint_weight.shape == (2 * out_target, 2 * in_target)
    a11 = checkpoint_weight[:out_target, :in_target]
    a12 = checkpoint_weight[:out_target, in_target:]
    a21 = checkpoint_weight[out_target:, :in_target]
    a22 = checkpoint_weight[out_target:, in_target:]
    mb_count = out_target // TILE64
    kb_count = in_target // TILE64
    codes = np.empty((mb_count * kb_count, TILE64, 4, 16), dtype=np.uint8)
    scales = np.empty((mb_count * kb_count, 4, 2), dtype=np.uint16)

    for mb in range(mb_count):
        rows = slice(mb * TILE64, (mb + 1) * TILE64)
        for kb in range(kb_count):
            cols = slice(kb * TILE64, (kb + 1) * TILE64)
            u_tile, w_tile, _ = canonical_quantize_wide_linear_tile64_script_bf16_f32(
                a11[rows, cols],
                a12[rows, cols],
                a21[rows, cols],
                a22[rows, cols],
            )
            tile_index = mb * kb_count + kb
            for branch, stage in enumerate(
                (u_tile.stage0, u_tile.stage1, w_tile.stage0, w_tile.stage1)
            ):
                codes[tile_index, :, branch, :] = _pack_scalar_oracle_stage_codes(
                    stage.codes
                )
                scales[tile_index, branch, :] = round_float32_to_bf16_bits(
                    np.asarray(
                        (stage.scale_real, stage.scale_imag),
                        dtype=np.float32,
                    )
                )

    return codes, scales


def test_script_bf16_f32_scalar_oracle_matches_vectorized_fixture_bits() -> None:
    tile_real, tile_imag = _script_profile_fixture()

    expected = canonical_quantize_complex_tile64_v2_script_bf16_f32(
        tile_real,
        tile_imag,
    )
    actual = quantize_complex_tile64_v2_script_bf16_f32(tile_real, tile_imag)

    digest = hashlib.sha256()
    for stage_name in ("stage0", "stage1"):
        expected_stage = getattr(expected, stage_name)
        actual_stage = getattr(actual, stage_name)
        np.testing.assert_array_equal(actual_stage.codes, expected_stage.codes)
        np.testing.assert_array_equal(
            actual_stage.quant_real.view(np.uint32),
            expected_stage.quant_real.view(np.uint32),
        )
        np.testing.assert_array_equal(
            actual_stage.quant_imag.view(np.uint32),
            expected_stage.quant_imag.view(np.uint32),
        )
        assert actual_stage.scale_real.view(np.uint32) == expected_stage.scale_real.view(
            np.uint32
        )
        assert actual_stage.scale_imag.view(np.uint32) == expected_stage.scale_imag.view(
            np.uint32
        )
        digest.update(actual_stage.codes.tobytes())
        digest.update(actual_stage.scale_real.tobytes())
        digest.update(actual_stage.scale_imag.tobytes())

    serialized_scale_bits = tuple(
        round_float32_to_bf16_bits(
            np.asarray((stage.scale_real, stage.scale_imag), dtype=np.float32)
        )
        for stage in (actual.stage0, actual.stage1)
    )
    reconstructed_real, reconstructed_imag = (
        reconstruct_complex_tile64_v2_script_bf16_f32(
            actual,
            serialized_scale_bits=serialized_scale_bits,
        )
    )
    legacy_reconstructed = reconstruct_complex_tile64_v2_script_bf16_f32(actual)
    expected_real = round_float32_to_bf16(
        np.add(actual.stage0.quant_real, actual.stage1.quant_real, dtype=np.float32)
    )
    expected_imag = round_float32_to_bf16(
        np.add(actual.stage0.quant_imag, actual.stage1.quant_imag, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        reconstructed_real.view(np.uint32),
        expected_real.view(np.uint32),
    )
    np.testing.assert_array_equal(
        reconstructed_imag.view(np.uint32),
        expected_imag.view(np.uint32),
    )
    np.testing.assert_array_equal(
        reconstructed_real.view(np.uint32),
        legacy_reconstructed[0].view(np.uint32),
    )
    np.testing.assert_array_equal(
        reconstructed_imag.view(np.uint32),
        legacy_reconstructed[1].view(np.uint32),
    )
    assert digest.hexdigest() == "42a3d98624c7b70e236b4db3f99c2bcd18ddcb08c190be8f22f6f0ccdb702d6c"


def test_script_bf16_f32_scalar_wide_oracle_matches_vectorized_u_w_and_a_bits() -> None:
    a11, a12 = _script_profile_fixture()
    a21 = np.roll(a11, shift=7, axis=0)
    a22 = np.roll(a12, shift=11, axis=1)

    expected_u, expected_w, expected_a = (
        canonical_quantize_wide_linear_tile64_script_bf16_f32(
            a11,
            a12,
            a21,
            a22,
        )
    )

    x11, x12, x21, x22 = (
        round_float32_to_bf16(component)
        for component in (a11, a12, a21, a22)
    )

    def bf16_half(value: np.ndarray) -> np.ndarray:
        return round_float32_to_bf16(
            np.multiply(value, np.float32(0.5), dtype=np.float32)
        )

    u_real = bf16_half(
        round_float32_to_bf16(np.add(x11, x22, dtype=np.float32))
    )
    u_imag = bf16_half(
        round_float32_to_bf16(np.subtract(x21, x12, dtype=np.float32))
    )
    w_real = bf16_half(
        round_float32_to_bf16(np.subtract(x11, x22, dtype=np.float32))
    )
    w_imag = bf16_half(
        round_float32_to_bf16(np.add(x12, x21, dtype=np.float32))
    )
    actual_u = quantize_complex_tile64_v2_script_bf16_f32(u_real, u_imag)
    actual_w = quantize_complex_tile64_v2_script_bf16_f32(w_real, w_imag)
    def serialized_scales(tile: tile64_v2.ScriptBF16F32Tile) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            round_float32_to_bf16_bits(
                np.asarray((stage.scale_real, stage.scale_imag), dtype=np.float32)
            )
            for stage in (tile.stage0, tile.stage1)
        )

    actual_a = reconstruct_wide_linear_tile64_script_bf16_f32(
        actual_u,
        actual_w,
        u_serialized_scale_bits=serialized_scales(actual_u),
        w_serialized_scale_bits=serialized_scales(actual_w),
    )

    for expected_tile, actual_tile in (
        (expected_u, actual_u),
        (expected_w, actual_w),
    ):
        for stage_name in ("stage0", "stage1"):
            expected_stage = getattr(expected_tile, stage_name)
            actual_stage = getattr(actual_tile, stage_name)
            np.testing.assert_array_equal(actual_stage.codes, expected_stage.codes)
            assert (
                actual_stage.scale_real.view(np.uint32)
                == expected_stage.scale_real.view(np.uint32)
            )
            assert (
                actual_stage.scale_imag.view(np.uint32)
                == expected_stage.scale_imag.view(np.uint32)
            )

    for expected_component, actual_component in zip(expected_a, actual_a):
        np.testing.assert_array_equal(
            actual_component.view(np.uint32),
            expected_component.view(np.uint32),
        )


@pytest.mark.parametrize(
    ("scale_bits", "match"),
    [
        (np.asarray([0x3F80, 0x3F00], dtype=np.int16), "BF16 uint16"),
        (np.asarray([0xBF80, 0x3F00], dtype=np.uint16), "non-negative"),
        (np.asarray([0x7F80, 0x3F00], dtype=np.uint16), "finite BF16"),
        (np.asarray([0x7FC1, 0x3F00], dtype=np.uint16), "finite BF16"),
    ],
)
def test_script_reconstruction_rejects_invalid_serialized_bf16_scales(
    scale_bits: np.ndarray,
    match: str,
) -> None:
    tile_real, tile_imag = _script_profile_fixture()
    tile = quantize_complex_tile64_v2_script_bf16_f32(tile_real, tile_imag)

    with pytest.raises(ValueError, match=match):
        reconstruct_complex_tile64_v2_script_bf16_f32(
            tile,
            serialized_scale_bits=(scale_bits, np.asarray([0x3F80, 0x3F00], dtype=np.uint16)),
        )


@pytest.mark.parametrize(
    ("tile_real", "tile_imag", "match"),
    [
        (
            np.ones((TILE64, TILE64), dtype=np.float32),
            np.zeros((TILE64, TILE64), dtype=np.float32),
            "empty imag-dominant category",
        ),
        (
            np.zeros((TILE64, TILE64), dtype=np.float32),
            np.zeros((TILE64, TILE64), dtype=np.float32),
            "empty real-dominant category",
        ),
        (
            np.full((TILE64, TILE64), np.nan, dtype=np.float32),
            np.zeros((TILE64, TILE64), dtype=np.float32),
            "requires finite",
        ),
        (
            np.full((TILE64, TILE64), np.inf, dtype=np.float32),
            np.zeros((TILE64, TILE64), dtype=np.float32),
            "requires finite",
        ),
        (
            np.full((TILE64, TILE64), -np.inf, dtype=np.float32),
            np.zeros((TILE64, TILE64), dtype=np.float32),
            "requires finite",
        ),
    ],
)
def test_script_bf16_f32_rejects_degenerate_or_nonfinite_tiles(
    tile_real: np.ndarray,
    tile_imag: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        quantize_complex_tile64_v2_script_bf16_f32(tile_real, tile_imag)


def test_script_bf16_f32_rejects_empty_stage1_dominant_group() -> None:
    tile_real = np.zeros((TILE64, TILE64), dtype=np.float32)
    tile_imag = np.ones_like(tile_real)
    tile_real[:, : TILE64 // 2] = 1.0
    tile_imag[:, : TILE64 // 2] = 0.0

    with pytest.raises(ValueError, match="stage1: empty real-dominant category"):
        quantize_complex_tile64_v2_script_bf16_f32(tile_real, tile_imag)


def test_script_bf16_f32_bundle_streaming_is_bit_identical_and_keeps_bf16_scale_bits() -> None:
    torch = pytest.importorskip("torch")

    out_target = 2 * TILE64
    in_target = 3 * TILE64
    weight = torch.from_numpy(
        _script_profile_wide_fixture(
            out_target // TILE64,
            in_target // TILE64,
        )
    ).to(torch.bfloat16)
    expected_codes, expected_scales = _canonical_bundle_from_wide_checkpoint(
        weight.to(torch.float32).numpy(),
        out_target,
        in_target,
    )
    nonstreaming_codes, nonstreaming_scales = (
        quantize_linear_to_fairy2i_bundle_v1_m64(
            weight,
            out_target,
            in_target,
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
            tensor_name="fixture.weight",
        )
    )
    strips = list(
        iter_quantize_linear_to_fairy2i_bundle_v1_m64(
            lambda rows, cols: weight[rows, cols],
            out_target,
            in_target,
            weight_shape=tuple(weight.shape),
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
            tensor_name="fixture.weight",
        )
    )
    streaming_codes, streaming_scales = merge_fairy2i_bundle_v1_m(strips)

    assert expected_codes.shape == (6, TILE64, 4, 16)
    assert expected_scales.shape == (6, 4, 2)
    assert len(strips) == 2
    assert all(strip_codes.shape == (3, TILE64, 4, 16) for strip_codes, _ in strips)
    assert all(strip_scales.shape == (3, 4, 2) for _, strip_scales in strips)
    np.testing.assert_array_equal(nonstreaming_codes, expected_codes)
    np.testing.assert_array_equal(nonstreaming_scales, expected_scales)
    np.testing.assert_array_equal(streaming_codes, expected_codes)
    np.testing.assert_array_equal(streaming_scales, expected_scales)

    expected_hash = hashlib.sha256(
        expected_codes.tobytes() + expected_scales.tobytes()
    ).hexdigest()
    assert expected_hash == "50933c0ea9e1616543da7364b9c03ceab81ecb4fa4a82bc0f4d064274044377f"
    assert (
        hashlib.sha256(
            nonstreaming_codes.tobytes() + nonstreaming_scales.tobytes()
        ).hexdigest()
        == expected_hash
    )
    assert (
        hashlib.sha256(
            streaming_codes.tobytes() + streaming_scales.tobytes()
        ).hexdigest()
        == expected_hash
    )
    assert streaming_scales.dtype == np.uint16
    assert np.all((streaming_scales & np.uint16(0x8000)) == 0)
    assert np.all((streaming_scales & np.uint16(0x7F80)) != np.uint16(0x7F80))


def test_script_bf16_f32_bundle_error_names_tensor_tile_and_branch() -> None:
    torch = pytest.importorskip("torch")
    weight = torch.ones((TILE64 * 2, TILE64 * 2), dtype=torch.bfloat16)

    with pytest.raises(
        ValueError,
        match=r"fixture\.weight: tile\(mb=0, kb=0\) branch=U: .*empty",
    ):
        quantize_linear_to_fairy2i_bundle_v1_m64(
            weight,
            TILE64,
            TILE64,
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
            tensor_name="fixture.weight",
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_script_bf16_f32_bundle_nonfinite_error_names_tensor_tile_and_branch(
    nonfinite: float,
) -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(20260729)
    weight = torch.randn(
        (TILE64 * 2, TILE64 * 2),
        generator=generator,
        dtype=torch.bfloat16,
    )
    weight[0, 0] = nonfinite

    with pytest.raises(
        ValueError,
        match=r"fixture\.weight: tile\(mb=0, kb=0\) branch=U: .*finite",
    ):
        quantize_linear_to_fairy2i_bundle_v1_m64(
            weight,
            TILE64,
            TILE64,
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
            tensor_name="fixture.weight",
        )


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
