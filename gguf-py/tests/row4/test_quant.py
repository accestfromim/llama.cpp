from __future__ import annotations

import numpy as np
import pytest

from row4.quant import (
    decode_row4_codes,
    pack_row4_m16k128,
    pack_row4_m32k256_pair2,
    pack_split8_group,
    pack_w8_m16k128,
    quantize_row4_codes,
    quantize_w8_rows,
)


def test_all_16_codebook_entries_round_trip() -> None:
    codes = np.arange(16, dtype=np.uint8)
    decoded = decode_row4_codes(codes)
    assert decoded.shape == (4, 16)
    expected_by_code = np.asarray(
        [
            [2, 0, 0, 0],
            [0, 0, 0, -2],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [0, 0, 0, 2],
            [-2, 0, 0, 0],
            [-1, -1, 1, 1],
            [-1, 1, -1, 1],
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [1, -1, -1, 1],
            [-1, -1, -1, -1],
            [0, -2, 0, 0],
            [0, 0, -2, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(decoded.T, expected_by_code)
    assert set(np.unique(decoded)).issubset({-2, -1, 0, 1, 2})
    np.testing.assert_array_equal(quantize_row4_codes(decoded), codes[None, :])


def test_ties_and_signed_zero_choose_positive_imaginary_axis() -> None:
    zero = np.asarray([[0.0], [-0.0], [0.0], [-0.0]], dtype=np.float32)
    assert quantize_row4_codes(zero).item() == 0xA

    # u=(+1,+1), v=(+1,-1): both ties must choose the imaginary axis.
    tie = np.asarray([[2.0], [-2.0], [0.0], [0.0]], dtype=np.float32)
    assert quantize_row4_codes(tie).item() == 0xE


def test_split8_known_k16_pack() -> None:
    packed = pack_split8_group(np.arange(16, dtype=np.uint8))
    assert packed.tobytes() == bytes.fromhex("80 91 a2 b3 c4 d5 e6 f7")


def test_m16k128_offsets_match_canonical_formula() -> None:
    codes = np.empty((8, 256), dtype=np.uint8)
    for group in range(codes.shape[0]):
        codes[group] = (group * 3 + np.arange(256)) & 0xF

    packed = pack_row4_m16k128(codes)
    assert packed.shape == (2, 2, 4, 64)
    flat = packed.reshape(-1)
    for output_tile in range(2):
        for k_tile in range(2):
            for group in range(4):
                for split in range(8):
                    for lane in range(8):
                        offset = ((((output_tile * 2 + k_tile) * 4 + group) * 8 + split) * 8 + lane)
                        source_group = output_tile * 4 + group
                        low_k = k_tile * 128 + split * 16 + lane
                        high_k = low_k + 8
                        expected = int(codes[source_group, low_k]) | (int(codes[source_group, high_k]) << 4)
                        assert int(flat[offset]) == expected


def _pair2_code_at(packed: np.ndarray, group: int, k: int) -> int:
    output_tile = group // 8
    output_group = group % 8
    k_tile = k // 256
    k_half = (k % 256) // 128
    r = k % 128
    lane = (r // 16) * 4 + ((r % 8) // 2)
    byte_index = lane * 4 + 2 * k_half + (r & 1)
    byte = int(packed[output_tile, k_tile, output_group, byte_index])
    return byte >> 4 if r & 8 else byte & 0xF


def _v1_code_at(packed: np.ndarray, group: int, k: int) -> int:
    output_tile = group // 4
    output_group = group % 4
    k_tile = k // 128
    r = k % 128
    byte = int(packed[output_tile, k_tile, output_group, (r // 16) * 8 + (r % 8)])
    return byte >> 4 if r & 8 else byte & 0xF


def test_m32k256_pair2_offsets_exhaustively_match_canonical_formula() -> None:
    codes = (
        np.arange(16 * 512, dtype=np.uint32).reshape(16, 512)
        + np.arange(16, dtype=np.uint32)[:, None] * 3
    ).astype(np.uint8) & np.uint8(0xF)

    packed = pack_row4_m32k256_pair2(codes)
    assert packed.shape == (2, 2, 8, 128)
    assert packed.nbytes == codes.size // 2
    for group in range(codes.shape[0]):
        for k in range(codes.shape[1]):
            assert _pair2_code_at(packed, group, k) == int(codes[group, k])


def test_m32k256_pair2_is_lossless_v1_payload_permutation() -> None:
    rng = np.random.default_rng(0x4A32)
    codes = rng.integers(0, 16, size=(16, 512), dtype=np.uint8)
    packed_v1 = pack_row4_m16k128(codes)
    packed_v2 = pack_row4_m32k256_pair2(codes)

    assert packed_v1.nbytes == packed_v2.nbytes == codes.size // 2
    expected_from_v1 = (
        packed_v1.reshape(2, 2, 2, 2, 4, 8, 4, 2)
        .transpose(0, 2, 1, 4, 5, 6, 3, 7)
        .reshape(2, 2, 8, 128)
    )
    np.testing.assert_array_equal(packed_v2, expected_from_v1)

    boundary_k = (0, 1, 7, 8, 15, 16, 127, 128, 255, 256, 383, 511)
    for group in (0, 3, 4, 7, 8, 11, 12, 15):
        for k in boundary_k:
            expected = int(codes[group, k])
            assert _v1_code_at(packed_v1, group, k) == expected
            assert _pair2_code_at(packed_v2, group, k) == expected

    unpacked_v1 = np.asarray(
        [
            [_v1_code_at(packed_v1, group, k) for k in range(codes.shape[1])]
            for group in range(codes.shape[0])
        ],
        dtype=np.uint8,
    )
    unpacked_v2 = np.asarray(
        [
            [_pair2_code_at(packed_v2, group, k) for k in range(codes.shape[1])]
            for group in range(codes.shape[0])
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(unpacked_v1, codes)
    np.testing.assert_array_equal(unpacked_v2, codes)


def test_w8_half_away_zero_row_and_tiled_order() -> None:
    row = np.asarray([-127.0, -2.5, -0.5, 0.0, 0.5, 2.5, 127.0], dtype=np.float32)
    weight = np.stack((row, np.zeros_like(row)))
    quantized, scales = quantize_w8_rows(weight)
    np.testing.assert_array_equal(
        quantized[0],
        np.asarray([-127, -3, -1, 0, 1, 3, 127], dtype=np.int8),
    )
    np.testing.assert_array_equal(quantized[1], np.zeros_like(quantized[1]))
    np.testing.assert_array_equal(scales, np.asarray([1.0, 0.0], dtype=np.float32))

    logical = np.arange(16 * 128, dtype=np.int16).astype(np.int8).reshape(16, 128)
    tiled = pack_w8_m16k128(logical)
    assert tiled.shape == (1, 1, 16, 128)
    np.testing.assert_array_equal(tiled[0, 0], logical)


@pytest.mark.parametrize("shape", [(3, 128), (4, 127)])
def test_row4_packer_rejects_unaligned_shapes(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="O16/K128"):
        pack_row4_m16k128(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(7, 256), (8, 255), (4, 128)])
def test_row4_pair2_packer_rejects_unaligned_shapes(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="O32/K256"):
        pack_row4_m32k256_pair2(np.zeros(shape, dtype=np.uint8))
