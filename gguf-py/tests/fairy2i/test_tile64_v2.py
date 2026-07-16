from __future__ import annotations

import numpy as np
import pytest

from fairy2i.quant.tile64_v2 import (
    TILE64,
    encode_stage_codes,
    pack_fairy2i_bundle_v1,
    pack_fairy2i_tile64_v2_stage,
    quantize_matrix_tile64_v2,
    quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale,
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
