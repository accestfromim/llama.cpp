from __future__ import annotations

import numpy as np
import pytest

from fairy2i.quant.tile64_v2 import TILE64, encode_stage_codes, pack_ifairy64_stage, quantize_matrix_tile64_v2


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


def test_pack_ifairy64_stage_size() -> None:
    stage_real = np.ones((TILE64, TILE64), dtype=np.float32)
    stage_imag = np.zeros((TILE64, TILE64), dtype=np.float32)
    scale_real = np.ones((1, 1), dtype=np.float32)
    scale_imag = np.zeros((1, 1), dtype=np.float32)

    packed = pack_ifairy64_stage(stage_real, stage_imag, scale_real, scale_imag)

    assert packed.shape == (TILE64, 20)
    assert packed.dtype == np.uint8


def test_quantize_matrix_tile64_requires_divisible_dims() -> None:
    real = np.zeros((TILE64, TILE64 + 1), dtype=np.float32)
    imag = np.zeros_like(real)

    with pytest.raises(ValueError, match="requires dims divisible"):
        quantize_matrix_tile64_v2(real, imag)
