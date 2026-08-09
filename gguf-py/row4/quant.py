from __future__ import annotations

import numpy as np


M_TILE = 16
K_TILE = 128


def _axis_code(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    real_dominant = np.abs(real) > np.abs(imag)
    real_code = np.where(real < 0, 1, 0)
    imag_code = np.where(imag < 0, 3, 2)
    return np.where(real_dominant, real_code, imag_code).astype(np.uint8)


def quantize_row4_codes(weight: np.ndarray) -> np.ndarray:
    """Return one 4-bit U/V axis code for each four-row weight group.

    ``weight`` is interpreted as exact BF16 values widened to F32 by the
    caller. Arithmetic and comparisons are performed in F32. Ties select the
    imaginary axis and both +0 and -0 select the positive signed axis.
    """

    weight_f32 = np.asarray(weight, dtype=np.float32)
    if weight_f32.ndim != 2:
        raise ValueError(f"Row4 weight must be 2D, got shape {weight_f32.shape}")
    out_features, _ = weight_f32.shape
    if out_features % 4 != 0:
        raise ValueError(f"Row4 output dimension must be divisible by 4, got {out_features}")
    if not np.isfinite(weight_f32).all():
        raise ValueError("Row4 weight contains a non-finite value")

    grouped = weight_f32.reshape(out_features // 4, 4, -1)
    a, b, c, d = (grouped[:, index, :] for index in range(4))
    half = np.float32(0.5)
    u_re = half * (a + d)
    u_im = half * (c - b)
    v_re = half * (a - d)
    v_im = half * (b + c)
    u_axis = _axis_code(u_re, u_im)
    v_axis = _axis_code(v_re, v_im)
    return np.bitwise_or(u_axis, np.left_shift(v_axis, 2)).astype(np.uint8)


def decode_row4_codes(codes: np.ndarray) -> np.ndarray:
    """Decode Row4 U/V axis codes to signed INT8 rows in {0, +/-1, +/-2}."""

    code_array = np.asarray(codes, dtype=np.uint8)
    if np.any(code_array > 15):
        raise ValueError("Row4 codes must be in [0, 15]")
    axes = np.asarray(
        ((1, 0), (-1, 0), (0, 1), (0, -1)),
        dtype=np.int8,
    )
    u = axes[code_array & np.uint8(3)]
    v = axes[(code_array >> np.uint8(2)) & np.uint8(3)]
    u_re, u_im = u[..., 0], u[..., 1]
    v_re, v_im = v[..., 0], v[..., 1]
    return np.stack(
        (u_re + v_re, -u_im + v_im, u_im + v_im, u_re - v_re),
        axis=-2,
    ).astype(np.int8)


def pack_split8_group(codes: np.ndarray) -> np.ndarray:
    """Pack a K stream with k=s*16+j low and k=s*16+8+j high."""

    code_array = np.asarray(codes, dtype=np.uint8)
    if code_array.ndim != 1 or code_array.size % 16 != 0:
        raise ValueError(f"split8 code stream must be 1D and K16 aligned, got {code_array.shape}")
    if np.any(code_array > 15):
        raise ValueError("Row4 codes must be in [0, 15]")
    split = code_array.reshape(-1, 16)
    return np.bitwise_or(split[:, :8], np.left_shift(split[:, 8:], 4)).astype(np.uint8)


def pack_row4_m16k128(codes: np.ndarray) -> np.ndarray:
    """Pack logical [O/4,K] codes to physical [O/16,K/128,4,64]."""

    code_array = np.asarray(codes, dtype=np.uint8)
    if code_array.ndim != 2:
        raise ValueError(f"Row4 codes must be 2D, got shape {code_array.shape}")
    groups, in_features = code_array.shape
    if groups % 4 != 0 or in_features % K_TILE != 0:
        raise ValueError(
            "Row4 codes require O16/K128 alignment, got "
            f"O={groups * 4}, K={in_features}"
        )
    if np.any(code_array > 15):
        raise ValueError("Row4 codes must be in [0, 15]")

    m_tiles = groups // 4
    k_tiles = in_features // K_TILE
    split = code_array.reshape(m_tiles, 4, k_tiles, 8, 16)
    packed = np.bitwise_or(split[..., :8], np.left_shift(split[..., 8:], 4))
    return np.ascontiguousarray(packed.transpose(0, 2, 1, 3, 4).reshape(m_tiles, k_tiles, 4, 64))


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + np.float32(0.5)), values)


def pack_w8_m16k128(quantized: np.ndarray) -> np.ndarray:
    """Pack logical signed INT8 [O,K] to physical [O/16,K/128,16,128]."""

    codes = np.asarray(quantized, dtype=np.int8)
    if codes.ndim != 2:
        raise ValueError(f"W8 codes must be 2D, got shape {codes.shape}")
    out_features, in_features = codes.shape
    if out_features % M_TILE != 0 or in_features % K_TILE != 0:
        raise ValueError(
            f"W8 codes require O16/K128 alignment, got O={out_features}, K={in_features}"
        )
    tiled = codes.reshape(out_features // M_TILE, M_TILE, in_features // K_TILE, K_TILE)
    return np.ascontiguousarray(tiled.transpose(0, 2, 1, 3))


def quantize_w8_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize BF16-widened lm_head rows to signed INT8 plus F32 scale."""

    weight_f32 = np.asarray(weight, dtype=np.float32)
    if weight_f32.ndim != 2:
        raise ValueError(f"W8 weight must be 2D, got shape {weight_f32.shape}")
    if not np.isfinite(weight_f32).all():
        raise ValueError("W8 weight contains a non-finite value")
    amax = np.max(np.abs(weight_f32), axis=1).astype(np.float32)
    scales = (amax / np.float32(127.0)).astype(np.float32)
    nonzero = amax != 0
    normalized = np.zeros_like(weight_f32, dtype=np.float32)
    np.divide(weight_f32, scales[:, None], out=normalized, where=nonzero[:, None])
    quantized = np.clip(_round_half_away_from_zero(normalized), -127, 127).astype(np.int8)
    quantized[~nonzero, :] = 0
    scales[~nonzero] = np.float32(0.0)
    return quantized, scales
