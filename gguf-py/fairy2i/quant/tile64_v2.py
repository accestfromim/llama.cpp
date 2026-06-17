from __future__ import annotations

import gc

import numpy as np
from gguf.constants import QK_IFAIRY

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - converter-only dependency
    torch = None  # type: ignore[assignment]


TILE64 = 64


def round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def phase_quant_v1(w_real: np.ndarray, w_imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    abs_real = np.abs(w_real)
    abs_imag = np.abs(w_imag)

    choose_real = abs_real > abs_imag
    choose_imag = abs_imag > abs_real

    ties = ~(choose_real | choose_imag)
    if np.any(ties):
        both_zero = ties & (abs_real == 0.0)
        same_sign = (w_real * w_imag) >= 0.0
        choose_imag |= ties & (~both_zero) & same_sign
        choose_real |= ties & (~both_zero) & (~same_sign)
        choose_real |= both_zero

    mask_real = choose_real
    mask_imag = choose_imag

    s_real = np.mean(abs_real[mask_real], dtype=np.float64) if np.any(mask_real) else 0.0
    s_imag = np.mean(abs_imag[mask_imag], dtype=np.float64) if np.any(mask_imag) else 0.0
    s_real = max(float(s_real), 1e-6) if np.isfinite(s_real) else 1e-6
    s_imag = max(float(s_imag), 1e-6) if np.isfinite(s_imag) else 1e-6

    q_real = np.zeros_like(w_real, dtype=np.float32)
    q_imag = np.zeros_like(w_imag, dtype=np.float32)

    q_real[mask_real] = np.where(w_real[mask_real] >= 0.0, s_real, -s_real)
    q_imag[mask_imag] = np.where(w_imag[mask_imag] >= 0.0, s_imag, -s_imag)

    return q_real, q_imag, s_real, s_imag


def phase_quant_v2(
    w_real: np.ndarray, w_imag: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, float, float], tuple[np.ndarray, np.ndarray, float, float]]:
    q0_real, q0_imag, s0_real, s0_imag = phase_quant_v1(w_real, w_imag)
    e_real = w_real - q0_real
    e_imag = w_imag - q0_imag
    q1_real, q1_imag, s1_real, s1_imag = phase_quant_v1(e_real, e_imag)
    return (q0_real, q0_imag, s0_real, s0_imag), (q1_real, q1_imag, s1_real, s1_imag)


def pad_complex_matrix(mat: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    out_src, in_src = mat.shape
    if out_src > out_dim or in_src > in_dim:
        raise ValueError(f"cannot pad from {(out_src, in_src)} to {(out_dim, in_dim)}")
    if out_src == out_dim and in_src == in_dim:
        return mat.astype(np.float32, copy=False)

    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    out[:out_src, :in_src] = mat
    return out


def quantize_tile64_once(tile_real: np.ndarray, tile_imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    abs_real = np.abs(tile_real)
    abs_imag = np.abs(tile_imag)

    is_real_dominant = abs_real > abs_imag
    is_imag_dominant = ~is_real_dominant

    real_count = int(np.count_nonzero(is_real_dominant))
    imag_count = int(np.count_nonzero(is_imag_dominant))

    real_scale = float(np.sum(abs_real[is_real_dominant], dtype=np.float64) / real_count) if real_count > 0 else 0.0
    imag_scale = float(np.sum(abs_imag[is_imag_dominant], dtype=np.float64) / imag_count) if imag_count > 0 else 0.0

    q_real = np.zeros_like(tile_real, dtype=np.float32)
    q_imag = np.zeros_like(tile_imag, dtype=np.float32)

    if real_count > 0:
        q_real[is_real_dominant] = np.where(tile_real[is_real_dominant] >= 0.0, real_scale, -real_scale)
    if imag_count > 0:
        q_imag[is_imag_dominant] = np.where(tile_imag[is_imag_dominant] >= 0.0, imag_scale, -imag_scale)

    return q_real, q_imag, real_scale, imag_scale


def quantize_matrix_tile64_v2(
    w_real: np.ndarray, w_imag: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if w_real.shape != w_imag.shape:
        raise ValueError(f"shape mismatch: {w_real.shape} vs {w_imag.shape}")
    if w_real.shape[0] % TILE64 != 0 or w_real.shape[1] % TILE64 != 0:
        raise ValueError(f"tile64 quantization requires dims divisible by {TILE64}, got {w_real.shape}")

    rows, cols = w_real.shape
    tile_rows = rows // TILE64
    tile_cols = cols // TILE64

    q0_real = np.zeros_like(w_real, dtype=np.float32)
    q0_imag = np.zeros_like(w_imag, dtype=np.float32)
    q1_real = np.zeros_like(w_real, dtype=np.float32)
    q1_imag = np.zeros_like(w_imag, dtype=np.float32)

    s0_real = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s0_imag = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s1_real = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s1_imag = np.zeros((tile_rows, tile_cols), dtype=np.float32)

    for tr in range(tile_rows):
        row_slice = slice(tr * TILE64, (tr + 1) * TILE64)
        for tc in range(tile_cols):
            col_slice = slice(tc * TILE64, (tc + 1) * TILE64)

            tile_real = w_real[row_slice, col_slice]
            tile_imag = w_imag[row_slice, col_slice]

            stage0_real, stage0_imag, scale0_real, scale0_imag = quantize_tile64_once(tile_real, tile_imag)
            resid_real = tile_real - stage0_real
            resid_imag = tile_imag - stage0_imag
            stage1_real, stage1_imag, scale1_real, scale1_imag = quantize_tile64_once(resid_real, resid_imag)

            q0_real[row_slice, col_slice] = stage0_real
            q0_imag[row_slice, col_slice] = stage0_imag
            q1_real[row_slice, col_slice] = stage1_real
            q1_imag[row_slice, col_slice] = stage1_imag

            s0_real[tr, tc] = scale0_real
            s0_imag[tr, tc] = scale0_imag
            s1_real[tr, tc] = scale1_real
            s1_imag[tr, tc] = scale1_imag

    return (q0_real, q0_imag, s0_real, s0_imag), (q1_real, q1_imag, s1_real, s1_imag)


def encode_stage_codes(stage_real: np.ndarray, stage_imag: np.ndarray) -> np.ndarray:
    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")
    if stage_real.shape[1] % TILE64 != 0:
        raise ValueError(f"tile64 code packing requires cols divisible by {TILE64}, got {stage_real.shape[1]}")

    abs_real = np.abs(stage_real)
    abs_imag = np.abs(stage_imag)
    choose_real = abs_real > abs_imag

    codes = np.empty(stage_real.shape, dtype=np.uint8)
    codes[choose_real] = np.where(stage_real[choose_real] >= 0.0, 1, 0)
    codes[~choose_real] = np.where(stage_imag[~choose_real] >= 0.0, 3, 2)

    rows, cols = codes.shape
    n_blocks = cols // TILE64
    codes = codes.reshape(rows, n_blocks, 4, 16)
    packed = (
        codes[:, :, 0, :]
        | (codes[:, :, 1, :] << 2)
        | (codes[:, :, 2, :] << 4)
        | (codes[:, :, 3, :] << 6)
    ).astype(np.uint8)
    return packed.reshape(rows, n_blocks * 16)


def pack_ifairy64_stage(
    stage_real: np.ndarray,
    stage_imag: np.ndarray,
    scale_real: np.ndarray,
    scale_imag: np.ndarray,
) -> np.ndarray:
    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")
    rows, cols = stage_real.shape
    if rows % TILE64 != 0 or cols % TILE64 != 0:
        raise ValueError(f"tile64 packing requires dims divisible by {TILE64}, got {stage_real.shape}")

    n_blocks = cols // TILE64
    if scale_real.shape != (rows // TILE64, n_blocks) or scale_imag.shape != (rows // TILE64, n_blocks):
        raise ValueError(
            f"scale shape mismatch: expected {(rows // TILE64, n_blocks)}, got {scale_real.shape} and {scale_imag.shape}"
        )

    codes = encode_stage_codes(stage_real, stage_imag).reshape(rows, n_blocks, 16)

    out = np.empty((rows, n_blocks, 20), dtype=np.uint8)
    out[:, :, :16] = codes

    scale_real_rows = np.repeat(np.ascontiguousarray(scale_real, dtype=np.float16), TILE64, axis=0)
    scale_imag_rows = np.repeat(np.ascontiguousarray(scale_imag, dtype=np.float16), TILE64, axis=0)
    out[:, :, 16:18] = scale_real_rows.view(np.uint8).reshape(rows, n_blocks, 2)
    out[:, :, 18:20] = scale_imag_rows.view(np.uint8).reshape(rows, n_blocks, 2)

    return out.reshape(rows, n_blocks * 20)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for Fairy2i tensor conversion")


def quantize_linear_to_ifairy64_stages(weight: "torch.Tensor", out_target: int, in_target: int) -> dict[str, np.ndarray]:
    _require_torch()

    a = weight.to(torch.float32).cpu().numpy()
    out_real, in_real = a.shape
    if out_real % 2 != 0 or in_real % 2 != 0:
        raise ValueError(f"linear weight shape must be even, got {a.shape}")

    out_c = out_real // 2
    in_c = in_real // 2

    a11 = a[:out_c, :in_c]
    a12 = a[:out_c, in_c:]
    a21 = a[out_c:, :in_c]
    a22 = a[out_c:, in_c:]

    u_real = pad_complex_matrix(0.5 * (a11 + a22), out_target, in_target)
    u_imag = pad_complex_matrix(0.5 * (a21 - a12), out_target, in_target)
    w_real = pad_complex_matrix(0.5 * (a11 - a22), out_target, in_target)
    w_imag = pad_complex_matrix(0.5 * (a12 + a21), out_target, in_target)

    (u0_real, u0_imag, u0_s_real, u0_s_imag), (u1_real, u1_imag, u1_s_real, u1_s_imag) = quantize_matrix_tile64_v2(
        u_real, u_imag
    )
    (w0_real, w0_imag, w0_s_real, w0_s_imag), (w1_real, w1_imag, w1_s_real, w1_s_imag) = quantize_matrix_tile64_v2(
        w_real, w_imag
    )

    out = {
        "U.s0": pack_ifairy64_stage(u0_real, u0_imag, u0_s_real, u0_s_imag),
        "U.s1": pack_ifairy64_stage(u1_real, u1_imag, u1_s_real, u1_s_imag),
        "W.s0": pack_ifairy64_stage(w0_real, w0_imag, w0_s_real, w0_s_imag),
        "W.s1": pack_ifairy64_stage(w1_real, w1_imag, w1_s_real, w1_s_imag),
    }

    del a
    del u_real, u_imag, w_real, w_imag
    del u0_real, u0_imag, u1_real, u1_imag
    del w0_real, w0_imag, w1_real, w1_imag
    gc.collect()

    return out


def pack_ifairy_stage(stage_real: np.ndarray, stage_imag: np.ndarray, d_real: float, d_imag: float) -> np.ndarray:
    stage_real = np.ascontiguousarray(stage_real, dtype=np.float32)
    stage_imag = np.ascontiguousarray(stage_imag, dtype=np.float32)

    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")

    rows, cols = stage_real.shape
    if cols % QK_IFAIRY != 0:
        raise ValueError(f"inner dim {cols} is not divisible by QK_IFAIRY={QK_IFAIRY}")

    mask_real = stage_real != 0.0
    mask_imag = stage_imag != 0.0
    both = mask_real & mask_imag
    if np.any(both):
        abs_real = np.abs(stage_real)
        abs_imag = np.abs(stage_imag)
        choose_real = abs_real > abs_imag
        choose_imag = abs_imag > abs_real
        ties = ~(choose_real | choose_imag)
        if np.any(ties):
            same_sign = (stage_real * stage_imag) >= 0.0
            choose_imag |= ties & same_sign
            choose_real |= ties & (~same_sign)
        mask_real = (mask_real & ~both) | (both & choose_real)
        mask_imag = (mask_imag & ~both) | (both & ~choose_real)

    d_real = 1e-6 if not np.isfinite(d_real) else max(float(d_real), 1e-6)
    d_imag = 1e-6 if not np.isfinite(d_imag) else max(float(d_imag), 1e-6)
    row_all_zero = ~np.any(mask_real | mask_imag, axis=1)

    d_real_arr = np.full(rows, d_real, dtype=np.float32)
    d_imag_arr = np.full(rows, d_imag, dtype=np.float32)
    d_real_arr[row_all_zero] = 0.0
    d_imag_arr[row_all_zero] = 0.0

    codes = np.zeros((rows, cols), dtype=np.uint8)

    real_pos = mask_real & (stage_real >= 0.0)
    real_neg = mask_real & (~real_pos)
    imag_pos = mask_imag & (stage_imag >= 0.0)
    imag_neg = mask_imag & (~imag_pos)

    codes[real_neg] = 0
    codes[real_pos] = 1
    codes[imag_neg] = 2
    codes[imag_pos] = 3

    zero_mask = ~(mask_real | mask_imag)
    prefer_real = d_real_arr <= d_imag_arr
    codes[zero_mask & prefer_real[:, None]] = 1
    codes[zero_mask & (~prefer_real)[:, None]] = 3

    n_blocks = cols // QK_IFAIRY
    codes = codes.reshape(rows, n_blocks, 4, 4, 16)
    packed = (
        codes[:, :, :, 0, :]
        | (codes[:, :, :, 1, :] << 2)
        | (codes[:, :, :, 2, :] << 4)
        | (codes[:, :, :, 3, :] << 6)
    ).astype(np.uint8)
    packed = packed.reshape(rows, n_blocks, 64)

    d_real_bytes = d_real_arr.astype(np.float16).view(np.uint8).reshape(rows, 2)
    d_imag_bytes = d_imag_arr.astype(np.float16).view(np.uint8).reshape(rows, 2)

    out = np.empty((rows, n_blocks, 68), dtype=np.uint8)
    out[:, :, :64] = packed
    out[:, :, 64:66] = d_real_bytes[:, None, :]
    out[:, :, 66:68] = d_imag_bytes[:, None, :]

    return out.reshape(rows, n_blocks * 68)


def quantize_linear_to_ifairy_stages_legacy(weight: "torch.Tensor", out_target: int, in_target: int) -> dict[str, np.ndarray]:
    _require_torch()

    a = weight.to(torch.float32).cpu().numpy()
    out_real, in_real = a.shape
    if out_real % 2 != 0 or in_real % 2 != 0:
        raise ValueError(f"linear weight shape must be even, got {a.shape}")

    out_c = out_real // 2
    in_c = in_real // 2

    a11 = a[:out_c, :in_c]
    a12 = a[:out_c, in_c:]
    a21 = a[out_c:, :in_c]
    a22 = a[out_c:, in_c:]

    u_real = 0.5 * (a11 + a22)
    u_imag = 0.5 * (a21 - a12)
    w_real = 0.5 * (a11 - a22)
    w_imag = 0.5 * (a12 + a21)

    (u0_real, u0_imag, u0_s_real, u0_s_imag), (u1_real, u1_imag, u1_s_real, u1_s_imag) = phase_quant_v2(
        u_real, u_imag
    )
    (w0_real, w0_imag, w0_s_real, w0_s_imag), (w1_real, w1_imag, w1_s_real, w1_s_imag) = phase_quant_v2(
        w_real, w_imag
    )

    u0_real = pad_complex_matrix(u0_real, out_target, in_target)
    u0_imag = pad_complex_matrix(u0_imag, out_target, in_target)
    u1_real = pad_complex_matrix(u1_real, out_target, in_target)
    u1_imag = pad_complex_matrix(u1_imag, out_target, in_target)
    w0_real = pad_complex_matrix(w0_real, out_target, in_target)
    w0_imag = pad_complex_matrix(w0_imag, out_target, in_target)
    w1_real = pad_complex_matrix(w1_real, out_target, in_target)
    w1_imag = pad_complex_matrix(w1_imag, out_target, in_target)

    out = {
        "U.s0": pack_ifairy_stage(u0_real, u0_imag, u0_s_real, u0_s_imag),
        "U.s1": pack_ifairy_stage(u1_real, u1_imag, u1_s_real, u1_s_imag),
        "W.s0": pack_ifairy_stage(w0_real, w0_imag, w0_s_real, w0_s_imag),
        "W.s1": pack_ifairy_stage(w1_real, w1_imag, w1_s_real, w1_s_imag),
    }

    del a
    del u_real, u_imag, w_real, w_imag
    del u0_real, u0_imag, u1_real, u1_imag
    del w0_real, w0_imag, w1_real, w1_imag
    gc.collect()

    return out
