from __future__ import annotations

import gc

import numpy as np
from gguf.constants import QK_IFAIRY

from fairy2i.quant.tile64_v2 import pad_complex_matrix, phase_quant_v2

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - converter-only dependency
    torch = None  # type: ignore[assignment]


LEGACY_IFAIRY_TILE = QK_IFAIRY


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for legacy iFairy tensor conversion")


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
