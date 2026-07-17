from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from typing import Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - converter-only dependency
    torch = None  # type: ignore[assignment]


FAIRY2I_TILE64 = 64
TILE64 = FAIRY2I_TILE64

FAIRY2I_BUNDLE_M = 64
FAIRY2I_BUNDLE_K = 64
FAIRY2I_BUNDLE_M_SUBTILE = 16
FAIRY2I_BUNDLE_Q4 = 16

Fairy2IBranch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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


def pad_scale_matrix(scale: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    out_tiles = out_dim // TILE64
    in_tiles = in_dim // TILE64
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim != 2:
        raise ValueError(f"scale must be a 2D tile matrix, got shape {scale.shape}")
    if scale.shape[0] > out_tiles or scale.shape[1] > in_tiles:
        raise ValueError(f"cannot pad scale from {scale.shape} to {(out_tiles, in_tiles)}")
    if scale.shape == (out_tiles, in_tiles):
        return scale.astype(np.float32, copy=False)

    out = np.zeros((out_tiles, in_tiles), dtype=np.float32)
    out[: scale.shape[0], : scale.shape[1]] = scale
    return out


def split_wide_linear_components(
    weight: "torch.Tensor", out_target: int, in_target: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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

    del a
    return u_real, u_imag, w_real, w_imag, out_c, in_c


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


def encode_stage_codes_bundle_v1(stage_real: np.ndarray, stage_imag: np.ndarray) -> np.ndarray:
    """Pack four consecutive K codes per byte for the LUT-oriented bundle layout."""

    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")
    rows, cols = stage_real.shape
    if rows % FAIRY2I_BUNDLE_M != 0 or cols % FAIRY2I_BUNDLE_K != 0:
        raise ValueError(f"bundle_v1 packing requires dims divisible by {TILE64}, got {stage_real.shape}")

    abs_real = np.abs(stage_real)
    abs_imag = np.abs(stage_imag)
    choose_real = abs_real > abs_imag

    codes = np.empty(stage_real.shape, dtype=np.uint8)
    codes[choose_real] = np.where(stage_real[choose_real] >= 0.0, 1, 0)
    codes[~choose_real] = np.where(stage_imag[~choose_real] >= 0.0, 3, 2)

    codes = codes.reshape(rows, cols // FAIRY2I_BUNDLE_K, FAIRY2I_BUNDLE_Q4, 4)
    return (
        codes[..., 0]
        | (codes[..., 1] << 2)
        | (codes[..., 2] << 4)
        | (codes[..., 3] << 6)
    ).astype(np.uint8)


def pack_fairy2i_bundle_v1(
    branches: Mapping[str, Fairy2IBranch], branch_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Pack branch codes/scales as [physical_tile, slot, branch, lane]."""

    if not branch_order:
        raise ValueError("bundle_v1 requires at least one branch")
    if set(branches) != set(branch_order) or len(branches) != len(branch_order):
        raise ValueError(f"branch mismatch: expected {tuple(branch_order)}, got {tuple(branches)}")

    first = branches[branch_order[0]]
    rows, cols = first[0].shape
    if rows % FAIRY2I_BUNDLE_M != 0 or cols % FAIRY2I_BUNDLE_K != 0:
        raise ValueError(f"bundle_v1 packing requires dims divisible by {TILE64}, got {(rows, cols)}")

    mb_count = rows // FAIRY2I_BUNDLE_M
    kb_count = cols // FAIRY2I_BUNDLE_K
    code_planes: list[np.ndarray] = []
    scale_planes: list[np.ndarray] = []

    for name in branch_order:
        stage_real, stage_imag, scale_real, scale_imag = branches[name]
        if stage_real.shape != (rows, cols) or stage_imag.shape != (rows, cols):
            raise ValueError(f"branch {name} shape mismatch: {stage_real.shape}, {stage_imag.shape}")
        if scale_real.shape != (mb_count, kb_count) or scale_imag.shape != (mb_count, kb_count):
            raise ValueError(
                f"branch {name} scale shape mismatch: expected {(mb_count, kb_count)}, "
                f"got {scale_real.shape} and {scale_imag.shape}"
            )

        packed = encode_stage_codes_bundle_v1(stage_real, stage_imag)
        packed = packed.reshape(mb_count, 4, FAIRY2I_BUNDLE_M_SUBTILE, kb_count, FAIRY2I_BUNDLE_Q4)
        code_planes.append(packed.transpose(0, 3, 1, 4, 2))
        scale_planes.append(
            np.stack(
                (
                    np.ascontiguousarray(scale_real, dtype=np.float16),
                    np.ascontiguousarray(scale_imag, dtype=np.float16),
                ),
                axis=-1,
            )
        )

    codes = np.stack(code_planes, axis=-2)
    codes = np.ascontiguousarray(codes.reshape(mb_count * kb_count, 64, len(branch_order), 16))
    scales = np.stack(scale_planes, axis=-2)
    scales = np.ascontiguousarray(scales.reshape(mb_count * kb_count, len(branch_order), 2))
    return codes, scales


def unpack_fairy2i_bundle_v1(
    codes: np.ndarray,
    scales: np.ndarray,
    rows: int,
    cols: int,
    branch_order: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode bundle bytes for converter round-trip tests and diagnostics."""

    if rows % FAIRY2I_BUNDLE_M != 0 or cols % FAIRY2I_BUNDLE_K != 0:
        raise ValueError(f"bundle_v1 unpack requires dims divisible by {TILE64}, got {(rows, cols)}")
    mb_count = rows // FAIRY2I_BUNDLE_M
    kb_count = cols // FAIRY2I_BUNDLE_K
    branch_count = len(branch_order)
    expected_codes = (mb_count * kb_count, 64, branch_count, 16)
    expected_scales = (mb_count * kb_count, branch_count, 2)
    if codes.shape != expected_codes or codes.dtype != np.uint8:
        raise ValueError(
            f"bundle code shape/type mismatch: expected {expected_codes}/uint8, got {codes.shape}/{codes.dtype}"
        )
    if scales.shape != expected_scales or scales.dtype != np.float16:
        raise ValueError(
            f"bundle scale shape/type mismatch: expected {expected_scales}/float16, got {scales.shape}/{scales.dtype}"
        )

    tiled_codes = codes.reshape(mb_count, kb_count, 4, 16, branch_count, 16)
    tiled_scales = scales.reshape(mb_count, kb_count, branch_count, 2)
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for branch, name in enumerate(branch_order):
        packed = tiled_codes[..., branch, :].transpose(0, 2, 4, 1, 3)
        packed = packed.reshape(rows, kb_count, FAIRY2I_BUNDLE_Q4)
        decoded = np.empty((rows, kb_count, FAIRY2I_BUNDLE_K), dtype=np.uint8)
        decoded[..., 0::4] = packed & 0x03
        decoded[..., 1::4] = (packed >> 2) & 0x03
        decoded[..., 2::4] = (packed >> 4) & 0x03
        decoded[..., 3::4] = (packed >> 6) & 0x03
        branch_scales = tiled_scales[:, :, branch, :]
        result[name] = (
            decoded.reshape(rows, cols),
            np.ascontiguousarray(branch_scales[..., 0]),
            np.ascontiguousarray(branch_scales[..., 1]),
        )
    return result


def pack_fairy2i_tile64_v2_stage(
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


def quantize_linear_to_fairy2i_tile64_v2_branch_data(
    weight: "torch.Tensor", out_target: int, in_target: int
) -> dict[str, Fairy2IBranch]:
    u_real, u_imag, w_real, w_imag, _, _ = split_wide_linear_components(weight, out_target, in_target)

    (u0_real, u0_imag, u0_s_real, u0_s_imag), (u1_real, u1_imag, u1_s_real, u1_s_imag) = quantize_matrix_tile64_v2(
        u_real, u_imag
    )
    (w0_real, w0_imag, w0_s_real, w0_s_imag), (w1_real, w1_imag, w1_s_real, w1_s_imag) = quantize_matrix_tile64_v2(
        w_real, w_imag
    )

    out: dict[str, Fairy2IBranch] = {
        "U.s0": (u0_real, u0_imag, u0_s_real, u0_s_imag),
        "U.s1": (u1_real, u1_imag, u1_s_real, u1_s_imag),
        "W.s0": (w0_real, w0_imag, w0_s_real, w0_s_imag),
        "W.s1": (w1_real, w1_imag, w1_s_real, w1_s_imag),
    }

    del u_real, u_imag, w_real, w_imag
    gc.collect()

    return out


def quantize_linear_to_fairy2i_tile64_v2_stages(
    weight: "torch.Tensor", out_target: int, in_target: int
) -> dict[str, np.ndarray]:
    branches = quantize_linear_to_fairy2i_tile64_v2_branch_data(weight, out_target, in_target)
    return {
        name: pack_fairy2i_tile64_v2_stage(*branch)
        for name, branch in branches.items()
    }


def quantize_linear_to_fairy2i_bundle_v1_stages(
    weight: "torch.Tensor", out_target: int, in_target: int
) -> tuple[np.ndarray, np.ndarray]:
    branches = quantize_linear_to_fairy2i_tile64_v2_branch_data(weight, out_target, in_target)
    return pack_fairy2i_bundle_v1(branches, ("U.s0", "U.s1", "W.s0", "W.s1"))


def quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale_branch_data(
    weight: "torch.Tensor", quant_scale: "torch.Tensor", out_target: int, in_target: int
) -> dict[str, Fairy2IBranch]:
    _require_torch()

    u_real, u_imag, w_real, w_imag, out_c, in_c = split_wide_linear_components(weight, out_target, in_target)

    if out_target % TILE64 != 0 or in_target % TILE64 != 0:
        raise ValueError(f"tile64 packing requires target dims divisible by {TILE64}, got {(out_target, in_target)}")

    expected_scale_shape = (4, out_c // TILE64, in_c // TILE64)
    scale = quant_scale.to(torch.float32).cpu().numpy()
    if scale.shape != expected_scale_shape:
        raise ValueError(f"learned scale shape mismatch: expected {expected_scale_shape}, got {scale.shape}")

    u_s_real = pad_scale_matrix(scale[0], out_target, in_target)
    u_s_imag = pad_scale_matrix(scale[1], out_target, in_target)
    w_s_real = pad_scale_matrix(scale[2], out_target, in_target)
    w_s_imag = pad_scale_matrix(scale[3], out_target, in_target)

    out: dict[str, Fairy2IBranch] = {
        "U.s0": (u_real, u_imag, u_s_real, u_s_imag),
        "W.s0": (w_real, w_imag, w_s_real, w_s_imag),
    }

    del scale
    gc.collect()

    return out


def quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale(
    weight: "torch.Tensor", quant_scale: "torch.Tensor", out_target: int, in_target: int
) -> dict[str, np.ndarray]:
    branches = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale_branch_data(
        weight, quant_scale, out_target, in_target
    )
    return {
        name: pack_fairy2i_tile64_v2_stage(*branch)
        for name, branch in branches.items()
    }


def quantize_linear_to_fairy2i_bundle_v1_w1_learned_scale(
    weight: "torch.Tensor", quant_scale: "torch.Tensor", out_target: int, in_target: int
) -> tuple[np.ndarray, np.ndarray]:
    branches = quantize_linear_to_fairy2i_tile64_v2_w1_learned_scale_branch_data(
        weight, quant_scale, out_target, in_target
    )
    return pack_fairy2i_bundle_v1(branches, ("U.s0", "W.s0"))
