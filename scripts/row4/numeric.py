"""Pure-Torch implementation of the frozen Qwen3 Row4 numeric contract.

This module deliberately does not import ``row4_qat`` or the converter.  It is
used as an independent oracle for both the converter's physical layout and the
runtime kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


M_TILE = 16
K_TILE = 128
ROW4_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class ActivationQuantized:
    carrier_f32: torch.Tensor
    bf16: torch.Tensor
    codes_i8: torch.Tensor
    scales_f32: torch.Tensor


@dataclass(frozen=True)
class LinearResult:
    accumulator_i32: torch.Tensor
    scaled_f32: torch.Tensor
    output_bf16: torch.Tensor
    carrier_f32: torch.Tensor


def _require_cpu(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor


def bf16_roundtrip(carrier_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the profile's BF16 RNE round-trip to an F32 carrier."""

    carrier_f32 = _require_cpu(carrier_f32, "carrier_f32")
    if carrier_f32.dtype != torch.float32:
        raise TypeError(f"carrier_f32 must be torch.float32, got {carrier_f32.dtype}")
    bf16 = carrier_f32.to(torch.bfloat16)
    return bf16, bf16.to(torch.float32)


def round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    """Round finite F32 values with ties away from zero."""

    values = _require_cpu(values, "values")
    if values.dtype != torch.float32:
        raise TypeError(f"values must be torch.float32, got {values.dtype}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot round a non-finite value")
    return torch.copysign(torch.floor(torch.abs(values) + 0.5), values)


def quantize_activation(carrier_f32: torch.Tensor) -> ActivationQuantized:
    """Quantize the last dimension with one F32 scale per token."""

    if carrier_f32.ndim < 1:
        raise ValueError("activation must have at least one dimension")
    bf16, rounded_carrier = bf16_roundtrip(carrier_f32)
    flat = rounded_carrier.reshape(-1, rounded_carrier.shape[-1])
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("activation contains a non-finite value")
    amax = torch.amax(torch.abs(flat), dim=1)
    floor = torch.tensor(1.0e-8, dtype=torch.float32)
    scales = torch.maximum(amax / 127.0, floor)
    normalized = flat / scales[:, None]
    codes = torch.clamp(round_half_away_from_zero(normalized), -127, 127).to(torch.int8)
    return ActivationQuantized(
        carrier_f32=rounded_carrier.reshape_as(carrier_f32),
        bf16=bf16.reshape_as(carrier_f32),
        codes_i8=codes.reshape_as(carrier_f32),
        scales_f32=scales.reshape(carrier_f32.shape[:-1]),
    )


def _axis_code(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    # Strict greater-than makes ties select imaginary.  Comparing with < 0
    # makes both +0 and -0 select the positive sign.
    real_dominant = torch.abs(real) > torch.abs(imag)
    real_code = torch.where(real < 0, 1, 0)
    imag_code = torch.where(imag < 0, 3, 2)
    return torch.where(real_dominant, real_code, imag_code).to(torch.uint8)


def quantize_row4_codes(weight_bf16: torch.Tensor) -> torch.Tensor:
    """Encode BF16 latent weights into logical ``[O/4, K]`` nibbles."""

    weight_bf16 = _require_cpu(weight_bf16, "weight_bf16")
    if weight_bf16.dtype != torch.bfloat16 or weight_bf16.ndim != 2:
        raise TypeError("weight_bf16 must be a two-dimensional BF16 tensor")
    out_features, _ = weight_bf16.shape
    if out_features % 4 != 0:
        raise ValueError(f"Row4 output size must be divisible by four, got {out_features}")
    weight = weight_bf16.to(torch.float32)
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Row4 latent weight contains a non-finite value")
    grouped = weight.reshape(out_features // 4, 4, -1)
    a, b, c, d = (grouped[:, index, :] for index in range(4))
    u_re = 0.5 * (a + d)
    u_im = 0.5 * (c - b)
    v_re = 0.5 * (a - d)
    v_im = 0.5 * (b + c)
    u_axis = _axis_code(u_re, u_im).to(torch.int64)
    v_axis = _axis_code(v_re, v_im).to(torch.int64)
    return (u_axis | (v_axis << 2)).to(torch.uint8)


def decode_row4_codes(codes_u8: torch.Tensor) -> torch.Tensor:
    """Decode logical nibbles into ``[O, K]`` values in {0,+/-1,+/-2}."""

    codes_u8 = _require_cpu(codes_u8, "codes_u8")
    if codes_u8.dtype != torch.uint8 or codes_u8.ndim != 2:
        raise TypeError("codes_u8 must be a two-dimensional uint8 tensor")
    if bool((codes_u8 > 15).any()):
        raise ValueError("Row4 code must be in [0, 15]")
    axes = torch.tensor(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=torch.int8)
    codes = codes_u8.to(torch.int64)
    u = axes[codes & 3]
    v = axes[(codes >> 2) & 3]
    u_re, u_im = u[..., 0], u[..., 1]
    v_re, v_im = v[..., 0], v[..., 1]
    decoded = torch.stack(
        (u_re + v_re, -u_im + v_im, u_im + v_im, u_re - v_re),
        dim=1,
    )
    return decoded.reshape(codes.shape[0] * 4, codes.shape[1]).to(torch.int8)


def pack_row4_m16k128(codes_u8: torch.Tensor) -> torch.Tensor:
    """Pack logical codes to ``[O/16, K/128, 4, 64]`` split8 bytes."""

    codes_u8 = _require_cpu(codes_u8, "codes_u8")
    if codes_u8.dtype != torch.uint8 or codes_u8.ndim != 2:
        raise TypeError("codes_u8 must be a two-dimensional uint8 tensor")
    groups, in_features = codes_u8.shape
    if groups % 4 != 0 or in_features % K_TILE != 0:
        raise ValueError(f"Row4 packing requires O16/K128, got O={groups * 4}, K={in_features}")
    if bool((codes_u8 > 15).any()):
        raise ValueError("Row4 code must be in [0, 15]")
    m_tiles = groups // 4
    k_tiles = in_features // K_TILE
    split = codes_u8.reshape(m_tiles, 4, k_tiles, 8, 16).to(torch.int16)
    packed = split[..., :8] | (split[..., 8:] << 4)
    return packed.permute(0, 2, 1, 3, 4).contiguous().reshape(m_tiles, k_tiles, 4, 64).to(torch.uint8)


def quantize_w8_rows(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline lm_head row quantization, including the all-zero-row rule."""

    weight_bf16 = _require_cpu(weight_bf16, "weight_bf16")
    if weight_bf16.dtype != torch.bfloat16 or weight_bf16.ndim != 2:
        raise TypeError("weight_bf16 must be a two-dimensional BF16 tensor")
    weight = weight_bf16.to(torch.float32)
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("W8 weight contains a non-finite value")
    amax = torch.amax(torch.abs(weight), dim=1)
    scales = amax / 127.0
    nonzero = amax != 0
    normalized = torch.zeros_like(weight)
    normalized[nonzero] = weight[nonzero] / scales[nonzero, None]
    codes = torch.clamp(round_half_away_from_zero(normalized), -127, 127).to(torch.int8)
    codes[~nonzero] = 0
    scales[~nonzero] = 0.0
    return codes, scales.to(torch.float32)


def pack_w8_m16k128(codes_i8: torch.Tensor) -> torch.Tensor:
    """Pack logical W8 rows to ``[O/16, K/128, 16, 128]``."""

    codes_i8 = _require_cpu(codes_i8, "codes_i8")
    if codes_i8.dtype != torch.int8 or codes_i8.ndim != 2:
        raise TypeError("codes_i8 must be a two-dimensional int8 tensor")
    out_features, in_features = codes_i8.shape
    if out_features % M_TILE != 0 or in_features % K_TILE != 0:
        raise ValueError(f"W8 packing requires O16/K128, got O={out_features}, K={in_features}")
    return (
        codes_i8.reshape(out_features // M_TILE, M_TILE, in_features // K_TILE, K_TILE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def int32_accumulate(activation_i8: torch.Tensor, weight_i8: torch.Tensor) -> torch.Tensor:
    """Small, exact oracle GEMM for selected output rows."""

    activation_i8 = _require_cpu(activation_i8, "activation_i8")
    weight_i8 = _require_cpu(weight_i8, "weight_i8")
    if activation_i8.dtype != torch.int8 or weight_i8.dtype != torch.int8:
        raise TypeError("oracle GEMM operands must be int8")
    if activation_i8.ndim != 2 or weight_i8.ndim != 2:
        raise ValueError("oracle GEMM operands must be two-dimensional")
    if activation_i8.shape[1] != weight_i8.shape[1]:
        raise ValueError("oracle GEMM K dimensions do not match")
    products = activation_i8[:, None, :].to(torch.int32) * weight_i8[None, :, :].to(torch.int32)
    return torch.sum(products, dim=-1, dtype=torch.int32)


def finish_linear(
    accumulator_i32: torch.Tensor,
    activation_scale_f32: torch.Tensor,
    row_scale_f32: torch.Tensor,
) -> LinearResult:
    """Apply ``(float(acc) * sx) * sw``, then BF16 RNE and widen to F32."""

    accumulator_i32 = _require_cpu(accumulator_i32, "accumulator_i32")
    activation_scale_f32 = _require_cpu(activation_scale_f32, "activation_scale_f32")
    row_scale_f32 = _require_cpu(row_scale_f32, "row_scale_f32")
    if accumulator_i32.dtype != torch.int32 or accumulator_i32.ndim != 2:
        raise TypeError("accumulator_i32 must be a two-dimensional int32 tensor")
    if activation_scale_f32.dtype != torch.float32 or row_scale_f32.dtype != torch.float32:
        raise TypeError("linear scales must be float32")
    sx = activation_scale_f32.reshape(-1)
    sw = row_scale_f32.reshape(-1)
    if accumulator_i32.shape != (sx.numel(), sw.numel()):
        raise ValueError(
            f"scale dimensions do not match accumulator {tuple(accumulator_i32.shape)}: "
            f"sx={sx.numel()}, sw={sw.numel()}"
        )
    scaled = accumulator_i32.to(torch.float32) * sx[:, None]
    scaled = scaled * sw[None, :]
    output_bf16 = scaled.to(torch.bfloat16)
    return LinearResult(accumulator_i32, scaled, output_bf16, output_bf16.to(torch.float32))


def row4_linear_selected(
    carrier_f32: torch.Tensor,
    weight_bf16: torch.Tensor,
    row_scale_bf16: torch.Tensor,
) -> tuple[ActivationQuantized, torch.Tensor, torch.Tensor, LinearResult]:
    """Evaluate selected Row4 rows using the complete frozen profile."""

    if row_scale_bf16.dtype != torch.bfloat16:
        raise TypeError("Row4 row scales must be saved BF16")
    activation = quantize_activation(carrier_f32)
    codes = quantize_row4_codes(weight_bf16)
    decoded = decode_row4_codes(codes)
    accumulator = int32_accumulate(
        activation.codes_i8.reshape(-1, carrier_f32.shape[-1]),
        decoded,
    )
    result = finish_linear(accumulator, activation.scales_f32, row_scale_bf16.to(torch.float32))
    return activation, codes, decoded, result


def w8_linear_selected(
    carrier_f32: torch.Tensor,
    weight_bf16: torch.Tensor,
) -> tuple[ActivationQuantized, torch.Tensor, torch.Tensor, LinearResult]:
    """Evaluate selected lm_head rows using offline W8 plus dynamic A8."""

    activation = quantize_activation(carrier_f32)
    weight_codes, row_scales = quantize_w8_rows(weight_bf16)
    accumulator = int32_accumulate(
        activation.codes_i8.reshape(-1, carrier_f32.shape[-1]),
        weight_codes,
    )
    result = finish_linear(accumulator, activation.scales_f32, row_scales)
    return activation, weight_codes, row_scales, result
