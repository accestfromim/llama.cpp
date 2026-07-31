from __future__ import annotations

from dataclasses import dataclass

import gguf


SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_EXACT_SCHEMA_VERSION = 3
ARCHITECTURE = "fairy2i"
QUANT_FORMAT_TILE64_V2 = "fairy2i_tile64_v2"
QUANT_VARIANT_TILE64_V2 = "tile64_v2"
QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE = "tile64_v2_w1_learned_scale"
NUMERIC_PROFILE_LEGACY_F16_V1 = "legacy_f16_v1"
NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1 = "script_f32reduce_bf16scale_v1"
WEIGHT_SCALE_DTYPE_F16 = "f16"
WEIGHT_SCALE_DTYPE_BF16 = "bf16"
CODEBOOK_ROOTS4 = "{+/-1,+/-i}"
SCALE_STAT_DOMINANT_MEAN_ABS = "dominant_mean_abs"
SCALE_SOURCE_LEARNED = "learned"
TILE_SIZE_TILE64_V2 = 64
WEIGHT_LAYOUT_TILE64_V2 = "tile64_v2"
WEIGHT_LAYOUT_BUNDLE_V1 = "bundle_v1"
BUNDLE_LAYOUT_NAME = "bundle_m64k64_v1"
BUNDLE_SCALE_SCOPE = "m64_k64"
BUNDLE_CODE_ORDER = "m16_q4_branch_lane"
BUNDLE_W1_BRANCH_ORDER = "U0,W0"
BUNDLE_W2_BRANCH_ORDER = "U0,U1,W0,W1"


@dataclass(frozen=True)
class Fairy2IMetadata:
    base_arch: str
    attn_layout: str
    tokenizer_profile: str
    quant_variant: str = QUANT_VARIANT_TILE64_V2
    residual_steps: int = 2
    quant_format: str | None = None
    base_model_type: str | None = None
    base_architecture: str | None = None
    scale_source: str | None = None
    numeric_profile: str | None = None
    weight_scale_dtype: str | None = None
    weight_layout: str = WEIGHT_LAYOUT_BUNDLE_V1
    vocab_original_size: int | None = None
    vocab_padded_size: int | None = None
    vocab_padding_multiple: int | None = None


def _quant_format_for_variant(variant: str) -> str:
    if variant in (QUANT_VARIANT_TILE64_V2, QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE):
        return QUANT_FORMAT_TILE64_V2
    raise ValueError(f"unsupported Fairy2i quant variant: {variant}")


def write_metadata(writer: gguf.GGUFWriter, metadata: Fairy2IMetadata) -> None:
    """Write the normalized Fairy2i GGUF schema."""

    expected_steps = 1 if metadata.quant_variant == QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE else 2
    if metadata.residual_steps != expected_steps:
        raise ValueError(
            f"Fairy2i {metadata.quant_variant} requires exactly {expected_steps} residual quantization step(s)"
        )

    quant_format = metadata.quant_format or _quant_format_for_variant(metadata.quant_variant)
    if quant_format != QUANT_FORMAT_TILE64_V2:
        raise ValueError(f"unsupported Fairy2i quant format: {quant_format}")

    if metadata.quant_variant == QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE:
        scale_source = metadata.scale_source or SCALE_SOURCE_LEARNED
        if scale_source != SCALE_SOURCE_LEARNED:
            raise ValueError(f"unsupported Fairy2i W1 scale source: {scale_source}")
    else:
        scale_source = metadata.scale_source

    if metadata.weight_layout not in (WEIGHT_LAYOUT_TILE64_V2, WEIGHT_LAYOUT_BUNDLE_V1):
        raise ValueError(f"unsupported Fairy2i weight layout: {metadata.weight_layout}")

    qwen2_bundle_w2 = (
        metadata.base_arch == "qwen2"
        and metadata.quant_variant == QUANT_VARIANT_TILE64_V2
        and metadata.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1
    )
    if qwen2_bundle_w2 and (
        metadata.numeric_profile,
        metadata.weight_scale_dtype,
    ) not in (
        (NUMERIC_PROFILE_LEGACY_F16_V1, WEIGHT_SCALE_DTYPE_F16),
        (NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1, WEIGHT_SCALE_DTYPE_BF16),
    ):
        raise ValueError(
            "Qwen2 bundle tile64_v2 metadata requires numeric_profile/weight_scale_dtype "
            "to be ('legacy_f16_v1', 'f16') or "
            "('script_f32reduce_bf16scale_v1', 'bf16')"
        )

    exact_profile = metadata.numeric_profile == NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1
    explicit_legacy_profile = metadata.numeric_profile == NUMERIC_PROFILE_LEGACY_F16_V1
    if exact_profile:
        if metadata.base_arch != "qwen2":
            raise ValueError("script_f32reduce_bf16scale_v1 is supported only for the Qwen2 Fairy2i path")
        if metadata.attn_layout != "qwen2_real":
            raise ValueError("script_f32reduce_bf16scale_v1 requires attn_layout='qwen2_real'")
        if metadata.tokenizer_profile != "qwen2":
            raise ValueError("script_f32reduce_bf16scale_v1 requires tokenizer_profile='qwen2'")
        if metadata.quant_variant != QUANT_VARIANT_TILE64_V2:
            raise ValueError("script_f32reduce_bf16scale_v1 requires the tile64_v2 W2 quantization variant")
        if metadata.weight_layout != WEIGHT_LAYOUT_BUNDLE_V1:
            raise ValueError("script_f32reduce_bf16scale_v1 requires the bundle_v1 weight layout")
        if metadata.weight_scale_dtype != WEIGHT_SCALE_DTYPE_BF16:
            raise ValueError("script_f32reduce_bf16scale_v1 requires weight_scale_dtype='bf16'")
    elif explicit_legacy_profile:
        if metadata.base_arch != "qwen2":
            raise ValueError("legacy_f16_v1 metadata is supported only for the Qwen2 Fairy2i path")
        if metadata.quant_variant != QUANT_VARIANT_TILE64_V2:
            raise ValueError("legacy_f16_v1 metadata requires the tile64_v2 W2 quantization variant")
        if metadata.weight_layout != WEIGHT_LAYOUT_BUNDLE_V1:
            raise ValueError("legacy_f16_v1 metadata requires the bundle_v1 weight layout")
        if metadata.weight_scale_dtype != WEIGHT_SCALE_DTYPE_F16:
            raise ValueError("legacy_f16_v1 requires weight_scale_dtype='f16'")
    elif metadata.numeric_profile is not None or metadata.weight_scale_dtype is not None:
        raise ValueError(
            "Fairy2i numeric_profile and weight_scale_dtype must form a supported pair"
        )

    if exact_profile:
        schema_version = BUNDLE_EXACT_SCHEMA_VERSION
    elif metadata.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        schema_version = BUNDLE_SCHEMA_VERSION
    else:
        schema_version = SCHEMA_VERSION
    writer.add_uint32("fairy2i.schema_version", schema_version)
    writer.add_string("fairy2i.base_arch", metadata.base_arch)
    writer.add_string("fairy2i.quant.format", quant_format)
    writer.add_uint32("fairy2i.quant.residual_steps", metadata.residual_steps)
    writer.add_string("fairy2i.quant.codebook", CODEBOOK_ROOTS4)
    writer.add_string("fairy2i.quant.variant", metadata.quant_variant)
    writer.add_string("fairy2i.attn.layout", metadata.attn_layout)
    writer.add_string("fairy2i.tokenizer.profile", metadata.tokenizer_profile)

    if metadata.base_model_type is not None:
        writer.add_string("fairy2i.base_model_type", metadata.base_model_type)
    if metadata.base_architecture is not None:
        writer.add_string("fairy2i.base_architecture", metadata.base_architecture)

    if metadata.quant_variant in (QUANT_VARIANT_TILE64_V2, QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE):
        writer.add_uint32("fairy2i.quant.tile_size", TILE_SIZE_TILE64_V2)
    if metadata.quant_variant == QUANT_VARIANT_TILE64_V2:
        writer.add_string("fairy2i.quant.scale_stat", SCALE_STAT_DOMINANT_MEAN_ABS)
    if exact_profile or explicit_legacy_profile:
        writer.add_string("fairy2i.quant.numeric_profile", metadata.numeric_profile)
    if scale_source is not None:
        writer.add_string("fairy2i.quant.scale_source", scale_source)

    if metadata.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        branch_order = (
            BUNDLE_W1_BRANCH_ORDER
            if metadata.quant_variant == QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE
            else BUNDLE_W2_BRANCH_ORDER
        )
        writer.add_string("fairy2i.weight.layout", BUNDLE_LAYOUT_NAME)
        writer.add_string("fairy2i.weight.scale_scope", BUNDLE_SCALE_SCOPE)
        if exact_profile or explicit_legacy_profile:
            writer.add_string("fairy2i.weight.scale_dtype", metadata.weight_scale_dtype)
        writer.add_string("fairy2i.weight.code_order", BUNDLE_CODE_ORDER)
        writer.add_string("fairy2i.weight.branch_order", branch_order)
        writer.add_uint32("fairy2i.weight.m_block", 64)
        writer.add_uint32("fairy2i.weight.k_block", 64)
        writer.add_uint32("fairy2i.weight.m_subtile", 16)

    if metadata.vocab_original_size is not None:
        writer.add_uint32("fairy2i.vocab.original_size", metadata.vocab_original_size)
    if metadata.vocab_padded_size is not None:
        writer.add_uint32("fairy2i.vocab.padded_size", metadata.vocab_padded_size)
    if metadata.vocab_padding_multiple is not None:
        writer.add_uint32("fairy2i.vocab.padding_multiple", metadata.vocab_padding_multiple)
