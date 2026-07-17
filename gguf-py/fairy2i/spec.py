from __future__ import annotations

from dataclasses import dataclass

import gguf


SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 2
ARCHITECTURE = "fairy2i"
QUANT_FORMAT_TILE64_V2 = "fairy2i_tile64_v2"
QUANT_VARIANT_TILE64_V2 = "tile64_v2"
QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE = "tile64_v2_w1_learned_scale"
CODEBOOK_ROOTS4 = "{+/-1,+/-i}"
SCALE_STAT_DOMINANT_MEAN_ABS = "dominant_mean_abs"
SCALE_SOURCE_LEARNED = "learned"
TILE_SIZE_TILE64_V2 = 64
WEIGHT_LAYOUT_TILE64_V2 = "tile64_v2"
WEIGHT_LAYOUT_BUNDLE_V1 = "bundle_v1"
BUNDLE_LAYOUT_NAME = "bundle_m64k64_v1"
BUNDLE_SCALE_SCOPE = "m64_k64"
BUNDLE_CODE_ORDER = "m16_q4_branch_lane"
BUNDLE_CODE_ORDER_M16_JOINT = "m16_q4_lane_joint"
BUNDLE_CODE_ORDER_M8_JOINT = "m8_q4_lane_joint"
BUNDLE_CODE_ORDER_NATIVE_BRANCH = "m32_k16_m8_q4_branch_lane"
BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16 = "m32_k16_m8_q4_branch_lane_inline16"
BUNDLE_CODE_ORDER_NATIVE_JOINT = "m32_k16_m8_q4_lane_joint"
BUNDLE_CODE_ORDER_M8_BITPLANE = "m8_q4_branch_bitplane"
BUNDLE_CODE_ORDER_ROW_JOINT = "m64_row_q4_lane_joint"
BUNDLE_CODE_ORDERS = (
    BUNDLE_CODE_ORDER,
    BUNDLE_CODE_ORDER_M16_JOINT,
    BUNDLE_CODE_ORDER_M8_JOINT,
    BUNDLE_CODE_ORDER_NATIVE_BRANCH,
    BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16,
    BUNDLE_CODE_ORDER_NATIVE_JOINT,
    BUNDLE_CODE_ORDER_M8_BITPLANE,
    BUNDLE_CODE_ORDER_ROW_JOINT,
)
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
    weight_layout: str = WEIGHT_LAYOUT_BUNDLE_V1
    bundle_code_order: str = BUNDLE_CODE_ORDER
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

    schema_version = BUNDLE_SCHEMA_VERSION if metadata.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1 else SCHEMA_VERSION
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
    if scale_source is not None:
        writer.add_string("fairy2i.quant.scale_source", scale_source)

    if metadata.weight_layout == WEIGHT_LAYOUT_BUNDLE_V1:
        if metadata.bundle_code_order not in BUNDLE_CODE_ORDERS:
            raise ValueError(f"unsupported Fairy2i bundle code order: {metadata.bundle_code_order}")
        branch_order = (
            BUNDLE_W1_BRANCH_ORDER
            if metadata.quant_variant == QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE
            else BUNDLE_W2_BRANCH_ORDER
        )
        writer.add_string("fairy2i.weight.layout", BUNDLE_LAYOUT_NAME)
        scale_scope = (
            "inline_m64_k64_header16"
            if metadata.bundle_code_order == BUNDLE_CODE_ORDER_NATIVE_BRANCH_INLINE16
            else BUNDLE_SCALE_SCOPE
        )
        writer.add_string("fairy2i.weight.scale_scope", scale_scope)
        writer.add_string("fairy2i.weight.code_order", metadata.bundle_code_order)
        writer.add_string("fairy2i.weight.branch_order", branch_order)
        writer.add_uint32("fairy2i.weight.m_block", 64)
        writer.add_uint32("fairy2i.weight.k_block", 64)
        m_subtile = 64 if metadata.bundle_code_order == BUNDLE_CODE_ORDER_ROW_JOINT else (
            16 if metadata.bundle_code_order in (BUNDLE_CODE_ORDER, BUNDLE_CODE_ORDER_M16_JOINT) else 8
        )
        writer.add_uint32("fairy2i.weight.m_subtile", m_subtile)

    if metadata.vocab_original_size is not None:
        writer.add_uint32("fairy2i.vocab.original_size", metadata.vocab_original_size)
    if metadata.vocab_padded_size is not None:
        writer.add_uint32("fairy2i.vocab.padded_size", metadata.vocab_padded_size)
    if metadata.vocab_padding_multiple is not None:
        writer.add_uint32("fairy2i.vocab.padding_multiple", metadata.vocab_padding_multiple)
