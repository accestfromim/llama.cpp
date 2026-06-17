from __future__ import annotations

from dataclasses import dataclass

import gguf


SCHEMA_VERSION = 1
ARCHITECTURE = "fairy2i"
QUANT_FORMAT_IFAIRY64 = "ifairy64"
QUANT_FORMAT_LEGACY_IFAIRY = "ifairy"
QUANT_VARIANT_TILE64_V2 = "tile64_v2"
QUANT_VARIANT_LEGACY = "legacy"
CODEBOOK_ROOTS4 = "{+/-1,+/-i}"
SCALE_STAT_DOMINANT_MEAN_ABS = "dominant_mean_abs"
TILE_SIZE_TILE64_V2 = 64


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
    vocab_original_size: int | None = None
    vocab_padded_size: int | None = None
    vocab_padding_multiple: int | None = None


def _quant_format_for_variant(variant: str) -> str:
    if variant == QUANT_VARIANT_TILE64_V2:
        return QUANT_FORMAT_IFAIRY64
    if variant == QUANT_VARIANT_LEGACY:
        return QUANT_FORMAT_LEGACY_IFAIRY
    return variant


def write_metadata(writer: gguf.GGUFWriter, metadata: Fairy2IMetadata) -> None:
    """Write the normalized Fairy2i GGUF schema plus legacy-compatible keys."""

    if metadata.residual_steps != 2:
        raise ValueError("Fairy2i currently supports exactly 2 residual quantization steps")

    quant_format = metadata.quant_format or _quant_format_for_variant(metadata.quant_variant)

    writer.add_uint32("fairy2i.schema_version", SCHEMA_VERSION)
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

    if metadata.quant_variant == QUANT_VARIANT_TILE64_V2:
        writer.add_uint32("fairy2i.quant.tile_size", TILE_SIZE_TILE64_V2)
        writer.add_string("fairy2i.quant.scale_stat", SCALE_STAT_DOMINANT_MEAN_ABS)

    if metadata.vocab_original_size is not None:
        writer.add_uint32("fairy2i.vocab.original_size", metadata.vocab_original_size)
    if metadata.vocab_padded_size is not None:
        writer.add_uint32("fairy2i.vocab.padded_size", metadata.vocab_padded_size)
    if metadata.vocab_padding_multiple is not None:
        writer.add_uint32("fairy2i.vocab.padding_multiple", metadata.vocab_padding_multiple)

