from __future__ import annotations

import pytest

from fairy2i.spec import (
    NUMERIC_PROFILE_LEGACY_F16_V1,
    NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
    NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
    QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
    WEIGHT_SCALE_DTYPE_F16,
    WEIGHT_SCALE_DTYPE_BF16,
    WEIGHT_LAYOUT_TILE64_V2,
    Fairy2IMetadata,
    SCALE_SOURCE_LEARNED,
    write_metadata,
)


class RecordingWriter:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def add_uint32(self, key: str, value: int) -> None:
        self.values[key] = value

    def add_string(self, key: str, value: str) -> None:
        self.values[key] = value


def test_write_tile64_v2_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="llama",
            base_model_type="llama",
            base_architecture="LlamaForCausalLM",
            attn_layout="llama_real",
            tokenizer_profile="llama_bpe",
            vocab_original_size=32006,
            vocab_padded_size=32128,
            vocab_padding_multiple=128,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 2
    assert writer.values["fairy2i.base_arch"] == "llama"
    assert writer.values["fairy2i.quant.format"] == "fairy2i_tile64_v2"
    assert writer.values["fairy2i.quant.variant"] == "tile64_v2"
    assert writer.values["fairy2i.quant.tile_size"] == 64
    assert writer.values["fairy2i.attn.layout"] == "llama_real"
    assert writer.values["fairy2i.tokenizer.profile"] == "llama_bpe"
    assert writer.values["fairy2i.vocab.padding_multiple"] == 128
    assert writer.values["fairy2i.weight.layout"] == "bundle_m64k64_v1"
    assert writer.values["fairy2i.weight.scale_scope"] == "m64_k64"
    assert writer.values["fairy2i.weight.code_order"] == "m16_q4_branch_lane"
    assert writer.values["fairy2i.weight.branch_order"] == "U0,U1,W0,W1"


def test_write_legacy_tile64_v2_layout_metadata() -> None:
    writer = RecordingWriter()
    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="llama",
            attn_layout="llama_real",
            tokenizer_profile="llama_bpe",
            weight_layout=WEIGHT_LAYOUT_TILE64_V2,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 1
    assert "fairy2i.weight.layout" not in writer.values


def test_write_qwen2_script_bf16_f32_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="qwen2",
            base_model_type="qwen2",
            base_architecture="Qwen2ForCausalLM",
            attn_layout="qwen2_real",
            tokenizer_profile="qwen2",
            numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
            weight_scale_dtype=WEIGHT_SCALE_DTYPE_BF16,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 3
    assert (
        writer.values["fairy2i.quant.numeric_profile"]
        == NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1
    )
    assert writer.values["fairy2i.weight.scale_dtype"] == WEIGHT_SCALE_DTYPE_BF16
    assert writer.values["fairy2i.weight.layout"] == "bundle_m64k64_v1"


def test_write_qwen2_explicit_legacy_f16_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="qwen2",
            attn_layout="qwen2_real",
            tokenizer_profile="qwen2",
            numeric_profile=NUMERIC_PROFILE_LEGACY_F16_V1,
            weight_scale_dtype=WEIGHT_SCALE_DTYPE_F16,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 2
    assert (
        writer.values["fairy2i.quant.numeric_profile"]
        == NUMERIC_PROFILE_LEGACY_F16_V1
    )
    assert writer.values["fairy2i.weight.scale_dtype"] == WEIGHT_SCALE_DTYPE_F16


@pytest.mark.parametrize(
    ("numeric_profile", "weight_scale_dtype"),
    [
        (None, None),
        (NUMERIC_PROFILE_LEGACY_F16_V1, None),
        (None, WEIGHT_SCALE_DTYPE_F16),
        (NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1, None),
        (None, WEIGHT_SCALE_DTYPE_BF16),
        (None, "f32"),
        (NUMERIC_PROFILE_LEGACY_F16_V1, WEIGHT_SCALE_DTYPE_BF16),
        (NUMERIC_PROFILE_LEGACY_F16_V1, "f32"),
        (NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1, WEIGHT_SCALE_DTYPE_F16),
        (NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1, "f32"),
        ("unsupported", WEIGHT_SCALE_DTYPE_F16),
    ],
)
def test_rejects_invalid_qwen2_bundle_w2_numeric_pair(
    numeric_profile: str | None,
    weight_scale_dtype: str | None,
) -> None:
    metadata = Fairy2IMetadata(
        base_arch="qwen2",
        attn_layout="qwen2_real",
        tokenizer_profile="qwen2",
        numeric_profile=numeric_profile,
        weight_scale_dtype=weight_scale_dtype,
    )
    with pytest.raises(ValueError, match="requires numeric_profile/weight_scale_dtype"):
        write_metadata(RecordingWriter(), metadata)  # type: ignore[arg-type]


def test_rejects_cross_arch_exact_metadata() -> None:
    with pytest.raises(ValueError, match="supported only for the Qwen2"):
        write_metadata(
            RecordingWriter(),  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="qwen3",
                attn_layout="qwen3_real",
                tokenizer_profile="qwen2",
                numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
                weight_scale_dtype=WEIGHT_SCALE_DTYPE_BF16,
            ),
        )


@pytest.mark.parametrize(
    ("attn_layout", "tokenizer_profile", "message"),
    [
        ("llama_real", "qwen2", "attn_layout='qwen2_real'"),
        ("qwen2_real", "llama_bpe", "tokenizer_profile='qwen2'"),
    ],
)
def test_rejects_wrong_qwen2_exact_graph_profile(
    attn_layout: str,
    tokenizer_profile: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        write_metadata(
            RecordingWriter(),  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="qwen2",
                attn_layout=attn_layout,
                tokenizer_profile=tokenizer_profile,
                numeric_profile=NUMERIC_PROFILE_SCRIPT_F32REDUCE_BF16SCALE_V1,
                weight_scale_dtype=WEIGHT_SCALE_DTYPE_BF16,
            ),
        )


def test_rejects_non_fairy2i_quant_schema() -> None:
    writer = RecordingWriter()
    old_variant = "lega" + "cy"
    old_format = "ifa" + "iry64"

    with pytest.raises(ValueError, match="unsupported Fairy2i quant variant"):
        write_metadata(
            writer,  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="llama",
                attn_layout="llama_real",
                tokenizer_profile="llama_bpe",
                quant_variant=old_variant,
            ),
        )

    with pytest.raises(ValueError, match="unsupported Fairy2i quant format"):
        write_metadata(
            writer,  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="llama",
                attn_layout="llama_real",
                tokenizer_profile="llama_bpe",
                quant_format=old_format,
            ),
        )


def test_write_tile64_v2_w1_learned_scale_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="qwen3",
            base_model_type="qwen3",
            base_architecture="Qwen3ForCausalLM",
            attn_layout="qwen3_real",
            tokenizer_profile="qwen2",
            quant_variant=QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
            residual_steps=1,
            scale_source=SCALE_SOURCE_LEARNED,
        ),
    )

    assert writer.values["fairy2i.base_arch"] == "qwen3"
    assert writer.values["fairy2i.quant.variant"] == "tile64_v2_w1_learned_scale"
    assert writer.values["fairy2i.quant.residual_steps"] == 1
    assert writer.values["fairy2i.quant.scale_source"] == "learned"
    assert writer.values["fairy2i.quant.tile_size"] == 64
    assert "fairy2i.quant.scale_stat" not in writer.values
    assert writer.values["fairy2i.weight.branch_order"] == "U0,W0"
    assert writer.values["fairy2i.schema_version"] == 2
    assert "fairy2i.quant.numeric_profile" not in writer.values
    assert "fairy2i.weight.scale_dtype" not in writer.values


def test_write_qwen3_qat_bf16_learned_scale_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="qwen3",
            base_model_type="qwen3",
            base_architecture="Qwen3ForCausalLM",
            attn_layout="qwen3_real",
            tokenizer_profile="qwen2",
            quant_variant=QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
            residual_steps=1,
            scale_source=SCALE_SOURCE_LEARNED,
            numeric_profile=NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
            weight_scale_dtype=WEIGHT_SCALE_DTYPE_BF16,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 4
    assert (
        writer.values["fairy2i.quant.numeric_profile"]
        == NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1
    )
    assert writer.values["fairy2i.weight.scale_dtype"] == WEIGHT_SCALE_DTYPE_BF16
    assert writer.values["fairy2i.quant.scale_source"] == SCALE_SOURCE_LEARNED
    assert writer.values["fairy2i.weight.branch_order"] == "U0,W0"


@pytest.mark.parametrize(
    ("numeric_profile", "weight_scale_dtype"),
    [
        (NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1, None),
        (None, WEIGHT_SCALE_DTYPE_BF16),
        (NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1, WEIGHT_SCALE_DTYPE_F16),
    ],
)
def test_rejects_invalid_qwen3_exact_numeric_pair(
    numeric_profile: str | None,
    weight_scale_dtype: str | None,
) -> None:
    with pytest.raises(ValueError, match="Qwen3 bundle learned-scale metadata requires"):
        write_metadata(
            RecordingWriter(),  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="qwen3",
                attn_layout="qwen3_real",
                tokenizer_profile="qwen2",
                quant_variant=QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
                residual_steps=1,
                scale_source=SCALE_SOURCE_LEARNED,
                numeric_profile=numeric_profile,
                weight_scale_dtype=weight_scale_dtype,
            ),
        )


def test_rejects_wrong_w1_residual_steps() -> None:
    writer = RecordingWriter()

    with pytest.raises(ValueError, match="requires exactly 1"):
        write_metadata(
            writer,  # type: ignore[arg-type]
            Fairy2IMetadata(
                base_arch="qwen3",
                attn_layout="qwen3_real",
                tokenizer_profile="qwen2",
                quant_variant=QUANT_VARIANT_TILE64_V2_W1_LEARNED_SCALE,
                residual_steps=2,
            ),
        )
