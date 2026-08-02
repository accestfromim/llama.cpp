from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest


def _write_tiny_qwen3_checkpoint(
    model_dir: Path,
    *,
    overrides: dict[str, object] | None = None,
    include_qkv_bias: bool = False,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 1,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 128,
    }
    tensors = {
        "model.embed_tokens.weight": torch.zeros((128, 128), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones((128,), dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros((128, 128), dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones((128,), dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones((128,), dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_norm.weight": torch.ones((64,), dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_norm.weight": torch.ones((64,), dtype=torch.bfloat16),
    }
    for suffix in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        base = f"model.layers.0.{suffix}"
        tensors[f"{base}.weight"] = torch.zeros((128, 128), dtype=torch.bfloat16)
        tensors[f"{base}.quant_scale"] = torch.ones((4, 1, 1), dtype=torch.bfloat16)
    if include_qkv_bias:
        for suffix in ("q_proj", "k_proj", "v_proj"):
            tensors[f"model.layers.0.self_attn.{suffix}.bias"] = torch.zeros(
                (128,),
                dtype=torch.bfloat16,
            )
    tensors.update(overrides or {})  # type: ignore[arg-type]

    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors_torch.save_file(tensors, model_dir / "model.safetensors")


def test_qwen3_converter_requires_quant_scale() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from convert_fairy2i_qwen3 import validate_checkpoint

    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 1,
    }
    weight_map = {
        "model.embed_tokens.weight": "model.safetensors",
        "model.norm.weight": "model.safetensors",
        "lm_head.weight": "model.safetensors",
        "model.layers.0.input_layernorm.weight": "model.safetensors",
        "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
        "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
        "model.layers.0.self_attn.k_norm.weight": "model.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.v_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.o_proj.weight": "model.safetensors",
        "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.down_proj.weight": "model.safetensors",
    }

    with pytest.raises(ValueError, match="quant_scale"):
        validate_checkpoint(config, weight_map)


def test_qwen3_exact_cli_accepts_profile_and_legacy_remains_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        main,
    )

    model_dir = tmp_path / "model"
    _write_tiny_qwen3_checkpoint(model_dir)

    main([str(model_dir), "--dry-run"])
    assert "numeric_profile=legacy_f16_v1" in capsys.readouterr().out

    main(
        [
            str(model_dir),
            "--dry-run",
            "--numeric-profile",
            NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        ]
    )
    assert (
        f"numeric_profile={NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1}"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("tensor_name", "dtype_name", "shape", "include_qkv_bias"),
    [
        ("model.layers.0.self_attn.q_proj.weight", "float16", (128, 128), False),
        ("model.layers.0.self_attn.q_proj.weight", "float32", (128, 128), False),
        ("model.layers.0.self_attn.q_proj.quant_scale", "float16", (4, 1, 1), False),
        ("lm_head.weight", "float16", (128, 128), False),
        ("model.embed_tokens.weight", "float16", (128, 128), False),
        ("model.layers.0.input_layernorm.weight", "float32", (128,), False),
        ("model.layers.0.self_attn.q_proj.bias", "float16", (128,), True),
    ],
)
def test_qwen3_exact_dry_run_rejects_non_bf16_source_tensor(
    tmp_path: Path,
    tensor_name: str,
    dtype_name: str,
    shape: tuple[int, ...],
    include_qkv_bias: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        main,
    )

    model_dir = tmp_path / "model"
    _write_tiny_qwen3_checkpoint(
        model_dir,
        overrides={
            tensor_name: torch.zeros(shape, dtype=getattr(torch, dtype_name)),
        },
        include_qkv_bias=include_qkv_bias,
    )

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(tensor_name)} dtype mismatch.*expected BF16",
    ):
        main(
            [
                str(model_dir),
                "--dry-run",
                "--numeric-profile",
                NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
            ]
        )


@pytest.mark.parametrize(
    ("tensor_name", "bad_shape"),
    [
        ("model.layers.0.self_attn.q_proj.weight", (128, 64)),
        ("model.layers.0.self_attn.q_proj.quant_scale", (4, 1, 2)),
        ("lm_head.weight", (127, 128)),
    ],
)
def test_qwen3_exact_dry_run_rejects_wrong_source_shape(
    tmp_path: Path,
    tensor_name: str,
    bad_shape: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        main,
    )

    model_dir = tmp_path / "model"
    _write_tiny_qwen3_checkpoint(
        model_dir,
        overrides={
            tensor_name: torch.zeros(bad_shape, dtype=torch.bfloat16),
        },
    )

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(tensor_name)} shape mismatch",
    ):
        main(
            [
                str(model_dir),
                "--dry-run",
                "--numeric-profile",
                NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
            ]
        )


@pytest.mark.parametrize(
    ("tensor_name", "shape", "include_qkv_bias"),
    [
        ("model.embed_tokens.weight", (128, 128), False),
        ("model.layers.0.self_attn.q_proj.weight", (128, 128), False),
        ("model.layers.0.self_attn.q_proj.bias", (128,), True),
    ],
)
def test_qwen3_exact_dry_run_rejects_nonfinite_bf16_source_tensor(
    tmp_path: Path,
    tensor_name: str,
    shape: tuple[int, ...],
    include_qkv_bias: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        main,
    )

    value = torch.zeros(shape, dtype=torch.bfloat16)
    value.reshape(-1)[-1] = float("nan")
    model_dir = tmp_path / "model"
    _write_tiny_qwen3_checkpoint(
        model_dir,
        overrides={tensor_name: value},
        include_qkv_bias=include_qkv_bias,
    )

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(tensor_name)} contains a non-finite BF16 value",
    ):
        main(
            [
                str(model_dir),
                "--dry-run",
                "--numeric-profile",
                NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
            ]
        )


def test_qwen3_legacy_dry_run_keeps_source_dtype_compatibility(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from convert_fairy2i_qwen3 import main

    model_dir = tmp_path / "model"
    _write_tiny_qwen3_checkpoint(
        model_dir,
        overrides={
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                (128, 128),
                dtype=torch.float16,
            ),
        },
    )

    main([str(model_dir), "--dry-run"])


def test_qwen3_exact_cli_rejects_tile64_f16_scale_carrier(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        main,
    )
    from fairy2i.spec import WEIGHT_LAYOUT_TILE64_V2

    with pytest.raises(ValueError, match="requires --weight-layout bundle_v1"):
        main(
            [
                str(tmp_path),
                "--dry-run",
                "--numeric-profile",
                NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
                "--weight-layout",
                WEIGHT_LAYOUT_TILE64_V2,
            ]
        )


def test_qwen3_exact_lm_head_uses_original_bf16_payload() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import gguf
    from convert_fairy2i_qwen3 import add_tensor_bf16

    class RecordingWriter:
        def __init__(self) -> None:
            self.name: str | None = None
            self.payload: np.ndarray | None = None
            self.raw_dtype: gguf.GGMLQuantizationType | None = None

        def add_tensor(
            self,
            name: str,
            payload: np.ndarray,
            *,
            raw_dtype: gguf.GGMLQuantizationType,
        ) -> None:
            self.name = name
            self.payload = payload.copy()
            self.raw_dtype = raw_dtype

    lm_head = torch.tensor(
        [[1.0, -2.0], [float.fromhex("0x1.02p0"), -0.0]],
        dtype=torch.bfloat16,
    )
    expected = lm_head.contiguous().view(torch.int16).numpy().view(np.uint16)
    writer = RecordingWriter()

    add_tensor_bf16(
        writer,  # type: ignore[arg-type]
        "output",
        lm_head,
        "lm_head.weight",
    )

    assert writer.name == "output"
    assert writer.raw_dtype == gguf.GGMLQuantizationType.BF16
    assert writer.payload is not None
    assert writer.payload.dtype == np.uint16
    np.testing.assert_array_equal(writer.payload, expected)


def test_qwen3_schema4_yarn_uses_training_equivalent_prefactor() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from convert_fairy2i_qwen3 import (
        NUMERIC_PROFILE_LEGACY_F16_V1,
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
        qwen3_yarn_gguf_attn_factor,
    )

    factor = 4.0
    default_mscale = 1.0 + 0.1 * np.log(factor)
    rope_params = {
        "factor": factor,
        # transformers 5.2.0 does not recognize this spelling.
        "attn_factor": 1.0 / default_mscale,
    }

    assert qwen3_yarn_gguf_attn_factor(
        rope_params,
        NUMERIC_PROFILE_LEGACY_F16_V1,
    ) == pytest.approx(1.0 / default_mscale)
    assert qwen3_yarn_gguf_attn_factor(
        rope_params,
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
    ) == pytest.approx(1.0)

    rope_params["attention_factor"] = 1.25
    assert qwen3_yarn_gguf_attn_factor(
        rope_params,
        NUMERIC_PROFILE_QAT_BF16_LEARNED_SCALE_V1,
    ) == pytest.approx(1.25 / default_mscale)
