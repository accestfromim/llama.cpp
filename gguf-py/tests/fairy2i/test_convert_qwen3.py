from __future__ import annotations

import pytest


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
