from __future__ import annotations

from .base import Fairy2IArchInfo


INFO = Fairy2IArchInfo(
    name="qwen3",
    hf_model_types=("qwen3",),
    hf_architectures=("Qwen3ForCausalLM",),
    tokenizer_profile="qwen2",
    attn_layout="qwen3_real",
)


def match(config: dict) -> bool:
    return config.get("model_type") in INFO.hf_model_types
