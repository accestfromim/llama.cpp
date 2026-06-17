from __future__ import annotations

from .base import Fairy2IArchInfo


INFO = Fairy2IArchInfo(
    name="qwen2",
    hf_model_types=("qwen2",),
    hf_architectures=("Qwen2ForCausalLM",),
    tokenizer_profile="qwen2",
    attn_layout="qwen2_real",
)


def match(config: dict) -> bool:
    return config.get("model_type") in INFO.hf_model_types
