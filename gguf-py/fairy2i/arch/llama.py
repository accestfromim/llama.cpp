from __future__ import annotations

from .base import Fairy2IArchInfo


INFO = Fairy2IArchInfo(
    name="llama",
    hf_model_types=("llama",),
    hf_architectures=("LlamaForCausalLM",),
    tokenizer_profile="llama_bpe",
    attn_layout="llama_real",
)


def match(config: dict) -> bool:
    return config.get("model_type") in INFO.hf_model_types

