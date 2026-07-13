from __future__ import annotations

from fairy2i.arch.registry import detect_arch_info, get_arch_info


def test_detects_qwen3_fairy2i_arch() -> None:
    info = detect_arch_info({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})

    assert info.name == "qwen3"
    assert info.attn_layout == "qwen3_real"
    assert info.tokenizer_profile == "qwen2"


def test_gets_qwen3_fairy2i_arch() -> None:
    assert get_arch_info("qwen3").name == "qwen3"
