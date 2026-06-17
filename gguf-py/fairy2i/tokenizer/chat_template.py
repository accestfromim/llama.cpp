from __future__ import annotations

import json
from pathlib import Path


FAIRY2I_DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
FAIRY2I_DEEPSEEK_USER_TOKEN = "<｜User｜>"
FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"
FAIRY2I_DEEPSEEK_EOS_TOKEN = "<｜end▁of▁sentence｜>"
FAIRY2I_DEEPSEEK_CHAT_TOKENS = (
    FAIRY2I_DEEPSEEK_BOS_TOKEN,
    FAIRY2I_DEEPSEEK_USER_TOKEN,
    FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN,
    FAIRY2I_DEEPSEEK_EOS_TOKEN,
)


def normalize_fairy2i_chat_template(chat_template: str) -> str:
    if "add_generation_prompt" in chat_template:
        return chat_template
    if not all(token in chat_template for token in FAIRY2I_DEEPSEEK_CHAT_TOKENS):
        return chat_template

    assistant_token_literal = repr(FAIRY2I_DEEPSEEK_ASSISTANT_TOKEN)
    generation_prompt = "{% if add_generation_prompt %}{{ " + assistant_token_literal + " }}{% endif %}"
    return chat_template.rstrip("\n") + generation_prompt


def normalize_fairy2i_chat_template_value(
    chat_template: str | list[dict[str, str]],
) -> str | list[dict[str, str]]:
    if isinstance(chat_template, str):
        return normalize_fairy2i_chat_template(chat_template)
    if isinstance(chat_template, list):
        normalized_templates: list[dict[str, str]] = []
        for choice in chat_template:
            if not isinstance(choice, dict):
                raise ValueError("chat_template list entries must be objects")
            normalized_choice = dict(choice)
            template = normalized_choice.get("template")
            if isinstance(template, str):
                normalized_choice["template"] = normalize_fairy2i_chat_template(template)
            normalized_templates.append(normalized_choice)
        return normalized_templates
    raise ValueError(f"bad chat_template type: {type(chat_template).__name__}")


def load_fairy2i_chat_template(model_dir: Path, tokenizer_config: dict) -> str | list[dict[str, str]] | None:
    chat_template = None
    chat_template_jinja = model_dir / "chat_template.jinja"
    chat_template_json = model_dir / "chat_template.json"

    if chat_template_jinja.is_file():
        chat_template = chat_template_jinja.read_text(encoding="utf-8")
    elif chat_template_json.is_file():
        chat_template_data = json.loads(chat_template_json.read_text(encoding="utf-8"))
        chat_template = chat_template_data.get("chat_template")
    else:
        chat_template = tokenizer_config.get("chat_template")

    if chat_template is None:
        return None
    if not isinstance(chat_template, (str, list)):
        raise ValueError(f"bad chat_template type: {type(chat_template).__name__}")
    return normalize_fairy2i_chat_template_value(chat_template)
