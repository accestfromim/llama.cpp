from __future__ import annotations


PROFILE = "qwen2"


def token_looks_special(token: str | bytes) -> bool:
    token_text = token.decode("utf-8") if isinstance(token, bytes) else token
    seems_special = token_text in (
        "<pad>",
        "<mask>",
        "<2mass>",
        "[@BOS@]",
    )
    seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
    seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))
    seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))
    return seems_special
