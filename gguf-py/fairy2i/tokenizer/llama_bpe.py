from __future__ import annotations


PROFILE = "llama_bpe"


def token_looks_special(token: str | bytes) -> bool:
    token_text = token.decode("utf-8") if isinstance(token, bytes) else token
    return (
        token_text in ("<unk>", "<s>", "</s>", "<pad>")
        or (token_text.startswith("<|") and token_text.endswith("|>"))
        or (token_text.startswith("<｜") and token_text.endswith("｜>"))
        or (token_text.startswith("<") and token_text.endswith(">"))
    )
