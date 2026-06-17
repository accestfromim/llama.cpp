from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Fairy2IArchInfo:
    name: str
    hf_model_types: tuple[str, ...]
    hf_architectures: tuple[str, ...]
    tokenizer_profile: str
    attn_layout: str


class Fairy2IArchAdapter(Protocol):
    info: Fairy2IArchInfo

    def match(self, config: dict) -> bool: ...
