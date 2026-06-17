from __future__ import annotations

from fairy2i.spec import Fairy2IMetadata, write_metadata


class RecordingWriter:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def add_uint32(self, key: str, value: int) -> None:
        self.values[key] = value

    def add_string(self, key: str, value: str) -> None:
        self.values[key] = value


def test_write_tile64_v2_metadata() -> None:
    writer = RecordingWriter()

    write_metadata(
        writer,  # type: ignore[arg-type]
        Fairy2IMetadata(
            base_arch="llama",
            base_model_type="llama",
            base_architecture="LlamaForCausalLM",
            attn_layout="llama_real",
            tokenizer_profile="llama_bpe",
            vocab_original_size=32006,
            vocab_padded_size=32128,
            vocab_padding_multiple=128,
        ),
    )

    assert writer.values["fairy2i.schema_version"] == 1
    assert writer.values["fairy2i.base_arch"] == "llama"
    assert writer.values["fairy2i.quant.format"] == "ifairy64"
    assert writer.values["fairy2i.quant.variant"] == "tile64_v2"
    assert writer.values["fairy2i.quant.tile_size"] == 64
    assert writer.values["fairy2i.attn.layout"] == "llama_real"
    assert writer.values["fairy2i.tokenizer.profile"] == "llama_bpe"
    assert writer.values["fairy2i.vocab.padding_multiple"] == 128

