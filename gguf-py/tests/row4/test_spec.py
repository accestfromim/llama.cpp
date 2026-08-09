from __future__ import annotations

import gguf

from row4.spec import METADATA, write_metadata


def test_python_enum_values_match_row4_v1() -> None:
    assert gguf.GGMLQuantizationType.ROW4_CODES == 47
    assert gguf.LlamaFileType.MOSTLY_ROW4 == 43
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.ROW4_CODES] == (1, 1)


def test_metadata_writer_emits_complete_strict_contract() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.values: dict[str, int | str] = {}

        def add_uint32(self, key: str, value: int) -> None:
            self.values[key] = value

        def add_string(self, key: str, value: str) -> None:
            self.values[key] = value

    writer = RecordingWriter()
    write_metadata(writer)  # type: ignore[arg-type]
    assert writer.values == METADATA
