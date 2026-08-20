from __future__ import annotations

import gguf
import pytest

from row4.spec import (
    EXECUTION_PROFILE_V2,
    METADATA,
    METADATA_V2,
    ROW4_LAYOUT_V2,
    WEIGHT_LAYOUT_V2,
    alignment_for_layout,
    metadata_for_layout,
    write_metadata,
)


def test_python_enum_values_match_row4_v1() -> None:
    assert gguf.GGMLQuantizationType.ROW4_CODES == 47
    assert gguf.LlamaFileType.MOSTLY_ROW4 == 43
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.ROW4_CODES] == (1, 1)


def test_python_enum_values_include_row4_pair2_v2() -> None:
    assert gguf.GGMLQuantizationType.ROW4_CODES_PAIR2 == 48
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.ROW4_CODES_PAIR2] == (1, 1)


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


def test_v2_metadata_writer_emits_metal_only_contract() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.values: dict[str, int | str] = {}

        def add_uint32(self, key: str, value: int) -> None:
            self.values[key] = value

        def add_string(self, key: str, value: str) -> None:
            self.values[key] = value

    writer = RecordingWriter()
    write_metadata(writer, ROW4_LAYOUT_V2)  # type: ignore[arg-type]
    assert writer.values == METADATA_V2
    assert writer.values["row4.schema_version"] == 2
    assert writer.values["row4.weight_layout"] == WEIGHT_LAYOUT_V2
    assert writer.values["row4.execution_profile"] == EXECUTION_PROFILE_V2
    assert alignment_for_layout(ROW4_LAYOUT_V2) == 128


def test_layout_helpers_reject_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unsupported Row4 layout"):
        metadata_for_layout("future")
    with pytest.raises(ValueError, match="unsupported Row4 layout"):
        alignment_for_layout("future")
