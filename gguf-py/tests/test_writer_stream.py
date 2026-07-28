#!/usr/bin/env python3

import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf


class TestGGUFWriterTensorStream(unittest.TestCase):
    def prepare_writer(
        self,
        path: Path,
        *,
        tensor_shape: tuple[int, ...],
        tensor_dtype: np.dtype,
        tensor_nbytes: int,
        endianess: gguf.GGUFEndian = gguf.GGUFEndian.LITTLE,
    ) -> tuple[gguf.GGUFWriter, int]:
        writer = gguf.GGUFWriter(path, "test", endianess=endianess)
        writer.add_custom_alignment(64)
        writer.add_tensor_info("stream", tensor_shape, tensor_dtype, tensor_nbytes)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        assert writer.fout is not None
        return writer, writer.fout[0].tell()

    def test_writes_uneven_chunks_with_one_64_byte_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream.gguf"
            values = np.arange(17, dtype=np.int8)
            writer, tensor_info_end = self.prepare_writer(
                path,
                tensor_shape=values.shape,
                tensor_dtype=values.dtype,
                tensor_nbytes=values.nbytes,
            )

            try:
                writer.write_tensor_data_stream((values[:3], values[3:8], values[8:]))
                writer.flush()

                data_offset = gguf.GGUFWriter.ggml_pad(tensor_info_end, 64)
                file_data = path.read_bytes()
                self.assertEqual(data_offset % 64, 0)
                self.assertEqual(file_data[data_offset:data_offset + values.nbytes], values.tobytes())
                self.assertEqual(file_data[data_offset + values.nbytes:], bytes(64 - values.nbytes))
                self.assertEqual(len(file_data), data_offset + 64)
                self.assertEqual(writer.state.name, "WEIGHTS")
                self.assertEqual(writer.tensors[0], {})
            finally:
                writer.close()

    def test_converts_each_chunk_to_big_endian(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream-big-endian.gguf"
            source_values = (0x0102, 0x1122, 0x3344, 0x5566, 0x7788)
            chunks = (
                np.asarray(source_values[:2], dtype=np.int16),
                np.asarray(source_values[2:], dtype=np.int16),
            )
            writer, tensor_info_end = self.prepare_writer(
                path,
                tensor_shape=(len(source_values),),
                tensor_dtype=chunks[0].dtype,
                tensor_nbytes=sum(chunk.nbytes for chunk in chunks),
                endianess=gguf.GGUFEndian.BIG,
            )

            try:
                writer.write_tensor_data_stream(chunks)
                writer.flush()

                data_offset = gguf.GGUFWriter.ggml_pad(tensor_info_end, 64)
                expected = b"".join(struct.pack(">h", value) for value in source_values)
                self.assertEqual(path.read_bytes()[data_offset:data_offset + len(expected)], expected)
            finally:
                writer.close()

    def test_big_endian_failure_does_not_mutate_chunks_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream-big-endian-retry.gguf"
            values = np.asarray((0x0102, 0x1122, 0x3344), dtype=np.int16)
            original = values.copy()
            writer, tensor_info_end = self.prepare_writer(
                path,
                tensor_shape=values.shape,
                tensor_dtype=values.dtype,
                tensor_nbytes=values.nbytes,
                endianess=gguf.GGUFEndian.BIG,
            )

            try:
                with self.assertRaisesRegex(ValueError, "underflow"):
                    writer.write_tensor_data_stream((values[:2],))
                np.testing.assert_array_equal(values, original)

                writer.write_tensor_data_stream((values[:1], values[1:]))
                writer.flush()
                data_offset = gguf.GGUFWriter.ggml_pad(tensor_info_end, 64)
                expected = b"".join(struct.pack(">h", value) for value in original)
                self.assertEqual(path.read_bytes()[data_offset:data_offset + len(expected)], expected)
                np.testing.assert_array_equal(values, original)
            finally:
                writer.close()

    def test_underflow_keeps_tensor_info_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream-underflow.gguf"
            values = np.arange(10, dtype=np.int8)
            writer, write_start = self.prepare_writer(
                path,
                tensor_shape=values.shape,
                tensor_dtype=values.dtype,
                tensor_nbytes=values.nbytes,
            )

            try:
                with self.assertRaisesRegex(ValueError, r"underflow: expected 10 bytes, got 9"):
                    writer.write_tensor_data_stream((values[:4], values[4:9]))

                assert writer.fout is not None
                self.assertEqual(writer.fout[0].tell(), write_start)
                self.assertEqual(list(writer.tensors[0]), ["stream"])
                self.assertEqual(writer.state.name, "TI_DATA")
            finally:
                writer.close()

    def test_overflow_keeps_tensor_info_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream-overflow.gguf"
            values = np.arange(11, dtype=np.int8)
            writer, write_start = self.prepare_writer(
                path,
                tensor_shape=(10,),
                tensor_dtype=values.dtype,
                tensor_nbytes=10,
            )

            try:
                with self.assertRaisesRegex(ValueError, r"overflow: expected 10 bytes, got at least 11"):
                    writer.write_tensor_data_stream((values[:6], values[6:]))

                assert writer.fout is not None
                self.assertEqual(writer.fout[0].tell(), write_start)
                self.assertEqual(list(writer.tensors[0]), ["stream"])
                self.assertEqual(writer.state.name, "TI_DATA")
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
