import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

ENDIAN = runpy.run_path(
    str(Path(__file__).parent.parent / "gguf" / "scripts" / "gguf_convert_endian.py"),
    run_name="gguf_convert_endian_test",
)
byteswap_noop = ENDIAN["byteswap_noop"]
byteswap_tq2_0 = ENDIAN["byteswap_tq2_0"]
byteswap_tensors = ENDIAN["byteswap_tensors"]


class TestQuantizedEndianConversion(unittest.TestCase):
    def test_tq2_0_swaps_only_f16_delta(self):
        block_size = GGML_QUANT_SIZES[GGMLQuantizationType.TQ2_0][1]
        self.assertEqual(block_size, 66)

        data = np.arange(2 * block_size, dtype=np.uint8)
        original = data.copy()
        tensor = SimpleNamespace(data=data)

        byteswap_tq2_0(tensor, 0)
        byteswap_tq2_0(tensor, block_size)

        expected = original.copy()
        expected[64:66] = expected[64:66][::-1]
        expected[130:132] = expected[130:132][::-1]
        np.testing.assert_array_equal(data, expected)
        self.assertIs(byteswap_tensors[GGMLQuantizationType.TQ2_0], byteswap_tq2_0)

        byteswap_tq2_0(tensor, 0)
        byteswap_tq2_0(tensor, block_size)
        np.testing.assert_array_equal(data, original)

    def test_mxfp4_data_is_endian_neutral(self):
        block_size = GGML_QUANT_SIZES[GGMLQuantizationType.MXFP4][1]
        self.assertEqual(block_size, 17)

        data = np.arange(block_size, dtype=np.uint8)
        original = data.copy()
        tensor = SimpleNamespace(data=data)

        byteswap_noop(tensor, 0)

        np.testing.assert_array_equal(data, original)
        self.assertIs(byteswap_tensors[GGMLQuantizationType.MXFP4], byteswap_noop)


if __name__ == "__main__":
    unittest.main()
