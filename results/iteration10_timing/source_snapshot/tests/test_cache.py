"""Integrity and timing-isolation checks for the evaluator-only disk cache."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from gbslab.cache import cached_reference


class ReferenceCacheTests(unittest.TestCase):
    def test_cache_reuse_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            V = 1.4 * np.eye(4)
            spec = {"fixture": "two_independent_thermal_modes"}
            first = cached_reference(spec, V, directory)
            second = cached_reference(spec, V, directory)
            self.assertFalse(first[3]); self.assertTrue(second[3])
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[1], second[1])
            path = Path(directory) / f"{first[4]}.npz"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "integrity"):
                cached_reference(spec, V, directory)


if __name__ == "__main__":
    unittest.main()
