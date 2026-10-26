"""Fast tests without training CTGAN."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).with_name("benchmark_v2.py")
spec = importlib.util.spec_from_file_location("benchmark_v2", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BenchmarkV2Tests(unittest.TestCase):
    def test_deterministic_seeds(self):
        self.assertEqual(mod.derived_seed(4, 10), mod.derived_seed(4, 10))
        self.assertNotEqual(mod.derived_seed(4, 10), mod.derived_seed(4, 11))

    def test_split_fingerprints(self):
        x, y = mod.sample(100, 7, "s1", "test")
        x2, y2 = mod.sample(100, 7, "s1", "test")
        self.assertEqual(mod.digest(x, y), mod.digest(x2, y2))
        x3, y3 = mod.sample(100, 8, "s1", "test")
        self.assertNotEqual(mod.digest(x, y), mod.digest(x3, y3))

    def test_structural_reference_nonzero(self):
        x, _ = mod.sample(100, 3, "s1", "train")
        self.assertTrue(np.isfinite(mod.residual(x)).all())
        self.assertGreater(float(mod.residual(x).mean()), 0)

    def test_smoke_baseline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            records, quality, failures = mod.run(
                1, ["s1"], ["baseline"], Path(directory), trees=10)
            self.assertEqual(len(failures), 0)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(quality), 0)
            self.assertEqual(len({r["test_sha256"] for r in records}), 1)


if __name__ == "__main__":
    unittest.main()
