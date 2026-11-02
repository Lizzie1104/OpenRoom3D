import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fill_table_layer import fill, pull_push, rotation_to_wxyz


class FillTest(unittest.TestCase):
    def setUp(self):
        self.texture = np.full((120, 120, 3), 180, np.uint8)
        self.hole = np.zeros((120, 120), bool)
        self.hole[40:80, 40:80] = True
        self.table = ~self.hole

    def test_pull_push_keeps_known_and_recovers_flat_color(self):
        result = pull_push(self.texture, self.table)
        self.assertTrue(np.array_equal(result[self.table], self.texture[self.table].astype(np.float32)))
        self.assertAlmostEqual(float(result[self.hole].mean()), 180, delta=1)

    def test_fill_leaves_texels_far_from_hole_unchanged(self):
        filled, _ = fill(self.texture, self.table, self.hole, 0)
        self.assertTrue(np.array_equal(filled[:20], self.texture[:20]))
        self.assertAlmostEqual(float(filled[self.hole].mean()), 180, delta=2)

    def test_quaternion_of_identity(self):
        self.assertTrue(np.allclose(rotation_to_wxyz(np.eye(3)), [1, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
