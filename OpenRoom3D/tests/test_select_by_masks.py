import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_by_masks import hull_support


class HullSupportTest(unittest.TestCase):
    def test_center_outside_mask_in_one_view_loses_support(self):
        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[8:12, 8:12] = 255
            cv2.imwrite(str(Path(directory) / "00000.png"), mask)
            cv2.imwrite(str(Path(directory) / "00001.png"), mask)
            # Nerfstudio cameras look down -z; camera 1 shifts the scene by half an image.
            front, shifted = np.eye(4), np.eye(4)
            front[2, 3] = shifted[2, 3] = 1.
            shifted[0, 3] = .5
            cameras = [(0, {"c2w": front}), (1, {"c2w": shifted})]
            intrinsics = np.array([[20., 0, 10], [0, 20, 10], [0, 0, 1]])
            means = torch.tensor([[0., 0, 0], [0., 0, 5.]])
            support, seen = hull_support(means, cameras, intrinsics, 20, 20, directory, 0)
        self.assertEqual(support[0].item(), .5)
        self.assertEqual(seen.tolist(), [2., 0.])


if __name__ == "__main__":
    unittest.main()
