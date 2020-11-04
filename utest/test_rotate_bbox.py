import torch
import unittest

from utils.general import rotate_bbox

class RotateBboxTestCase(unittest.TestCase):
    def test_case_1(self):
        xyxy = torch.tensor([5., 6., 7., 8.]).view(1, 4)
        image_size = (100, 200, 3)
        new_xyxy = rotate_bbox(image_size, xyxy, k=-1).squeeze(0)
        print(new_xyxy)
        expected_new_xyxy = torch.tensor([6., 93., 8., 95.])
        print(expected_new_xyxy)
        for i in range(4):
            self.assertEqual(expected_new_xyxy[i], new_xyxy[i])