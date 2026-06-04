import unittest

import cv2
import numpy as np

from src.mask_features import extract_mask_features


class MaskFeatureTests(unittest.TestCase):
    def test_extract_mask_features_uses_raw_hu_moments(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:20, 8:24] = 1

        features = extract_mask_features(mask)

        binary = mask * 255
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        largest = max(contours, key=cv2.contourArea)
        raw_hu = cv2.HuMoments(cv2.moments(largest)).flatten()

        self.assertEqual(len(features), 12)
        self.assertTrue(np.allclose(features[5:], raw_hu))

    def test_extract_mask_features_uses_legacy_ellipse_fallback(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1

        features = extract_mask_features(mask)

        self.assertEqual(features[0], -1.0)
        self.assertEqual(features[1], -1.0)


if __name__ == "__main__":
    unittest.main()
