import unittest

from src.detection_metrics import iou, match_detection_boxes, one_to_one_iou_matches


class DetectionMetricsTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_detection_matching_is_one_to_one(self):
        gt = {1: [(0, 0, 10, 10)]}
        preds = {1: [(0, 0, 10, 10), (0, 0, 10, 10)]}

        summary = match_detection_boxes(gt, preds, iou_threshold=0.5)

        self.assertEqual(summary.tp, 1)
        self.assertEqual(summary.fp, 1)
        self.assertEqual(summary.fn, 0)
        self.assertEqual(summary.mean_iou, 1.0)

    def test_matching_prefers_best_iou_pair(self):
        gt = [(0, 0, 10, 10), (30, 0, 40, 10)]
        preds = [(1, 0, 11, 10), (0, 0, 10, 10)]

        matches = one_to_one_iou_matches(gt, preds, threshold=0.9)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1], 1)
        self.assertAlmostEqual(matches[0][2], 1.0)


if __name__ == "__main__":
    unittest.main()

