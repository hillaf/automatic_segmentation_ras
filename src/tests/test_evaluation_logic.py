import unittest

from submission_code.box_generation import box_for_prediction
from submission_code.evaluate_detection import gt_boxes_for_test_fold, predicted_boxes_for_test_fold
from submission_code.mask_classifier import FoldPrediction


class EvaluationLogicTests(unittest.TestCase):
    def test_gt_fold_contains_only_held_out_images(self):
        predictions = {
            1: FoldPrediction(1, image_id=10, fold=1, true_label="fish", predicted_label="fish"),
            2: FoldPrediction(2, image_id=20, fold=2, true_label="fish", predicted_label="fish"),
        }
        gt = {10: [(0, 0, 1, 1)], 20: [(2, 2, 3, 3)]}

        fold_gt = gt_boxes_for_test_fold(gt, predictions, fold=1)

        self.assertEqual(set(fold_gt), {10})
        self.assertEqual(fold_gt[10], [(0, 0, 1, 1)])

    def test_predicted_head_uses_sam_bbox(self):
        annotation = {"id": 1, "image_id": 10, "bbox": [5, 6, 7, 8]}
        image = {"id": 10, "width": 100, "height": 100}

        box = box_for_prediction(annotation, image, predicted_label="head", head_fraction=0.25)

        self.assertEqual(box, (5.0, 6.0, 12.0, 14.0))

    def test_filter_uses_predicted_label_not_true_label(self):
        coco = {
            "images": [{"id": 10, "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 10, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 10, "bbox": [20, 20, 10, 10]},
            ],
        }
        predictions = {
            1: FoldPrediction(1, image_id=10, fold=1, true_label="fish", predicted_label="bad"),
            2: FoldPrediction(2, image_id=10, fold=1, true_label="bad", predicted_label="head"),
        }

        boxes = predicted_boxes_for_test_fold(
            coco,
            predictions,
            keep_classes=frozenset({"fish", "head"}),
            fold=1,
            head_fraction=0.25,
        )

        self.assertEqual(boxes, {10: [(20.0, 20.0, 30.0, 30.0)]})


if __name__ == "__main__":
    unittest.main()

