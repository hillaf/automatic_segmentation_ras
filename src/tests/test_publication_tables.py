import unittest

from src.mask_classifier import FoldPrediction
from src.publication_results import all_sam_detection, mask_filtering_by_fold


class PublicationTableTests(unittest.TestCase):
    def test_all_sam_detection_excludes_images_without_sam_predictions(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "with_predictions.png", "width": 10, "height": 10},
                {"id": 2, "file_name": "without_predictions.png", "width": 10, "height": 10},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_name": "fish"},
            ],
        }
        generated_boxes = {10: (0.0, 0.0, 5.0, 5.0)}

        result = all_sam_detection(coco, generated_boxes, iou_threshold=0.30)

        self.assertEqual(result["n_predictions"], 1)
        self.assertEqual(result["fn"], 0)

    def test_mask_filtering_precision_is_binary_relevance(self):
        predictions = {
            1: FoldPrediction(1, image_id=10, fold=0, true_label="fish", predicted_label="fish"),
            2: FoldPrediction(2, image_id=10, fold=0, true_label="head", predicted_label="fish"),
            3: FoldPrediction(3, image_id=10, fold=0, true_label="bad", predicted_label="fish"),
            4: FoldPrediction(4, image_id=10, fold=0, true_label="double", predicted_label="bad"),
        }

        result = mask_filtering_by_fold(predictions, frozenset({"fish", "head"}))

        self.assertEqual(result["folds"][0]["n_before_filter"], 4)
        self.assertEqual(result["folds"][0]["n_after_filter"], 3)
        self.assertAlmostEqual(result["folds"][0]["precision"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
