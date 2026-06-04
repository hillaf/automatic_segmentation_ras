import unittest

from submission_code.publication_results import all_sam_detection


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


if __name__ == "__main__":
    unittest.main()
