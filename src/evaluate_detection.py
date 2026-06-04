from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .box_generation import box_for_prediction
from .config import DATASET_FILES, FISH_AND_HEAD_CONFIG, FISH_ONLY_CONFIG, GT_BBOX_DIR, EvaluationConfig
from .data_loading import gt_boxes_by_image_id, image_table, load_coco_annotations
from .detection_metrics import DetectionSummary, match_detection_boxes
from .mask_classifier import FoldPrediction, predict_mask_classes_by_cv, validate_prediction_labels


def predicted_boxes_for_test_fold(
    coco: dict[str, Any],
    predictions: dict[int, FoldPrediction],
    keep_classes: frozenset[str],
    fold: int,
    head_fraction: float,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Build detection predictions for one held-out fold."""

    images = image_table(coco)
    output: dict[int, list[tuple[float, float, float, float]]] = {}
    for annotation in coco.get("annotations", []):
        prediction = predictions[annotation["id"]]
        if prediction.fold != fold:
            continue
        if prediction.predicted_label not in keep_classes:
            continue
        box = box_for_prediction(annotation, images[annotation["image_id"]], prediction.predicted_label, head_fraction)
        output.setdefault(annotation["image_id"], []).append(box)
    return output


def gt_boxes_for_test_fold(
    all_gt_boxes: dict[int, list[tuple[float, float, float, float]]],
    predictions: dict[int, FoldPrediction],
    fold: int,
) -> dict[int, list[tuple[float, float, float, float]]]:
    image_ids = {
        prediction.image_id
        for prediction in predictions.values()
        if prediction.fold == fold
    }
    return {image_id: all_gt_boxes.get(image_id, []) for image_id in image_ids}


def summarize_fold_values(values: list[float]) -> dict[str, float]:
    import numpy as np

    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def evaluate_dataset_detection(
    dataset_name: str,
    coco_path: Path,
    gt_bbox_dir: Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Run classifier CV and detection evaluation for one SAM prediction dataset."""

    coco = load_coco_annotations(coco_path)
    # Fold generation happens inside predict_mask_classes_by_cv(), which uses
    # StratifiedGroupKFold with image id as the group. The returned prediction
    # for each mask carries its held-out fold id; detection below uses those
    # same fold ids, so classifier prediction, mask precision, and detection
    # are evaluated on the same held-out images.
    predictions = predict_mask_classes_by_cv(coco, n_splits=config.n_splits, random_state=config.random_state)
    validate_prediction_labels(predictions)

    all_gt_boxes = gt_boxes_by_image_id(coco, gt_bbox_dir)
    fold_ids = sorted({prediction.fold for prediction in predictions.values()})
    per_threshold: dict[str, dict[str, Any]] = {}

    for iou_threshold in config.iou_thresholds:
        fold_summaries: list[dict[str, Any]] = []
        for fold in fold_ids:
            gt_fold = gt_boxes_for_test_fold(all_gt_boxes, predictions, fold)
            pred_fold = predicted_boxes_for_test_fold(coco, predictions, config.keep_classes, fold, config.head_fraction)
            summary = match_detection_boxes(gt_fold, pred_fold, iou_threshold)
            row = asdict(summary)
            row["fold"] = fold
            row["n_gt_boxes"] = sum(len(boxes) for boxes in gt_fold.values())
            row["n_predictions_before_filter"] = sum(1 for prediction in predictions.values() if prediction.fold == fold)
            row["n_predictions_after_filter"] = sum(len(boxes) for boxes in pred_fold.values())
            fold_summaries.append(row)
        per_threshold[f"iou_{iou_threshold:.2f}"] = summarize_detection_folds(fold_summaries)

    return {
        "dataset": dataset_name,
        "coco_path": str(coco_path),
        "gt_bbox_dir": str(gt_bbox_dir),
        "keep_classes": sorted(config.keep_classes),
        "head_fraction": config.head_fraction,
        "n_splits": config.n_splits,
        "random_state": config.random_state,
        "metrics": per_threshold,
    }


def summarize_detection_folds(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "n_gt_boxes",
        "n_predictions_before_filter",
        "n_predictions_after_filter",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_iou",
    ]
    summary = {key: summarize_fold_values([float(row[key]) for row in fold_rows]) for key in keys}
    return {"folds": fold_rows, "summary": summary}


def run_all_datasets(config: EvaluationConfig) -> list[dict[str, Any]]:
    return [
        evaluate_dataset_detection(dataset_name, coco_path, GT_BBOX_DIR, config)
        for dataset_name, coco_path in DATASET_FILES.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated detection evaluation for SAM mask post-processing.")
    parser.add_argument("--target", choices=["fish", "fish-head"], default="fish-head")
    parser.add_argument("--output-json", type=Path, default=Path("submission_detection_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FISH_ONLY_CONFIG if args.target == "fish" else FISH_AND_HEAD_CONFIG
    results = run_all_datasets(config)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
