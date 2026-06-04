from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .box_generation import box_for_prediction
from .config import DATASET_FILES, DEFAULT_HEAD_FRACTION, FISH_AND_HEAD_CONFIG, GT_BBOX_DIR
from .data_loading import gt_boxes_by_image_id, image_table, load_coco_annotations
from .detection_metrics import Box, DetectionSummary, match_detection_boxes
from .evaluate_detection import gt_boxes_for_test_fold, summarize_detection_folds
from .mask_classifier import FoldPrediction, predict_mask_classes_by_cv


PROMPT_ORDER = ["manual", "automatic"]


def summarize_values(values: list[float]) -> dict[str, float]:
    import numpy as np

    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def generated_box_lookup(coco: dict[str, Any]) -> dict[int, Box]:
    """Recompute one contour-guided head box per SAM annotation from masks."""

    images = image_table(coco)
    return {
        annotation["id"]: box_for_prediction(annotation, images[annotation["image_id"]], "fish", DEFAULT_HEAD_FRACTION)
        for annotation in coco["annotations"]
    }


def mask_class_distribution(coco: dict[str, Any]) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(annotation["category_name"] for annotation in coco["annotations"])
    total = sum(counts.values())
    return {
        "n": total,
        "fish": counts["fish"] / total,
        "bad": counts["bad"] / total,
        "head": counts["head"] / total,
        "double": counts["double"] / total,
    }


def all_sam_detection(coco: dict[str, Any], generated_boxes: dict[int, Box], iou_threshold: float = 0.30) -> dict[str, Any]:
    """Evaluate all SAM masks as predictions.

    The scope is images that have at least one SAM annotation. This matches the
    submitted no-postprocessing table and avoids adding false negatives from
    images where SAM produced no mask to evaluate.
    """

    all_gt = gt_boxes_by_image_id(coco, GT_BBOX_DIR)
    predictions: dict[int, list[Box]] = {}
    for annotation in coco["annotations"]:
        predictions.setdefault(annotation["image_id"], []).append(generated_boxes[annotation["id"]])
    gt = {image_id: all_gt.get(image_id, []) for image_id in predictions}
    summary = match_detection_boxes(gt, predictions, iou_threshold)
    return {"n_predictions": sum(len(boxes) for boxes in predictions.values()), **asdict(summary)}


def unfiltered_predictions(coco: dict[str, Any], fold_predictions: dict[int, FoldPrediction]) -> dict[int, FoldPrediction]:
    """Route every SAM annotation through the generated-box path while preserving CV folds."""

    return {
        annotation["id"]: FoldPrediction(
            annotation_id=annotation["id"],
            image_id=annotation["image_id"],
            fold=fold_predictions[annotation["id"]].fold,
            true_label=annotation["category_name"],
            predicted_label="fish",
        )
        for annotation in coco["annotations"]
    }


def mask_filtering_by_fold(predictions: dict[int, FoldPrediction], keep_classes: frozenset[str]) -> dict[str, Any]:
    """Summarize filtering as binary relevance, not multiclass accuracy.

    The final table asks how pure the kept mask set is for the filtering
    target. A kept mask is therefore correct when its true class is one of the
    kept classes, even if the classifier confused `fish` and `head`.
    """

    rows = []
    for fold in sorted({prediction.fold for prediction in predictions.values()}):
        fold_predictions = [prediction for prediction in predictions.values() if prediction.fold == fold]
        kept = [prediction for prediction in fold_predictions if prediction.predicted_label in keep_classes]
        correct = sum(1 for prediction in kept if prediction.true_label in keep_classes)
        rows.append(
            {
                "fold": fold,
                "n_before_filter": len(fold_predictions),
                "n_after_filter": len(kept),
                "precision": correct / len(kept) if kept else 0.0,
            }
        )

    summary = {
        f"{key}_{stat}": value
        for key in ["n_before_filter", "n_after_filter", "precision"]
        for stat, value in summarize_values([float(row[key]) for row in rows]).items()
    }
    return {"folds": rows, "summary": summary}


def predicted_boxes_for_fold(
    coco: dict[str, Any],
    predictions: dict[int, FoldPrediction],
    keep_classes: frozenset[str],
    generated_boxes: dict[int, Box],
    fold: int,
) -> dict[int, list[Box]]:
    images = image_table(coco)
    boxes_by_image: dict[int, list[Box]] = {}
    for annotation in coco["annotations"]:
        prediction = predictions[annotation["id"]]
        if prediction.fold != fold or prediction.predicted_label not in keep_classes:
            continue
        if prediction.predicted_label == "head":
            box = box_for_prediction(annotation, images[annotation["image_id"]], "head", DEFAULT_HEAD_FRACTION)
        else:
            box = generated_boxes[annotation["id"]]
        boxes_by_image.setdefault(annotation["image_id"], []).append(box)
    return boxes_by_image


def fold_detection(
    coco: dict[str, Any],
    predictions: dict[int, FoldPrediction],
    keep_classes: frozenset[str],
    generated_boxes: dict[int, Box],
    iou_threshold: float = 0.30,
) -> dict[str, Any]:
    all_gt_boxes = gt_boxes_by_image_id(coco, GT_BBOX_DIR)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted({prediction.fold for prediction in predictions.values()}):
        gt_fold = gt_boxes_for_test_fold(all_gt_boxes, predictions, fold)
        pred_fold = predicted_boxes_for_fold(coco, predictions, keep_classes, generated_boxes, fold)
        summary: DetectionSummary = match_detection_boxes(gt_fold, pred_fold, iou_threshold)
        row = asdict(summary)
        row["fold"] = fold
        row["n_gt_boxes"] = sum(len(boxes) for boxes in gt_fold.values())
        row["n_predictions_before_filter"] = sum(1 for prediction in predictions.values() if prediction.fold == fold)
        row["n_predictions_after_filter"] = sum(len(boxes) for boxes in pred_fold.values())
        fold_rows.append(row)
    return summarize_detection_folds(fold_rows)


def final_detection_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for dataset in PROMPT_ORDER:
        coco = load_coco_annotations(DATASET_FILES[dataset])
        generated_boxes = generated_box_lookup(coco)
        classifier_predictions = predict_mask_classes_by_cv(
            coco,
            n_splits=FISH_AND_HEAD_CONFIG.n_splits,
            random_state=FISH_AND_HEAD_CONFIG.random_state,
        )
        none_predictions = unfiltered_predictions(coco, classifier_predictions)
        rows[f"{dataset}/None"] = {
            "mask": mask_filtering_by_fold(none_predictions, frozenset({"fish", "head"})),
            "detection": fold_detection(coco, none_predictions, frozenset({"fish", "head"}), generated_boxes),
        }
        rows[f"{dataset}/Ours"] = {
            "mask": mask_filtering_by_fold(classifier_predictions, frozenset({"fish", "head"})),
            "detection": fold_detection(coco, classifier_predictions, frozenset({"fish", "head"}), generated_boxes),
        }
    return rows


def all_sam_rows() -> dict[str, dict[str, Any]]:
    rows = {}
    for dataset in PROMPT_ORDER:
        coco = load_coco_annotations(DATASET_FILES[dataset])
        rows[dataset] = all_sam_detection(coco, generated_box_lookup(coco))
    return rows


def class_distribution_rows() -> dict[str, dict[str, Any]]:
    return {
        dataset: mask_class_distribution(load_coco_annotations(DATASET_FILES[dataset]))
        for dataset in PROMPT_ORDER
    }
