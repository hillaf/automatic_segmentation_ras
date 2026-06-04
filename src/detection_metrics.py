from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionSummary:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou: float


def xywh_to_xyxy(box: Iterable[float]) -> Box:
    x, y, width, height = box
    return float(x), float(y), float(x + width), float(y + height)


def yolo_to_xyxy(box: Iterable[float], image_width: int, image_height: int) -> Box:
    cx, cy, width, height = box
    x1 = (cx - width / 2.0) * image_width
    y1 = (cy - height / 2.0) * image_height
    x2 = (cx + width / 2.0) * image_width
    y2 = (cy + height / 2.0) * image_height
    return float(x1), float(y1), float(x2), float(y2)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def iou(box_a: Box, box_b: Box) -> float:
    intersection = intersection_area(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def one_to_one_iou_matches(gt_boxes: list[Box], pred_boxes: list[Box], threshold: float) -> list[tuple[int, int, float]]:
    """Return maximum-weight one-to-one matches with IoU >= threshold."""

    if not gt_boxes or not pred_boxes:
        return []

    score_matrix = [
        [iou(pred_box, gt_box) for pred_box in pred_boxes]
        for gt_box in gt_boxes
    ]

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return greedy_one_to_one_matches(score_matrix, threshold)

    # The reward term forces valid pairs to be preferred over invalid pairs,
    # while still maximizing IoU among valid assignments.
    valid_reward = 10.0
    cost_matrix = [
        [
            -(valid_reward + score if score >= threshold else score)
            for score in row
        ]
        for row in score_matrix
    ]
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, float]] = []
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        score = score_matrix[int(gt_idx)][int(pred_idx)]
        if score >= threshold:
            matches.append((int(gt_idx), int(pred_idx), float(score)))
    return matches


def greedy_one_to_one_matches(score_matrix: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    """Fallback matcher used only if scipy is unavailable."""

    candidates = [
        (gt_idx, pred_idx, score)
        for gt_idx, row in enumerate(score_matrix)
        for pred_idx, score in enumerate(row)
        if score >= threshold
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gt_idx, pred_idx, score in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, float(score)))
    return matches


def match_detection_boxes(
    gt_by_image: dict[int, list[Box]],
    pred_by_image: dict[int, list[Box]],
    iou_threshold: float,
) -> DetectionSummary:
    """Compare predicted and GT boxes for detection-style evaluation.

    This is the submission package's shared object-detection matcher. It is
    used for SAM detection evaluation and final table summaries. Inputs are
    grouped by image id; within each image, boxes are matched one-to-one by IoU.
    """

    tp = fp = fn = 0
    matched_ious: list[float] = []

    for image_id in sorted(set(gt_by_image) | set(pred_by_image)):
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])
        matches = one_to_one_iou_matches(gt_boxes, pred_boxes, iou_threshold)
        tp += len(matches)
        fp += len(pred_boxes) - len(matches)
        fn += len(gt_boxes) - len(matches)
        matched_ious.extend(score for _gt_idx, _pred_idx, score in matches)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return DetectionSummary(tp, fp, fn, precision, recall, f1, mean_iou)
