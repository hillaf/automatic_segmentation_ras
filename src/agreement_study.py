from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import DATA_DIR


AGREEMENT_LABEL_DIR = DATA_DIR / "agreement_study_labels_sampled"
AGREEMENT_SAMPLE_FILE = DATA_DIR / "agreement_sample_common211_scaled_to_camera3.txt"
AGREEMENT_IOU_THRESHOLD = 0.30
ANNOTATOR_ORDER = ("annotator_a", "annotator_b", "annotator_c")
PAIR_ORDER = (
    ("annotator_a", "annotator_b"),
    ("annotator_c", "annotator_a"),
    ("annotator_c", "annotator_b"),
)
PAIR_LABELS = {
    ("annotator_a", "annotator_b"): "A vs B",
    ("annotator_c", "annotator_a"): "C vs A",
    ("annotator_c", "annotator_b"): "C vs B",
}


@dataclass(frozen=True)
class AgreementBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.width / 2.0,
            self.center_y - self.height / 2.0,
            self.center_x + self.width / 2.0,
            self.center_y + self.height / 2.0,
        )


def load_sample_ids(path: Path = AGREEMENT_SAMPLE_FILE) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_yolo_boxes(path: Path) -> list[AgreementBox]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        center_x, center_y, width, height = map(float, parts[1:5])
        boxes.append(AgreementBox(class_id, center_x, center_y, width, height))
    return boxes


def load_agreement_labels(label_dir: Path = AGREEMENT_LABEL_DIR) -> dict[str, dict[str, list[AgreementBox]]]:
    labels: dict[str, dict[str, list[AgreementBox]]] = {}
    for annotator in ANNOTATOR_ORDER:
        annotator_dir = label_dir / annotator
        labels[annotator] = {
            path.stem: read_yolo_boxes(path)
            for path in sorted(annotator_dir.glob("*.txt"))
        }
    return labels


def box_iou(box_a: AgreementBox, box_b: AgreementBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.xyxy()
    bx1, by1, bx2, by2 = box_b.xyxy()
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def hungarian_same_class_matches(
    boxes_a: list[AgreementBox],
    boxes_b: list[AgreementBox],
    iou_threshold: float,
) -> list[float]:
    """Return IoUs for one-to-one same-class matches above the threshold."""

    if not boxes_a or not boxes_b:
        return []

    scores = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for row, box_a in enumerate(boxes_a):
        for col, box_b in enumerate(boxes_b):
            if box_a.class_id != box_b.class_id:
                continue
            iou = box_iou(box_a, box_b)
            if iou >= iou_threshold:
                scores[row, col] = iou

    row_indices, col_indices = linear_sum_assignment(1.0 - scores)
    return [
        float(scores[row, col])
        for row, col in zip(row_indices, col_indices)
        if scores[row, col] > 0.0
    ]


def pairwise_agreement_rows(
    sample_ids: Iterable[str] | None = None,
    iou_threshold: float = AGREEMENT_IOU_THRESHOLD,
) -> list[dict[str, float | int | str]]:
    labels = load_agreement_labels()
    image_ids = list(sample_ids) if sample_ids is not None else load_sample_ids()
    rows: list[dict[str, float | int | str]] = []

    for first, second in PAIR_ORDER:
        n_first = 0
        n_second = 0
        matched_ious: list[float] = []

        for image_id in image_ids:
            boxes_first = labels[first].get(image_id, [])
            boxes_second = labels[second].get(image_id, [])
            n_first += len(boxes_first)
            n_second += len(boxes_second)
            matched_ious.extend(hungarian_same_class_matches(boxes_first, boxes_second, iou_threshold))

        matched = len(matched_ious)
        precision = matched / n_first if n_first else math.nan
        recall = matched / n_second if n_second else math.nan
        rows.append(
            {
                "pair": PAIR_LABELS[(first, second)],
                "images": len(image_ids),
                "n_first": n_first,
                "n_second": n_second,
                "matched": matched,
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * matched / (n_first + n_second) if (n_first + n_second) else math.nan,
                "mean_iou": float(np.mean(matched_ious)) if matched_ious else math.nan,
            }
        )

    return rows
