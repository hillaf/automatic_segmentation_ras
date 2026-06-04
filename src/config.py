from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DATASET_FILES = {
    "automatic": DATA_DIR / "masks_annotations_automatic" / "annotations_automatic.json",
    "manual": DATA_DIR / "masks_bounding_box_prompted" / "annotations_bounding_box_prompted.json",
}

GT_BBOX_DIR = DATA_DIR / "bounding_boxes_human_annotations"
IMAGE_DIR = DATA_DIR / "images"

CLASS_ORDER = ["fish", "bad", "head", "double"]
DEFAULT_IOU_THRESHOLDS = [0.30]
DEFAULT_HEAD_FRACTION = 0.25
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class EvaluationConfig:
    """Parameters that define one detection evaluation run."""

    keep_classes: frozenset[str]
    iou_thresholds: tuple[float, ...] = tuple(DEFAULT_IOU_THRESHOLDS)
    head_fraction: float = DEFAULT_HEAD_FRACTION
    n_splits: int = 5
    random_state: int = DEFAULT_RANDOM_STATE


FISH_ONLY_CONFIG = EvaluationConfig(keep_classes=frozenset({"fish"}))
FISH_AND_HEAD_CONFIG = EvaluationConfig(keep_classes=frozenset({"fish", "head"}))
