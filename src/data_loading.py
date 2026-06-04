from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .detection_metrics import Box, yolo_to_xyxy


def load_coco_annotations(path: Path) -> dict[str, Any]:
    """Load COCO-style SAM mask annotations and attach category names to annotations."""

    coco = json.loads(path.read_text())
    id_to_name = {category["id"]: category["name"] for category in coco.get("categories", [])}
    result = copy.deepcopy(coco)
    for annotation in result.get("annotations", []):
        annotation["category_name"] = id_to_name[annotation["category_id"]]
    return result


def image_table(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {image["id"]: image for image in coco.get("images", [])}


def read_yolo_gt_boxes(yolo_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Read human-annotated detection boxes from YOLO txt files."""

    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for path in sorted(yolo_dir.glob("*.txt")):
        image_name = path.stem
        boxes[image_name] = []
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _class_id, cx, cy, width, height = map(float, parts)
            boxes[image_name].append((cx, cy, width, height))
    return boxes


def gt_boxes_by_image_id(coco: dict[str, Any], yolo_dir: Path) -> dict[int, list[Box]]:
    """Map each COCO image id to its human-annotated detection boxes."""

    yolo_boxes = read_yolo_gt_boxes(yolo_dir)
    output: dict[int, list[Box]] = {}
    for image in coco.get("images", []):
        image_name = Path(image["file_name"]).stem
        output[image["id"]] = [
            yolo_to_xyxy(box, image["width"], image["height"])
            for box in yolo_boxes.get(image_name, [])
        ]
    return output

