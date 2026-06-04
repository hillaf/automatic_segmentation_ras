from __future__ import annotations

from typing import Any


def decode_coco_mask(annotation: dict[str, Any], image: dict[str, Any]) -> Any:
    """Decode one COCO segmentation annotation into a binary mask."""

    from pycocotools import mask as mask_utils

    height, width = image["height"], image["width"]
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        rle = {"counts": segmentation["counts"], "size": segmentation.get("size", [height, width])}
        compressed = mask_utils.frPyObjects(rle, height, width) if isinstance(rle["counts"], list) else rle
        return mask_utils.decode(compressed)
    return mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))


def extract_mask_features(mask: Any) -> list[float]:
    """Legacy feature vector used by the mask-quality classifiers."""

    import cv2
    import numpy as np

    binary = (mask > 0).astype("uint8") * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [0.0] * 12

    points = np.array([point for contour in contours for point in contour])
    hull = cv2.convexHull(points)
    if len(hull) > 5:
        (_x, _y), (major_axis, minor_axis), _angle = cv2.fitEllipse(hull)
    else:
        major_axis = minor_axis = -1.0

    area = float(sum(cv2.contourArea(contour) for contour in contours))
    n_defects = 0
    for contour in contours:
        hull_idx = cv2.convexHull(np.array([point for point in contour]), returnPoints=False)
        if hull_idx is None:
            continue
        hull_idx[::-1].sort(axis=0)
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            n_defects += len(defects)

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()

    return [
        float(major_axis),
        float(minor_axis),
        area,
        float(len(contours)),
        float(n_defects),
        *[float(value) for value in hu],
    ]
