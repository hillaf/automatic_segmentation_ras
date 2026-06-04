from __future__ import annotations

from typing import Any

from .detection_metrics import Box, xywh_to_xyxy
from .mask_features import decode_coco_mask


def contour_guided_head_box(annotation: dict[str, Any], image: dict[str, Any], head_fraction: float) -> Box:
    """Generate the square head box from a SAM mask, returned as xyxy."""

    import cv2
    import numpy as np

    mask = decode_coco_mask(annotation, image)
    binary = (mask > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # Empty or undecodable masks cannot define a contour. Keep the original
        # SAM box so the annotation is still evaluated instead of silently
        # dropping a prediction.
        return xywh_to_xyxy(annotation["bbox"])

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 2:
        # A line cannot be fit to a one-point contour. Fall back to the square
        # bounding box around that tiny contour, preserving a generated-box
        # prediction with minimal geometry.
        x, y, width, height = cv2.boundingRect(contour)
        side = float(max(width, height))
        return xywh_to_xyxy((float(x), float(y), side, side))

    distance_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    endpoints = line_mask_intersections(line, binary)
    if len(endpoints) != 2:
        # The fitted centerline should enter and exit the mask exactly once.
        # If it misses the mask or only touches one point, the head candidate
        # is ambiguous, so use the contour's square bounding box.
        x, y, width, height = cv2.boundingRect(contour)
        side = float(max(width, height))
        return xywh_to_xyxy((float(x), float(y), side, side))

    candidates: list[tuple[float, float, float, float]] = []
    for endpoint in endpoints:
        inside = point_fraction_along_line(line, binary, endpoint, head_fraction)
        if inside is None:
            # This endpoint is skipped if the mask boundary could not be
            # projected onto the fitted line. The other endpoint may still
            # produce a valid candidate.
            continue
        square = square_from_two_points(endpoint, inside, binary.shape)
        if square is not None:
            candidates.append(square)

    if not candidates:
        # If both endpoint candidates fail, keep the annotation by using the
        # square contour box rather than returning no prediction.
        x, y, width, height = cv2.boundingRect(contour)
        side = float(max(width, height))
        return xywh_to_xyxy((float(x), float(y), side, side))

    masses = [count_mask_pixels_in_rectangle(distance_transform, candidate) for candidate in candidates]
    chosen = int(np.argmax(masses))
    return xywh_to_xyxy(candidates[chosen])


def line_mask_intersections(line: Any, mask: Any) -> list[tuple[int, int]]:
    import numpy as np

    vx, vy, x0, y0 = np.asarray(line).reshape(-1)[:4].astype(float)
    height, width = mask.shape[:2]
    norm = float(np.sqrt(vx * vx + vy * vy))
    if norm == 0:
        # Degenerate fitted line; caller will fall back to the contour box.
        return []
    vx, vy = vx / norm, vy / norm

    hits: list[tuple[int, int]] = []
    for t in np.arange(-max(width, height), max(width, height), 1.0):
        x = int(round(x0 + t * vx))
        y = int(round(y0 + t * vy))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            point = (x, y)
            if not hits or hits[-1] != point:
                hits.append(point)

    if not hits:
        # No intersection with mask pixels; caller will fall back.
        return []
    return [hits[0], hits[-1]] if hits[0] != hits[-1] else [hits[0]]


def point_fraction_along_line(
    line: Any,
    mask: Any,
    endpoint: tuple[int, int],
    fraction: float,
) -> tuple[float, float] | None:
    import cv2
    import numpy as np

    vx, vy, x0, y0 = np.asarray(line).reshape(-1)[:4].astype(float)
    endpoint_x, endpoint_y = endpoint
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # No boundary to project onto the fitted line.
        return None

    boundary = np.vstack(contours).squeeze()
    if boundary.ndim != 2 or boundary.shape[0] < 2:
        # Need at least two boundary points to locate the opposite end.
        return None

    t_boundary = (boundary[:, 0] - x0) * vx + (boundary[:, 1] - y0) * vy
    t_endpoint = (endpoint_x - x0) * vx + (endpoint_y - y0) * vy
    t_other = np.min(t_boundary) if abs(t_endpoint - np.max(t_boundary)) < abs(t_endpoint - np.min(t_boundary)) else np.max(t_boundary)
    t_target = t_endpoint + fraction * (t_other - t_endpoint)
    return float(x0 + vx * t_target), float(y0 + vy * t_target)


def square_from_two_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    image_height, image_width = image_shape[:2]
    cx = min(max(0.5 * (p1[0] + p2[0]), 0.0), float(image_width - 1))
    cy = min(max(0.5 * (p1[1] + p2[1]), 0.0), float(image_height - 1))
    half_side = max(abs(p1[0] - cx), abs(p1[1] - cy), abs(p2[0] - cx), abs(p2[1] - cy))
    if half_side <= 0:
        # The two points collapsed to the same image location; no box can be
        # formed, so caller tries the other endpoint or falls back.
        return None
    side = 2.0 * half_side
    x = min(max(cx - half_side, 0.0), float(image_width - 1))
    y = min(max(cy - half_side, 0.0), float(image_height - 1))
    side = min(side, float(image_width) - x, float(image_height) - y)
    return float(x), float(y), float(side), float(side)


def count_mask_pixels_in_rectangle(mask: Any, rect_xywh: tuple[float, float, float, float]) -> int:
    x, y, width, height = rect_xywh
    image_height, image_width = mask.shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + width)))
    y2 = min(image_height, int(round(y + height)))
    if x1 >= x2 or y1 >= y2:
        # Candidate is outside the image after rounding; it contributes no mass.
        return 0
    return int((mask[y1:y2, x1:x2] > 0).sum())


def box_for_prediction(annotation: dict[str, Any], image: dict[str, Any], predicted_label: str, head_fraction: float) -> Box:
    """Use SAM bbox for predicted heads, generated head bbox otherwise."""

    if predicted_label == "head":
        return xywh_to_xyxy(annotation["bbox"])
    return contour_guided_head_box(annotation, image, head_fraction)
