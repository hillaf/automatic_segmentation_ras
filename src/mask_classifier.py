from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CLASS_ORDER
from .data_loading import image_table
from .mask_features import decode_coco_mask, extract_mask_features


@dataclass(frozen=True)
class FoldPrediction:
    annotation_id: int
    image_id: int
    fold: int
    true_label: str
    predicted_label: str


def build_feature_matrix(coco: dict[str, Any]) -> tuple[Any, Any, Any, list[int]]:
    """Return X, labels, image-group ids, and annotation ids."""

    import numpy as np

    images = image_table(coco)
    features: list[list[float]] = []
    labels: list[str] = []
    groups: list[int] = []
    annotation_ids: list[int] = []

    for annotation in coco.get("annotations", []):
        image = images[annotation["image_id"]]
        mask = decode_coco_mask(annotation, image)
        features.append(extract_mask_features(mask))
        labels.append(annotation["category_name"])
        groups.append(annotation["image_id"])
        annotation_ids.append(annotation["id"])

    return (
        np.asarray(features, dtype=float),
        np.asarray(labels),
        np.asarray(groups),
        annotation_ids,
    )


def grouped_cv_splits(labels: Any, groups: Any, n_splits: int, random_state: int) -> list[tuple[Any, Any]]:
    """Split masks by image id, stratifying by mask label when possible.

    The group is the COCO image id, so masks from the same image cannot appear
    in both train and test folds. This is the main guard against same-image
    leakage in classifier training.
    """

    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two image groups are required for cross-validation.")

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(labels)), labels, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(labels)), labels, groups))


def make_quality_classifier(random_state: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            max_depth=10,
            n_estimators=100,
            max_features=3,
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def predict_mask_classes_by_cv(coco: dict[str, Any], n_splits: int, random_state: int) -> dict[int, FoldPrediction]:
    """Train inside each grouped fold and return one out-of-fold prediction per mask."""

    X, labels, groups, annotation_ids = build_feature_matrix(coco)
    predictions: dict[int, FoldPrediction] = {}

    for fold, (train_idx, test_idx) in enumerate(grouped_cv_splits(labels, groups, n_splits, random_state), start=1):
        classifier_random_state = random_state + fold
        classifier = make_quality_classifier(classifier_random_state)
        classifier.fit(X[train_idx], labels[train_idx])
        predicted_labels = classifier.predict(X[test_idx])

        for local_idx, feature_idx in enumerate(test_idx):
            annotation_id = annotation_ids[feature_idx]
            predictions[annotation_id] = FoldPrediction(
                annotation_id=int(annotation_id),
                image_id=int(groups[feature_idx]),
                fold=fold,
                true_label=str(labels[feature_idx]),
                predicted_label=str(predicted_labels[local_idx]),
            )

    return predictions


def validate_prediction_labels(predictions: dict[int, FoldPrediction]) -> None:
    observed = {prediction.true_label for prediction in predictions.values()}
    missing = observed - set(CLASS_ORDER)
    if missing:
        raise ValueError(f"Unexpected mask labels in predictions: {sorted(missing)}")
