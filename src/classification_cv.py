from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CLASS_ORDER, DATASET_FILES, DEFAULT_RANDOM_STATE
from .data_loading import load_coco_annotations
from .mask_classifier import build_feature_matrix, grouped_cv_splits


@dataclass(frozen=True)
class ClassificationSummary:
    mean: float
    std: float


def summarize(values: list[float]) -> ClassificationSummary:
    import numpy as np

    return ClassificationSummary(float(np.mean(values)), float(np.std(values)))


def classifier_specs(random_state: int) -> list[tuple[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # StandardScaler is inside each pipeline, so it is fit only on the
    # training split for each fold. This avoids the global scaling leak in the
    # original one-vs-all table script.
    return [
        ("KNN", make_pipeline(StandardScaler(), KNeighborsClassifier(3))),
        (
            "MLP",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(alpha=0.05, max_iter=5000, random_state=random_state),
            ),
        ),
        (
            "RF",
            make_pipeline(
                StandardScaler(),
                RandomForestClassifier(
                    max_depth=10,
                    n_estimators=20,
                    max_features=3,
                    random_state=random_state,
                ),
            ),
        ),
    ]


def one_vs_all_rows_for_dataset(
    dataset: str,
    coco: dict[str, Any],
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.base import clone

    features, labels, groups, _annotation_ids = build_feature_matrix(coco)
    # The split is shared across the four one-vs-all targets and grouped by
    # image id, so masks from one image cannot land in both train and test.
    splits = grouped_cv_splits(labels, groups, n_splits=n_splits, random_state=random_state)
    rows = []

    for class_name in CLASS_ORDER:
        binary_labels = np.asarray([class_name if label == class_name else "other" for label in labels])
        for model_name, classifier in classifier_specs(random_state):
            scores = []
            fold_sizes = []
            for train_idx, test_idx in splits:
                fold_classifier = clone(classifier)
                fold_classifier.fit(features[train_idx], binary_labels[train_idx])
                predicted = fold_classifier.predict(features[test_idx])
                scores.append(float(np.mean(predicted == binary_labels[test_idx])))
                fold_sizes.append(int(len(test_idx)))

            score_summary = summarize(scores)
            size_summary = summarize([float(size) for size in fold_sizes])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "class": class_name,
                    "n_masks_test_fold_mean": size_summary.mean,
                    "n_masks_test_fold_std": size_summary.std,
                    "fold_scores": scores,
                    "fold_n_masks": fold_sizes,
                    "accuracy_mean": score_summary.mean,
                    "accuracy_std": score_summary.std,
                }
            )

    return rows


def one_vs_all_rows(
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    rows = []
    for dataset, path in DATASET_FILES.items():
        rows.extend(
            one_vs_all_rows_for_dataset(
                dataset,
                load_coco_annotations(path),
                n_splits=n_splits,
                random_state=random_state,
            )
        )
    return rows


def aggregate_one_vs_all_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    aggregated = []
    for dataset in ["manual", "automatic"]:
        for model in ["KNN", "MLP", "RF"]:
            group = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            if not group:
                continue

            fold_scores = []
            for fold_idx in range(len(group[0]["fold_scores"])):
                numerator = sum(row["fold_scores"][fold_idx] * row["fold_n_masks"][fold_idx] for row in group)
                denominator = sum(row["fold_n_masks"][fold_idx] for row in group)
                fold_scores.append(numerator / denominator if denominator else 0.0)

            row = {
                "dataset": dataset,
                "model": model,
                "n_masks_test_fold_mean": float(np.mean([row["n_masks_test_fold_mean"] for row in group])),
                "n_masks_test_fold_std": float(np.mean([row["n_masks_test_fold_std"] for row in group])),
                "pooled_accuracy_mean": float(np.mean(fold_scores)),
                "pooled_accuracy_std": float(np.std(fold_scores)),
            }
            for class_name in CLASS_ORDER:
                class_row = next(row for row in group if row["class"] == class_name)
                row[f"{class_name}_accuracy_mean"] = class_row["accuracy_mean"]
                row[f"{class_name}_accuracy_std"] = class_row["accuracy_std"]
            aggregated.append(row)
    return aggregated
