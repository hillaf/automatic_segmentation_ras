# Submission Evaluation Code

This directory is a reviewable evaluation code base for the paper
revision.

The main scope is the table-producing evaluation code:

- train the mask-quality classifier inside each cross-validation fold;
- filter SAM masks using the out-of-fold predicted class labels;
- convert retained SAM masks to detection boxes;
- compare those boxes with the human-annotated detection bounding boxes;
- recompute the mask-class distribution, one-vs-all classification, and
  detection result tables.

## Datasets

The evaluation uses the two SAM prediction datasets:

- `data/masks_bounding_box_prompted/annotations_bounding_box_prompted.json`
  as `manual`;
- `data/masks_annotations_automatic/annotations_automatic.json` as `automatic`.

The human detection ground truth is read from:

- `data/bounding_boxes_human_annotations/*.txt`

## Detection Logic

For the fish-only setting, only masks predicted as `fish` by the out-of-fold
classifier are retained. These masks are transformed to contour-guided head
bounding boxes before detection matching.

For the fish+head setting, masks predicted as `fish` use the contour-guided
generated box, while masks predicted as `head` keep the original SAM-produced
bounding box.

Detection metrics use one-to-one matching between prediction boxes and human
ground-truth boxes at the configured IoU threshold.

The mask-quality classifier uses grouped cross-validation by image id, so masks
from one image cannot appear in both the train and test partitions.

Mask features intentionally use the raw-Hu legacy feature definition: ellipse
axes, contour area, contour count, convexity-defect count, and raw Hu moments.

The one-vs-all classification table uses the same image-grouped folds and
per-fold scaling inside sklearn pipelines.

## Entry Point

```bash
.venv/bin/python -m submission_code.evaluate_detection --target fish --output-json submission_code/fish_results.json
.venv/bin/python -m submission_code.evaluate_detection --target fish-head --output-json submission_code/fish_head_results.json
.venv/bin/python -m submission_code.publication_tables --output-dir submission_code/generated_tables
```

`publication_tables` writes one `.tex` file per final table plus
`diagnostics.md`. Numeric computation lives in `publication_results.py`; the
table module only renders those results to TeX.

## Agreement Labels

The publication data folder includes `data/agreement_study_labels_sampled/`, 76 label files from three annotators.

