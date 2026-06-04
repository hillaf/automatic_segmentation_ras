# Evaluation diagnostics

- Generated one-vs-all classification values differ from `final_tables.tex` because the cleaned code uses image-grouped folds and per-fold scaling instead of the legacy mask-level folds with global scaling.
- Generated final post-processing values now differ from `final_tables.tex` after switching `extract_mask_features` to the raw-Hu legacy feature extractor.
- `final_tables.tex` appears to match a signed-log Hu feature extractor for final post-processing, while the one-vs-all classification table uses raw Hu moments.
