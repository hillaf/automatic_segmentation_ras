from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .agreement_study import pairwise_agreement_rows
from .classification_cv import aggregate_one_vs_all_rows, one_vs_all_rows
from .publication_results import (
    PROMPT_ORDER,
    all_sam_rows,
    class_distribution_rows,
    diagnostics_text,
    final_detection_rows,
    summarize_values,
)


PROMPT_LABEL = {"manual": "Manual", "automatic": "Automatic"}
PROMPT_SHORT = {"manual": "M", "automatic": "A"}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt1(value: float) -> str:
    return f"{value:.1f}"


def fmt_int(value: float) -> str:
    return str(int(round(value)))


def fmt_sd_int(value: float) -> str:
    return f"$\\pm$ {int(round(value))}"


def fmt_sd(value: float, decimals: int = 3, scale: float = 1.0) -> str:
    return f"$\\pm$ {value * scale:.{decimals}f}"


def all_sam_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{llrrrrlll}",
        "\\hline",
        "Prompt & N predictions & TP & FP & FN & Precision & Recall & F1 & Mean IoU \\\\",
        "\\hline",
    ]
    for dataset in PROMPT_ORDER:
        row = rows[dataset]
        lines.append(
            f"{PROMPT_LABEL[dataset]} & {row['n_predictions']} & {row['tp']} & {row['fp']} & {row['fn']} & "
            f"{fmt(row['precision'])} & {fmt(row['recall'])} & {fmt(row['f1'])} & {fmt(row['mean_iou'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def agreement_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Annotators & Images & $N_A$ & $N_B$ & Precision & Recall & F1 & Mean IoU \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['pair']} & {row['images']} & "
            f"{int(row['n_first']):,} & {int(row['n_second']):,} & "
            f"{fmt(row['precision'])} & {fmt(row['recall'])} & "
            f"{fmt(row['f1'])} & {fmt(row['mean_iou'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def class_distribution_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[]",
        "\\centering",
        "\\begin{tabular}{ l c c c c c }",
        "\\hline",
        "Prompt & N & Fish & Incorrect & Head & Multiple\\\\",
        "\\hline",
    ]
    for dataset in PROMPT_ORDER:
        row = rows[dataset]
        lines.append(
            f"{PROMPT_LABEL[dataset]} & {row['n']} & {fmt1(100 * row['fish'])}\\% & "
            f"{fmt1(100 * row['bad'])}\\% & {fmt1(100 * row['head'])}\\% & {fmt1(100 * row['double'])}\\% \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def one_vs_all_classification_table(rows: list[dict[str, Any]]) -> str:
    aggregated = aggregate_one_vs_all_rows(rows)
    lines = [
        "\\begin{table*}[]",
        "\\centering",
        "\\begin{tabular}{ l l c c c c}",
        "\\hline",
        "Prompt & Model & Fish $\\%$ & Incorrect $\\%$ & Head $\\%$ & Multiple $\\%$\\\\",
        "\\hline",
    ]
    for dataset in PROMPT_ORDER:
        if dataset == "automatic":
            lines.append("\\hline")
        lines.append(f"\\multirow{{3}}{{*}}{{{PROMPT_LABEL[dataset]}}}")
        for model in ["KNN", "MLP", "RF"]:
            row = next(item for item in aggregated if item["dataset"] == dataset and item["model"] == model)
            lines.append(
                f" & \\textbf{{{model}}} & "
                f"${row['fish_accuracy_mean'] * 100:.1f} \\pm {row['fish_accuracy_std'] * 100:.1f}$ & "
                f"${row['bad_accuracy_mean'] * 100:.1f} \\pm {row['bad_accuracy_std'] * 100:.1f}$ & "
                f"${row['head_accuracy_mean'] * 100:.1f} \\pm {row['head_accuracy_std'] * 100:.1f}$ & "
                f"${row['double_accuracy_mean'] * 100:.1f} \\pm {row['double_accuracy_std'] * 100:.1f}$ \\\\"
            )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def final_postprocessing_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{1.5pt}",
        "\\renewcommand{\\arraystretch}{0.9}",
        "\\begin{tabular}{l|rrr|r|rrrrrrr}",
        "\\hline",
        " & \\multicolumn{3}{c|}{Mask count} &  & \\multicolumn{7}{c}{Detection at IoU 0.30} \\\\",
        "  \\shortstack{Prompt/\\\\filter}  & GT & \\shortstack{Before\\\\filter} & \\shortstack{After\\\\filter} & \\shortstack{Mask\\\\P\\\\(\\%)}  & TP & FP & FN & P & R & F1 & \\shortstack{Mean\\\\IoU} \\\\",
        "\\hline",
    ]
    for dataset in PROMPT_ORDER:
        if dataset == "automatic":
            lines.append("\\hline")
        for method in ["None", "Ours"]:
            row = rows[f"{dataset}/{method}"]
            filtering = row["mask"]["summary"]
            metric = row["detection"]["summary"]
            gt_counts = summarize_values([float(fold["tp"] + fold["fn"]) for fold in row["detection"]["folds"]])
            lines.append(
                " & ".join(
                    [
                        f"{PROMPT_SHORT[dataset]}/{method}",
                        fmt_int(gt_counts["mean"]),
                        fmt_int(filtering["n_before_filter_mean"]),
                        fmt_int(filtering["n_after_filter_mean"]),
                        fmt1(filtering["precision_mean"] * 100.0),
                        fmt_int(metric["tp"]["mean"]),
                        fmt_int(metric["fp"]["mean"]),
                        fmt_int(metric["fn"]["mean"]),
                        fmt(metric["precision"]["mean"]),
                        fmt(metric["recall"]["mean"]),
                        fmt(metric["f1"]["mean"]),
                        fmt(metric["mean_iou"]["mean"]),
                    ]
                )
                + " \\\\"
            )
            lines.append(
                " & ".join(
                    [
                        "",
                        fmt_sd_int(gt_counts["std"]),
                        fmt_sd_int(filtering["n_before_filter_std"]),
                        fmt_sd_int(filtering["n_after_filter_std"]),
                        fmt_sd(filtering["precision_std"], decimals=1, scale=100.0),
                        fmt_sd_int(metric["tp"]["std"]),
                        fmt_sd_int(metric["fp"]["std"]),
                        fmt_sd_int(metric["fn"]["std"]),
                        fmt_sd(metric["precision"]["std"]),
                        fmt_sd(metric["recall"]["std"]),
                        fmt_sd(metric["f1"]["std"]),
                        fmt_sd(metric["mean_iou"]["std"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\hline", "\\end{tabular}%", "\\end{table}", ""])
    return "\n".join(lines)


def generate_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table_pairwise_annotator_agreement.tex").write_text(agreement_table(pairwise_agreement_rows()))
    (output_dir / "table_detection_no_postprocessing.tex").write_text(all_sam_table(all_sam_rows()))
    (output_dir / "table_mask_class_distribution.tex").write_text(class_distribution_table(class_distribution_rows()))
    (output_dir / "table_one_vs_all_classification.tex").write_text(one_vs_all_classification_table(one_vs_all_rows()))
    (output_dir / "table_final_postprocessing.tex").write_text(final_postprocessing_table(final_detection_rows()))
    (output_dir / "diagnostics.md").write_text(diagnostics_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission result tables from computed result dictionaries.")
    parser.add_argument("--output-dir", type=Path, default=Path("submission_code/generated_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all(args.output_dir)
    print(f"Wrote generated tables to {args.output_dir}")


if __name__ == "__main__":
    main()
