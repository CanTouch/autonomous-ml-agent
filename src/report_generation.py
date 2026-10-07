"""
Report generation helper for the report-generation skill.
Usage: ./.venv/bin/python src/report_generation.py <dataset_name>
"""
import argparse
import glob
import re
from pathlib import Path
from datetime import date


STAGE_ORDER = [
    ("profile", "Data Profile"),
    ("cleaning_log", "Data Cleaning"),
    ("feature_engineering_log", "Feature Engineering"),
    ("feature_selection", "Feature Selection"),
    ("class_balancing", "Class Balancing"),
    ("pca", "Principal Component Analysis"),
    ("kmeans", "K-Means Clustering"),
    ("hierarchical", "Hierarchical Clustering"),
    ("model_selection_regression", "Model Selection"),
    ("model_selection", "Model Selection"),
    ("tuning_regression", "Hyperparameter Tuning"),
    ("tuning", "Hyperparameter Tuning"),
    ("evaluation_regression", "Final Evaluation"),
    ("evaluation", "Final Evaluation"),
]


def find_latest(pattern: str, reports_dir: str = "reports"):
    matches = sorted(glob.glob(f"{reports_dir}/{pattern}"))
    return matches[-1] if matches else None


def find_stage_files(dataset_name: str, reports_dir: str = "reports"):
    found = []
    seen_stage_labels = set()
    for prefix, label in STAGE_ORDER:
        if label in seen_stage_labels and prefix in ["model_selection", "tuning", "evaluation"]:
            continue
        pattern = f"{prefix}_{dataset_name}_*.md" if prefix != "profile" else f"profile_{dataset_name}_*.md"
        if not find_latest(pattern, reports_dir):
            pattern = f"{prefix}_*_{dataset_name}_*.md"
        path = find_latest(pattern, reports_dir)
        if path:
            found.append((label, path))
            seen_stage_labels.add(label)
    return found
        


def find_plots(dataset_name: str, reports_dir: str = "reports"):
    plots = []
    plot_patterns = [
        f"roc_curve_*_{dataset_name}_*.png",
        f"residuals_*_{dataset_name}_*.png",
        f"pca_scree_{dataset_name}_*.png",
        f"pca_2d_{dataset_name}_*.png",
        f"kmeans_elbow_{dataset_name}_*.png",
        f"hierarchical_dendrogram_{dataset_name}_*.png",
    ]
    for pattern in plot_patterns:
        matches = sorted(glob.glob(f"{reports_dir}/{pattern}"))
        plots.extend(matches)
    return plots


def extract_summary_hints(stage_files: list):
    hints = {"shape": None, "top_issue": None, "model": None}
    for label, path in stage_files:
        text = Path(path).read_text()
        if label == "Data Profile" and hints["shape"] is None:
            m = re.search(r"(\d+) rows × (\d+) columns", text)
            if m:
                hints["shape"] = f"{m.group(1)} rows, {m.group(2)} columns"
            m2 = re.search(r"⚠️ high missingness", text)
            if m2:
                line_match = re.search(r"\|\s*(\w+)\s*\|[^\n]*⚠️ high missingness", text)
                if line_match:
                    hints["top_issue"] = f"`{line_match.group(1)}` has high missingness"
        if label == "Final Evaluation" and hints["model"] is None:
            m = re.search(r"Report: (\w+) on", text)
            if m:
                hints["model"] = m.group(1)
    return hints


def build_report(dataset_name: str, output_dir: str = "reports"):
    stage_files = find_stage_files(dataset_name, output_dir)
    plots = find_plots(dataset_name, output_dir)

    if not stage_files:
        raise FileNotFoundError(f"No reports found for dataset '{dataset_name}' in {output_dir}/")

    hints = extract_summary_hints(stage_files)
    today = date.today().isoformat()

    lines = [f"# Final Report: {dataset_name}", f"Generated: {today}", ""]
    lines.append("## Summary")
    summary_parts = []
    if hints["shape"]:
        summary_parts.append(f"Dataset: {hints['shape']}.")
    if hints["top_issue"]:
        summary_parts.append(f"Main data quality issue: {hints['top_issue']}.")
    if hints["model"]:
        summary_parts.append(f"Final model evaluated: {hints['model']}.")
    summary_parts.append(f"This report combines {len(stage_files)} pipeline stage(s): "
                          + ", ".join(label for label, _ in stage_files) + ".")
    lines.append(" ".join(summary_parts))
    lines.append("")

    if plots:
        lines.append("## Plots")
        for p in plots:
            lines.append(f"- {p}")
        lines.append("")

    lines.append("---")
    lines.append("")

    for label, path in stage_files:
        lines.append(f"## Stage: {label}")
        lines.append(f"*Source: `{path}`*")
        lines.append("")
        lines.append(Path(path).read_text())
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(output_dir) / f"FINAL_REPORT_{dataset_name}_{today}.md"
    out_path.write_text("\n".join(lines))
    print(f"Final report written to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    args = parser.parse_args()

    build_report(args.dataset_name)
