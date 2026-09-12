"""
Data profiling helper for the data-profiling skill.
Usage: ./.venv/bin/python src/data_profiling.py <path_to_dataset> [--target COLUMN_NAME]
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import date


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".csv"]:
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    report = {}

    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    report["duplicates"] = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include="number").columns
    report["numeric_stats"] = {}
    for col in numeric_cols:
        desc = df[col].describe()
        q75 = desc["75%"]
        flag = bool(q75 > 0 and desc["max"] > 10 * q75)
        report["numeric_stats"][col] = {
            "min": float(desc["min"]),
            "max": float(desc["max"]),
            "mean": float(desc["mean"]),
            "median": float(df[col].median()),
            "std": float(desc["std"]),
            "outlier_flag": flag,
        }

    cat_cols = df.select_dtypes(include=["str", "category"]).columns
    report["categorical_stats"] = {}
    for col in cat_cols:
        nunique = df[col].nunique()
        high_cardinality = bool(nunique > 0.5 * len(df))
        report["categorical_stats"][col] = {
            "unique_count": int(nunique),
            "high_cardinality_flag": high_cardinality,
        }

    if target and target in df.columns:
        suspicious_keywords = ["outcome", "result", "final", "post", "after"]
        leakage_flags = [
            col for col in df.columns
            if col != target
            and any(kw in col.lower() for kw in suspicious_keywords)
        ]
        report["leakage_candidates"] = leakage_flags
        
        target_counts = df[target].value_counts()
        target_pct = (target_counts / len(df) * 100).round(2)
        report["target_balance"] = {
            str(val): {"count": int(target_counts[val]), "pct": float(target_pct[val])}
            for val in target_counts.index
        }

    return report


def write_markdown_report(report: dict, dataset_name: str, output_dir: str = "reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"profile_{dataset_name}_{today}.md"

    lines = [f"# Data Profile: {dataset_name}", f"Generated: {today}", ""]
    lines.append(f"## Shape\n{report['shape'][0]} rows × {report['shape'][1]} columns\n")

    lines.append("## Dtypes")
    lines.append("| Column | Dtype |")
    lines.append("|---|---|")
    for col, dtype in report["dtypes"].items():
        lines.append(f"| {col} | {dtype} |")
    lines.append("")

    lines.append("## Missing Values")
    if report["missing"]:
        lines.append("| Column | Missing Count | Missing % |")
        lines.append("|---|---|---|")
        for col, info in sorted(report["missing"].items(), key=lambda x: -x[1]["pct"]):
            flag = " ⚠️ high missingness" if info["pct"] > 40 else ""
            lines.append(f"| {col} | {info['count']} | {info['pct']}%{flag} |")
    else:
        lines.append("No missing values found.")
    lines.append("")

    lines.append(f"## Duplicate Rows\n{report['duplicates']} duplicate rows found.\n")

    lines.append("## Numeric Column Stats")
    if report["numeric_stats"]:
        lines.append("| Column | Min | Max | Mean | Median | Std | Flag |")
        lines.append("|---|---|---|---|---|---|---|")
        for col, s in report["numeric_stats"].items():
            flag = "⚠️ possible outliers" if s["outlier_flag"] else ""
            lines.append(f"| {col} | {s['min']:.2f} | {s['max']:.2f} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | {flag} |")
    else:
        lines.append("No numeric columns found.")
    lines.append("")

    lines.append("## Categorical Column Stats")
    if report["categorical_stats"]:
        lines.append("| Column | Unique Values | Flag |")
        lines.append("|---|---|---|")
        for col, s in report["categorical_stats"].items():
            flag = "⚠️ high cardinality" if s["high_cardinality_flag"] else ""
            lines.append(f"| {col} | {s['unique_count']} | {flag} |")
    else:
        lines.append("No categorical columns found.")
    lines.append("")

    if "leakage_candidates" in report:
        lines.append("## Potential Target Leakage Candidates")
        if report["leakage_candidates"]:
            for col in report["leakage_candidates"]:
                lines.append(f"- ⚠️ `{col}` — name suggests it may be computed after the target is known. Investigate before using as a feature.")
        else:
            lines.append("No obviously suspicious column names found.")
        lines.append("")
        
    if "target_balance" in report:
        lines.append("## Target Balance")
        lines.append("| Value | Count | Percentage |")
        lines.append("|---|---|---|")
        for val, info in report["target_balance"].items():
            lines.append(f"| {val} | {info['count']} | {info['pct']}% |")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Report written to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem
    report = profile(df, target=args.target)
    write_markdown_report(report, dataset_name)
