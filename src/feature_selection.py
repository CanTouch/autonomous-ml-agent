"""
Feature selection helper for the feature-selection skill.
Usage:
  ./.venv/bin/python src/feature_selection.py data/processed/titanic_features.csv \
      --target Survived --task classification
"""
import argparse
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def variance_filter(df: pd.DataFrame, target: str, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    feature_cols = [c for c in df.columns if c != target]
    variances = df[feature_cols].var()
    drop = variances[variances < threshold].index.tolist()
    log = [{"column": c, "variance": float(variances[c]), "action": "dropped — below variance threshold"}
           for c in drop]
    kept = [{"column": c, "variance": float(variances[c]), "action": "kept"} for c in feature_cols if c not in drop]
    return df.drop(columns=drop), log + kept


def correlation_filter(df: pd.DataFrame, target: str, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    feature_cols = [c for c in df.columns if c != target]
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [(col, row, upper.loc[row, col])
             for col in upper.columns for row in upper.index
             if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > threshold]

    to_drop = set()
    log = []
    for col_a, col_b, corr_val in pairs:
        if col_a in to_drop or col_b in to_drop:
            continue
        # Drop the one with lower mean absolute correlation to all others
        mean_a = corr[col_a].drop(index=col_a).mean()
        mean_b = corr[col_b].drop(index=col_b).mean()
        dropped = col_a if mean_a <= mean_b else col_b
        kept = col_b if dropped == col_a else col_a
        to_drop.add(dropped)
        log.append({
            "pair": f"{col_a} ↔ {col_b}",
            "correlation": round(float(corr_val), 4),
            "dropped": dropped,
            "kept": kept,
        })

    return df.drop(columns=list(to_drop)), log


def permutation_importance_filter(
    df: pd.DataFrame, target: str, task: str, threshold: float
) -> tuple[pd.DataFrame, list[dict]]:
    feature_cols = [c for c in df.columns if c != target]
    # Cast bools to int for sklearn
    X = df[feature_cols].astype(float)
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        "column": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    to_drop = importance_df[importance_df["importance_mean"] <= threshold]["column"].tolist()
    log = importance_df.to_dict(orient="records")
    for entry in log:
        entry["action"] = "dropped — importance ≤ 0" if entry["column"] in to_drop else "kept"

    return df.drop(columns=to_drop), log


def write_report(
    dataset_name: str,
    original_cols: list[str],
    final_cols: list[str],
    variance_log: list[dict],
    corr_log: list[dict],
    perm_log: list[dict],
    output_dir: str = "reports",
):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"feature_selection_{dataset_name}_{today}.md"

    lines = [
        f"# Feature Selection Report: {dataset_name}",
        f"Generated: {today}",
        "",
        f"**Original feature count**: {len(original_cols)}",
        f"**Final feature count**: {len(final_cols)}",
        f"**Features removed**: {len(original_cols) - len(final_cols)}",
        "",
    ]

    # Variance
    lines += ["## Stage 1: Variance Threshold",
              "| Column | Variance | Action |",
              "|---|---|---|"]
    for entry in variance_log:
        lines.append(f"| {entry['column']} | {entry['variance']:.6f} | {entry['action']} |")
    lines.append("")

    # Correlation
    lines += ["## Stage 2: Correlation Drop"]
    if corr_log:
        lines += ["| Pair | Correlation | Dropped | Kept |",
                  "|---|---|---|---|"]
        for entry in corr_log:
            lines.append(f"| {entry['pair']} | {entry['correlation']} | {entry['dropped']} | {entry['kept']} |")
    else:
        lines.append("No highly-correlated pairs found above threshold.")
    lines.append("")

    # Permutation importance
    lines += ["## Stage 3: Permutation Importance",
              "| Column | Mean Importance | Std | Action |",
              "|---|---|---|---|"]
    for entry in perm_log:
        lines.append(
            f"| {entry['column']} | {entry['importance_mean']:.6f} | {entry['importance_std']:.6f} | {entry['action']} |"
        )
    lines.append("")

    lines += ["## Final Feature Set",
              ", ".join(f"`{c}`" for c in final_cols)]

    out_path.write_text("\n".join(lines))
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--variance-threshold", type=float, default=0.01)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--importance-threshold", type=float, default=0.0)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem.replace("_features", "")
    original_cols = [c for c in df.columns if c != args.target]

    print(f"[1/3] Variance filter (threshold={args.variance_threshold})...")
    df, variance_log = variance_filter(df, args.target, args.variance_threshold)

    print(f"[2/3] Correlation filter (threshold={args.corr_threshold})...")
    df, corr_log = correlation_filter(df, args.target, args.corr_threshold)

    print(f"[3/3] Permutation importance filter (threshold={args.importance_threshold})...")
    df, perm_log = permutation_importance_filter(df, args.target, args.task, args.importance_threshold)

    final_cols = [c for c in df.columns if c != args.target]

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = f"data/processed/{dataset_name}_selected.csv"
    df.to_csv(out_path, index=False)
    print(f"Selected dataset written to: {out_path}")

    write_report(dataset_name, original_cols, final_cols, variance_log, corr_log, perm_log)
    print(f"Features kept: {len(final_cols)}/{len(original_cols)}")
