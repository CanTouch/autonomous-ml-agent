"""
Feature engineering helper for the feature-engineering skill.
Usage: ./.venv/bin/python src/feature_engineering.py <path_to_cleaned_dataset> --config <path_to_config.json>
"""
import argparse
import json
from pathlib import Path
from datetime import date
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def engineer_features(df: pd.DataFrame, config: dict):
    log = []
    df = df.copy()
    target = config.get("target_column")

    for col, mapping in config.get("binary_encode", {}).items():
        if col in df.columns and col != target:
            df[col] = df[col].map(mapping)
            log.append(f"Binary-encoded `{col}` using mapping {mapping}.")

    one_hot_cols = [c for c in config.get("one_hot_encode", []) if c in df.columns and c != target]
    if one_hot_cols:
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
        new_cols = set(df.columns) - before_cols
        log.append(f"One-hot encoded {one_hot_cols} → new columns: {sorted(new_cols)}.")

    for col in config.get("frequency_encode", []):
        if col in df.columns and col != target:
            freq_map = df[col].value_counts(normalize=True)
            new_col = f"{col}_freq"
            df[new_col] = df[col].map(freq_map)
            df = df.drop(columns=[col])
            log.append(f"Frequency-encoded `{col}` -> `{new_col}` (replaced raw column with its occurrence rate).")

    return df, log


def write_log(log, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"feature_engineering_log_{dataset_name}_{today}.md"
    lines = [f"# Feature Engineering Log: {dataset_name}", f"Generated: {today}", ""]
    for entry in log:
        lines.append(f"- {entry}")
    out_path.write_text("\n".join(lines))
    print(f"Feature engineering log written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    with open(args.config) as f:
        config = json.load(f)

    featured_df, log = engineer_features(df, config)

    dataset_name = Path(args.dataset_path).stem.replace("_cleaned", "")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = f"data/processed/{dataset_name}_features.csv"
    featured_df.to_csv(out_path, index=False)
    print(f"Feature-engineered dataset written to: {out_path}")

    write_log(log, dataset_name)
