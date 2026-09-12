"""
Data cleaning helper for the data-cleaning skill.
Usage: ./.venv/bin/python src/data_cleaning.py <path_to_dataset> --config <path_to_cleaning_config.json>
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


def clean(df: pd.DataFrame, config: dict):
    log = []
    df = df.copy()

    for col, flag_name in config.get("add_flags", {}).items():
        if col in df.columns:
            df[flag_name] = df[col].notna().astype(int)
            log.append(f"Added binary flag `{flag_name}` from presence/absence of `{col}`.")

    for col, spec in config.get("impute", {}).items():
        if col not in df.columns:
            continue
        strategy = spec.get("strategy")
        before_missing = df[col].isnull().sum()
        if strategy == "median":
            value = df[col].median()
            df[col] = df[col].fillna(value)
            log.append(f"Imputed `{col}` ({before_missing} missing) with median = {value:.2f}.")
        elif strategy == "mean":
            value = df[col].mean()
            df[col] = df[col].fillna(value)
            log.append(f"Imputed `{col}` ({before_missing} missing) with mean = {value:.2f}.")
        elif strategy == "mode":
            value = df[col].mode()[0]
            df[col] = df[col].fillna(value)
            log.append(f"Imputed `{col}` ({before_missing} missing) with mode = '{value}'.")
        elif strategy == "unknown":
            df[col] = df[col].fillna("Unknown")
            log.append(f"Filled `{col}` ({before_missing} missing) with 'Unknown' category.")

    if config.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            log.append(f"Dropped {removed} exact duplicate rows.")

    drop_cols = [c for c in config.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.append(f"Dropped columns: {', '.join(drop_cols)}.")

    return df, log


def write_log(log, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"cleaning_log_{dataset_name}_{today}.md"
    lines = [f"# Cleaning Log: {dataset_name}", f"Generated: {today}", ""]
    for entry in log:
        lines.append(f"- {entry}")
    out_path.write_text("\n".join(lines))
    print(f"Cleaning log written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    with open(args.config) as f:
        config = json.load(f)

    cleaned_df, log = clean(df, config)

    dataset_name = Path(args.dataset_path).stem
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = f"data/processed/{dataset_name}_cleaned.csv"
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned dataset written to: {out_path}")

    write_log(log, dataset_name)
