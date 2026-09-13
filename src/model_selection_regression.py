"""
Regression model selection helper.
Usage: ./.venv/bin/python src/model_selection_regression.py <path_to_features_dataset> --target Price [--folds 10]
"""
import argparse
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def run_model_selection(df: pd.DataFrame, target: str, folds: int = 10):
    X = df.drop(columns=[target])
    y = df[target]

    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    results = {}

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "r2_mean": scores["test_r2"].mean(),
            "mae_mean": -scores["test_mae"].mean(),
            "rmse_mean": -scores["test_rmse"].mean(),
        }

    return results


def write_report(results: dict, dataset_name: str, target: str, output_dir: str = "reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"model_selection_regression_{dataset_name}_{today}.md"

    lines = [f"# Regression Model Selection Report: {dataset_name}", f"Generated: {today}", ""]
    lines.append(f"Target: `{target}`")
    lines.append("")
    lines.append("| Model | R² (mean) | MAE (mean) | RMSE (mean) |")
    lines.append("|---|---|---|---|")
    for name, r in results.items():
        lines.append(f"| {name} | {r['r2_mean']:.4f} | {r['mae_mean']:,.2f} | {r['rmse_mean']:,.2f} |")
    lines.append("")
    lines.append("No model has been auto-selected. Higher R² is better; lower MAE/RMSE "
                  "is better. A large gap between MAE and RMSE suggests the model makes "
                  "a few very large errors alongside mostly small ones.")

    out_path.write_text("\n".join(lines))
    print(f"Model selection report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    results = run_model_selection(df, args.target, args.folds)

    dataset_name = Path(args.dataset_path).stem.replace("_features", "")
    write_report(results, dataset_name, args.target)
