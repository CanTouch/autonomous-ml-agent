"""
Model selection helper for the model-selection skill.
Usage: ./.venv/bin/python src/model_selection.py <path_to_features_dataset> --target Survived [--folds 10]
"""
import argparse
from pathlib import Path
from datetime import date
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


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

    if y.isnull().any():
        n_missing = y.isnull().sum()
        raise ValueError(
            f"Target column '{target}' has {n_missing} missing value(s). "
            f"Model selection requires a complete target column — "
            f"clean or drop these rows before running this skill."
        )
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"Feature columns {bad_cols} still contain missing values. "
            f"Run data-cleaning fully before model-selection."
        )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "SVM": SVC(random_state=42),
    }

    scoring = ["accuracy", "precision", "recall", "f1"]
    results = {}

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_min": scores["test_accuracy"].min(),
            "accuracy_max": scores["test_accuracy"].max(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
        }

    return results


def write_report(results: dict, dataset_name: str, output_dir: str = "reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"model_selection_{dataset_name}_{today}.md"

    lines = [f"# Model Selection Report: {dataset_name}", f"Generated: {today}", ""]
    lines.append("| Model | Accuracy (mean) | Accuracy (min) | Accuracy (max) | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['accuracy_mean']:.4f} | {r['accuracy_min']:.4f} | "
            f"{r['accuracy_max']:.4f} | {r['precision_mean']:.4f} | "
            f"{r['recall_mean']:.4f} | {r['f1_mean']:.4f} |"
        )
    lines.append("")
    lines.append("No model has been auto-selected. Review the metrics above — "
                  "consider both mean performance and the accuracy range (min/max) "
                  "before choosing which model to proceed with for tuning.")

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
    write_report(results, dataset_name)
