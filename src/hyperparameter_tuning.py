"""
Hyperparameter tuning helper for the hyperparameter-tuning skill.
Usage: ./.venv/bin/python src/hyperparameter_tuning.py <path_to_features_dataset> --target Survived --model random_forest [--scoring f1]
"""
import argparse
from pathlib import Path
from datetime import date
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

PARAM_GRIDS = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "grid": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000),
        "grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
        },
    },
    "adaboost": {
        "model": AdaBoostClassifier(random_state=42),
        "grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
    "xgboost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss"),
        "grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "gamma": [0, 0.1],
            "reg_lambda": [0, 1],
        },
    },
    "svm": {
        "model": SVC(random_state=42),
        "grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"],  # linear kernel is prohibitively slow on this machine
            "gamma": ["scale", "auto"],
        },
    },
}


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def tune(df: pd.DataFrame, target: str, model_key: str, scoring: str = "f1", folds: int = 10):
    X = df.drop(columns=[target])
    y = df[target]

    if y.isnull().any():
        n_missing = y.isnull().sum()
        raise ValueError(
            f"Target column '{target}' has {n_missing} missing value(s). "
            f"Clean the target before tuning."
        )
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Feature columns {bad_cols} still contain missing values.")

    spec = PARAM_GRIDS[model_key]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # SVM forks workers that are slow to initialise in this environment; run single-threaded.
    n_jobs = 1 if model_key == "svm" else -1
    grid_search = GridSearchCV(
        estimator=spec["model"],
        param_grid=spec["grid"],
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    top5 = results_df.sort_values("rank_test_score").head(5)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ]

    return grid_search.best_params_, grid_search.best_score_, top5


def write_report(model_key, best_params, best_score, top5, dataset_name, scoring, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"tuning_{model_key}_{dataset_name}_{today}.md"

    lines = [f"# Hyperparameter Tuning Report: {model_key} on {dataset_name}", f"Generated: {today}", ""]
    lines.append(f"**Scoring metric optimized**: {scoring}")
    lines.append(f"**Best score**: {best_score:.4f}")
    lines.append(f"**Best parameters**: `{best_params}`")
    lines.append("")
    lines.append("## Top 5 parameter combinations")
    lines.append("| Rank | Params | Mean Score | Std |")
    lines.append("|---|---|---|---|")
    for _, row in top5.iterrows():
        lines.append(f"| {int(row['rank_test_score'])} | `{row['params']}` | {row['mean_test_score']:.4f} | {row['std_test_score']:.4f} |")
    lines.append("")
    lines.append("Note: compare the spread between rank 1 and rank 2-5. A narrow gap "
                  "means the 'best' result may not be meaningfully better than the "
                  "runner-ups — treat it as one reasonable choice, not a guaranteed optimum.")

    out_path.write_text("\n".join(lines))
    print(f"Tuning report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True, choices=list(PARAM_GRIDS.keys()))
    parser.add_argument("--scoring", default="f1")
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    best_params, best_score, top5 = tune(df, args.target, args.model, args.scoring, args.folds)

    dataset_name = (
        Path(args.dataset_path).stem
        .replace("_features", "")
        .replace("_selected", "")
        .replace("_balanced", "")
        .replace("_cleaned", "")
    )
    write_report(args.model, best_params, best_score, top5, dataset_name, args.scoring)
