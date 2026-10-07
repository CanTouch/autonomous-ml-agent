"""
Evaluation helper for the evaluation skill.
Usage: ./.venv/bin/python src/evaluation.py <path_to_features_dataset> --target Survived --model random_forest \
    --params '{"max_depth": null, "min_samples_leaf": 1, "min_samples_split": 5, "n_estimators": 400}'
"""
import argparse
import json
import ast
from pathlib import Path
from datetime import date
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)

MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "adaboost": AdaBoostClassifier,
    "xgboost": XGBClassifier,
    "svm": SVC,
}


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def evaluate(df: pd.DataFrame, target: str, model_key: str, params: dict, test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]

    if y.isnull().any():
        n_missing = y.isnull().sum()
        raise ValueError(
            f"Target column '{target}' has {n_missing} missing value(s). "
            f"Clean the target before evaluation."
        )
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Feature columns {bad_cols} still contain missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    model_cls = MODEL_CLASSES[model_key]
    clean_params = {k: v for k, v in params.items() if v is not None}
    if model_key == "logistic_regression":
        model = model_cls(max_iter=1000, **clean_params)
    elif model_key == "xgboost":
        model = model_cls(random_state=42, eval_metric="logloss", **clean_params)
    else:
        model = model_cls(random_state=42, **clean_params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": auc,
    }

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "metrics": metrics,
        "fpr": fpr, "tpr": tpr,
    }


def write_roc_plot(fpr, tpr, auc, model_key, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"roc_curve_{model_key}_{dataset_name}_{today}.png"

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_key} on {dataset_name}")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"ROC curve plot written to: {out_path}")
    return str(out_path)


def write_report(result, model_key, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"evaluation_{model_key}_{dataset_name}_{today}.md"

    m = result["metrics"]
    lines = [f"# Evaluation Report: {model_key} on {dataset_name}", f"Generated: {today}", ""]
    lines.append("## Confusion Matrix (test set — held out, never trained on)")
    lines.append("|  | Predicted: Positive | Predicted: Negative |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual: Positive** | TP = {result['tp']} | FN = {result['fn']} |")
    lines.append(f"| **Actual: Negative** | FP = {result['fp']} | TN = {result['tn']} |")
    lines.append("")
    lines.append("## Test Set Metrics")
    lines.append(f"- Accuracy: {m['accuracy']:.4f}")
    lines.append(f"- Precision: {m['precision']:.4f}")
    lines.append(f"- Recall: {m['recall']:.4f}")
    lines.append(f"- F1: {m['f1']:.4f}")
    lines.append(f"- AUC: {m['auc']:.4f}")
    lines.append("")
    lines.append("Compare these to the cross-validated numbers from model-selection "
                  "and hyperparameter-tuning reports. A large gap (test performance "
                  "notably worse) can indicate overfitting during tuning.")

    out_path.write_text("\n".join(lines))
    print(f"Evaluation report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--params", required=True, help="JSON string of hyperparameters")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    try:
        params = json.loads(args.params)
    except Exception:
        params = ast.literal_eval(args.params)

    result = evaluate(df, args.target, args.model, params, args.test_size)

    dataset_name = (
        Path(args.dataset_path).stem
        .replace("_features", "")
        .replace("_selected", "")
        .replace("_balanced", "")
        .replace("_cleaned", "")
    )
    write_roc_plot(result["fpr"], result["tpr"], result["metrics"]["auc"], args.model, dataset_name)
    write_report(result, args.model, dataset_name)
