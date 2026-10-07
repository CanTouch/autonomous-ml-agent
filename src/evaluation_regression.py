"""
Regression evaluation helper.
Usage: ./.venv/bin/python src/evaluation_regression.py <path> --target Price --model random_forest \
    --params '{"max_depth": null, "min_samples_split": 2, "n_estimators": 400}'
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "linear_regression": LinearRegression,
    "adaboost": AdaBoostRegressor,
    "xgboost": XGBRegressor,
    "svm": SVR,
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
        X, y, test_size=test_size, random_state=42
    )

    model_cls = MODEL_CLASSES[model_key]
    clean_params = {k: v for k, v in params.items() if v is not None}
    if model_key == "svm":
        clean_params.setdefault("max_iter", 2000)
    if model_key in ("linear_regression", "svm"):
        model = model_cls(**clean_params)
    else:
        model = model_cls(random_state=42, **clean_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
    }

    residuals = y_test - y_pred

    return metrics, y_test, y_pred, residuals


def write_residual_plot(y_test, y_pred, residuals, model_key, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"residuals_{model_key}_{dataset_name}_{today}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Residual plot written to: {out_path}")
    return str(out_path)


def write_report(metrics, model_key, dataset_name, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"evaluation_regression_{model_key}_{dataset_name}_{today}.md"

    lines = [f"# Regression Evaluation Report: {model_key} on {dataset_name}", f"Generated: {today}", ""]
    lines.append("## Test Set Metrics (held out, never trained on)")
    lines.append(f"- R²: {metrics['r2']:.4f}")
    lines.append(f"- MAE: {metrics['mae']:,.2f}")
    lines.append(f"- RMSE: {metrics['rmse']:,.2f}")
    lines.append("")
    lines.append("Compare these to the cross-validated numbers from model-selection-regression "
                  "and hyperparameter-tuning-regression. A large gap can indicate overfitting. "
                  "Check the residual plot: a funnel shape (residuals growing with predicted value) "
                  "suggests the model is less reliable at the high end of the price range.")

    out_path.write_text("\n".join(lines))
    print(f"Evaluation report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--params", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    try:
        params = json.loads(args.params)
    except Exception:
        params = ast.literal_eval(args.params)

    metrics, y_test, y_pred, residuals = evaluate(df, args.target, args.model, params, args.test_size)

    dataset_name = (
        Path(args.dataset_path).stem
        .replace("_features", "")
        .replace("_selected", "")
        .replace("_balanced", "")
        .replace("_cleaned", "")
    )
    write_residual_plot(y_test, y_pred, residuals, args.model, dataset_name)
    write_report(metrics, args.model, dataset_name)
