"""
Class balancing helper for the class-balancing skill.
Usage:
  ./.venv/bin/python src/class_balancing.py data/processed/titanic_selected.csv \
      --target Survived --strategy smote
"""
import argparse
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    total = len(y)
    return {str(cls): {"count": int(cnt), "pct": round(100 * cnt / total, 2)}
            for cls, cnt in counts.items()}


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install it with: ./.venv/bin/pip install imbalanced-learn"
        )
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def apply_undersample(X_train: pd.DataFrame, y_train: pd.Series):
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for undersampling. "
            "Install it with: ./.venv/bin/pip install imbalanced-learn"
        )
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def compute_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    n_classes = len(classes)
    total = len(y)
    weights = {int(cls): round(float(total / (n_classes * (y == cls).sum())), 4) for cls in classes}
    return weights


def write_report(
    dataset_name: str,
    strategy: str,
    dist_before: dict,
    dist_after: dict | None,
    train_shape: tuple,
    test_shape: tuple,
    class_weights: dict | None,
    output_dir: str = "reports",
):
    Path(output_dir).mkdir(exist_ok=True)
    today = date.today().isoformat()
    out_path = Path(output_dir) / f"class_balancing_{dataset_name}_{today}.md"

    lines = [
        f"# Class Balancing Report: {dataset_name}",
        f"Generated: {today}",
        f"Strategy: `{strategy}`",
        "",
        "## Class Distribution — Before Balancing (full dataset)",
        "| Class | Count | Percentage |",
        "|---|---|---|",
    ]
    for cls, info in dist_before.items():
        lines.append(f"| {cls} | {info['count']} | {info['pct']}% |")
    lines.append("")

    if strategy == "class_weight":
        lines += [
            "## Class Weights (pass to model)",
            "No resampling applied. Use the following weight dict in your model:",
            "",
            f"```python",
            f"class_weight = {class_weights}",
            f"```",
            "",
            "Pass as `class_weight` parameter to `RandomForestClassifier`, "
            "`LogisticRegression`, etc.",
        ]
    else:
        lines += [
            "## Class Distribution — After Balancing (training set only)",
            "| Class | Count | Percentage |",
            "|---|---|---|",
        ]
        for cls, info in dist_after.items():
            lines.append(f"| {cls} | {info['count']} | {info['pct']}% |")
        lines.append("")
        lines += [
            "## Output Sizes",
            f"- Balanced training set: {train_shape[0]} rows × {train_shape[1]} columns",
            f"- Test set (untouched): {test_shape[0]} rows × {test_shape[1]} columns",
            "",
            "> ⚠️ The test set was never resampled. "
            "Evaluate your model on `*_balanced_test.csv` only.",
        ]

    out_path.write_text("\n".join(lines))
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--strategy", choices=["smote", "undersample", "class_weight"],
                        default="smote")
    args = parser.parse_args()

    df = load_data(args.dataset_path)
    dataset_name = Path(args.dataset_path).stem.replace("_selected", "").replace("_features", "")

    X = df.drop(columns=[args.target]).astype(float)
    y = df[args.target]

    dist_before = class_distribution(y)
    print("Class distribution before:")
    for cls, info in dist_before.items():
        print(f"  Class {cls}: {info['count']} ({info['pct']}%)")

    if args.strategy == "class_weight":
        weights = compute_class_weights(y)
        print(f"\nClass weights: {weights}")
        write_report(dataset_name, args.strategy, dist_before,
                     None, (0, 0), (0, 0), weights)
    else:
        # Stratified split — test set stays untouched
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if args.strategy == "smote":
            print("\nApplying SMOTE to training set...")
            X_res, y_res = apply_smote(X_train, y_train)
        else:
            print("\nApplying random undersampling to training set...")
            X_res, y_res = apply_undersample(X_train, y_train)

        dist_after = class_distribution(y_res)
        print("Class distribution after (training set only):")
        for cls, info in dist_after.items():
            print(f"  Class {cls}: {info['count']} ({info['pct']}%)")

        # Save balanced train set
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        train_out = f"data/processed/{dataset_name}_balanced.csv"
        test_out = f"data/processed/{dataset_name}_balanced_test.csv"

        train_df = X_res.copy()
        train_df[args.target] = y_res.values
        train_df.to_csv(train_out, index=False)
        print(f"Balanced training set written to: {train_out}")

        test_df = X_test.copy()
        test_df[args.target] = y_test.values
        test_df.to_csv(test_out, index=False)
        print(f"Test set (untouched) written to: {test_out}")

        write_report(dataset_name, args.strategy, dist_before, dist_after,
                     train_df.shape, test_df.shape, None)
