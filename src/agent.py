"""
ML Agent: Autonomous and Interactive ML Assistant.
Provides automated data inspection, task inference, configuration generation,
and end-to-end pipeline execution for tabular datasets.

Usage:
  Interactive wizard:
    ./.venv/bin/python src/agent.py

  CLI automation:
    ./.venv/bin/python src/agent.py --dataset data/raw/heart_disease.csv --target target --model xgboost
    ./.venv/bin/python src/agent.py --dataset data/raw/wine.csv --task unsupervised
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np


class MLAgent:
    """Core ML Agent assistant for automated tabular machine learning."""

    def __init__(self, raw_data_dir: str = "data/raw", config_dir: str = "."):
        self.raw_data_dir = Path(raw_data_dir)
        self.config_dir = Path(config_dir)

    def list_available_datasets(self) -> list[Path]:
        """List all CSV files in the raw data directory."""
        if not self.raw_data_dir.exists():
            return []
        return sorted(list(self.raw_data_dir.glob("*.csv")))

    def inspect_dataset(self, csv_path: Path) -> Dict[str, Any]:
        """Inspect a dataset and extract high-level metadata."""
        df = pd.read_csv(csv_path)
        missing_count = int(df.isnull().sum().sum())
        missing_cols = df.columns[df.isnull().any()].tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Potential target candidates (columns named target, survived, price, etc. or last column)
        candidate_targets = []
        common_target_names = ["target", "survived", "price", "exited", "class", "label", "outcome", "medhouseval"]
        for col in df.columns:
            if col.lower() in common_target_names:
                candidate_targets.append(col)
        if not candidate_targets and len(df.columns) > 0:
            candidate_targets.append(df.columns[-1])

        return {
            "path": csv_path,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "missing_total": missing_count,
            "missing_columns": missing_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "candidate_targets": candidate_targets,
        }

    def infer_task_type(self, df: pd.DataFrame, target: Optional[str] = None) -> str:
        """Infer whether a dataset / target represents classification, regression, or unsupervised."""
        if target is None or target not in df.columns:
            return "unsupervised"

        y = df[target].dropna()
        n_unique = y.nunique()

        # If target is non-numeric, it's classification
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"

        # If integer or float with small unique values (< 15 or <= 5% of dataset size), classify
        if n_unique <= 10 or (n_unique <= 20 and n_unique / max(len(y), 1) < 0.05):
            return "classification"

        return "regression"

    def auto_generate_configs(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        dataset_stem: str = "dataset"
    ) -> Tuple[Path, Path]:
        """
        Auto-generates clean, sensible cleaning and feature engineering configs
        for any unseen dataset.
        """
        drop_cols = []
        impute_spec = {}
        id_like_terms = ["id", "rownumber", "customerid", "passengerid"]

        for col in df.columns:
            if col == target:
                continue
            col_lower = col.lower()
            # Drop ID columns or ultra-high cardinality string columns
            if any(term in col_lower for term in id_like_terms):
                drop_cols.append(col)
            elif not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 100:
                drop_cols.append(col)

            # Imputation for columns with nulls
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    impute_spec[col] = {"strategy": "median"}
                else:
                    impute_spec[col] = {"strategy": "mode"}

        cleaning_config = {
            "drop_columns": drop_cols,
            "impute": impute_spec,
            "drop_duplicates": True,
        }

        # Feature engineering: one-hot encode remaining low-cardinality categoricals
        one_hot_cols = []
        for col in df.columns:
            if col == target or col in drop_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]) and 2 <= df[col].nunique() <= 20:
                one_hot_cols.append(col)

        feature_config: Dict[str, Any] = {
            "one_hot_encode": one_hot_cols,
        }
        if target:
            feature_config["target_column"] = target

        clean_path = self.config_dir / f"{dataset_stem}_cleaning_config.json"
        feat_path = self.config_dir / f"{dataset_stem}_feature_engineering_config.json"

        clean_path.write_text(json.dumps(cleaning_config, indent=2))
        feat_path.write_text(json.dumps(feature_config, indent=2))

        return clean_path, feat_path

    def run_pipeline(
        self,
        dataset_path: Path,
        target: Optional[str],
        task: str,
        model: str = "xgboost",
        cleaning_config: Optional[Path] = None,
        feature_config: Optional[Path] = None,
        feature_selection: bool = False,
        class_balancing: Optional[str] = None,
    ) -> int:
        """Invokes workflows/run_pipeline.sh with the configured parameters."""
        cmd = [
            "./workflows/run_pipeline.sh",
            "--dataset", str(dataset_path),
            "--task", task,
        ]

        if target:
            cmd.extend(["--target", target])

        if cleaning_config:
            cmd.extend(["--cleaning-config", str(cleaning_config)])

        if feature_config:
            cmd.extend(["--feature-config", str(feature_config)])

        if task != "unsupervised":
            cmd.extend(["--model", model])

        if feature_selection:
            cmd.append("--feature-selection")

        if class_balancing:
            cmd.extend(["--class-balancing", class_balancing])

        print("\n" + "=" * 60)
        print(" ML Agent Orchestrator Launching Pipeline")
        print(f" Command: {' '.join(cmd)}")
        print("=" * 60 + "\n")

        res = subprocess.run(cmd)
        return res.returncode


def run_interactive_wizard():
    """Runs a friendly terminal wizard prompting user for dataset & task choices."""
    agent = MLAgent()
    datasets = agent.list_available_datasets()

    print("=" * 60)
    print(" Welcome to the ML Agent Interactive Assistant!")
    print("=" * 60)

    if not datasets:
        print("No datasets found in data/raw/. Please place a CSV file there first.")
        sys.exit(1)

    print("\nAvailable datasets in data/raw/:")
    for idx, dpath in enumerate(datasets, 1):
        print(f"  [{idx}] {dpath.name}")

    choice_str = input(f"\nSelect a dataset [1-{len(datasets)}] (default 1): ").strip()
    idx = int(choice_str) - 1 if choice_str.isdigit() and 1 <= int(choice_str) <= len(datasets) else 0
    selected_dataset = datasets[idx]

    info = agent.inspect_dataset(selected_dataset)
    print(f"\nInspecting '{selected_dataset.name}':")
    print(f"  Rows: {info['rows']}, Columns: {info['columns']}")
    print(f"  Missing values: {info['missing_total']} across {len(info['missing_columns'])} columns")
    print(f"  Candidate targets: {', '.join(info['candidate_targets']) if info['candidate_targets'] else 'None'}")

    df = pd.read_csv(selected_dataset)

    # Prompt target
    default_target = info['candidate_targets'][0] if info['candidate_targets'] else None
    target_prompt = f"\nEnter target column name (leave empty for unsupervised, default: {default_target}): "
    target_input = input(target_prompt).strip()
    target = target_input if target_input else default_target

    # Infer or prompt task
    inferred_task = agent.infer_task_type(df, target)
    print(f"\nInferred Task: {inferred_task.upper()}")
    task_input = input(f"Select task [classification/regression/unsupervised] (default: {inferred_task}): ").strip().lower()
    task = task_input if task_input in ["classification", "regression", "unsupervised"] else inferred_task

    model = "xgboost"
    feature_selection = False
    class_balancing = None

    if task != "unsupervised":
        model_input = input("\nSelect model [xgboost/random_forest/svm/adaboost/gradient_boosting] (default: xgboost): ").strip().lower()
        if model_input in ["xgboost", "random_forest", "svm", "adaboost", "gradient_boosting", "logistic_regression", "linear_regression"]:
            model = model_input

        fs_input = input("Apply automated feature selection (variance, collinearity, permutation)? [y/N]: ").strip().lower()
        feature_selection = fs_input in ["y", "yes"]

        if task == "classification":
            cb_input = input("Apply class balancing strategy [none/undersample/smote/class_weight] (default: none): ").strip().lower()
            if cb_input in ["undersample", "smote", "class_weight"]:
                class_balancing = cb_input

    # Check for existing configs or auto-generate
    stem = selected_dataset.stem
    clean_cfg = Path(f"{stem}_cleaning_config.json")
    feat_cfg = Path(f"{stem}_feature_engineering_config.json")

    if not clean_cfg.exists() or not feat_cfg.exists():
        print("\nAuto-generating cleaning and feature engineering configs...")
        clean_cfg, feat_cfg = agent.auto_generate_configs(df, target, stem)
        print(f"  Generated: {clean_cfg}")
        print(f"  Generated: {feat_cfg}")

    returncode = agent.run_pipeline(
        dataset_path=selected_dataset,
        target=target,
        task=task,
        model=model,
        cleaning_config=clean_cfg,
        feature_config=feat_cfg,
        feature_selection=feature_selection,
        class_balancing=class_balancing,
    )

    if returncode == 0:
        print("\n✨ Run successfully completed by ML Agent!")
    else:
        print(f"\n❌ Pipeline exited with status code {returncode}.")


def main():
    parser = argparse.ArgumentParser(description="ML Agent: Autonomous Tabular ML Assistant")
    parser.add_argument("--dataset", type=str, help="Path to input dataset in data/raw/")
    parser.add_argument("--target", type=str, help="Target column name")
    parser.add_argument("--task", choices=["classification", "regression", "unsupervised", "auto"], default="auto")
    parser.add_argument("--model", default="xgboost", help="Model to evaluate (xgboost, random_forest, svm, adaboost, etc.)")
    parser.add_argument("--cleaning-config", type=str, help="Path to cleaning config JSON")
    parser.add_argument("--feature-config", type=str, help="Path to feature engineering config JSON")
    parser.add_argument("--feature-selection", action="store_true", help="Apply 3-stage feature selection filter")
    parser.add_argument("--class-balancing", choices=["undersample", "smote", "class_weight"], help="Class balancing strategy")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive wizard")

    args = parser.parse_args()

    # If no dataset is specified or --interactive is passed, launch the wizard
    if args.interactive or not args.dataset:
        run_interactive_wizard()
        return

    agent = MLAgent()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset file '{dataset_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    target = args.target

    task = args.task
    if task == "auto":
        task = agent.infer_task_type(df, target)
        print(f"Auto-detected task type: {task}")

    stem = dataset_path.stem
    clean_cfg = Path(args.cleaning_config) if args.cleaning_config else Path(f"{stem}_cleaning_config.json")
    feat_cfg = Path(args.feature_config) if args.feature_config else Path(f"{stem}_feature_engineering_config.json")

    if not clean_cfg.exists() or not feat_cfg.exists():
        print("Auto-generating dataset cleaning and feature configs...")
        clean_cfg, feat_cfg = agent.auto_generate_configs(df, target, stem)

    code = agent.run_pipeline(
        dataset_path=dataset_path,
        target=target,
        task=task,
        model=args.model,
        cleaning_config=clean_cfg,
        feature_config=feat_cfg,
        feature_selection=args.feature_selection,
        class_balancing=args.class_balancing,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
