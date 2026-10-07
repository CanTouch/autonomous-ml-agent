#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end ML pipeline orchestrator
#
# Usage (classification):
#   ./workflows/run_pipeline.sh \
#     --dataset data/raw/titanic.csv \
#     --target Survived \
#     --cleaning-config data_cleaning_config.json \
#     --feature-config feature_engineering_config.json \
#     --model random_forest \
#     --task classification
#
# Usage (regression):
#   ./workflows/run_pipeline.sh \
#     --dataset data/raw/melbourne_housing.csv \
#     --target Price \
#     --cleaning-config melbourne_cleaning_config.json \
#     --feature-config melbourne_feature_engineering_config.json \
#     --model random_forest \
#     --task regression
#
# Usage (unsupervised):
#   ./workflows/run_pipeline.sh \
#     --dataset data/raw/titanic.csv \
#     --cleaning-config data_cleaning_config.json \
#     --feature-config feature_engineering_config.json \
#     --task unsupervised
#
# Optional flags:
#   --feature-selection         Run feature selection (variance, correlation, importance)
#   --class-balancing <strat>   Run class balancing (smote, undersample, class_weight)
#
# Each stage writes its output to data/processed/ and reports/.
# =============================================================================

set -euo pipefail

PYTHON="./.venv/bin/python"

# ---- Defaults ----------------------------------------------------------------
TASK="classification"
MODEL="random_forest"
TARGET=""
SCORING=""          # empty = use task default (f1 for classification, r2/neg_mae for regression)
FOLDS=10
DO_FEATURE_SELECTION=false
CLASS_BALANCING=""
CLEANING_CONFIG=""
FEATURE_CONFIG=""
DATASET=""

# ---- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)           DATASET="$2";            shift 2 ;;
    --target)            TARGET="$2";             shift 2 ;;
    --cleaning-config)   CLEANING_CONFIG="$2";    shift 2 ;;
    --feature-config)    FEATURE_CONFIG="$2";     shift 2 ;;
    --model)             MODEL="$2";              shift 2 ;;
    --task)              TASK="$2";               shift 2 ;;
    --scoring)           SCORING="$2";            shift 2 ;;
    --folds)             FOLDS="$2";              shift 2 ;;
    --feature-selection) DO_FEATURE_SELECTION=true; shift 1 ;;
    --class-balancing)   CLASS_BALANCING="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---- Validate required args --------------------------------------------------
: "${DATASET:?--dataset is required}"
if [[ "$TASK" != "unsupervised" && -z "$TARGET" ]]; then
  echo "Error: --target is required for classification and regression tasks."
  exit 1
fi

DATASET_STEM=$(basename "$DATASET" .csv)
DATE=$(date +%F)

echo ""
echo "============================================================"
echo " ML Pipeline"
echo " Dataset : $DATASET"
echo " Target  : ${TARGET:-(none)}"
echo " Task    : $TASK"
if [[ "$TASK" != "unsupervised" ]]; then
  echo " Model   : $MODEL"
fi
echo "============================================================"
echo ""

# ---- Stage 1: Data Profiling -------------------------------------------------
echo "[1] Data Profiling..."
TARGET_ARG=""
if [[ -n "$TARGET" ]]; then
  TARGET_ARG="--target $TARGET"
fi
$PYTHON src/data_profiling.py "$DATASET" $TARGET_ARG
echo "      → reports/profile_${DATASET_STEM}_${DATE}.md"
echo ""

# ---- Stage 2: Data Cleaning --------------------------------------------------
CLEANED="$DATASET"
if [[ -n "$CLEANING_CONFIG" && -f "$CLEANING_CONFIG" ]]; then
  echo "[2] Data Cleaning..."
  $PYTHON src/data_cleaning.py "$DATASET" --config "$CLEANING_CONFIG"
  CLEANED="data/processed/${DATASET_STEM}_cleaned.csv"
  echo "      → $CLEANED"
  echo "      → reports/cleaning_log_${DATASET_STEM}_${DATE}.md"
  echo ""
fi

# ---- Stage 3: Feature Engineering -------------------------------------------
FEATURES="$CLEANED"
if [[ -n "$FEATURE_CONFIG" && -f "$FEATURE_CONFIG" ]]; then
  echo "[3] Feature Engineering..."
  $PYTHON src/feature_engineering.py "$CLEANED" --config "$FEATURE_CONFIG"
  FEATURES="data/processed/${DATASET_STEM}_features.csv"
  echo "      → $FEATURES"
  echo "      → reports/feature_engineering_log_${DATASET_STEM}_${DATE}.md"
  echo ""
fi

# ---- Optional Stage: Feature Selection ---------------------------------------
if [[ "$DO_FEATURE_SELECTION" == true && -n "$TARGET" ]]; then
  echo "[+] Feature Selection..."
  $PYTHON src/feature_selection.py "$FEATURES" --target "$TARGET" --task "$TASK"
  FEATURES="data/processed/${DATASET_STEM}_selected.csv"
  echo "      → $FEATURES"
  echo "      → reports/feature_selection_${DATASET_STEM}_${DATE}.md"
  echo ""
fi

# ---- Optional Stage: Class Balancing -----------------------------------------
if [[ -n "$CLASS_BALANCING" && "$TASK" == "classification" && -n "$TARGET" ]]; then
  echo "[+] Class Balancing ($CLASS_BALANCING)..."
  $PYTHON src/class_balancing.py "$FEATURES" --target "$TARGET" --strategy "$CLASS_BALANCING"
  if [[ "$CLASS_BALANCING" != "class_weight" ]]; then
    FEATURES="data/processed/${DATASET_STEM}_balanced.csv"
  fi
  echo "      → reports/class_balancing_${DATASET_STEM}_${DATE}.md"
  echo ""
fi

# ---- Branch: Unsupervised Workflow -------------------------------------------
if [[ "$TASK" == "unsupervised" ]]; then
  echo "[4] Unsupervised Analysis..."
  TARGET_FLAG=""
  if [[ -n "$TARGET" ]]; then
    TARGET_FLAG="--target $TARGET"
  fi

  echo "  - Running Principal Component Analysis..."
  $PYTHON src/pca_analysis.py "$FEATURES" $TARGET_FLAG
  echo "      → reports/pca_${DATASET_STEM}_${DATE}.md"

  echo "  - Running K-Means Clustering..."
  $PYTHON src/clustering_kmeans.py "$FEATURES" $TARGET_FLAG
  echo "      → reports/kmeans_${DATASET_STEM}_${DATE}.md"

  echo "  - Running Hierarchical Clustering..."
  $PYTHON src/clustering_hierarchical.py "$FEATURES" $TARGET_FLAG
  echo "      → reports/hierarchical_${DATASET_STEM}_${DATE}.md"
  echo ""

  echo "[+] Generating combined final report..."
  $PYTHON src/report_generation.py "$DATASET_STEM"
  echo "      → reports/FINAL_REPORT_${DATASET_STEM}_${DATE}.md"
  echo ""
  echo "============================================================"
  echo " Unsupervised pipeline complete!"
  echo " Final report: reports/FINAL_REPORT_${DATASET_STEM}_${DATE}.md"
  echo "============================================================"
  exit 0
fi

# ---- Stage 4: Model Selection ------------------------------------------------
echo "[4/6] Model Selection (${FOLDS}-fold CV)..."
if [[ "$TASK" == "regression" ]]; then
  $PYTHON src/model_selection_regression.py "$FEATURES" --target "$TARGET" --folds "$FOLDS"
  echo "      → reports/model_selection_regression_${DATASET_STEM}_${DATE}.md"
else
  $PYTHON src/model_selection.py "$FEATURES" --target "$TARGET" --folds "$FOLDS"
  echo "      → reports/model_selection_${DATASET_STEM}_${DATE}.md"
fi
echo ""

# ---- Stage 5: Hyperparameter Tuning -----------------------------------------
echo "[5/6] Hyperparameter Tuning ($MODEL)..."
SCORING_ARG=""
if [[ -n "$SCORING" ]]; then
  SCORING_ARG="--scoring $SCORING"
fi
if [[ "$TASK" == "regression" ]]; then
  $PYTHON src/hyperparameter_tuning_regression.py "$FEATURES" --target "$TARGET" --model "$MODEL" $SCORING_ARG
  echo "      → reports/tuning_regression_${MODEL}_${DATASET_STEM}_${DATE}.md"
else
  $PYTHON src/hyperparameter_tuning.py "$FEATURES" --target "$TARGET" --model "$MODEL" $SCORING_ARG
  echo "      → reports/tuning_${MODEL}_${DATASET_STEM}_${DATE}.md"
fi
echo ""

# ---- Stage 6: Evaluation -----------------------------------------------------
echo "[6/6] Final Evaluation..."
# Read best params from the tuning report
if [[ "$TASK" == "regression" ]]; then
  TUNING_REPORT="reports/tuning_regression_${MODEL}_${DATASET_STEM}_${DATE}.md"
else
  TUNING_REPORT="reports/tuning_${MODEL}_${DATASET_STEM}_${DATE}.md"
fi
BEST_PARAMS=$(grep "Best parameters" "$TUNING_REPORT" | sed "s/.*\`\(.*\)\`.*/\1/")
if [[ "$TASK" == "regression" ]]; then
  $PYTHON src/evaluation_regression.py "$FEATURES" --target "$TARGET" --model "$MODEL" --params "$BEST_PARAMS"
  echo "      → reports/evaluation_regression_${MODEL}_${DATASET_STEM}_${DATE}.md"
else
  $PYTHON src/evaluation.py "$FEATURES" --target "$TARGET" --model "$MODEL" --params "$BEST_PARAMS"
  echo "      → reports/evaluation_${MODEL}_${DATASET_STEM}_${DATE}.md"
fi
echo ""

# ---- Final Report ------------------------------------------------------------
echo "[+] Generating combined final report..."
$PYTHON src/report_generation.py "$DATASET_STEM"
echo "      → reports/FINAL_REPORT_${DATASET_STEM}_${DATE}.md"

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo " Final report: reports/FINAL_REPORT_${DATASET_STEM}_${DATE}.md"
echo "============================================================"
echo ""
