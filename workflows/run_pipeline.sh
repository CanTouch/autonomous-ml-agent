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
# Each stage writes its output to data/processed/ and reports/.
# Run individual scripts directly for more control over each stage.
# =============================================================================

set -euo pipefail

PYTHON="./.venv/bin/python"

# ---- Defaults ----------------------------------------------------------------
TASK="classification"
MODEL="random_forest"
SCORING=""          # empty = use task default (f1 for classification, r2 for regression)
FOLDS=10

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
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---- Validate required args --------------------------------------------------
: "${DATASET:?--dataset is required}"
: "${TARGET:?--target is required}"
: "${CLEANING_CONFIG:?--cleaning-config is required}"
: "${FEATURE_CONFIG:?--feature-config is required}"

DATASET_STEM=$(basename "$DATASET" .csv)
DATE=$(date +%F)

echo ""
echo "============================================================"
echo " ML Pipeline"
echo " Dataset : $DATASET"
echo " Target  : $TARGET"
echo " Task    : $TASK"
echo " Model   : $MODEL"
echo "============================================================"
echo ""

# ---- Stage 1: Data Profiling -------------------------------------------------
echo "[1/6] Data Profiling..."
$PYTHON src/data_profiling.py "$DATASET" --target "$TARGET"
echo "      → reports/profile_${DATASET_STEM}_${DATE}.md"
echo ""

# ---- Stage 2: Data Cleaning --------------------------------------------------
echo "[2/6] Data Cleaning..."
$PYTHON src/data_cleaning.py "$DATASET" --config "$CLEANING_CONFIG"
CLEANED="data/processed/${DATASET_STEM}_cleaned.csv"
echo "      → $CLEANED"
echo "      → reports/cleaning_log_${DATASET_STEM}_${DATE}.md"
echo ""

# ---- Stage 3: Feature Engineering -------------------------------------------
echo "[3/6] Feature Engineering..."
$PYTHON src/feature_engineering.py "$CLEANED" --config "$FEATURE_CONFIG"
FEATURES="data/processed/${DATASET_STEM}_features.csv"
echo "      → $FEATURES"
echo "      → reports/feature_engineering_log_${DATASET_STEM}_${DATE}.md"
echo ""

# ---- Stage 4: Model Selection ------------------------------------------------
echo "[4/6] Model Selection (${FOLDS}-fold CV)..."
if [[ "$TASK" == "regression" ]]; then
  $PYTHON src/model_selection_regression.py "$FEATURES" --target "$TARGET" --folds "$FOLDS"
else
  $PYTHON src/model_selection.py "$FEATURES" --target "$TARGET" --folds "$FOLDS"
fi
echo "      → reports/model_selection_${DATASET_STEM}_${DATE}.md"
echo ""

# ---- Stage 5: Hyperparameter Tuning -----------------------------------------
echo "[5/6] Hyperparameter Tuning ($MODEL)..."
SCORING_ARG=""
if [[ -n "$SCORING" ]]; then
  SCORING_ARG="--scoring $SCORING"
fi
if [[ "$TASK" == "regression" ]]; then
  $PYTHON src/hyperparameter_tuning_regression.py "$FEATURES" --target "$TARGET" --model "$MODEL" $SCORING_ARG
else
  $PYTHON src/hyperparameter_tuning.py "$FEATURES" --target "$TARGET" --model "$MODEL" $SCORING_ARG
fi
echo "      → reports/tuning_${MODEL}_${DATASET_STEM}_${DATE}.md"
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
