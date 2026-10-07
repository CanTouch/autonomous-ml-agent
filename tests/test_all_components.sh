#!/usr/bin/env bash
# =============================================================================
# test_all_components.sh — Automated verification of all ML Agent components
# =============================================================================

set -euo pipefail

export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

PYTHON="./.venv/bin/python"
DATE=$(date +%F)

echo "============================================================"
echo " Starting Full Component Readiness Verification"
echo "============================================================"

# Test 1: Data Profiling
echo "[TEST 1] Data Profiling..."
$PYTHON src/data_profiling.py data/raw/titanic.csv --target Survived
test -f "reports/profile_titanic_${DATE}.md"
echo "  ✓ PASS: Data profiling report generated"

# Test 2: Data Cleaning
echo "[TEST 2] Data Cleaning..."
$PYTHON src/data_cleaning.py data/raw/titanic.csv --config data_cleaning_config.json
test -f "data/processed/titanic_cleaned.csv"
test -f "reports/cleaning_log_titanic_${DATE}.md"
echo "  ✓ PASS: Data cleaning completed"

# Test 3: Feature Engineering
echo "[TEST 3] Feature Engineering..."
$PYTHON src/feature_engineering.py data/processed/titanic_cleaned.csv --config feature_engineering_config.json
test -f "data/processed/titanic_features.csv"
test -f "reports/feature_engineering_log_titanic_${DATE}.md"
echo "  ✓ PASS: Feature engineering completed"

# Test 4: Feature Selection
echo "[TEST 4] Feature Selection..."
$PYTHON src/feature_selection.py data/processed/titanic_features.csv --target Survived --task classification
test -f "data/processed/titanic_selected.csv"
test -f "reports/feature_selection_titanic_${DATE}.md"
echo "  ✓ PASS: Feature selection completed"

# Test 5: Class Balancing
echo "[TEST 5] Class Balancing..."
$PYTHON src/class_balancing.py data/processed/titanic_selected.csv --target Survived --strategy undersample
test -f "data/processed/titanic_balanced.csv"
test -f "reports/class_balancing_titanic_${DATE}.md"
echo "  ✓ PASS: Class balancing completed"

# Test 6: PCA Analysis
echo "[TEST 6] PCA Analysis..."
$PYTHON src/pca_analysis.py data/processed/titanic_features.csv --target Survived --n-components 5
test -f "data/processed/titanic_pca.csv"
test -f "reports/pca_scree_titanic_${DATE}.png"
test -f "reports/pca_2d_titanic_${DATE}.png"
test -f "reports/pca_titanic_${DATE}.md"
echo "  ✓ PASS: PCA analysis completed"

# Test 7: K-Means Clustering
echo "[TEST 7] K-Means Clustering..."
$PYTHON src/clustering_kmeans.py data/processed/titanic_features.csv --target Survived --k 3
test -f "data/processed/titanic_kmeans.csv"
test -f "reports/kmeans_elbow_titanic_${DATE}.png"
test -f "reports/kmeans_titanic_${DATE}.md"
echo "  ✓ PASS: K-Means clustering completed"

# Test 8: Hierarchical Clustering
echo "[TEST 8] Hierarchical Clustering..."
$PYTHON src/clustering_hierarchical.py data/processed/titanic_features.csv --target Survived --n-clusters 3
test -f "data/processed/titanic_hierarchical.csv"
test -f "reports/hierarchical_dendrogram_titanic_${DATE}.png"
test -f "reports/hierarchical_titanic_${DATE}.md"
echo "  ✓ PASS: Hierarchical clustering completed"

# Test 9: Supervised Classification Evaluation (Random Forest, XGBoost, SVM)
echo "[TEST 9] Classification Evaluation..."
for MODEL in random_forest xgboost svm; do
  TUNING_REP=$(ls -t reports/tuning_${MODEL}_titanic_*.md | head -n 1)
  test -f "$TUNING_REP"
  BEST_PARAMS=$(grep "Best parameters" "$TUNING_REP" | sed "s/.*\`\(.*\)\`.*/\1/")
  $PYTHON src/evaluation.py data/processed/titanic_features.csv --target Survived --model "$MODEL" --params "$BEST_PARAMS"
  test -f "reports/evaluation_${MODEL}_titanic_${DATE}.md"
  test -f "reports/roc_curve_${MODEL}_titanic_${DATE}.png"
  echo "  ✓ PASS: Classification evaluation for $MODEL completed"
done

# Evaluate AdaBoost on Titanic
$PYTHON src/evaluation.py data/processed/titanic_features.csv --target Survived --model adaboost --params '{"n_estimators": 50, "learning_rate": 0.1}'
test -f "reports/evaluation_adaboost_titanic_${DATE}.md"
echo "  ✓ PASS: Classification evaluation for adaboost completed"

# Test 10: Supervised Regression Evaluation (Random Forest, XGBoost, SVM)
echo "[TEST 10] Regression Evaluation..."

for MODEL in random_forest xgboost svm; do
  TUNING_REP=$(ls -t reports/tuning_regression_${MODEL}_melbourne_housing_*.md | head -n 1)
  BEST_PARAMS=$(grep "Best parameters" "$TUNING_REP" | sed "s/.*\`\(.*\)\`.*/\1/")
  $PYTHON src/evaluation_regression.py data/processed/melbourne_housing_features.csv --target Price --model "$MODEL" --params "$BEST_PARAMS"
  test -f "reports/evaluation_regression_${MODEL}_melbourne_housing_${DATE}.md"
  test -f "reports/residuals_${MODEL}_melbourne_housing_${DATE}.png"
  echo "  ✓ PASS: Regression evaluation for $MODEL completed"
done

# Evaluate AdaBoost on Melbourne Housing
$PYTHON src/evaluation_regression.py data/processed/melbourne_housing_features.csv --target Price --model adaboost --params '{"n_estimators": 50, "learning_rate": 0.1}'
test -f "reports/evaluation_regression_adaboost_melbourne_housing_${DATE}.md"
echo "  ✓ PASS: Regression evaluation for adaboost completed"

# Test 11: Combined Final Report Generation
echo "[TEST 11] Report Generation..."
$PYTHON src/report_generation.py titanic
test -f "reports/FINAL_REPORT_titanic_${DATE}.md"
$PYTHON src/report_generation.py melbourne_housing
test -f "reports/FINAL_REPORT_melbourne_housing_${DATE}.md"
echo "  ✓ PASS: Final report generation completed"

echo ""
echo "============================================================"
echo " ALL 11 COMPONENT TESTS PASSED! Readiness is 100%."
echo "============================================================"
