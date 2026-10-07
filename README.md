---
title: Autonomous ML Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.65.0
app_file: app.py
pinned: false
license: mit
short_description: Zero-config tabular ML agent for automated EDA, modeling & scoring
---

# 🤖 Autonomous ML Agent

An autonomous, zero-configuration Automated Data Scientist and machine learning pipeline for tabular data. Simply upload any raw CSV or select a benchmark dataset — the agent automatically inspects data quality, cleans anomalies, balances skewed distributions, runs a multi-algorithm tournament, tunes the champion model, and generates interactive reports, feature importance drivers, and downloadable artifacts.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](app.py)

---

### 🌟 Key Capabilities
- **Zero-Config Autonomy**: Auto-detects target columns, task type (Classification, Regression, or Unsupervised), missing values, and data types without manual intervention.
- **Robust Data Cleaning**: Automatic median/mode imputation, deduplication, and outlier filtering.
- **Class Balancing**: Automatically handles severe target imbalance via SMOTE, undersampling, or class weighting.
- **6-Model Tournament**: Evaluates Random Forest, Gradient Boosting, XGBoost, AdaBoost, Logistic/Linear Regression, and SVM across 10-fold cross-validation.
- **Hyperparameter Optimization**: Grid search tuning for the tournament champion.
- **Unsupervised Suite**: Principal Component Analysis (PCA with scree & 2D projection), K-Means (elbow curve & silhouette analysis), and Hierarchical Agglomerative Clustering (dendrograms).
- **Driver Analysis & Export Tray**:
  - 📥 **Download Markdown Report**: Comprehensive executive audit and performance metrics.
  - 💾 **Download Trained Model (`.joblib`)**: Production-ready Scikit-learn pipeline for downstream deployment.
  - 📊 **Download Predictions (`.csv`)**: Held-out test set predictions with ground truth and probabilities.
- **🔮 Batch Scoring Hub**: Upload new unlabelled CSVs directly in the UI to generate instant batch predictions.

---

---

## Project Layout

```
ml-agent-project/
├── data/
│   ├── raw/                  # Original, unmodified datasets
│   ├── processed/            # Cleaned, feature-engineered, and transformed datasets
│   ├── transcripts/          # Raw YouTube transcripts (markdown)
│   └── knowledge/            # Extracted methodology notes
├── reports/                  # All output markdown reports and PNG plots
├── skills/                   # Agent skill definitions (26 markdown specifications)
├── src/                      # Production Python helper scripts
├── tests/                    # Unit tests and end-to-end integration test suite
├── workflows/                # End-to-end pipeline orchestrator (run_pipeline.sh)
├── urls.txt                  # YouTube URLs for batch ingestion
├── failed_ingests.log        # URLs that could not be ingested
└── .venv/                    # Python virtual environment
```

---

## Skills Registry (26 Skills)

### 1. Data Preparation & Preprocessing
| Skill | File | Helper Script | Description |
|---|---|---|---|
| Ingest YouTube | `skills/ingest-youtube.md` | `src/ingest_youtube.py` | Ingest transcript + metadata from YouTube videos |
| Extract Methodology | `skills/extract-methodology.md` | — | Extract structured knowledge notes with strict source fidelity |
| Data Profiling | `skills/data-profiling.md` | `src/data_profiling.py` | Profile raw datasets — shape, types, missingness, cardinality, outliers |
| Data Cleaning | `skills/data-cleaning.md` | `src/data_cleaning.py` | Execute config-driven cleaning (imputation, dropping, flags) |
| Feature Engineering | `skills/feature-engineering.md` | `src/feature_engineering.py` | Encode categoricals, create binary flags, frequency encoding |
| Feature Selection | `skills/feature-selection.md` | `src/feature_selection.py` | Variance, collinearity, and permutation importance filtering |
| Class Balancing | `skills/class-balancing.md` | `src/class_balancing.py` | Address class imbalance via SMOTE, undersampling, or class weights |

### 2. Supervised Learning & Evaluation
| Skill | File | Helper Script | Description |
|---|---|---|---|
| Model Selection | `skills/model-selection.md` | `src/model_selection.py` | Stratified k-fold CV across classifiers (RF, GB, AdaBoost, XGBoost, SVM, LR) |
| Model Selection (Regression) | `skills/model-selection-regression.md` | `src/model_selection_regression.py` | K-fold CV across regressors (RF, GB, AdaBoost, XGBoost, LR) |
| Hyperparameter Tuning | `skills/hyperparameter-tuning.md` | `src/hyperparameter_tuning.py` | Grid search with cross-validation for classifiers |
| Hyperparameter Tuning (Regression) | `skills/hyperparameter-tuning-regression.md` | `src/hyperparameter_tuning_regression.py` | Grid search with cross-validation for regressors |
| Evaluation | `skills/evaluation.md` | `src/evaluation.py` | Held-out test evaluation (confusion matrix, ROC curves, AUC, F1) |
| Evaluation (Regression) | `skills/evaluation-regression.md` | `src/evaluation_regression.py` | Held-out test evaluation (MAE, RMSE, R², residual plots) |
| Report Generation | `skills/report-generation.md` | `src/report_generation.py` | Combines up to 11 pipeline stage reports and all plots into a master report |

### 3. Unsupervised Learning
| Skill | File | Helper Script | Description |
|---|---|---|---|
| PCA | `skills/pca.md` | `src/pca_analysis.py` | SVD dimensionality reduction, scree plots, 2D projections, loading scores |
| K-Means Clustering | `skills/k-means-clustering.md` | `src/clustering_kmeans.py` | K-means clustering with automated elbow curve scan and silhouette scores |
| Hierarchical Clustering | `skills/hierarchical-clustering.md` | `src/clustering_hierarchical.py` | Agglomerative clustering (Euclidean & Manhattan) with dendrogram generation |

### 4. Machine Learning Algorithm Methodology
| Skill | File | Core Techniques Extracted |
|---|---|---|
| Decision Trees | `skills/decision-trees.md` | Gini impurity, numeric thresholds, missing data surrogate splits |
| Random Forests | `skills/random-forests.md` | Bootstrapping, feature sub-sampling, out-of-bag (OOB) error estimation |
| Gradient Boost | `skills/gradient-boost.md` | Pseudo-residuals, shrinkage, log-odds transformation, Taylor expansion |
| AdaBoost | `skills/adaboost.md` | Sample weight updates, amount-of-say calculation, decision stumps |
| CatBoost | `skills/catboost.md` | Ordered target encoding, symmetric oblivious trees, cosine similarity |
| XGBoost | `skills/xgboost.md` | Similarity scores, gain criteria, cover values, regularization |
| SVM | `skills/svm.md` | Maximal margin hyperplanes, soft margin slack, polynomial and RBF kernels |

---

## Available Datasets (`data/raw/`)

| Dataset | File | Rows | Cols | Task / Benchmark Purpose | Target |
|---|---|---|---|---|---|
| **Titanic** | `data/raw/titanic.csv` | 891 | 12 | Binary classification with mixed data types and missing values | `Survived` |
| **Melbourne Housing** | `data/raw/melbourne_housing.csv` | 13,580 | 21 | Continuous regression with categorical & geographical features | `Price` |
| **Heart Disease (Cleveland)** | `data/raw/heart_disease.csv` | 303 | 14 | Clinical classification (from StatQuest Random Forest tutorial) | `target` |
| **Churn Modelling** | `data/raw/churn_modelling.csv` | 10,000 | 14 | Customer churn prediction (from Krish Naik XGBoost tutorial) | `Exited` |
| **California Housing** | `data/raw/california_housing.csv` | 20,640 | 9 | Continuous geographic regression benchmark | `MedHouseVal` |
| **Breast Cancer** | `data/raw/breast_cancer.csv` | 569 | 31 | High-dimensional classification, PCA dimensionality reduction | `target` |
| **Wine** | `data/raw/wine.csv` | 178 | 14 | Multi-class classification & unsupervised clustering | `target` |
| **Stress Test** | `data/raw/stress_test.csv` | 10 | 6 | Edge-case validation for mixed numeric types & nulls | `target` |

---

## Autonomous & Interactive ML Assistant (`src/agent.py`)

The ML Agent provides an interactive wizard and automated entrypoint that inspects datasets, automatically infers tasks, auto-generates cleaning and feature engineering configs, and runs the entire end-to-end pipeline:

### Interactive Wizard
```bash
./.venv/bin/python src/agent.py
```

### Scriptable Execution (Auto-Inference & Auto-Config)
```bash
# Auto-detects classification, auto-generates configs, and tunes XGBoost
./.venv/bin/python src/agent.py --dataset data/raw/heart_disease.csv --target target --model xgboost --feature-selection

# Auto-runs unsupervised PCA, K-Means, and Hierarchical clustering
./.venv/bin/python src/agent.py --dataset data/raw/wine.csv --task unsupervised
```

---

## Automated Pipeline (`workflows/run_pipeline.sh`)

### Classification Pipeline (Titanic)
```bash
./workflows/run_pipeline.sh \
  --dataset data/raw/titanic.csv \
  --target Survived \
  --cleaning-config data_cleaning_config.json \
  --feature-config feature_engineering_config.json \
  --model xgboost \
  --task classification \
  --feature-selection \
  --class-balancing undersample
```

### Regression Pipeline (Melbourne Housing)
```bash
./workflows/run_pipeline.sh \
  --dataset data/raw/melbourne_housing.csv \
  --target Price \
  --cleaning-config melbourne_cleaning_config.json \
  --feature-config melbourne_feature_engineering_config.json \
  --model xgboost \
  --task regression
```

### Unsupervised Pipeline (PCA + K-Means + Hierarchical)
```bash
./workflows/run_pipeline.sh \
  --dataset data/raw/titanic.csv \
  --target Survived \
  --cleaning-config data_cleaning_config.json \
  --feature-config feature_engineering_config.json \
  --task unsupervised
```

---

## Testing & Verification

The project includes an end-to-end integration test runner and an isolated unit test suite:

```bash
# 1. Run unit tests (edge cases, input validation, math correctness)
./.venv/bin/python -m unittest tests/test_unit.py

# 2. Run full 11-stage integration test suite
./tests/test_all_components.sh
```

---

## Setup & Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl scikit-learn matplotlib xgboost imbalanced-learn scipy yt-dlp youtube-transcript-api
```

---

## Design Principles

- **Strict Source Fidelity**: Methodology notes contain strictly what the primary source explains — zero external assumptions added.
- **Read-Only Profiling**: Profiling never mutates raw data.
- **Config-Driven Cleaning**: Transformations are auditable and reproducible via explicit configuration JSONs.
- **Auditability**: Every stage writes timestamped markdown logs and metrics.
- **Production Guardrails**: SVR iterations and parallelism are tuned per model to prevent worker deadlocks on large datasets.
 
---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
