# ML Agent Project

An agent-driven machine learning pipeline with a knowledge base built from
YouTube video transcripts. Every stage is implemented as a **skill** (a
plain-markdown instruction file) paired with a **Python helper script**
— so the agent knows *what* to do and has a tool to do it.

---

## Project Layout

```
ml-agent-project/
├── data/
│   ├── raw/                  # Original, unmodified datasets
│   ├── processed/            # Cleaned and feature-engineered datasets
│   ├── transcripts/          # Raw YouTube transcripts (markdown)
│   └── knowledge/            # Extracted methodology notes
├── reports/                  # All output reports and plots
├── skills/                   # Agent skill definitions (markdown)
├── src/                      # Python helper scripts
├── workflows/                # End-to-end pipeline orchestration
├── urls.txt                  # YouTube URLs for batch ingestion
├── failed_ingests.log        # URLs that could not be ingested
└── .venv/                    # Python virtual environment
```

---

## Skills

| Skill | File | Description |
|---|---|---|
| Ingest YouTube | `skills/ingest-youtube.md` | Download transcript + metadata from a YouTube video |
| Extract Methodology | `skills/extract-methodology.md` | Extract structured knowledge notes from a transcript |
| Data Profiling | `skills/data-profiling.md` | Profile a raw dataset — shape, dtypes, missing values, cardinality, outlier flags |
| Data Cleaning | `skills/data-cleaning.md` | Propose and execute a cleaning plan (imputation, drops, flags) |
| Feature Engineering | `skills/feature-engineering.md` | Encode categoricals, create binary flags, apply frequency encoding |
| Model Selection | `skills/model-selection.md` | Compare classifiers via k-fold CV |
| Model Selection (Regression) | `skills/model-selection-regression.md` | Compare regressors via k-fold CV |
| Hyperparameter Tuning | `skills/hyperparameter-tuning.md` | Tune classifier hyperparameters via RandomizedSearchCV |
| Hyperparameter Tuning (Regression) | `skills/hyperparameter-tuning-regression.md` | Tune regressor hyperparameters via RandomizedSearchCV |
| Evaluation | `skills/evaluation.md` | Final held-out test evaluation for classifiers (confusion matrix, ROC, AUC) |
| Evaluation (Regression) | `skills/evaluation-regression.md` | Final held-out test evaluation for regressors (MAE, RMSE, R², residuals plot) |
| Report Generation | `skills/report-generation.md` | Combine all stage reports into a single final report |

---

## Full Pipeline

### Option A — Automated (single command)

```bash
# Classification example (Titanic)
./workflows/run_pipeline.sh \
  --dataset data/raw/titanic.csv \
  --target Survived \
  --cleaning-config data_cleaning_config.json \
  --feature-config feature_engineering_config.json \
  --model random_forest \
  --task classification

# Regression example (Melbourne housing)
./workflows/run_pipeline.sh \
  --dataset data/raw/melbourne_housing.csv \
  --target Price \
  --cleaning-config melbourne_cleaning_config.json \
  --feature-config melbourne_feature_engineering_config.json \
  --model random_forest \
  --task regression
```

### Option B — Stage by stage

```bash
PYTHON="./.venv/bin/python"

# 1. Profile
$PYTHON src/data_profiling.py data/raw/titanic.csv --target Survived

# 2. Clean  (edit data_cleaning_config.json first based on profile findings)
$PYTHON src/data_cleaning.py data/raw/titanic.csv --config data_cleaning_config.json

# 3. Feature engineer  (edit feature_engineering_config.json based on cleaning output)
$PYTHON src/feature_engineering.py data/processed/titanic_cleaned.csv --config feature_engineering_config.json

# 4. Model selection
$PYTHON src/model_selection.py data/processed/titanic_features.csv --target Survived

# 5. Hyperparameter tuning
$PYTHON src/hyperparameter_tuning.py data/processed/titanic_features.csv \
  --target Survived --model random_forest --scoring f1

# 6. Evaluation  (paste best params from tuning report)
$PYTHON src/evaluation.py data/processed/titanic_features.csv \
  --target Survived --model random_forest \
  --params '{"max_depth": null, "min_samples_leaf": 1, "min_samples_split": 5, "n_estimators": 400}'

# 7. Final report
$PYTHON src/report_generation.py titanic
```

---

## YouTube Ingestion

```bash
# Ingest a single video
./.venv/bin/python src/ingest_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Batch ingest from urls.txt
./src/batch_ingest.sh urls.txt
# Failures are logged to failed_ingests.log
```

Then run the extract-methodology skill on the new transcript:
> "Run the extract-methodology skill on data/transcripts/<filename>.md"

---

## Knowledge Base

9 methodology notes extracted from StatQuest and other ML educators:

| Topic | File |
|---|---|
| Bias and Variance | `data/knowledge/bias-and-variance.md` |
| Cross-Validation | `data/knowledge/cross-validation.md` |
| Train/Test Split vs K-Fold vs Stratified K-Fold | `data/knowledge/train-test-split-vs-k-fold-vs-stratified-k-fold.md` |
| Ridge Regression (L2) | `data/knowledge/regularization-part-1-ridge-l2-regression.md` |
| Lasso Regression (L1) | `data/knowledge/regularization-part-2-lasso-l1-regression.md` |
| GridSearchCV Hyperparameter Tuning | `data/knowledge/gridsearchcv-hyperparameter-tuning.md` |
| Gradient Boost — Regression Main Ideas | `data/knowledge/gradient-boost-part-1-regression-main-ideas.md` |
| Confusion Matrix | `data/knowledge/confusion-matrix.md` |
| ROC and AUC | `data/knowledge/roc-and-auc.md` |

---

## Validated Pipelines

### Titanic (Classification)
- **Model**: Random Forest
- **Test Accuracy**: 0.7709 | **AUC**: 0.8370 | **F1**: 0.6870
- **Best params**: `n_estimators=400, max_depth=None, min_samples_split=5, min_samples_leaf=1`
- **Final report**: `reports/FINAL_REPORT_titanic_2026-09-13.md`

### Melbourne Housing (Regression)
- **Model**: Random Forest
- **Final report**: `reports/FINAL_REPORT_melbourne_housing_2026-09-13.md`

---

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install pandas openpyxl scikit-learn matplotlib yt-dlp youtube-transcript-api
```

---

## Design Principles

- **Source fidelity**: Knowledge notes contain only what the video actually says — no external knowledge added.
- **Read-only profiling**: The profiling stage never modifies data.
- **Plan before acting**: The cleaning skill proposes a plan and waits for approval before touching data.
- **Auditability**: Every pipeline stage writes a report. The final report combines them all.
- **No silent drops**: Outliers are flagged, never automatically removed.
