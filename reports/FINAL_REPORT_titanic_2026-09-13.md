# Final Report: titanic
Generated: 2026-09-13

## Summary
Dataset: 891 rows, 12 columns. Main data quality issue: `Cabin` has high missingness. Final model evaluated: random_forest. This report combines 6 pipeline stage(s): Data Profile, Data Cleaning, Feature Engineering, Model Selection, Hyperparameter Tuning, Final Evaluation.

## Plots
- reports/roc_curve_random_forest_titanic_2026-09-13.png

---

## Stage: Data Profile
*Source: `reports/profile_titanic_2026-09-13.md`*

# Data Profile: titanic
Generated: 2026-09-13

## Shape
891 rows × 12 columns

## Dtypes
| Column | Dtype |
|---|---|
| PassengerId | int64 |
| Survived | int64 |
| Pclass | int64 |
| Name | str |
| Sex | str |
| Age | float64 |
| SibSp | int64 |
| Parch | int64 |
| Ticket | str |
| Fare | float64 |
| Cabin | str |
| Embarked | str |

## Missing Values
| Column | Missing Count | Missing % |
|---|---|---|
| Cabin | 687 | 77.1% ⚠️ high missingness |
| Age | 177 | 19.87% |
| Embarked | 2 | 0.22% |

## Duplicate Rows
0 duplicate rows found.

## Numeric Column Stats
| Column | Min | Max | Mean | Median | Std | Flag |
|---|---|---|---|---|---|---|
| PassengerId | 1.00 | 891.00 | 446.00 | 446.00 | 257.35 |  |
| Survived | 0.00 | 1.00 | 0.38 | 0.00 | 0.49 |  |
| Pclass | 1.00 | 3.00 | 2.31 | 3.00 | 0.84 |  |
| Age | 0.42 | 80.00 | 29.70 | 28.00 | 14.53 |  |
| SibSp | 0.00 | 8.00 | 0.52 | 0.00 | 1.10 |  |
| Parch | 0.00 | 6.00 | 0.38 | 0.00 | 0.81 |  |
| Fare | 0.00 | 512.33 | 32.20 | 14.45 | 49.69 | ⚠️ possible outliers |

## Categorical Column Stats
| Column | Unique Values | Flag |
|---|---|---|
| Name | 891 | ⚠️ high cardinality |
| Sex | 2 |  |
| Ticket | 681 | ⚠️ high cardinality; ⚠️ looks numeric but contains non-numeric values — investigate before treating as categorical |
| Cabin | 147 |  |
| Embarked | 3 |  |

## Potential Target Leakage Candidates
No obviously suspicious column names found.

## Target Balance
| Value | Count | Percentage |
|---|---|---|
| 0 | 549 | 61.62% |
| 1 | 342 | 38.38% |


---

## Stage: Data Cleaning
*Source: `reports/cleaning_log_titanic_2026-09-13.md`*

# Cleaning Log: titanic
Generated: 2026-09-13

- Added binary flag `has_cabin` from presence/absence of `Cabin`.
- Imputed `Age` (177 missing) with median = 28.00.
- Imputed `Embarked` (2 missing) with mode = 'S'.
- Dropped columns: Ticket, Name, PassengerId, Cabin.

---

## Stage: Feature Engineering
*Source: `reports/feature_engineering_log_titanic_2026-09-13.md`*

# Feature Engineering Log: titanic
Generated: 2026-09-13

- Binary-encoded `Sex` using mapping {'male': 0, 'female': 1}.
- One-hot encoded ['Embarked'] → new columns: ['Embarked_Q', 'Embarked_S'].

---

## Stage: Model Selection
*Source: `reports/model_selection_titanic_2026-09-13.md`*

# Model Selection Report: titanic
Generated: 2026-09-13

| Model | Accuracy (mean) | Accuracy (min) | Accuracy (max) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7968 | 0.7303 | 0.8427 | 0.7449 | 0.7161 | 0.7287 |
| Random Forest | 0.8215 | 0.7753 | 0.8444 | 0.7780 | 0.7514 | 0.7624 |
| Gradient Boosting | 0.8260 | 0.7753 | 0.8889 | 0.8298 | 0.6900 | 0.7513 |

No model has been auto-selected. Review the metrics above — consider both mean performance and the accuracy range (min/max) before choosing which model to proceed with for tuning.

---

## Stage: Hyperparameter Tuning
*Source: `reports/tuning_random_forest_titanic_2026-09-13.md`*

# Hyperparameter Tuning Report: random_forest on titanic
Generated: 2026-09-13

**Scoring metric optimized**: f1
**Best score**: 0.7714
**Best parameters**: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}`

## Top 5 parameter combinations
| Rank | Params | Mean Score | Std |
|---|---|---|---|
| 1 | `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}` | 0.7714 | 0.0402 |
| 2 | `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}` | 0.7694 | 0.0372 |
| 3 | `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}` | 0.7690 | 0.0465 |
| 4 | `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}` | 0.7683 | 0.0438 |
| 5 | `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}` | 0.7682 | 0.0416 |

Note: compare the spread between rank 1 and rank 2-5. A narrow gap means the 'best' result may not be meaningfully better than the runner-ups — treat it as one reasonable choice, not a guaranteed optimum.

---

## Stage: Final Evaluation
*Source: `reports/evaluation_random_forest_titanic_2026-09-13.md`*

# Evaluation Report: random_forest on titanic
Generated: 2026-09-13

## Confusion Matrix (test set — held out, never trained on)
|  | Predicted: Positive | Predicted: Negative |
|---|---|---|
| **Actual: Positive** | TP = 45 | FN = 24 |
| **Actual: Negative** | FP = 17 | TN = 93 |

## Test Set Metrics
- Accuracy: 0.7709
- Precision: 0.7258
- Recall: 0.6522
- F1: 0.6870
- AUC: 0.8370

Compare these to the cross-validated numbers from model-selection and hyperparameter-tuning reports. A large gap (test performance notably worse) can indicate overfitting during tuning.

---
