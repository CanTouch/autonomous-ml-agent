# Regression Hyperparameter Tuning Report: random_forest on melbourne_housing
Generated: 2026-09-13

**Scoring metric optimized**: neg_mean_absolute_error (note: sklearn reports this negated, higher/less-negative is better)
**Best score**: -167,455.12
**Best parameters**: `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 400}`

## Top 5 parameter combinations
| Rank | Params | Mean Score | Std |
|---|---|---|---|
| 1 | `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 400}` | -167,455.12 | 2,307.55 |
| 2 | `{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 400}` | -167,476.32 | 2,235.01 |
| 3 | `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 400}` | -167,560.68 | 2,220.35 |
| 4 | `{'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 400}` | -167,629.96 | 2,278.04 |
| 5 | `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}` | -167,925.09 | 2,391.36 |

Note: compare the spread between rank 1 and rank 2-5. A narrow gap means the 'best' result may not be meaningfully better than the runner-ups.