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