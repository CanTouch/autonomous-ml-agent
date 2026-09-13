# Final Report: melbourne_housing
Generated: 2026-09-13

## Summary
Dataset: 13580 rows, 21 columns. Main data quality issue: `BuildingArea` has high missingness. Final model evaluated: random_forest. This report combines 6 pipeline stage(s): Data Profile, Data Cleaning, Feature Engineering, Model Selection, Hyperparameter Tuning, Final Evaluation.

## Plots
- reports/residuals_random_forest_melbourne_housing_2026-09-13.png

---

## Stage: Data Profile
*Source: `reports/profile_melbourne_housing_2026-09-13.md`*

# Data Profile: melbourne_housing
Generated: 2026-09-13

## Shape
13580 rows × 21 columns

## Dtypes
| Column | Dtype |
|---|---|
| Suburb | str |
| Address | str |
| Rooms | int64 |
| Type | str |
| Price | float64 |
| Method | str |
| SellerG | str |
| Date | str |
| Distance | float64 |
| Postcode | float64 |
| Bedroom2 | float64 |
| Bathroom | float64 |
| Car | float64 |
| Landsize | float64 |
| BuildingArea | float64 |
| YearBuilt | float64 |
| CouncilArea | str |
| Lattitude | float64 |
| Longtitude | float64 |
| Regionname | str |
| Propertycount | float64 |

## Missing Values
| Column | Missing Count | Missing % |
|---|---|---|
| BuildingArea | 6450 | 47.5% ⚠️ high missingness |
| YearBuilt | 5375 | 39.58% |
| CouncilArea | 1369 | 10.08% |
| Car | 62 | 0.46% |

## Duplicate Rows
0 duplicate rows found.

## Numeric Column Stats
| Column | Min | Max | Mean | Median | Std | Flag |
|---|---|---|---|---|---|---|
| Rooms | 1.00 | 10.00 | 2.94 | 3.00 | 0.96 |  |
| Price | 85000.00 | 9000000.00 | 1075684.08 | 903000.00 | 639310.72 |  |
| Distance | 0.00 | 48.10 | 10.14 | 9.20 | 5.87 |  |
| Postcode | 3000.00 | 3977.00 | 3105.30 | 3084.00 | 90.68 |  |
| Bedroom2 | 0.00 | 20.00 | 2.91 | 3.00 | 0.97 |  |
| Bathroom | 0.00 | 8.00 | 1.53 | 1.00 | 0.69 |  |
| Car | 0.00 | 10.00 | 1.61 | 2.00 | 0.96 |  |
| Landsize | 0.00 | 433014.00 | 558.42 | 440.00 | 3990.67 | ⚠️ possible outliers |
| BuildingArea | 0.00 | 44515.00 | 151.97 | 126.00 | 541.01 | ⚠️ possible outliers |
| YearBuilt | 1196.00 | 2018.00 | 1964.68 | 1970.00 | 37.27 |  |
| Lattitude | -38.18 | -37.41 | -37.81 | -37.80 | 0.08 |  |
| Longtitude | 144.43 | 145.53 | 145.00 | 145.00 | 0.10 |  |
| Propertycount | 249.00 | 21650.00 | 7454.42 | 6555.00 | 4378.58 |  |

## Categorical Column Stats
| Column | Unique Values | Flag |
|---|---|---|
| Suburb | 314 |  |
| Address | 13378 | ⚠️ high cardinality |
| Type | 3 |  |
| Method | 5 |  |
| SellerG | 268 |  |
| Date | 58 |  |
| CouncilArea | 33 |  |
| Regionname | 8 |  |

## Potential Target Leakage Candidates
- ⚠️ `Postcode` — name suggests it may be computed after the target is known. Investigate before using as a feature.

## Target Type
Target `2204` unique values — treated as continuous (regression target), not categorical. Class-balance reporting skipped.


---

## Stage: Data Cleaning
*Source: `reports/cleaning_log_melbourne_housing_2026-09-13.md`*

# Cleaning Log: melbourne_housing
Generated: 2026-09-13

- Added binary flag `has_building_area` from presence/absence of `BuildingArea`.
- Imputed `YearBuilt` (5375 missing) with median = 1970.00.
- Filled `CouncilArea` (1369 missing) with 'Unknown' category.
- Imputed `Car` (62 missing) with median = 2.00.
- Dropped columns: Address, BuildingArea, Date.

---

## Stage: Feature Engineering
*Source: `reports/feature_engineering_log_melbourne_housing_2026-09-13.md`*

# Feature Engineering Log: melbourne_housing
Generated: 2026-09-13

- One-hot encoded ['Type', 'Method', 'CouncilArea', 'Regionname'] → new columns: ['CouncilArea_Bayside', 'CouncilArea_Boroondara', 'CouncilArea_Brimbank', 'CouncilArea_Cardinia', 'CouncilArea_Casey', 'CouncilArea_Darebin', 'CouncilArea_Frankston', 'CouncilArea_Glen Eira', 'CouncilArea_Greater Dandenong', 'CouncilArea_Hobsons Bay', 'CouncilArea_Hume', 'CouncilArea_Kingston', 'CouncilArea_Knox', 'CouncilArea_Macedon Ranges', 'CouncilArea_Manningham', 'CouncilArea_Maribyrnong', 'CouncilArea_Maroondah', 'CouncilArea_Melbourne', 'CouncilArea_Melton', 'CouncilArea_Monash', 'CouncilArea_Moonee Valley', 'CouncilArea_Moorabool', 'CouncilArea_Moreland', 'CouncilArea_Nillumbik', 'CouncilArea_Port Phillip', 'CouncilArea_Stonnington', 'CouncilArea_Unavailable', 'CouncilArea_Unknown', 'CouncilArea_Whitehorse', 'CouncilArea_Whittlesea', 'CouncilArea_Wyndham', 'CouncilArea_Yarra', 'CouncilArea_Yarra Ranges', 'Method_S', 'Method_SA', 'Method_SP', 'Method_VB', 'Regionname_Eastern Victoria', 'Regionname_Northern Metropolitan', 'Regionname_Northern Victoria', 'Regionname_South-Eastern Metropolitan', 'Regionname_Southern Metropolitan', 'Regionname_Western Metropolitan', 'Regionname_Western Victoria', 'Type_t', 'Type_u'].
- Frequency-encoded `Suburb` -> `Suburb_freq` (replaced raw column with its occurrence rate).
- Frequency-encoded `SellerG` -> `SellerG_freq` (replaced raw column with its occurrence rate).

---

## Stage: Model Selection
*Source: `reports/model_selection_regression_melbourne_housing_2026-09-13.md`*

# Regression Model Selection Report: melbourne_housing
Generated: 2026-09-13

Target: `Price`

| Model | R² (mean) | MAE (mean) | RMSE (mean) |
|---|---|---|---|
| Linear Regression | 0.6246 | 258,682.90 | 391,465.45 |
| Random Forest | 0.7980 | 166,753.43 | 286,733.74 |
| Gradient Boosting | 0.7649 | 190,027.47 | 309,671.21 |

No model has been auto-selected. Higher R² is better; lower MAE/RMSE is better. A large gap between MAE and RMSE suggests the model makes a few very large errors alongside mostly small ones.

---

## Stage: Hyperparameter Tuning
*Source: `reports/tuning_regression_random_forest_melbourne_housing_2026-09-13.md`*

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

---

## Stage: Final Evaluation
*Source: `reports/evaluation_regression_random_forest_melbourne_housing_2026-09-13.md`*

# Regression Evaluation Report: random_forest on melbourne_housing
Generated: 2026-09-13

## Test Set Metrics (held out, never trained on)
- R²: 0.8200
- MAE: 165,391.25
- RMSE: 267,372.55

Compare these to the cross-validated numbers from model-selection-regression and hyperparameter-tuning-regression. A large gap can indicate overfitting. Check the residual plot: a funnel shape (residuals growing with predicted value) suggests the model is less reliable at the high end of the price range.

---
