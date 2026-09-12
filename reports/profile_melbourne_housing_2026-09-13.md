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
