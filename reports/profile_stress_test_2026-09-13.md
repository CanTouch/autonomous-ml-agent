# Data Profile: stress_test
Generated: 2026-09-13

## Shape
11 rows × 6 columns

## Dtypes
| Column | Dtype |
|---|---|
| id | int64 |
| age | str |
| income | float64 |
| category | str |
| target | float64 |
| notes | str |

## Missing Values
| Column | Missing Count | Missing % |
|---|---|---|
| notes | 5 | 45.45% ⚠️ high missingness |
| age | 2 | 18.18% |
| income | 2 | 18.18% |
| category | 2 | 18.18% |
| target | 2 | 18.18% |

## Duplicate Rows
1 duplicate rows found.

## Numeric Column Stats
| Column | Min | Max | Mean | Median | Std | Flag |
|---|---|---|---|---|---|---|
| id | 1.00 | 10.00 | 5.09 | 5.00 | 3.18 |  |
| income | 30000.00 | 999999999.00 | 111150222.11 | 45000.00 | 333318666.45 | ⚠️ possible outliers |
| target | 0.00 | 1.00 | 0.67 | 1.00 | 0.50 |  |

## Categorical Column Stats
| Column | Unique Values | Flag |
|---|---|---|
| age | 8 | ⚠️ high cardinality; ⚠️ looks numeric but contains non-numeric values — investigate before treating as categorical |
| category | 4 |  |
| notes | 6 | ⚠️ high cardinality |

## Potential Target Leakage Candidates
No obviously suspicious column names found.

## Target Balance
| Value | Count | Percentage |
|---|---|---|
| 1.0 | 6 | 54.55% |
| 0.0 | 3 | 27.27% |
