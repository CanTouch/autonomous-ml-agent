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
