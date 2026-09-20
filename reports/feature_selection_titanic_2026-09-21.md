# Feature Selection Report: titanic
Generated: 2026-09-21

**Original feature count**: 9
**Final feature count**: 7
**Features removed**: 2

## Stage 1: Variance Threshold
| Column | Variance | Action |
|---|---|---|
| Pclass | 0.699015 | kept |
| Sex | 0.228475 | kept |
| Age | 169.512498 | kept |
| SibSp | 1.216043 | kept |
| Parch | 0.649728 | kept |
| Fare | 2469.436846 | kept |
| has_cabin | 0.176734 | kept |
| Embarked_Q | 0.079040 | kept |
| Embarked_S | 0.199586 | kept |

## Stage 2: Correlation Drop
No highly-correlated pairs found above threshold.

## Stage 3: Permutation Importance
| Column | Mean Importance | Std | Action |
|---|---|---|---|
| Sex | 0.158101 | 0.025729 | kept |
| Fare | 0.040782 | 0.011187 | kept |
| Pclass | 0.026257 | 0.012994 | kept |
| Embarked_S | 0.017318 | 0.009497 | kept |
| Age | 0.015642 | 0.023807 | kept |
| has_cabin | 0.013408 | 0.014177 | kept |
| SibSp | 0.005028 | 0.006345 | kept |
| Embarked_Q | -0.004469 | 0.004870 | dropped — importance ≤ 0 |
| Parch | -0.010056 | 0.008211 | dropped — importance ≤ 0 |

## Final Feature Set
`Pclass`, `Sex`, `Age`, `SibSp`, `Fare`, `has_cabin`, `Embarked_S`