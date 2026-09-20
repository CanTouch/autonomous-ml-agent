# Class Balancing Report: titanic
Generated: 2026-09-21
Strategy: `class_weight`

## Class Distribution — Before Balancing (full dataset)
| Class | Count | Percentage |
|---|---|---|
| 0 | 549 | 61.62% |
| 1 | 342 | 38.38% |

## Class Weights (pass to model)
No resampling applied. Use the following weight dict in your model:

```python
class_weight = {0: 0.8115, 1: 1.3026}
```

Pass as `class_weight` parameter to `RandomForestClassifier`, `LogisticRegression`, etc.