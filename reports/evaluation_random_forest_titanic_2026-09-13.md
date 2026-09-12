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