# Regression Evaluation Report: random_forest on melbourne_housing
Generated: 2026-09-13

## Test Set Metrics (held out, never trained on)
- R²: 0.8200
- MAE: 165,391.25
- RMSE: 267,372.55

Compare these to the cross-validated numbers from model-selection-regression and hyperparameter-tuning-regression. A large gap can indicate overfitting. Check the residual plot: a funnel shape (residuals growing with predicted value) suggests the model is less reliable at the high end of the price range.