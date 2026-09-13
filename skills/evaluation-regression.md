---
name: evaluation-regression
description: Trains the final chosen regression model with tuned hyperparameters on a train split, then evaluates on a genuinely held-out test set — R², MAE, RMSE, plus a residual plot. Regression counterpart to evaluation.md.
---

# Evaluation Skill (Regression)

## When to use this
Trigger after hyperparameter-tuning-regression has produced final
tuned parameters. This is the last step before report-generation, and
the only skill to evaluate on truly unseen data.

## What this skill must do
1. Split into train/test BEFORE fitting anything (no stratify needed
   for continuous targets — plain random split).
2. Fit the model with tuned hyperparameters on TRAIN only.
3. Predict on the test set, compute R², MAE, RMSE.
4. Generate a residual plot (predicted vs. actual, and residuals vs.
   predicted) — the regression equivalent of a confusion matrix. A
   good model's residuals should scatter randomly around zero with no
   obvious pattern; a funnel shape (residuals growing with predicted
   value) suggests the model is less reliable for expensive properties.
5. Compare test metrics to the cross-validated numbers from tuning.
   Flag a large gap honestly, same as the classification version.

## What NOT to do
- Do not evaluate on training data.
- Do not re-tune here.
- Do not hide a large train/test performance gap.
