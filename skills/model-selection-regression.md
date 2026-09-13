---
name: model-selection-regression
description: Trains and compares multiple candidate regression models on a feature-engineered dataset using k-fold cross-validation. Reports R-squared, MAE, and RMSE per model. This is the regression counterpart to model-selection.md, used when the target is continuous rather than categorical.
---

# Model Selection Skill (Regression)

## When to use this
Trigger this skill instead of model-selection.md when the target column
is continuous (a regression problem — e.g. price, count, duration) rather
than categorical. Check the data-profiling report's "Target Type" section
to confirm: if it says "continuous", use this skill.

## Inputs required
- Path to the feature-engineered dataset
- Target column name
- Optional: number of folds (default 10)

## What this skill must do

1. **Use plain KFold, not StratifiedKFold** — stratification by class
   proportion is a classification-only concept and does not apply to a
   continuous target.

2. **Train a small battery of baseline regression models**:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor

3. **Report three metrics per model**:
   - R² (proportion of variance explained — higher is better, 1.0 is perfect)
   - MAE (mean absolute error, in the target's original units — directly
     interpretable, e.g. "average error of $150,000")
   - RMSE (root mean squared error — penalizes large errors more than MAE;
     a large gap between MAE and RMSE suggests some big misses)

4. **Do not pick a "winner" automatically** — same principle as the
   classification version. Present the comparison and let the user decide.

## What NOT to do
- Do not use StratifiedKFold or any classification-specific logic.
- Do not report accuracy/precision/recall/F1 — these do not apply to
  continuous targets.
- Do not declare a "best" model without the user's input.
