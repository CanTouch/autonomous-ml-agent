---
name: hyperparameter-tuning-regression
description: Tunes a single chosen regression model's hyperparameters using GridSearchCV with plain k-fold cross-validation. Regression counterpart to hyperparameter-tuning.md.
---

# Hyperparameter Tuning Skill (Regression)

## When to use this
Trigger after model-selection-regression has identified a candidate
model. Use plain KFold, not StratifiedKFold — this is a continuous
target.

## What this skill must do
1. Use KFold for the CV inside GridSearchCV.
2. Define a reasonable parameter grid per model type.
3. Optimize for a chosen metric (default: neg_mean_absolute_error,
   since MAE is directly interpretable in the target's units).
4. Report best params, best score, and top 5 combinations — same
   discipline as the classification version: don't over-trust a
   single "best" result if the margin over runner-ups is small.
5. Compare tuned score against the untuned baseline from
   model-selection-regression, stated honestly.

## What NOT to do
- Do not use StratifiedKFold.
- Do not silently accept "best" params without showing runner-ups.
