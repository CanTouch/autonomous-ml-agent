---
name: model-selection
description: Trains and compares multiple candidate models on a feature-engineered dataset using stratified k-fold cross-validation. Reports mean, min, and max accuracy per model so the best candidate can be chosen before tuning. Does not tune hyperparameters — that is a separate skill.
---

# Model Selection Skill

## When to use this
Trigger this skill after feature-engineering has produced a model-ready
dataset. This skill compares baseline models — it does not tune any of
them beyond their defaults.

## Inputs required
- Path to the feature-engineered dataset
- Target column name
- Optional: number of folds (default 10, per standard practice)

## What this skill must do

1. **Split features from target.**

2. **Use StratifiedKFold, not plain KFold**, when the target is
   categorical/imbalanced — this preserves class proportions across
   folds, per the reasoning in the project's own knowledge base
   (train-test-split-vs-k-fold-vs-stratified-k-fold note).

3. **Train a small battery of baseline models** with default
   hyperparameters:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting (sklearn's GradientBoostingClassifier)

4. **Evaluate each with cross_val_score across all folds**, and report
   not just the mean accuracy but also min and max — per the project's
   own knowledge base note on reporting to stakeholders (average
   accuracy alone hides how much a model's performance varies across
   folds).

5. **Do not report only accuracy** — also compute cross-validated
   precision, recall, and F1 (via `cross_validate` with multiple
   scoring metrics), since accuracy alone can be misleading on
   imbalanced targets.

6. **Do not pick a "winner" automatically.** Present the comparison
   table and let the user decide, or explicitly ask "which model would
   you like to proceed with for tuning?" — do not assume higher mean
   accuracy alone is the right choice without considering variance and
   the other metrics.

7. **Output** a markdown comparison report to
   `reports/model_selection_<name>_<date>.md` with one row per model,
   columns for each metric (mean/min/max accuracy, precision, recall,
   F1).

## What NOT to do
- Do not tune hyperparameters here — this skill uses model defaults
  only. Tuning happens in the hyperparameter-tuning skill, after a
  model is chosen.
- Do not use plain KFold on an imbalanced or categorical target — use
  StratifiedKFold.
- Do not silently drop the target column's fold structure — always
  stratify by it for classification tasks.
- Do not declare a "best" model — present the data and let the user
  choose.
