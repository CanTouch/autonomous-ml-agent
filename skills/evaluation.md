---
name: evaluation
description: Trains the final chosen model with its tuned hyperparameters on a train split, then evaluates it on a genuinely held-out test set — producing a confusion matrix, ROC/AUC, and classification metrics. This is the only skill in the pipeline that reports performance on truly unseen data.
---

# Evaluation Skill

## When to use this
Trigger this skill only after hyperparameter-tuning has produced final
chosen parameters for one model. This is the last step before
report-generation. Everything before this skill used cross-validation
scores — this skill is the first and only one to hold out a real test
set the model never sees during training.

## Inputs required
- Path to the feature-engineered dataset
- Target column name
- The model type and its tuned hyperparameters (from hyperparameter-tuning)
- Test set size (default 20%)

## What this skill must do

1. **Split the data into train/test BEFORE fitting anything**, using
   stratified splitting (`stratify=y`) to preserve class proportions —
   consistent with the project's own reasoning about imbalanced
   targets in earlier skills.

2. **Fit the model with the tuned hyperparameters on the TRAIN set
   only.** Never let the test set touch training in any way.

3. **Predict on the test set** and build a confusion matrix, per the
   structure in the project's confusion-matrix knowledge note: rows =
   predicted, columns = actual, report TP/TN/FP/FN explicitly with
   labels, not just raw numbers in a grid.

4. **Compute ROC curve and AUC** on the test set predictions
   (using predicted probabilities, not just hard class labels), per
   the project's roc-and-auc knowledge note. Report the AUC as a
   single number for easy comparison to other runs.

5. **Report precision, recall, F1, and accuracy on the test set**, and
   explicitly compare these to the cross-validated numbers from
   model-selection and hyperparameter-tuning. If test performance is
   notably worse than cross-val performance, flag this clearly as a
   possible sign of overfitting during tuning — don't bury it.

6. **Output**:
   - A markdown report to `reports/evaluation_<model>_<name>_<date>.md`
     with the confusion matrix, ROC/AUC, and metrics comparison table.
   - A saved plot of the ROC curve to
     `reports/roc_curve_<model>_<name>_<date>.png`.

## What NOT to do
- Do not evaluate on data the model was trained on — that defeats the
  purpose of this skill entirely.
- Do not re-tune anything here — hyperparameters are fixed, coming
  from the hyperparameter-tuning skill's output.
- Do not report only accuracy — the whole point of this skill,
  informed by the project's own knowledge notes, is that accuracy
  alone can be misleading, especially on imbalanced data.
- Do not hide a large gap between cross-validated and test performance
  — that gap is exactly the kind of signal a real data scientist would
  flag, not smooth over.
