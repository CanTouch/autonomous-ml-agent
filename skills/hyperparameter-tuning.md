---
name: hyperparameter-tuning
description: Tunes a single chosen model's hyperparameters using GridSearchCV with stratified cross-validation, based on the pattern extracted from the project's own GridSearchCV knowledge note. Runs only after model-selection has identified a candidate model to tune.
---

# Hyperparameter Tuning Skill

## When to use this
Trigger this skill only after model-selection has produced a comparison
and the user has picked ONE model to proceed with. This skill does not
compare models — it only tunes the one specified.

## Inputs required
- Path to the feature-engineered dataset
- Target column name
- Which model to tune (Logistic Regression / Random Forest / Gradient Boosting)
- Optional: scoring metric to optimize for (default: f1, since the
  project's model-selection results showed accuracy alone can hide
  precision/recall tradeoffs)

## What this skill must do

1. **Use StratifiedKFold for the CV inside GridSearchCV**, consistent
   with model-selection — do not switch to plain KFold here.

2. **Define a reasonable default parameter grid per model type**:
   - Random Forest: n_estimators, max_depth, min_samples_split,
     min_samples_leaf
   - Gradient Boosting: n_estimators, learning_rate, max_depth
   - Logistic Regression: C, penalty

3. **Run GridSearchCV** with the chosen scoring metric, and report:
   - Best parameters found
   - Best cross-validated score
   - The full `cv_results_` comparison for the top 5 parameter
     combinations (not just the single best) — so the user can see how
     much difference tuning actually made, per the gotcha noted in the
     project's own GridSearchCV knowledge note about not over-trusting
     a single "best" result.

4. **Compare the tuned score against the untuned baseline** from the
   model-selection report for the same model, and state the
   improvement (or lack of it) explicitly. Sometimes tuning barely
   helps — report that honestly rather than implying tuning is always
   worthwhile.

5. **Output** a markdown report to
   `reports/tuning_<model>_<name>_<date>.md`.

## What NOT to do
- Do not tune multiple models in one run — one model per invocation,
  chosen deliberately by the user based on the model-selection report.
- Do not silently accept the "best" params without showing the
  runner-up combinations — a narrow margin between best and 2nd/3rd
  best means the "best" params may just be noise.
- Do not use the full dataset for final evaluation here — that's the
  evaluation skill's job, on a held-out test set. This skill only
  reports cross-validated tuning scores.
