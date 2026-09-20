---
name: feature-selection
description: Removes low-signal features from an engineered dataset before modeling. Uses three methods in order — variance threshold, correlation drop, and permutation importance — and writes a report explaining every column kept or dropped. Runs AFTER feature-engineering and BEFORE model-selection.
---

# Feature Selection Skill

## When to use this
Run this skill after feature-engineering and before model-selection.
Input must be a fully numeric dataset (all categoricals already encoded).

## Inputs required
- Path to the features dataset (output of feature-engineering)
- Name of the target column
- `--task`: `classification` or `regression`
- Optional thresholds (defaults shown):
  - `--variance-threshold 0.01` — drop columns with variance below this
  - `--corr-threshold 0.95`     — drop one of any pair correlated above this
  - `--importance-threshold 0.0` — drop features with permutation importance ≤ 0 (i.e. actively harmful)

## What this skill must do

1. **Variance threshold**: Drop any feature column whose variance is below
   `--variance-threshold`. Near-zero variance means the column carries
   almost no information. Always report which columns were dropped and
   their actual variance values.

2. **Correlation drop**: Compute the pairwise Pearson correlation matrix.
   For any pair of features correlated above `--corr-threshold` in
   absolute value, drop the one with the lower mean absolute correlation
   to all other features (keep the more "central" one). Report every
   pair found and which column was dropped.

3. **Permutation importance**: Fit a lightweight Random Forest (100 trees,
   no hyperparameter tuning) on the surviving features. Compute
   permutation importance on a held-out validation split (20%). Drop any
   feature with mean permutation importance ≤ 0 — these actively hurt
   the model. Report all importance scores, sorted descending.

4. **Write output**: Save the reduced feature set to
   `data/processed/<name>_selected.csv`. The target column is always
   preserved regardless of its scores.

5. **Write report**: Save `reports/feature_selection_<name>_<date>.md`
   with three sections (one per method), listing what was dropped, why,
   and the final column set.

## What NOT to do
- Do not drop the target column.
- Do not drop columns based on domain assumptions — only on the
  numeric criteria above.
- Do not tune the Random Forest used for permutation importance — it is
  a diagnostic tool, not the final model.
