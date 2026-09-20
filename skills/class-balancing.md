---
name: class-balancing
description: Detects and corrects class imbalance in a classification dataset using SMOTE (oversample minority), random undersampling, or a class_weight flag. Runs AFTER feature-selection and BEFORE model-selection. Always reports class distribution before and after. Never modifies the original file.
---

# Class Balancing Skill

## When to use this
Run this skill only for classification tasks where the data-profiling
report shows a meaningful class imbalance (minority class < 35% of total).
Skip this skill entirely for regression tasks.

## Inputs required
- Path to the selected features dataset (output of feature-selection)
- Name of the target column
- `--strategy`: one of `smote`, `undersample`, or `class_weight`
  - `smote`       — oversample the minority class synthetically (default)
  - `undersample` — randomly remove majority class rows to match minority
  - `class_weight`— do not resample; instead print the weight dict to pass
                    to the model (no output file written — just a report)

## What this skill must do

1. **Report class distribution before** balancing: counts and percentages
   per class value.

2. **Apply the chosen strategy**:
   - `smote`: Use `SMOTE` from `imbalanced-learn`. Output shape will be
     roughly 2 × minority_count rows. Only applied to training split —
     NEVER to the test set. Split is 80/20, stratified.
   - `undersample`: Use `RandomUnderSampler`. Majority class is reduced to
     match minority class count.
   - `class_weight`: Compute `{class: total/(n_classes * count)}` weights.
     Print the dict. Write it to the report. No file is written.

3. **Report class distribution after** balancing (skip for `class_weight`).

4. **Save output** to `data/processed/<name>_balanced.csv` for `smote` and
   `undersample` strategies. The test split is saved separately as
   `data/processed/<name>_balanced_test.csv` — it is NEVER resampled.

5. **Write report** to `reports/class_balancing_<name>_<date>.md` with:
   - Strategy used and rationale
   - Before/after class distribution table
   - Final row counts (train balanced + test untouched)

## What NOT to do
- Do not apply SMOTE or undersampling to the test set — ever.
- Do not use this skill for regression targets.
- Do not invent new samples outside the range of the training data.
