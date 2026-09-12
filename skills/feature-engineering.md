---
name: feature-engineering
description: Transforms a cleaned dataset into a model-ready dataset by encoding categorical columns and creating derived features. Runs AFTER data-cleaning. Proposes an encoding plan before applying it, and never modifies the cleaned dataset in place.
---

# Feature Engineering Skill

## When to use this
Trigger this skill only after data-cleaning has already produced a
cleaned dataset. This skill does not clean data — it assumes missing
values are already handled.

## Inputs required
- Path to the cleaned dataset (from data-cleaning)
- Optional: target column name (so it's never accidentally encoded)

## What this skill must do

1. **Identify remaining categorical columns** — any column with dtype
   object/string still present after cleaning.

2. **Propose an encoding strategy per column, before applying anything:**
   - **Low cardinality (2 unique values)**: propose binary encoding
     (0/1) — e.g. Sex → 0/1.
   - **Low-to-medium cardinality (3-10 unique values), no natural order**:
     propose one-hot encoding, and note it will increase column count.
   - **High cardinality (>10 unique values)**: propose target encoding
     or frequency encoding instead of one-hot, and explain why (one-hot
     would create too many sparse columns).
   - **Ordinal columns** (where order matters, e.g. a rating scale):
     propose ordinal/integer encoding that preserves the order — ask
     the user to confirm the order if it's not obvious from the data.

3. **Wait for approval of the plan** before transforming anything.

4. **Once approved, apply the encoding** using
   `src/feature_engineering.py`, outputting to
   `data/processed/<name>_features.csv`. The cleaned input file is
   never modified.

5. **Log every transformation** to
   `reports/feature_engineering_log_<name>_<date>.md` — which column
   was encoded, which method, and the resulting new column names.

## What NOT to do
- Do not encode the target column.
- Do not scale/normalize numeric columns here — that's a modeling-time
  decision, not a feature-engineering-time one, and scaling before
  train/test split risks data leakage. Note this explicitly if the user
  asks about it.
- Do not drop or impute anything — if missing values remain, stop and
  point back to data-cleaning; don't silently patch over it here.
