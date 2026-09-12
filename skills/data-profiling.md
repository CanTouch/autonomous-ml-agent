---
name: data-profiling
description: Profiles a raw dataset (CSV/Excel) before any cleaning or modeling begins. Reports shape, dtypes, missing values, duplicates, distributions, cardinality, and flags likely target leakage. Produces a markdown report and does not modify the source data.
---

# Data Profiling Skill

## When to use this
Trigger this skill whenever the user provides a new dataset and wants to
understand it before cleaning, feature engineering, or modeling. This is
always the FIRST step in the pipeline — never skip straight to cleaning
or modeling without profiling first.

## Inputs required
- Path to a dataset file (CSV or Excel)
- Optional: the name of the target column (if known) — enables leakage checks

## What this skill must do

1. **Load the data** using `src/data_profiling.py`, which handles both
   CSV and Excel via pandas.

2. **Report structural facts** — shape (rows × columns), column names,
   and dtypes. State these plainly; do not editorialize.

3. **Report missing values** — count and percentage per column, sorted
   descending. Flag any column with >40% missing as "high missingness —
   consider dropping or requires domain-specific imputation strategy."

4. **Report duplicate rows** — exact duplicate count. If duplicates exist,
   note it but do NOT remove them (profiling is read-only).

5. **Report numeric column stats** — min, max, mean, median, std for every
   numeric column. Flag columns where the max is more than 10x the 75th
   percentile as "possible outliers — verify before modeling."

6. **Report categorical column stats** — unique value count (cardinality)
   per categorical column. Flag any column with unique values > 50% of
   row count as "high cardinality — likely an ID column, not a feature"
   or, if named like a real category, "consider target encoding instead
   of one-hot."

7. **Target leakage check** (only if a target column is specified) —
   flag any column whose name suggests it was computed AFTER the target
   is known (e.g. contains words like "outcome", "result", "final",
   "post", or has suspiciously perfect correlation with the target).
   State this as a caution to investigate, not a certainty.

8. **Do not draw conclusions about what model to use or what cleaning
   steps to take.** This skill's job is to describe the data accurately,
   not prescribe next steps. Recommendations belong in later skills
   (data-cleaning, feature-engineering).

## Output
Write a markdown report to `reports/profile_<dataset_name>_<date>.md`
with the sections above, in order. Use tables where the data is tabular
(dtypes, missing values, numeric stats). Keep prose between tables brief
— this is a reference document, not an essay.

## What NOT to do
- Do not modify or save over the original dataset file.
- Do not impute missing values, drop columns, or transform data — this
  skill only observes and reports.
- Do not use outside knowledge about "best practices" beyond what's
  needed to flag the specific numeric thresholds above (e.g. don't
  invent additional checks not listed here without asking the user first).
