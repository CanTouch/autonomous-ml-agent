---
name: data-cleaning
description: Cleans a dataset based on the findings from a data-profiling report — handles missing values, drops or flags unusable columns, and prepares categorical columns for encoding. Always runs AFTER data-profiling and requires the profiling report as input. Never runs silently; proposes a plan and waits for approval before modifying data.
---

# Data Cleaning Skill

## When to use this
Trigger this skill only after data-profiling has already run on the
dataset. This skill reads the profiling report and proposes a cleaning
plan — it does not re-profile the data itself.

## Inputs required
- Path to the raw dataset
- Path to the corresponding profiling report (from data-profiling)

## What this skill must do

1. **Read the profiling report** and extract: missing value columns +
   percentages, high-cardinality columns, outlier flags, duplicate count.

2. **Propose a cleaning plan BEFORE touching any data.** For each issue
   found in profiling, propose a specific action and the reasoning:

   - **Missing values >40%**: propose dropping the column, unless the
     user indicates the column is important (e.g. Cabin on Titanic —
     propose dropping, but note that presence/absence of a cabin number
     could itself be informative and offer creating a binary
     "has_cabin" flag as an alternative).
   - **Missing values <40%, numeric**: propose median imputation as the
     default (robust to outliers), or mean if the profiling report
     shows no outlier flag on that column.
   - **Missing values <40%, categorical**: propose mode imputation or a
     new "Unknown" category, and explain the tradeoff (mode imputation
     assumes missingness is random; "Unknown" preserves the signal that
     data was missing).
   - **High-cardinality columns flagged as ID-like** (e.g. PassengerId,
     Ticket, Name): propose dropping, UNLESS the user wants to engineer
     a feature from them first (e.g. extracting a title from Name) —
     ask before dropping if the column name suggests extractable
     structure.
   - **Duplicate rows**: propose dropping exact duplicates, and ask for
     confirmation only if the duplicate count is a meaningful fraction
     of the dataset (>1%).
   - **Outlier-flagged numeric columns**: do NOT propose automatically
     removing outliers. Propose investigating them first (they may be
     real, valid data points, e.g. legitimate first-class fares) and
     offer capping/winsorizing only as an option, not a default.

3. **Wait for explicit approval of the plan** before writing any code
   that modifies data. Present the plan as a numbered list.

4. **Once approved, execute the cleaning** using
   `src/data_cleaning.py`, which takes the dataset and a cleaning
   config (built from the approved plan) and outputs a cleaned dataset
   to `data/processed/<name>_cleaned.csv`. The original raw file is
   NEVER modified or overwritten.

5. **Log every transformation applied** — write a short markdown log
   to `reports/cleaning_log_<name>_<date>.md` stating exactly what was
   done to each column and why, so the steps are auditable later.

## What NOT to do
- Do not silently drop columns or impute values without the plan being
  approved first.
- Do not perform feature engineering (creating new features, encoding
  categoricals) — that belongs to the feature-engineering skill, which
  runs after this one.
- Do not remove outliers by default — flag and ask, never assume.
- Do not modify the original raw dataset file under any circumstances.
