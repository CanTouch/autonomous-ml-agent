---
name: report-generation
description: Combines every report produced during a pipeline run (profiling, cleaning, feature engineering, model selection, tuning, evaluation) into one final markdown document. Runs last, after evaluation. Works for both classification and regression runs.
---

# Report Generation Skill

## When to use this
Trigger this skill after evaluation (or evaluation-regression) has
completed — it is always the last step in a pipeline run for a given
dataset.

## Inputs required
- The dataset name (the same name used throughout the pipeline, e.g.
  "titanic" or "melbourne_housing")

## What this skill must do

1. **Locate every report file** in `reports/` whose filename contains
   the dataset name, across all pipeline stages: profile, cleaning
   log, feature engineering log, model selection (classification or
   regression), tuning report, evaluation report.

2. **Assemble them in pipeline order**, not alphabetical or by
   filename — profiling first, then cleaning, then feature
   engineering, then model selection, then tuning, then evaluation.
   If a stage's report doesn't exist (e.g. a run that never reached
   tuning), skip it silently rather than erroring.

3. **Include a one-paragraph summary at the top** before the detailed
   sections — key findings only: shape, biggest data quality issue,
   which model was evaluated, and the headline metric (accuracy/F1 for
   classification, R²/MAE for regression). This is the part a busy
   reader sees first.

4. **Reference plot files by path** (ROC curve or residual plot) rather
   than trying to embed images inline in markdown text — link them so
   they're viewable.

5. **Do not re-analyze or reinterpret the underlying data.** This
   skill's job is to combine what already exists, not generate new
   conclusions. Any interpretation already written into the individual
   reports carries forward as-is.

## Output
Write the combined document to `reports/FINAL_REPORT_<dataset_name>_<date>.md`.

## What NOT to do
- Do not skip the summary paragraph — a wall of six concatenated
  reports with no orientation at the top defeats the purpose.
- Do not fabricate results for a stage that was never run — only
  include what's actually on disk.
