---
title: "The Confusion Matrix"
source_transcript: "data/transcripts/machine-learning-fundamentals-the-confusion-matrix.md"
source_url: "https://www.youtube.com/watch?v=Kdsp6soqA7o"
video_title: "Machine Learning Fundamentals: The Confusion Matrix"
date_extracted: "2026-09-13"
technique: "Confusion Matrix"
tags: ["confusion-matrix", "classification", "model-evaluation", "true-positive", "false-negative", "false-positive", "true-negative"]
---

# The Confusion Matrix

> **Summary**: A confusion matrix summarises what a machine learning classification algorithm got right and what it got wrong. Rows represent what the algorithm predicted; columns represent the known truth. Correct predictions run along the diagonal (green); misclassifications are off-diagonal (red). The matrix scales to any number of classes.

---

## 1. Core Technique Taught

### Overview & Concepts

- **Motivating problem**: Medical dataset with clinical measurements (chest pain, blood circulation, blocked arteries, weight) used to predict whether a patient will develop heart disease. Multiple algorithms (logistic regression, K-nearest neighbors, random forest) are candidates, and the confusion matrix is used to compare them on the same test set.
- **Workflow described**:
  1. Divide data into training and testing sets (the speaker notes "this would be an excellent opportunity to use cross-validation").
  2. Train all candidate methods on the training data.
  3. Test each method on the test set.
  4. Build a confusion matrix per method to summarise performance.

#### Structure of a Binary Confusion Matrix (as explained in the video)

Rows = what the algorithm predicted. Columns = the known truth.

|  | Actually Has Disease | Actually No Disease |
|---|---|---|
| **Predicted: Has Disease** | True Positives (TP) — top-left | False Positives (FP) — top-right |
| **Predicted: No Disease** | False Negatives (FN) — bottom-left | True Negatives (TN) — bottom-right |

- **True Positives**: Patients with heart disease correctly identified by the algorithm.
- **True Negatives**: Patients without heart disease correctly identified.
- **False Negatives**: Patient has heart disease but algorithm said they didn't.
- **False Positives**: Patient does not have heart disease but algorithm said they did.
- **Diagonal (green boxes)**: Correctly classified samples.
- **Off-diagonal (red boxes)**: Samples the algorithm "messed up."

#### Example numbers from the video (Random Forest)
- 142 True Positives
- 110 True Negatives
- 29 False Negatives
- 22 False Positives

#### Comparison used in the video
K-nearest neighbors produced 107 TPs and 79 TNs vs. the random forest's 142 and 110 — the random forest was clearly better on both metrics, making it the preferred choice.

#### Multi-class confusion matrices
When there are more than two categories (e.g., predicting favourite movie among Troll 2, Gore Police, or Cool as Ice), the matrix has one row and one column per class. The diagonal still represents correct predictions; everything else is a misclassification. A 40-class problem produces a 40×40 matrix.

- **Teaser**: The speaker explicitly mentions that sensitivity, specificity, ROC, and AUC are more sophisticated metrics covered in follow-up videos when two confusion matrices are too similar to distinguish between methods.

---

## 2. Code Patterns Shown

No code shown in this video — conceptual explanation only.

---

## 3. Key Warnings & Gotchas

- **Confusion matrices alone may be insufficient**: When two confusion matrices look very similar, the speaker states it becomes "hard to choose which machine learning method is a better fit for this data" and explicitly recommends more sophisticated metrics (sensitivity, specificity, ROC, AUC) instead.
- **Use cross-validation when splitting data**: The speaker specifically calls out that splitting into training and test sets is "an excellent opportunity to use cross-validation."
