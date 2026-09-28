---
name: catboost
description: Complete guide to CatBoost (Categorical Boosting) covering Ordered Target Encoding, target binning, sequential non-leaking leaf outputs, threshold selection via Cosine Similarity, and Symmetric (Oblivious) Decision Trees. Extracted strictly from the 2-part StatQuest CatBoost series.
---

# CatBoost Methodology & Reference

> **Summary**: CatBoost (Categorical Boosting) is a gradient boosting framework designed specifically to handle categorical features without target leakage. It introduces Ordered Target Encoding by processing observations in randomized sequential order, computes sequential tree leaf outputs to prevent residual leakage, evaluates split thresholds using Cosine Similarity, and builds Symmetric (Oblivious) Trees that evaluate identical split conditions across all nodes at each depth for fast vectorized inference.

---

## Part 1: Ordered target encoding (source: catboost-part-1-ordered-target-encoding.md)

### Motivation & Target Leakage
- **Categorical Feature Challenge**: Features with multiple categories need to be converted to numbers without causing exponential column growth (like one-hot encoding).
- **Target Leakage**: Standard target encoding uses a row's target value to calculate its own feature encoding. This allows models to achieve high training accuracy by memorizing targets, but results in severe overfitting on test data.
- **CatBoost Approach**: CatBoost (short for *Categorical Boosting*) eliminates leakage by treating observations as if they arrive sequentially, one row at a time.

### Ordered Target Encoding Formula
To encode a category in the current row, CatBoost only considers preceding rows in the sequence:

$$\text{Encoded Value} = \frac{\text{Option Count} + \text{Prior}}{N_{\text{previous}} + 1}$$

- $\text{Option Count}$: Number of times the positive target (e.g., Target = 1) occurred for that specific category in preceding rows.
- $\text{Prior}$: A defined constant guess or prior baseline (e.g., $0.05$).
- $N_{\text{previous}}$: Total number of preceding rows that contained that category.
- **First Occurrence**: When a category appears for the first time ($N_{\text{previous}} = 0, \text{Option Count} = 0$), the formula evaluates to:
  $$\text{Encoded Value} = \frac{0 + 0.05}{0 + 1} = 0.05$$

### Encoding Unseen / Test Data
- During training, the sequential ordering prevents leakage.
- Once training is complete, the entire training dataset is aggregated to compute the fixed target encoding applied to new test data.

---

## Part 2: Symmetric trees, building and using (source: catboost-part-2-building-and-using-trees.md)

*(Note: The core concepts of target leakage and Ordered Target Encoding are established in Part 1 and referenced here).*

### Target Binning for Continuous Variables
When predicting a continuous target (regression), CatBoost converts the continuous target into discrete bins (e.g., Bin $0$ and Bin $1$) to enable Ordered Target Encoding on categorical features. The worked example uses two bins. With more than two bins, the source says a slightly different encoding equation is used, [not fully specified in source].

### Step-by-Step Tree Construction

1. **Randomize Order & Re-encode**:
   - Randomize the row order of the training set before building each tree.
   - Re-apply Ordered Target Encoding based on the new randomized sequence. Ordered Target Encoding is applied to discrete columns with more than two options (features with only two options are simply replaced with 0s and 1s).
2. **Initialize Predictions & Residuals**:
   - Initial predictions are set to $0$ for all samples.
   - Initial residuals = $\text{Observed} - \text{Predicted} = \text{Observed} - 0 = \text{Observed}$.
3. **Generate Candidate Split Thresholds**:
   - Sort the encoded feature values and identify candidate midpoints between adjacent values.
   - *(Note: For large datasets, candidate thresholds are constrained by binning close values)*.
4. **Evaluate Thresholds via Cosine Similarity**:
   - To avoid leakage, leaf output values are tracked sequentially as samples flow down the tree:
     - Initialize leaf output to $0$.
     - When a row enters a leaf, assign its predicted output as the current leaf state *before* incorporating that row's residual.
     - Update the leaf output to the average of all residuals in that leaf up to that point.
   - Compute the **Cosine Similarity** between the vector of observed residuals ($A$) and the vector of sequential leaf outputs ($B$):
     $$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
   - Select the candidate threshold with the highest Cosine Similarity.
5. **Update Model Predictions & Residuals**:
   - Update each prediction:
     $$\text{New Prediction} = \text{Previous Prediction} + (\text{Learning Rate} \times \text{Leaf Output})$$
   - Recompute residuals: $\text{Residual} = \text{Observed} - \text{New Prediction}$.
   - Iterate to build sequential trees.

---

## Symmetric (Oblivious) Decision Trees

### Definition
- In a **Symmetric Decision Tree**, all decision nodes at the same tree depth/level use the **exact same splitting feature and threshold**.
- For example, if level 2 tests $\text{Age} < 12$, both branches at depth 2 test $\text{Age} < 12$.

### Rationale & Advantages
1. **Weaker Individual Learners**: Constraining trees to be symmetric prevents individual trees from overfitting complex branch combinations, aligning with gradient boosting's principle of combining weak learners.
2. **Fast Vectorized Inference**: Because every sample evaluates the exact same sequence of questions regardless of which branch it takes, tree evaluation can be executed simultaneously across all rows using single vector operations without conditional branching overhead.
