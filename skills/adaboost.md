---
name: adaboost
description: Complete guide to AdaBoost (Adaptive Boosting) covering decision stumps (weak learners), total error calculations, amount of say formulas, exponential sample reweighting, normalization, and weighted ensemble classification. Extracted strictly from StatQuest AdaBoost.
---

# AdaBoost Methodology & Reference

> **Summary**: AdaBoost (Adaptive Boosting) is an ensemble classification algorithm that combines sequentially built weak learners—typically single-split decision stumps. Each stump's voting power ("amount of say") is calculated from its weighted classification error. Misclassified samples receive increased sample weights while correctly classified samples receive decreased weights, forcing subsequent stumps to correct prior mistakes.

---

## 1. Core Concepts & Comparison with Random Forests

| Property | Random Forests | AdaBoost |
|---|---|---|
| **Tree Size** | Full-sized decision trees (no predetermined fixed depth) | Stumps (single node with 2 leaves evaluating only 1 variable) |
| **Learner Type** | Strong individual learners | Weak learners combined into a strong ensemble |
| **Voting Power** | Equal vote per tree | Unequal vote: amount of say depends on individual accuracy |
| **Tree Creation Order** | Independent: build order does not matter | Sequential (order is important; each stump depends on errors of preceding stumps) |

---

## 2. Three Core Principles of AdaBoost
1. **Ensemble of Weak Learners**: Combines many simple decision stumps.
2. **Variable Amount of Say**: Stumps with lower classification error have a larger influence on the final decision.
3. **Sequential Error Correction**: Previous stumps' misclassifications directly guide how subsequent stumps and datasets are formed.

---

## 3. Step-by-Step Algorithm

### Step 1: Initialize Sample Weights
- Assign every sample an equal starting weight:
  $$w_i = \frac{1}{N}$$
  *(where $N$ is the total number of training samples)*.

### Step 2: Build the First Stump
1. Evaluate candidate stumps across all individual features (e.g. using Gini index / threshold evaluation).
2. Select the single feature and threshold that yields the lowest Gini index.

### Step 3: Compute Total Error
- Calculate the total error of the selected stump by summing the weights of all misclassified samples:
  $$\text{Total Error} = \sum_{i \in \text{Misclassified}} w_i$$
- Because sample weights sum to 1, $\text{Total Error} \in [0, 1]$.

### Step 4: Calculate Amount of Say ($\alpha$)
- Compute the voting weight ("amount of say") for the stump:
  $$\text{Amount of Say} = \frac{1}{2} \log\left(\frac{1 - \text{Total Error}}{\text{Total Error}}\right)$$
  - When $\text{Total Error}$ is small $\implies \text{Amount of Say}$ is a large positive value.
  - When $\text{Total Error} = 0.5$ (random guess) $\implies \text{Amount of Say} = 0$.
  - When $\text{Total Error}$ is close to $1 \implies \text{Amount of Say}$ is a large negative value.
  - *(Note: In practice, a small epsilon error term is added to prevent undefined log operations when total error is 0 or 1)*.

### Step 5: Update & Normalize Sample Weights
- **Increase weight for misclassified samples**:
  $$w_{\text{new}, i} = w_{\text{old}, i} \times e^{\text{Amount of Say}}$$
- **Decrease weight for correctly classified samples**:
  $$w_{\text{new}, i} = w_{\text{old}, i} \times e^{-\text{Amount of Say}}$$
- **Normalize weights**:
  Divide each updated weight by the sum of all new weights so they sum to 1:
  $$w_{\text{normalized}, i} = \frac{w_{\text{new}, i}}{\sum_{j=1}^N w_{\text{new}, j}}$$

### Step 6: Create the Next Dataset / Stump
Two equivalent approaches for applying sample weights to the next stump:
1. **Weighted Gini Index**: Use $w_{\text{normalized}}$ directly in the impurity calculations.
2. **Resampling with Replacement (Distribution Sampling)**:
   - Build a new dataset of size $N$ by drawing random numbers in $[0, 1]$ and mapping them to the cumulative distribution of normalized weights.
   - Samples with larger weights are duplicated multiple times in the new dataset.
   - Reset sample weights equally ($1/N$) in the resampled dataset.

### Step 7: Repeat
- Repeat steps 2 through 6 sequentially to build the specified number of stumps in the forest.

---

## 4. Making Predictions with an AdaBoost Forest
To classify a new sample:
1. Pass the sample through every stump in the forest.
2. Group the stumps by their predicted class (e.g., Class A vs Class B).
3. Sum the **Amounts of Say** for each class group:
   $$\text{Score}_{\text{Class A}} = \sum_{\text{stumps voting A}} \text{Amount of Say}$$
   $$\text{Score}_{\text{Class B}} = \sum_{\text{stumps voting B}} \text{Amount of Say}$$
4. Assign the final classification to the class with the higher aggregate score.
