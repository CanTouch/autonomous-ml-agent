---
name: xgboost
description: Complete guide to XGBoost covering regression, classification, mathematical derivation via 2nd-order Taylor approximation, and hardware/algorithmic optimizations. Extracted strictly from the 4-part StatQuest XGBoost series.
---

# XGBoost Methodology & Reference

> **Summary**: XGBoost (Extreme Gradient Boosting) is an ensemble tree algorithm that fits trees sequentially to the residuals of prior predictions. It optimizes an objective consisting of a specific loss function plus L2 leaf regularization ($\lambda$) using a second-order Taylor expansion (gradients and Hessians), pruned via a complexity threshold ($\gamma$) and scaled by a learning rate ($\eta$). For large datasets, it employs an approximate greedy algorithm with weighted quantile sketches, sparsity-aware split finding, cache-aware access, and out-of-core block computation.

---

## Part 1: Regression objective (source: xgboost-part-1-of-4-regression.md)

### What is XGBoost / Gradient Boosting
XGBoost stands for Extreme Gradient Boosting. Like unextreme gradient boost, it builds an ensemble of trees sequentially. The algorithm starts with an initial baseline prediction, computes residuals (the differences between observed values and predictions), and fits a tree to predict those residuals. Subsequent predictions are updated by scaling the tree's leaf outputs by a learning rate and adding them to the running prediction, taking small steps that shrink the residuals. Unlike unextreme gradient boost (which typically uses off-the-shelf regression trees), XGBoost uses custom regression trees built with similarity scores, gain, regularization, and pruning.

### Initial Prediction
- By default, the initial prediction is $0.5$ (regardless of whether XGBoost is used for regression or classification).
- Residuals are calculated as: $\text{Residual} = \text{Observed} - \text{Predicted}$.

### Similarity Score
Each tree starts as a single leaf containing all residuals. Splitting and leaf quality are evaluated using a **Similarity Score**:

$$\text{Similarity Score} = \frac{\left(\sum \text{Residuals}\right)^2}{\text{Number of Residuals} + \lambda}$$

- **Numerator**: The residuals are summed *before* squaring. As a result, positive and negative residuals cancel each other out when mixed in the same leaf. When residuals are similar (or isolated), they do not cancel out and the similarity score is relatively large.
- **Denominator**: The count of residuals in the leaf plus $\lambda$.
- $\lambda$ is a regularization parameter (default is $\lambda = 0$).

### Gain & Splitting Criteria
Candidate split thresholds are evaluated by calculating the **Gain**:

$$\text{Gain} = \text{Similarity}_{\text{Left}} + \text{Similarity}_{\text{Right}} - \text{Similarity}_{\text{Root}}$$

- Thresholds are tested at the average of adjacent sorted feature values.
- The threshold yielding the largest positive Gain is chosen for the branch.
- Tree depth: In the video example, tree depth is limited to 2 levels; the default maximum depth in XGBoost is 6 levels.

### Pruning via Gamma ($\gamma$)
After growing the tree, branches are pruned from the bottom up based on a user-defined tree complexity parameter called **gamma** ($\gamma$):
- Calculate: $\text{Gain} - \gamma$ for the lowest branch.
- If $\text{Gain} - \gamma < 0$ (negative difference): **prune** the branch.
- If $\text{Gain} - \gamma \ge 0$ (positive difference): **do not prune** the branch, and stop pruning along that path (the parent/root is not removed even if its own gain is less than $\gamma$).
- If $\text{Gain} - \gamma$ is negative for a branch and it is removed, proceed to test the next branch up the tree.

### Role of Lambda ($\lambda$) in Regularization & Pruning
- When $\lambda > 0$, similarity scores decrease.
- The shrinkage is inversely proportional to the number of residuals in the node (a node with 1 residual experiences a larger percentage reduction than a node with 4 residuals).
- Lower similarity scores produce lower Gain values, making it easier for $\text{Gain} - \gamma$ to be negative, thus encouraging more pruning.
- If $\lambda > 0$, a branch can have a negative Gain value. In such cases, setting $\gamma = 0$ still results in pruning ($\text{Gain} - 0 < 0$). Thus, setting $\gamma = 0$ does not turn off pruning.

### Output Value of Leaves
The final prediction value assigned to each terminal leaf is:

$$\text{Output Value} = \frac{\sum \text{Residuals}}{\text{Number of Residuals} + \lambda}$$

- Unlike the similarity score, the numerator is the sum of residuals (not squared).
- When $\lambda = 0$, the output value is simply the average of the residuals in the leaf.
- When $\lambda > 0$, the output value is shrunk toward zero, reducing the model's sensitivity to individual observations.

### Updating Predictions & Learning Rate ($\eta$)
Predictions are updated by adding the scaled tree output:

$$\text{New Prediction} = \text{Previous Prediction} + (\eta \times \text{Output Value})$$

- The learning rate is called **eta** ($\eta$), with a default value of $0.3$.
- Trees are built iteratively until residuals become very small or the user-specified maximum number of trees is reached.

---

## Part 2: Classification objective (source: xgboost-part-2-of-4-classification.md)

*(Note: The general gradient boosting framework and tree-building concepts are defined in Part 1 and referenced here.)*

### Initial Prediction
- By default, the initial prediction is a probability of $0.5$ (representing a 50% chance). It can also be set to an empirical probability (e.g., $0.75$).
- Target observations are binary: $1$ for positive/effective, $0$ for negative/ineffective.
- Residuals are calculated as: $\text{Residual} = \text{Observed} - \text{Predicted Probability}$.

### Similarity Score for Classification
$$\text{Similarity Score} = \frac{\left(\sum \text{Residuals}\right)^2}{\sum \left[\text{Previous Probability}_i \times (1 - \text{Previous Probability}_i)\right] + \lambda}$$

- **Numerator**: The sum of residuals squared (identical to regression).
- **Denominator**: The sum over each observation in the leaf of $\text{Previous Probability} \times (1 - \text{Previous Probability})$, plus $\lambda$.

### Cover Metric & `min_child_weight`
Cover determines the minimum weight required to allow a leaf:
- **Definition**: Cover is defined as the denominator of the Similarity Score minus $\lambda$.
- For classification:
  $$\text{Cover} = \sum \left[\text{Previous Probability}_i \times (1 - \text{Previous Probability}_i)\right]$$
- (In contrast, for regression, $\text{Cover} = \text{Number of Residuals}$).
- Parameter: Controlled by `min_child_weight`, which has a default value of $1$.
- Because an individual observation evaluated at default previous probability $0.5$ contributes $0.5 \times (1 - 0.5) = 0.25$ to cover, a leaf requires at least 4 such observations to satisfy the default minimum cover of 1. If cover is below `min_child_weight`, the split is not allowed.

### Output Value for Classification
$$\text{Output Value} = \frac{\sum \text{Residuals}}{\sum \left[\text{Previous Probability}_i \times (1 - \text{Previous Probability}_i)\right] + \lambda}$$

- When $\lambda = 0$, this matches unextreme gradient boost for classification.
- When $\lambda > 0$, the output value is shrunk closer to zero, reducing sensitivity to isolated observations.

### Updating Predictions (Log-Odds & Logistic Function)
1. **Convert initial probability to log(odds)**:
   $$\log(\text{odds}) = \log\left(\frac{P}{1 - P}\right)$$
   For default $P = 0.5$, $\log(\text{odds}) = \log(1) = 0$. (If $P = 0.75$, $\log(\text{odds}) = 1.1$).
2. **Update in log-odds space**:
   $$\text{New } \log(\text{odds}) = \text{Previous } \log(\text{odds}) + (\eta \times \text{Output Value})$$
   (Default $\eta = 0.3$).
3. **Convert back to probability via logistic function**:
   $$P = \frac{e^{\log(\text{odds})}}{1 + e^{\log(\text{odds})}}$$
   [Step-by-step mathematical derivation of converting logistic regression log-odds not fully specified in source; speaker references a separate StatQuest on logistic regression details].
4. For subsequent trees, observations have distinct previous probabilities, so the denominator of similarity and output scores uses each observation's own updated probability.

---

## Part 3: Mathematical derivation (source: xgboost-part-3-of-4-mathematical-details.md)

### Objective Function
XGBoost builds trees by minimizing an objective function consisting of a loss function and an L2 regularization term for leaf output values:

$$\text{Objective} = \sum_{i=1}^n L\left(y_i, p_i^{(0)} + O\right) + \frac{1}{2} \lambda O^2$$

- $O$ is the Output Value for the leaf.
- $p_i^{(0)}$ is the initial/previous prediction.
- $\frac{1}{2} \lambda O^2$ penalizes large leaf output values, shrinking optimal values toward zero as $\lambda$ increases.
- *(Note: The original XGBoost manuscript includes an additional term $\gamma T$, where $T$ is the number of terminal leaves and $\gamma$ is a pruning penalty. The speaker explicitly omits $\gamma T$ from this derivation because pruning occurs after the tree is built and plays no role in deriving optimal output values or similarity scores).*

### Loss Functions
- **Regression**:
  $$L(y_i, p_i) = \frac{1}{2}(y_i - p_i)^2$$
- **Classification**: Negative log-likelihood.
  [Exact formula not fully specified in source; speaker refers to Gradient Boost Part 4].

### Second-Order Taylor Approximation
To minimize the objective for any differentiable loss function, XGBoost approximates the loss using a second-order Taylor polynomial:

$$L(y_i, p_i^{(0)} + O) \approx L(y_i, p_i^{(0)}) + g_i O + \frac{1}{2} h_i O^2$$

where:
- $g_i = \frac{\partial L(y_i, p_i)}{\partial p_i}$ is the **gradient** (first derivative with respect to predicted value).
- $h_i = \frac{\partial^2 L(y_i, p_i)}{\partial p_i^2}$ is the **Hessian** (second derivative with respect to predicted value).
- [General derivation of Taylor series approximations is not fully specified in source — stated as out of scope by speaker].

Dropping the constant terms $L(y_i, p_i^{(0)})$ (which do not depend on $O$), the approximate objective to minimize is:

$$\sum_{i=1}^n \left[ g_i O + \frac{1}{2} h_i O^2 \right] + \frac{1}{2} \lambda O^2 = \left(\sum_{i=1}^n g_i\right) O + \frac{1}{2} \left(\sum_{i=1}^n h_i + \lambda\right) O^2$$

### Derivation of Optimal Output Value ($O^*$)
1. Take the first derivative with respect to $O$:
   $$\frac{d}{dO} = \sum g_i + \left(\sum h_i + \lambda\right) O$$
2. Set the derivative equal to $0$ and solve for $O$:
   $$\sum g_i + \left(\sum h_i + \lambda\right) O = 0$$
   $$O^* = -\frac{\sum g_i}{\sum h_i + \lambda}$$

### Derivation of Similarity Score
1. XGBoost substitutes the simplified objective and multiplies by $-1$ so the minimum becomes the highest point of a parabola:
   $$-\left[ \left(\sum g_i\right) O + \frac{1}{2} \left(\sum h_i + \lambda\right) O^2 \right]$$
2. Substitute $O^* = -\frac{\sum g_i}{\sum h_i + \lambda}$:
   $$-\left[ \left(\sum g_i\right) \left(-\frac{\sum g_i}{\sum h_i + \lambda}\right) + \frac{1}{2} \left(\sum h_i + \lambda\right) \left(-\frac{\sum g_i}{\sum h_i + \lambda}\right)^2 \right]$$
   $$= \frac{(\sum g_i)^2}{\sum h_i + \lambda} - \frac{1}{2} \frac{(\sum g_i)^2}{\sum h_i + \lambda} = \frac{1}{2} \frac{(\sum g_i)^2}{\sum h_i + \lambda}$$
3. In the original XGBoost manuscript, the similarity score includes the $\frac{1}{2}$ factor. In practical XGBoost implementations, the $\frac{1}{2}$ is omitted (multiplied by 2) because it is a relative metric used strictly for comparison:
   $$\text{Similarity Score} = \frac{(\sum g_i)^2}{\sum h_i + \lambda}$$

### Applying to Specific Loss Functions

#### 1. Regression
- Loss: $L = \frac{1}{2}(y_i - p_i)^2$
- First derivative (gradient): $g_i = -(y_i - p_i) = -\text{Residual}_i$
- Second derivative (Hessian): $h_i = 1$
- **Optimal Output Value**:
  $$O^* = -\frac{\sum (-\text{Residual}_i)}{\sum 1 + \lambda} = \frac{\sum \text{Residual}_i}{\text{Number of Residuals} + \lambda}$$
- **Similarity Score**:
  $$\text{Similarity Score} = \frac{(-\sum \text{Residual}_i)^2}{\sum 1 + \lambda} = \frac{(\sum \text{Residuals})^2}{\text{Number of Residuals} + \lambda}$$
- **Cover**: $\sum h_i = \text{Number of Residuals}$.

#### 2. Classification
- First derivative (gradient): $g_i = -\text{Residual}_i$
  [Derivation taking derivatives with respect to log-odds and converting back to probabilities not fully specified in source; speaker refers to Gradient Boost Part 4].
- Second derivative (Hessian): $h_i = p_i(1 - p_i)$
  [Full derivative calculation from log-odds to probabilities not fully specified in source; speaker refers to Gradient Boost Part 4].
- **Optimal Output Value**:
  $$O^* = \frac{\sum \text{Residual}_i}{\sum \left[p_i(1 - p_i)\right] + \lambda}$$
- **Similarity Score**:
  $$\text{Similarity Score} = \frac{(\sum \text{Residuals})^2}{\sum \left[p_i(1 - p_i)\right] + \lambda}$$
- **Cover**: $\sum h_i = \sum [p_i(1 - p_i)]$.

---

## Part 4: Optimization tricks (source: xgboost-part-4-of-4-crazy-cool-optimizations.md)

### 1. Approximate Greedy Algorithm
- **Greedy Algorithm**: The standard tree builder evaluates every single threshold across every feature and selects the one with the highest Gain, without looking ahead. This becomes prohibitive on large datasets with many features and observations.
- **Approximate Greedy Algorithm**: Instead of testing all possible thresholds, divides feature values into quantiles and tests only the quantiles as candidate split points.
- By default, the approximate greedy algorithm uses **about 33 quantiles**.

### 2. Parallel Learning & Quantile Sketches
- When data is too large to fit in memory, sketch algorithms create approximate solutions.
- [Exact algorithmic details of sketch algorithms are not fully specified in source — stated as out of scope by speaker].
- **Workflow**: The dataset is divided across multiple machines on a network. The quantile sketch merges values from each computer to construct an approximate histogram, which is used to calculate approximate quantiles.

### 3. Weighted Quantile Sketch
- Standard quantiles place an equal *number of observations* into each bin.
- **Weighted quantiles** place an equal *sum of weights* into each bin.
- **Weight definition**: The weight for each observation is its Hessian ($h_i$, the second derivative of the loss function, equivalent to cover):
  - **Regression**: $h_i = 1$. The weights are identical for all observations, so weighted quantiles are identical to standard quantiles with equal numbers of observations.
  - **Classification**: $h_i = p_i(1 - p_i)$.
    - When previous predicted probability is near $0.5$ (low confidence), $p(1 - p)$ is large (e.g., at $p=0.6$, weight $= 0.24$), resulting in higher weight.
    - When previous predicted probability is near $0$ or $1$ (high confidence), $p(1 - p)$ is small (e.g., at $p=0.1$ or $0.9$, weight $= 0.09$), resulting in lower weight.
  - **Impact**: Observations with low-confidence predictions receive higher weight and are placed into smaller, narrower quantiles (fewer observations per bin), allowing XGBoost to focus split resolution where prediction confidence is lowest.
- *Activation note*: XGBoost uses the approximate greedy algorithm, parallel learning, and weighted quantile sketch only when the dataset is huge; for small datasets, it defaults to the exact greedy algorithm.

### 4. Sparsity-Aware Split Finding (Missing Data Handling)
When continuous features contain missing values, XGBoost determines an optimal default path:
1. Divide observations for a feature into two tables: known values vs. missing values.
2. Sort only the known values and evaluate candidate thresholds (or quantiles).
3. For each candidate threshold:
   - Calculate $\text{Gain}_{\text{Left}}$ by placing all missing-value residuals into the left leaf.
   - Calculate $\text{Gain}_{\text{Right}}$ by placing all missing-value residuals into the right leaf.
4. Select the overall combination of threshold and direction ($\text{Gain}_{\text{Left}}$ vs $\text{Gain}_{\text{Right}}$) that maximizes total Gain.
5. The winning direction becomes the permanent **default path** for that split when evaluating future observations with missing values.

### 5. Cache-Aware Access
- Computer memory hierarchy: CPU cache memory is fastest, followed by main memory (RAM), followed by disk storage.
- XGBoost allocates gradients ($g_i$) and Hessians ($h_i$) directly into CPU cache memory so it can rapidly compute similarity scores and leaf output values.

### 6. Blocks for Out-of-Core Computation
When a dataset exceeds RAM and must reside on disk:
- **Block Compression**: Data on disk is compressed. The CPU decompressing data from disk into memory is faster than the disk reading uncompressed data.
- **Block Sharding**: When multiple disks are available, data is sharded across drives so multiple drives can read data concurrently in parallel.

### 7. Subsampling
- **Data Subsampling**: Speeds up training by building each tree using only a random subset of observations.
- **Feature Subsampling**: Speeds up tree building and reduces overfitting by selecting from a random subset of features at each split.
