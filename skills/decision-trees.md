---
name: decision-trees
description: Complete guide to Classification Trees, Gini impurity calculations, numeric feature splitting, automatic feature selection, and missing data imputation strategies. Extracted strictly from StatQuest Decision Trees Parts 1 and 2.
---

# Decision Trees Methodology & Reference

> **Summary**: Decision trees make predictions by evaluating feature statements from a root node down through internal branches to terminal leaf nodes. Classification trees split data to minimize leaf impurity using metrics like Gini impurity. Features that fail to reduce impurity are automatically excluded, and missing values can be imputed using mode/mean/median or regression against correlated features.

---

## Part 1: Classification trees (source: decision-and-classification-trees-clearly-explaine.md)

### Tree Structure & Terminology
- **Classification Tree vs Regression Tree**: A decision tree that classifies samples into categories is a *classification tree*; a decision tree that predicts continuous numeric values is a *regression tree*.
- **Mixing Data Types**: A single tree can mix continuous numeric data and categorical (e.g., Yes/No) features. The same numeric feature can be tested multiple times with different thresholds at different depths.
- **Traversal**: Evaluation starts at the top and moves downward until reaching a terminal node. By convention, if a condition statement is true, follow the branch to the left; if false, follow to the right.
- **Node Definitions**:
  - **Root Node (Root)**: The topmost node of the tree with no incoming arrows, where the first decision is evaluated.
  - **Internal Nodes (Branches)**: Intermediate decision nodes with arrows pointing to them and arrows pointing away from them.
  - **Leaf Nodes (Leaves)**: Terminal nodes with incoming arrows but no outgoing arrows. They provide the final prediction/classification.

### Impurity & Gini Impurity
- **Purity**: A leaf containing only a single class is *pure*. A leaf containing a mixture of classes is *impure*.
- **Metrics for Quantifying Impurity**: The primary metric covered is **Gini Impurity**. (The speaker notes alternative metrics exist, such as *Entropy* and *Information Gain*, which are numerically similar).
- **Single Leaf Gini Impurity Formula**:
  $$\text{Gini Impurity} = 1 - P(\text{Class}_1)^2 - P(\text{Class}_2)^2$$
  *(For a binary classification, $1 - (\text{probability of Yes})^2 - (\text{probability of No})^2$)*.
  - A completely pure leaf ($100\%$ one class) has a Gini impurity of $1 - 1^2 - 0^2 = 0$.
- **Total Feature Gini Impurity (Weighted Average)**:
  Because candidate child leaves often contain different numbers of samples, the total impurity for a split is the weighted average of the individual leaf impurities:
  $$\text{Total Gini Impurity} = \left(\frac{N_{\text{left}}}{N_{\text{total}}}\right) \times \text{Gini}_{\text{left}} + \left(\frac{N_{\text{right}}}{N_{\text{total}}}\right) \times \text{Gini}_{\text{right}}$$

### Splitting Continuous Numeric Features
1. **Sort**: Sort the numeric values in ascending order.
2. **Calculate Adjacent Averages**: Compute candidate split thresholds by calculating the average between each pair of adjacent values:
   $$\text{Candidate Threshold}_i = \frac{\text{Value}_i + \text{Value}_{i+1}}{2}$$
3. **Evaluate Each Threshold**: For each candidate average, place the condition (e.g., $\text{Feature} < \text{Threshold}$) in the node, split all samples, and calculate the weighted total Gini impurity.
4. **Select Lowest Impurity**: Choose the candidate threshold that yields the lowest Gini impurity. If there is a tie, either threshold can be chosen.

### Step-by-Step Tree Construction
1. **Calculate Impurity for All Candidate Features**:
   - For categorical features, compute Gini impurity directly from the split.
   - For numeric features, evaluate all adjacent midpoints and find the best threshold.
2. **Select the Root Node**: Place the feature (or feature threshold) with the lowest overall Gini impurity at the root.
3. **Split Data**: Route samples into left (True) and right (False) child nodes.
4. **Recursive Splitting**: For each impure child node, re-evaluate all remaining candidate features and thresholds using only the subset of samples that reached that node. Split the node using the feature/threshold with the lowest impurity.
5. **Terminal Criteria (Leaves)**: When further splitting does not reduce impurity, or when leaves become pure, stop splitting and declare the node a leaf.
6. **Assign Leaf Output Class**: The output classification assigned to a leaf is the majority class of the samples residing in that leaf.

### Overfitting & Mitigation Strategies
- **Overfitting**: Occurs when a tree fits training data too closely (e.g., creating leaves containing only 1 sample), resulting in poor predictive performance on unseen data.
- **Mitigation Techniques**:
  1. **Pruning**: [referenced as a separate topic in the source].
  2. **Minimum Samples per Leaf**: Require a minimum number of samples (e.g., 3 or more) to form a leaf.
  3. **Cross-Validation**: Because the optimal minimum sample threshold is unknown in advance, test candidate thresholds via cross-validation and choose the value that performs best.

---

## Part 2: Feature selection and missing data (source: statquest-decision-trees-part-2-feature-selection-.md)

*(Note: Tree structure, impurity metrics, and overfitting concepts are established in Part 1 and referenced here).*

### Automatic Feature Selection
- **Natural Selection via Impurity**: During tree construction, candidate splits are evaluated by comparing the impurity of splitting on a feature against the impurity of not splitting (leaving as a leaf). If a feature never yields a reduction in impurity at any branch, it is never included in the tree.
- **Impurity Reduction Threshold**: A minimum threshold can be enforced requiring each split to achieve a sufficiently large reduction in impurity. This simplifies the tree structure and prevents overfitting.

### Handling Missing Data
When samples in the dataset have missing values, decision trees can handle them using several strategies:

1. **Categorical / Discrete Missing Values**:
   - **Most Common Value (Mode)**: Replace the missing value with the most frequent category across that column.
   - **Correlated Column Matching**: Identify another column with the highest correlation to the column with missing data, and copy the corresponding value/state from that correlated column for that sample.

2. **Continuous / Numeric Missing Values**:
   - **Mean or Median**: Replace missing numeric values with the column mean or median.
   - **Linear Regression Imputation**: Identify another numeric column that has the highest correlation with the missing feature, fit a linear regression model (least squares line) between the two features, and predict the missing value from the correlated feature.
