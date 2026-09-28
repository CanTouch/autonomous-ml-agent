---
name: gradient-boost
description: Complete guide to Gradient Boost covering regression and classification algorithms, mathematical derivations using loss function gradients, Taylor polynomial approximations, pseudo-residuals, learning rate scaling, and leaf output value calculations. Extracted strictly from the 4-part StatQuest Gradient Boost series.
---

# Gradient Boost Methodology & Reference

> **Summary**: Gradient Boost is an ensemble machine learning algorithm that iteratively builds fixed-size decision trees to predict pseudo-residuals (the negative gradients of a differentiable loss function). For regression, it initializes with the mean target value and minimizes squared error loss; for classification, it initializes with log-odds and minimizes negative log-likelihood loss using second-order Taylor approximations for leaf outputs. Each tree is scaled by a learning rate $\nu$ to ensure low variance on unseen data.

---

## Part 1: Regression main ideas (source: gradient-boost-part-1-of-4-regression-main-ideas.md)

### Overview & Comparison to AdaBoost
- **Task**: Predicting a continuous target variable (Regression) using an ensemble of decision trees.
- **Comparison to AdaBoost**:
  - *Starting state*: AdaBoost starts with a short decision tree (stump); Gradient Boost starts with a single leaf representing the average value of the target.
  - *Tree size*: AdaBoost builds single-split stumps; Gradient Boost builds trees with multiple leaves (restricted in practice, commonly between 8 and 32 leaves).
  - *Scaling*: AdaBoost scales each stump differently based on error compensation ("amount of say"); Gradient Boost scales all trees equally using a fixed learning rate.

### Step-by-Step Procedure for Regression
1. **Initialize Base Prediction**:
   - Compute the average of the observed continuous target values:
     $$F_0 = \text{Average}(y)$$
   - This single leaf serves as the baseline prediction for all samples.
2. **Calculate Pseudo-Residuals**:
   - Compute the difference between each observed value and the current prediction:
     $$\text{Pseudo-Residual} = \text{Observed} - \text{Predicted}$$
3. **Fit a Tree to Predict Residuals**:
   - Build a regression tree using feature variables to predict the calculated pseudo-residuals.
   - Restrict the tree size (e.g., maximum 8 to 32 leaves).
4. **Calculate Leaf Output Values**:
   - For leaves containing multiple samples, replace the residuals with their average value:
     $$\text{Leaf Output} = \text{Average}(\text{Residuals in Leaf})$$
5. **Update Model with Learning Rate**:
   - Scale the contribution of the newly built tree by a learning rate ($\nu \in (0, 1]$, commonly $0.1$):
     $$\text{New Prediction} = \text{Previous Prediction} + (\text{Learning Rate} \times \text{Leaf Output})$$
   - *Jerome Friedman Heuristic*: Taking many small steps scaled by a learning rate lowers model variance and improves generalization on test data.
6. **Iterate**:
   - Recompute pseudo-residuals from the updated predictions.
   - Fit a new tree on the new residuals, scale by the learning rate, and add to the ensemble.
   - Continue until reaching a specified number of trees or when additional trees fail to reduce residuals significantly.
7. **Predicting Unseen Data**:
   - Sum the initial base prediction and the scaled outputs from all sequential trees in the ensemble.

---

## Part 2: Regression details (source: gradient-boost-part-2-of-4-regression-details.md)

*(Note: Core regression concepts, pseudo-residuals, learning rate scaling, and leaf averaging are defined in Part 1 and referenced here).*

### Mathematical Formulation & Derivation

#### Input & Loss Function
- **Input Data**: Training set $\{(x_i, y_i)\}_{i=1}^n$, where $x_i$ represents feature rows and $y_i$ represents observed target values.
- **Differentiable Loss Function**:
  $$L(y_i, f(x)) = \frac{1}{2}(y_i - f(x))^2$$
  *(The factor of $\frac{1}{2}$ simplifies the derivative during differentiation without altering the optimal solution)*.
- **Loss Derivative**:
  $$\frac{\partial L(y_i, f(x))}{\partial f(x)} = -(y_i - f(x))$$
  - The negative derivative (negative gradient) equals $(y_i - f(x))$, which is the residual.

#### Algorithmic Steps
1. **Step 1: Initialize Model with Constant Value**:
   $$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$$
   - Differentiating with respect to $\gamma$ and setting to zero yields:
     $$\sum_{i=1}^n -(y_i - \gamma) = 0 \implies \gamma = \frac{1}{n}\sum_{i=1}^n y_i = \text{Average}(y)$$
   - Output $F_0(x) = \text{Average}(y)$.

2. **Step 2: Iterative Tree Building ($m = 1 \dots M$)**:
   *(Where $M$ is the number of trees, commonly set to 100)*.
   - **(a) Compute Pseudo-Residuals ($r_{im}$)**:
     $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)} = y_i - F_{m-1}(x_i), \quad \text{for } i = 1 \dots n$$
   - **(b) Fit Regression Tree**:
     - Fit a regression tree to the targets $r_{im}$ using features $x_i$.
     - Define terminal regions (leaves) $R_{jm}$ for $j = 1 \dots J_m$ (where $J_m$ is the number of leaves in tree $m$).
   - **(c) Compute Leaf Output Values ($\gamma_{jm}$)**:
     - For each leaf $j$, find $\gamma_{jm}$ that minimizes the loss:
       $$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$
     - Differentiating and solving yields:
       $$\gamma_{jm} = \text{Average}(r_{im} \text{ for all } x_i \in R_{jm})$$
   - **(d) Update Model ($F_m(x)$)**:
     $$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$
     - Where $\nu$ is the learning rate ($0 < \nu \le 1$).

3. **Step 3: Final Output Model**:
   $$F_M(x) = F_0(x) + \nu \sum_{m=1}^M \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$

---

## Part 3: Classification main ideas (source: gradient-boost-part-3-of-4-classification.md)

### Classification Concepts & Setup
- **Task**: Classifying samples into discrete binary categories (e.g., $0$ or $1$).
- **Log-Odds Baseline**: Instead of the mean, the baseline prediction is the log of the odds:
  $$\text{Log-Odds} = \log\left(\frac{\text{Count}(\text{Positive Class})}{\text{Count}(\text{Negative Class})}\right)$$
- **Converting Log-Odds to Probability**:
  $$P(\text{Class} = 1) = \frac{e^{\text{Log-Odds}}}{1 + e^{\text{Log-Odds}}}$$
- **Classification Threshold**: Default threshold is $P \ge 0.5$ (can be adjusted via ROC/AUC analysis).

### Step-by-Step Procedure for Classification
1. **Initialize Base Prediction**:
   - Compute initial log-odds from training counts and store in initial leaf $F_0$.
   - Convert initial log-odds to initial baseline probability $P_0$.
2. **Calculate Pseudo-Residuals**:
   - Pseudo-residuals are calculated on the probability scale:
     $$\text{Residual}_i = \text{Observed}_i - P_i$$
     *(where $\text{Observed}_i = 1$ for positive class, $0$ for negative class)*.
3. **Fit a Tree to Predict Residuals**:
   - Build a fixed-size regression tree to predict the calculated probability residuals.
4. **Compute Leaf Output Values (Log-Odds Transformation)**:
   - Because tree predictions are added to log-odds while residuals are differences in probabilities, leaf outputs are transformed using:
     $$\text{Leaf Output} = \frac{\sum \text{Residual}_i}{\sum [P_{\text{previous}, i} \times (1 - P_{\text{previous}, i})]}$$
5. **Update Log-Odds Prediction**:
   $$\text{New Log-Odds}_i = \text{Previous Log-Odds}_i + (\nu \times \text{Leaf Output})$$
6. **Iterate**:
   - Convert new log-odds to updated probabilities $P_i$.
   - Recompute residuals $(\text{Observed}_i - P_i)$ and fit the next tree.
7. **Predicting New Samples**:
   - Compute final predicted log-odds: $\text{Log-Odds} = F_0 + \nu \sum_{m=1}^M \text{Tree}_m(\text{sample})$.
   - Convert final log-odds to probability via the logistic function and compare against decision threshold.

---

## Part 4: Classification details (source: gradient-boost-part-4-of-4-classification-details.md)

*(Note: Core classification setup, log-odds transformations, and residual calculations are defined in Part 3 and referenced here).*

### Mathematical Derivation for Classification

#### Loss Function (Negative Log-Likelihood)
- **Log-Likelihood Function**:
  $$\text{Log-Likelihood} = \sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \right]$$
- **Loss Function in Log-Odds**:
  To minimize loss rather than maximize likelihood, multiply by $-1$ and substitute $p = \frac{e^{\log(\text{odds})}}{1 + e^{\log(\text{odds})}}$:
  $$L(y_i, \log(\text{odds})) = -y_i \cdot \log(\text{odds}) + \log(1 + e^{\log(\text{odds})})$$

#### Loss Function Derivatives
- **First Derivative (Gradient)**:
  $$\frac{\partial L}{\partial \log(\text{odds})} = -y_i + \frac{e^{\log(\text{odds})}}{1 + e^{\log(\text{odds})}} = -(y_i - p_i)$$
  - Multiplying by $-1$ gives the pseudo-residual $r_{im} = y_i - p_i$.
- **Second Derivative (Hessian)**:
  $$\frac{\partial^2 L}{\partial \log(\text{odds})^2} = p_i(1 - p_i)$$

#### Step-by-Step Algorithm for Classification
1. **Step 1: Initialize Model with Constant Log-Odds**:
   $$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$$
   - Setting derivative to 0 gives initial probability $p = \frac{\sum y_i}{n}$ and initial log-odds $F_0(x) = \log\left(\frac{\sum y_i}{\sum (1 - y_i)}\right)$.

2. **Step 2: Iterative Tree Building ($m = 1 \dots M$)**:
   - **(a) Compute Pseudo-Residuals**:
     $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)} = y_i - p_{i, m-1}, \quad \text{where } p_{i, m-1} = \frac{e^{F_{m-1}(x_i)}}{1 + e^{F_{m-1}(x_i)}}$$
   - **(b) Fit Regression Tree**:
     - Fit a regression tree to targets $r_{im}$, yielding terminal regions $R_{jm}$ for $j = 1 \dots J_m$.
   - **(c) Compute Leaf Output Values ($\gamma_{jm}$) via Second-Order Taylor Approximation**:
     - Approximating $L(y_i, F_{m-1}(x_i) + \gamma)$ with a second-order Taylor polynomial:
       $$L(y_i, F_{m-1}(x_i) + \gamma) \approx L(y_i, F_{m-1}(x_i)) + \gamma \left[\frac{\partial L}{\partial F}\right] + \frac{\gamma^2}{2}\left[\frac{\partial^2 L}{\partial F^2}\right]$$
     - Differentiating with respect to $\gamma$ and setting to zero yields:
       $$\sum_{x_i \in R_{jm}} \left[ -(y_i - p_{i, m-1}) + \gamma \cdot p_{i, m-1}(1 - p_{i, m-1}) \right] = 0$$
       $$\gamma_{jm} = \frac{\sum_{x_i \in R_{jm}} (y_i - p_{i, m-1})}{\sum_{x_i \in R_{jm}} p_{i, m-1}(1 - p_{i, m-1})} = \frac{\sum_{x_i \in R_{jm}} r_{im}}{\sum_{x_i \in R_{jm}} p_{i, m-1}(1 - p_{i, m-1})}$$
   - **(d) Update Model**:
     $$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$

3. **Step 3: Final Classification**:
   - Predict probability: $P(Y = 1 | x) = \frac{e^{F_M(x)}}{1 + e^{F_M(x)}}$.
   - Classify as positive class if $P \ge \text{threshold}$ (default $0.5$).
