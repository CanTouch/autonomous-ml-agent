---
name: svm
description: Complete guide to Support Vector Machines covering maximal margin classifiers, soft margins, support vector classifiers, the kernel trick, polynomial kernels, and radial (RBF) kernels in infinite dimensions. Extracted strictly from the 3-part StatQuest SVM series.
---

# Support Vector Machines (SVM) Methodology & Reference

> **Summary**: Support Vector Machines classify data by finding an optimal separating boundary (hyperplane). When data is separable with outliers or overlapping classes, a Support Vector Classifier creates a soft margin parameterized by cross-validation. When data cannot be separated linearly, Support Vector Machines apply the kernel trick (e.g., polynomial or radial/RBF kernels) to compute relationships between observations in higher (or infinite) dimensions without directly transforming coordinates.

---

## Part 1: Main ideas (source: support-vector-machines-part-1-of-3-main-ideas.md)

### What is Margin & Maximal Margin Classifiers
- **Margin**: The shortest distance between the observations and the decision threshold.
- Choosing the midpoint between the closest edge observations of each cluster maximizes this distance, producing a **Maximal Margin Classifier**.
- **Limitation**: Maximal margin classifiers have low bias and high variance; they are hypersensitive to outliers in the training data, leading to poor generalization on new data.

### Soft Margins & Support Vector Classifiers
- To reduce sensitivity to outliers, misclassifications must be permitted.
- **Soft Margin**: The distance between observations and the decision threshold when allowing points to fall inside the margin or on the wrong side of the threshold.
  - Soft margins introduce higher bias on training data but lower variance on unseen data.
  - The optimal number of misclassifications and observations allowed within the soft margin is selected using **cross-validation**.
- **Support Vectors**: The observations that lie on the edge of the soft margin or within the soft margin are called support vectors.
- **Support Vector Classifier Geometry across Dimensions**:
  - **1D Data**: A single point (flat affine 0-dimensional subspace).
  - **2D Data**: A line (flat affine 1-dimensional subspace).
  - **3D Data**: A plane (flat affine 2-dimensional subspace).
  - **4D or Higher Data**: A hyperplane (flat affine subspace).
  *(Note: Technically, all flat affine subspaces are hyperplanes).*

### Limitation of Linear Support Vector Classifiers
When categories overlap heavily in their raw feature space (e.g., drug effectiveness where only intermediate dosages cure a disease, while low and high dosages do not), no straight threshold, line, or flat hyperplane can separate the classes without excessive misclassifications.

### Support Vector Machines (SVM) Concept
When data cannot be linearly separated:
1. Start with data in a relatively low dimension.
2. Move the data into a higher dimension (for example, taking 1D dosage $x$ and adding a 2nd axis $y = x^2$).
3. Find a support vector classifier (line, plane, or hyperplane) that separates the higher-dimensional data into two groups.

### Kernel Functions & The Kernel Trick
- Instead of manually guessing transformations ($x^2, x^3$, etc.), SVMs use **kernel functions** to systematically find support vector classifiers in higher dimensions.
- **The Kernel Trick**: Kernel functions calculate the mathematical relationships between every pair of points *as if* they were located in a higher-dimensional space, without explicitly calculating the high-dimensional coordinates or performing the transformation.
- This dramatically reduces computation and enables finding classifiers in higher or even infinite dimensions.
- Two primary kernels introduced:
  - **Polynomial Kernel**: Systematically increases dimensions using a degree parameter $d$.
  - **Radial (RBF) Kernel**: Finds support vector classifiers in infinite dimensions; behaves like a weighted nearest-neighbor model where closer observations have strong influence and distant observations have near-zero influence.

---

## Part 2: Polynomial kernel (source: support-vector-machines-part-2-the-polynomial-kern.md)

*(Note: The general concepts of margins, support vectors, and overlapping data limitations are defined in Part 1 and referenced here).*

### Polynomial Kernel Formula
The polynomial kernel computes the relationship between two observations $a$ and $b$:

$$(a \times b + r)^d$$

- $a, b$: Two observations in the dataset.
- $r$: Determines the coefficient of the polynomial.
- $d$: Determines the degree of the polynomial.
- Both $r$ and $d$ are tuned via **cross-validation**.

### Expansion & Coordinate Dot Product Representation
The polynomial evaluates directly to a dot product representing higher-dimensional coordinates.

#### Example 1: $r = \frac{1}{2}, d = 2$
1. Expand the polynomial:
   $$\left(ab + \frac{1}{2}\right)^2 = (ab)^2 + ab + \frac{1}{4} = ab + (ab)^2 + \frac{1}{4}$$
2. Express as a dot product of coordinates for observation $a$ and observation $b$:
   $$\left(a, a^2, \frac{1}{2}\right) \cdot \left(b, b^2, \frac{1}{2}\right) = (a \cdot b) + (a^2 \cdot b^2) + \left(\frac{1}{2} \cdot \frac{1}{2}\right) = ab + (ab)^2 + \frac{1}{4}$$
   - First terms ($a, b$): $X$-axis coordinates (original values).
   - Second terms ($a^2, b^2$): $Y$-axis coordinates (squared values).
   - Third terms ($\frac{1}{2}, \frac{1}{2}$): $Z$-axis coordinates (constant value for all points, which can be ignored).

#### Example 2: $r = 1, d = 2$
1. Expand the polynomial:
   $$(ab + 1)^2 = (ab)^2 + 2ab + 1 = 2ab + (ab)^2 + 1$$
2. Express as a dot product:
   $$(\sqrt{2}a, a^2, 1) \cdot (\sqrt{2}b, b^2, 1) = 2ab + (ab)^2 + 1$$
   - New $X$-axis coordinates: $\sqrt{2} \times \text{original value}$.
   - New $Y$-axis coordinates: $\text{original value}^2$.
   - $Z$-axis coordinates: Constant $1$ (ignored).

### Calculating Relationships via Dot Products
- To determine the high-dimensional support vector classifier, the algorithm only requires the dot products between each pair of observations.
- Because $(ab + r)^d$ equals this dot product, evaluating the kernel on any pair of observations (e.g., yielding $16,002.25$ in the video's numeric example) gives the high-dimensional relationship directly without converting data points into higher dimensions.
- [Why the optimization only requires the pairwise dot products to solve for the classifier is not fully specified in source; stated as out of scope by speaker].

---

## Part 3: Radial/RBF kernel (source: support-vector-machines-part-3-the-radial-rbf-kern.md)

*(Note: The general concepts of margins, support vectors, and kernel functions are defined in Part 1 and referenced here).*

### Radial (RBF) Kernel Formula
The Radial Basis Function (RBF) kernel evaluates the relationship between two observations $a$ and $b$:

$$e^{-\gamma (a - b)^2}$$

- $(a - b)^2$: The squared distance between observation $a$ and observation $b$.
- $\gamma$ (gamma): A scaling parameter determined via **cross-validation** that scales the squared distance and controls the influence points have on each other.

### Behavior as a Weighted Nearest-Neighbor
- The kernel output reflects the amount of influence two observations have on each other:
  - When observations are close, $(a - b)^2$ is small, and the kernel output is relatively large (e.g., in the video's numeric example, two close observations yield $0.11$ when $\gamma = 1$).
  - When observations are far apart, $(a - b)^2$ is large, and the kernel output approaches $0$.
- **Effect of Gamma ($\gamma$)**:
  - Increasing $\gamma$ (e.g., from $\gamma = 1$ to $\gamma = 2$) scales up the exponent, reducing the kernel value (from $0.11$ down to $0.01$). Higher $\gamma$ narrows the geographic span of influence.

### Mathematical Derivation of Infinite Dimensions via Taylor Series
The video provides an explicit proof demonstrating that the radial kernel is equivalent to an infinite-dimensional dot product.

#### Step 1: Simplifying with $\gamma = \frac{1}{2}$
Set $\gamma = \frac{1}{2}$ to remove the coefficient 2:
$$e^{-\frac{1}{2}(a - b)^2} = e^{-\frac{1}{2}(a^2 - 2ab + b^2)} = e^{-\frac{1}{2}a^2} e^{-\frac{1}{2}b^2} e^{ab}$$

#### Step 2: Taylor Series Expansion of $e^x$
A function $f(x)$ expanded in a Taylor series around point $A$ is:
$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(A)}{n!} (x - A)^n$$
For $f(x) = e^x$, all derivatives equal $e^x$. Setting $A = 0$ (since $e^0 = 1$ exists):
$$e^x = 1 + \frac{x}{1!} + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots = \sum_{n=0}^\infty \frac{x^n}{n!}$$

#### Step 3: Expanding $e^{ab}$
Substituting $x = ab$ into the Taylor series:
$$e^{ab} = 1 + ab + \frac{(ab)^2}{2!} + \frac{(ab)^3}{3!} + \dots$$

#### Step 4: Connecting to Polynomial Kernels ($r = 0, d \to \infty$)
Notice that a polynomial kernel $(ab + r)^d$ with $r = 0$:
- For $d = 0$: $(ab + 0)^0 = 1$
- For $d = 1$: $(ab + 0)^1 = ab$
- For $d = 2$: $(ab + 0)^2 = (ab)^2$
- For degree $n$: $(ab)^n$
Thus, each term in the Taylor expansion of $e^{ab}$ contains a polynomial kernel with $r = 0$ as $d$ ranges from $0$ to $\infty$.

#### Step 5: Expressing as an Infinite Dot Product
The expansion of $e^{ab}$ can be written as a dot product:
$$e^{ab} = \left(1, \frac{a}{\sqrt{1!}}, \frac{a^2}{\sqrt{2!}}, \frac{a^3}{\sqrt{3!}}, \dots\right) \cdot \left(1, \frac{b}{\sqrt{1!}}, \frac{b^2}{\sqrt{2!}}, \frac{b^3}{\sqrt{3!}}, \dots\right)$$

#### Step 6: Completing the Radial Kernel
Returning to the full expression:
$$e^{-\frac{1}{2}(a - b)^2} = \left(e^{-\frac{1}{2}a^2} e^{-\frac{1}{2}b^2}\right) \left[ \left(1, \frac{a}{\sqrt{1!}}, \frac{a^2}{\sqrt{2!}}, \dots\right) \cdot \left(1, \frac{b}{\sqrt{1!}}, \frac{b^2}{\sqrt{2!}}, \dots\right) \right]$$
Let $S = \sqrt{e^{-\frac{1}{2}a^2} e^{-\frac{1}{2}b^2}}$ and distribute $S$ into both vectors of the dot product.
This demonstrates that the Radial (RBF) Kernel is mathematically identical to an inner dot product with coordinates for an **infinite number of dimensions**. Calculating $e^{-\gamma(a - b)^2}$ directly computes the high-dimensional relationship between points in infinite-dimensional space without evaluating coordinates.
