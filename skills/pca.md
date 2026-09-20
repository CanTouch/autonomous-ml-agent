---
name: pca
description: Dimensionality reduction technique that projects high-dimensional data onto orthogonal axes of maximal variance (principal components) using Singular Value Decomposition, evaluated via eigenvalues and scree plots. Extracted strictly from StatQuest.
---

# Principal Component Analysis (PCA) Methodology & Reference

> **Summary**: Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique computed via Singular Value Decomposition (SVD). It centers the data at the origin and identifies orthogonal axes—Principal Components—that maximize the sum of squared distances of projected points from the origin (maximizing explained variance). The proportion of total variance explained by each component is plotted on a scree plot to determine how well a low-dimensional representation captures the data.

---

## 1. Core Technique Taught

### Overview & Motivation
- **Dimensionality Reduction**: Visualizing samples across 4 or more measured variables requires 4 or more dimensions, which cannot be graphed directly. PCA projects $N$-dimensional data into a 2D or lower-dimensional plot while preserving clusters of similar samples.
- **Variable Importance**: PCA determines which variables contribute most to the variance along each principal component.
- **Number of Components**: In theory, there is 1 Principal Component per variable. In practice, the maximum number of PCs is:
  $$\text{Number of PCs} = \min(\text{Number of Variables}, \text{Number of Samples})$$
  [Mathematical explanation for why this is the case is not fully specified in source; speaker notes it is covered in a separate video].

### Step-by-Step Procedure (using SVD)

1. **Center the Data**:
   - Compute the mean for each variable (e.g., Gene 1 and Gene 2).
   - Shift the data so the center (mean) lies directly on the origin $(0, 0)$. Shifting preserves the relative spatial relationships between points.

2. **Fit Principal Component 1 (PC1)**:
   - Draw a line through the origin and rotate it to find the line of best fit.
   - **Optimization Criterion**: For any point with fixed distance $A$ to the origin, the distance to the line ($B$) and the distance from the projected point to the origin ($C$) satisfy the Pythagorean theorem:
     $$A^2 = B^2 + C^2$$
     Because $A^2$ is fixed, minimizing the perpendicular distance to the line ($B$) is mathematically equivalent to maximizing the distance from the projected point to the origin ($C$).
   - PCA finds the best-fitting line by **maximizing the sum of squared distances (SS distances)** between the projected points and the origin:
     $$\text{SS Distances} = \sum_{i=1}^n d_i^2$$
     (Distances are squared so negative values do not cancel positive values).
   - The line that maximizes this sum is **PC1**.

3. **Compute Eigenvector (Singular Vector) & Loading Scores**:
   - The slope of PC1 represents a linear combination of variables (e.g., 4 parts Gene 1 to 1 part Gene 2).
   - The vector length is computed via the Pythagorean theorem:
     $$\text{Length} = \sqrt{4^2 + 1^2} = 4.12$$
   - To convert this into a **unit vector** (length = 1), divide each component by the total length:
     $$\text{Component 1} = \frac{4}{4.12} \approx 0.97, \quad \text{Component 2} = \frac{1}{4.12} \approx 0.242$$
   - This 1-unit vector is called the **Singular Vector** or **Eigenvector for PC1**.
   - The individual proportions ($0.97, 0.242$) are called the **loading scores**.

4. **Identify PC2 and Subsequent Components**:
   - In 2D, **PC2** is simply the line through the origin that is perpendicular to PC1 (no further optimization needed).
   - For PC2, the recipe flips with a negative sign (e.g., $-1$ part Gene 1 to $4$ parts Gene 2; unit vector loading scores: $-0.242$ Gene 1 and $0.97$ Gene 2).
   - For 3 or more dimensions, PC3 is the line through the origin perpendicular to both PC1 and PC2 that maximizes the remaining projected sum of squared distances.

5. **Construct the 2D PCA Plot**:
   - Rotate the axes so PC1 is horizontal and PC2 is vertical.
   - Project original sample points onto PC1 and PC2; the projected positions define the sample coordinates in the new 2D plot.

---

## 2. Mathematical Definitions & Terminology

- **Eigenvalue**: The average of the sums of squared distances for the best-fit line (represents the variation captured by that PC).
- **Singular Value**: The square root of the sum of squared distances for that PC:
  $$\text{Singular Value} = \sqrt{\text{SS Distances}}$$
- **Eigenvector / Singular Vector**: The unit-length vector along the PC line.
- **Loading Scores**: The coefficients (elements) of the eigenvector describing the relative weight of each variable.

---

## 3. Scree Plots & Explained Variance

### Calculating Explained Variance
- Because eigenvalues represent variation, the percentage of total variation explained by $\text{PC}_k$ is:
  $$\text{Variation Explained by } \text{PC}_k = \frac{\text{Eigenvalue}_k}{\sum_i \text{Eigenvalue}_i}$$
  *(Example from video: if PC1 variation = 15 and PC2 variation = 3, total variation = 18. PC1 accounts for $\frac{15}{18} = 83\%$, and PC2 accounts for $\frac{3}{18} = 17\%$).*

### Scree Plot Interpretation
- A **scree plot** is a bar/scatter chart displaying the percentage of variation accounted for by each Principal Component.
- **Assessing Plot Accuracy**:
  - If the cumulative variance of the first two PCs is very high (e.g., $90\%$ or $94\%$), the 2D PCA plot is an accurate representation of the high-dimensional data.
  - If later PCs (e.g., PC3, PC4) account for substantial variation, a 2D projection is noisy, but can still be used to identify clusters of similar samples.
