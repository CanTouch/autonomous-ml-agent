---
name: k-means-clustering
description: Partitioning algorithm that clusters data into K groups by iteratively assigning points to the nearest mean and recalculating cluster centers, selecting K via an elbow plot. Extracted strictly from StatQuest.
---

# K-Means Clustering Methodology & Reference

> **Summary**: K-means clustering is an iterative unsupervised algorithm that clusters observations into a user-specified number of groups ($K$). It randomly initializes $K$ cluster centers, assigns points to the nearest cluster based on Euclidean distance, recalculates the mean of each cluster, and repeats until cluster assignments stabilize. The optimal $K$ is determined by evaluating the within-cluster variation across candidate $K$ values using an elbow plot.

---

## 1. Core Technique Taught

### Overview & Concepts
- **Goal**: Group raw, unlabeled data into $K$ distinct clusters without relying on visual inspection.
- **Comparison with Hierarchical Clustering**: K-means specifically partitions data into the exact number of clusters ($K$) you specify up front, whereas hierarchical clustering evaluates pairwise similarity across all samples.
- **Data representations**: Can be applied to points on a 1D number line, 2D $(X, Y)$ scatter plots, multi-dimensional tables, or heat maps. Data does not need to be visually plotted—clustering relies strictly on pairwise distances between observations.

### Step-by-Step Algorithm
1. **Select $K$**: Choose the number of clusters to identify (e.g., $K = 3$).
2. **Initialize Centers**: Randomly select $K$ distinct data points from the dataset as initial cluster centers.
3. **Measure Distances**: Calculate the distance from each remaining data point to each of the $K$ cluster centers.
4. **Assign to Nearest Cluster**: Assign each point to the cluster center it is closest to.
5. **Recalculate Means**: Once all points are assigned, calculate the mean (center) of each cluster.
6. **Iterate**: Re-measure distances from every point to the new cluster means, and re-assign points to the nearest mean.
7. **Convergence**: Repeat step 6 until cluster assignments no longer change between iterations.

---

## 2. Mathematical Formulas & Distance Metrics

### Distance Metric (Euclidean Distance)
Distance is computed using Euclidean distance (the Pythagorean theorem in 2D):
- **2 Dimensions (or 2 samples/axes)**:
  $$\text{Distance} = \sqrt{x^2 + y^2}$$
- **3 Dimensions (or 3 samples/axes)**:
  $$\text{Distance} = \sqrt{x^2 + y^2 + z^2}$$
- **4 Dimensions (or 4 samples/axes)**:
  $$\text{Distance} = \sqrt{x^2 + y^2 + z^2 + a^2}$$
- **Higher Dimensions**:
  [not fully specified in source — speaker notes "Et cetera, et cetera, et cetera"]

---

## 3. Selecting $K$ & Evaluation

### Within-Cluster Variation & Multiple Runs
- The quality of a clustering result is assessed by summing the variation (variance) within each cluster:
  $$\text{Total Variation} = \sum_{\text{clusters}} \text{Variation within cluster}$$
  [Exact mathematical formula for within-cluster variation is not fully specified in source; described conceptually as adding up the variation within each cluster].
- **Local Optima & Random Restarts**: Because initial random selections can lead to poor clusters, the computer repeats the entire clustering procedure multiple times with different random starting points, tracks the total variation for each run, and selects the run with the lowest total within-cluster variation.

### The Elbow Plot
To choose the best value for $K$:
1. Fit K-means across a range of $K$ values (e.g., $K = 1, 2, 3, 4, \dots$).
   - $K = 1$ has the largest total variation.
   - Each time $K$ increases, the total within-cluster variation decreases (reaching 0 when every point is its own cluster).
2. Plot the reduction in variation against each value of $K$.
3. Identify the **elbow in the plot**: the point where the rate of variation reduction drops off sharply. Choose this value as the optimal $K$.
