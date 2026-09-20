---
name: hierarchical-clustering
description: Agglomerative clustering method that builds a dendrogram by iteratively merging the most similar items or clusters based on chosen distance metrics and linkage criteria. Extracted strictly from StatQuest.
---

# Hierarchical Clustering Methodology & Reference

> **Summary**: Hierarchical clustering is an unsupervised agglomerative algorithm that iteratively merges the most similar observations or clusters until all data points belong to a single hierarchy. It is commonly displayed alongside heat maps and dendrograms, where branch lengths indicate the order of formation and degree of similarity. The clustering structure depends on the chosen pairwise distance metric (e.g., Euclidean or Manhattan) and the cluster linkage criterion (e.g., centroid, single, or complete linkage).

---

## 1. Core Technique Taught

### Overview & Visual Representation
- **Purpose**: Orders rows (e.g., genes) and/or columns (e.g., samples) by similarity to reveal correlations in multi-dimensional data.
- **Heat Maps & Dendrograms**:
  - Unclustered heat maps display raw data order; hierarchical clustering reorganizes rows/columns so similar patterns are adjacent.
  - A **dendrogram** visualizes the hierarchy:
    - **Branch Height / Length**: Reflects both the order of formation and degree of similarity.
    - Shorter branches represent clusters formed first (highest similarity).
    - Longest branches represent clusters formed last (lowest similarity).

### Step-by-Step Agglomerative Algorithm
1. **Compute Pairwise Similarity**: Compare every item to every other item using a defined distance metric.
2. **Merge Closest Items**: Identify the two items with the smallest distance (highest similarity) and merge them into the first cluster.
3. **Update Representation**: Treat the newly formed cluster as a single composite item.
4. **Iterate**: Recompute similarities between all items and clusters, find the closest pair, and merge them.
5. **Terminate**: Repeat the process until all items are merged into one final cluster.

---

## 2. Distance Metrics

The method for defining similarity between individual items is user-selected:

### 1. Euclidean Distance
Computed via the Pythagorean theorem:
- **2 Dimensions (e.g., 2 samples)**:
  $$\text{Distance} = \sqrt{(\text{Sample}_1 \text{ difference})^2 + (\text{Sample}_2 \text{ difference})^2}$$
  *(Numeric example from video: Difference in sample 1 = $1.6 - (-0.5) = 2.1$; difference in sample 2 = $0.5 - (-1.9) = 2.4$. Distance $= \sqrt{2.1^2 + 2.4^2} = 3.2$).*
- **More than 2 Dimensions**:
  Extended by summing the squared differences across all samples under the square root:
  $$\text{Distance} = \sqrt{\sum_{i=1}^n (\text{Sample}_i \text{ difference})^2}$$

### 2. Manhattan Distance
- Defined as the sum of the absolute values of the differences:
  $$\text{Distance} = \sum_{i=1}^n |\text{Sample}_i \text{ difference}|$$
- **Geometric Interpretation**: Can be visualized as placing individual difference line segments together head-to-tail and measuring the total combined length.

### Choosing a Distance Metric
- The choice of distance metric is arbitrary (neither is universally superior); practitioners should select the metric that provides the clearest insight into their specific dataset.

---

## 3. Linkage Methods (Comparing Clusters)

When comparing a single point to an existing cluster, or comparing two clusters to each other, different **linkage methods** can be used:

1. **Centroid**:
   - Compares items to the **average** of the measurements in each cluster (the cluster center/centroid).
2. **Single Linkage**:
   - Compares items to the **closest point** in each cluster.
3. **Complete Linkage**:
   - Compares items to the **furthest point** in each cluster.
   - *(Note: Complete linkage is the default setting for R's `hclust` function).*

Different linkage methods produce slight differences in heat map clustering and dendrogram branch layouts.
