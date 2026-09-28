---
name: random-forests
description: Complete guide to Random Forests covering bootstrap aggregating (bagging), random feature subsets, Out-of-Bag (OOB) error estimation, parameter tuning (ntree, mtry), missing data imputation via rfImpute, and MDS clustering with proximity matrices. Extracted strictly from StatQuest Random Forests Parts 1 and 2.
---

# Random Forests Methodology & Reference

> **Summary**: Random Forests combine bootstrap aggregation (bagging) with random feature selection at each split to build an ensemble of diverse decision trees. Predictions are made via majority vote (or averaging in regression). Model accuracy is evaluated using Out-of-Bag (OOB) samples without a separate validation set, and sample similarities can be visualized via proximity matrices and MDS scaling.

---

## Part 1: Building, using, evaluating (source: statquest-random-forests-part-1-building-using-and.md)

### Motivation
- **Individual Decision Tree Limitation**: Individual decision trees are easy to build and interpret, but suffer from high variance and inaccuracy on new samples (they fit training data well but lack flexibility for generalizing).
- **Random Forest Advantage**: Combines the simplicity of decision trees with the flexibility of ensembling, yielding a substantial improvement in predictive accuracy.

### Algorithm: Building a Random Forest
1. **Create a Bootstrapped Dataset**:
   - Randomly select $N$ samples from the original dataset of size $N$ with replacement (allowing the same sample to be selected more than once).
2. **Build a Decision Tree with Random Feature Subsets**:
   - At each split (including the root node and all internal nodes), randomly select a subset of candidate variables/columns (e.g., 2 variables instead of all available features).
   - Evaluate only that random subset to choose the best splitting variable and threshold.
3. **Repeat**:
   - Repeat steps 1 and 2 hundreds of times to create a forest of distinct trees.
   - The combination of bootstrapping samples and selecting random feature subsets produces variety across trees, which makes the ensemble robust and accurate.

### Making Predictions
- To predict the class of a new observation, run the observation down every tree in the forest.
- Tally the predictions from all trees.
- Assign the class that receives the majority vote.
- **Terminology**: The combination of bootstrapping datasets and aggregating predictions across the ensemble is called **Bagging** (Bootstrap Aggregating).

### Out-of-Bag (OOB) Evaluation
- **Out-of-Bag Dataset**: When creating each bootstrap dataset, approximately **one-third** ($1/3$) of the original samples are left out. These unselected samples form the *Out-of-Bag (OOB)* dataset for that specific tree.
- **OOB Error Estimation**:
  1. Take an original sample and pass it through only the subset of trees that were built *without* that sample.
  2. Aggregate the votes from those trees to predict the sample's label.
  3. Repeat for all samples in the original dataset.
  4. The proportion of out-of-bag samples correctly classified measures accuracy; the proportion incorrectly classified is the **Out-of-Bag (OOB) Error**.
- **Hyperparameter Optimization**:
  - The number of features considered at each split can be optimized by comparing the OOB error across different settings.
  - Typical starting heuristic: start with $\sqrt{\text{total variables}}$, then test values above and below to select the setting with the lowest OOB error.

---

## Part 2: Random Forests in R, hands-on implementation (source: statquest-random-forests-in-r.md)

*(Note: The core concepts of bootstrap datasets, bagging, random feature subsets, and OOB error are established in Part 1 and referenced here).*

### Hands-On Workflow in R

#### 1. Loading Libraries
```R
library(ggplot2)
library(cowplot)
library(randomForest)
```

#### 2. Reading Data
```R
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" # [exact URL format not fully specified in source]
data <- read.csv(url, header=FALSE)
head(data)

# Assign column names from UCI repository documentation
colnames(data) <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"
)
```

#### 3. Inspecting & Cleaning Data
```R
# Check structure
str(data)

# Replace '?' with NA
data[data == "?"] <- NA

# Format sex: 0 = F (female), 1 = M (male), convert to factor
data$sex[data$sex == 0] <- "F"
data$sex[data$sex == 1] <- "M"
data$sex <- as.factor(data$sex)

# Convert categorical columns into factors
# Columns with strings from '?' conversion are converted to integer first, then factor
data$ca <- as.factor(as.integer(data$ca))
data$thal <- as.factor(as.integer(data$thal))
data$cp <- as.factor(data$cp)

# Convert target hd (heart disease) to factor: 0 = healthy, 1 = unhealthy
data$hd <- ifelse(test=data$hd == 0, yes="healthy", no="unhealthy")
data$hd <- as.factor(data$hd)

# Re-inspect cleaned structure
str(data)
```

#### 4. Imputing Missing Values with `rfImpute`
```R
set.seed(42) # [exact seed value not fully specified in source]

# rfImpute predicts hd using all other columns (hd ~ .)
# Builds random forests iteratively (4-6 iterations typically sufficient, tested up to 20)
# Prints OOB error rate after each iteration
data.imputed <- rfImpute(hd ~ ., data = data, iter = 6)
```

#### 5. Building the Random Forest Model
```R
model <- randomForest(hd ~ ., data = data.imputed, proximity = TRUE)
model
```
- **Model Output Summary**:
  - Task type: Classification (or Regression / Unsupervised if predicting continuous or omitted target).
  - Number of trees (`ntree`): default is `500`.
  - Number of variables tried at each split (`mtry`): default for classification is $\sqrt{\text{variables}}$; default for regression is $\text{variables} / 3$.
  - Out-of-Bag (OOB) error rate estimate.
  - Confusion matrix of correct and incorrect predictions.

#### 6. Evaluating Number of Trees & Error Convergence
```R
# Extract error rate matrix (OOB error, healthy error, unhealthy error over tree count)
head(model$err.rate)

# Format into a data frame and plot error rates across number of trees using ggplot2
# Lines: green = overall OOB error, blue = unhealthy error, red = healthy error

# Test with 1000 trees to verify error stabilization
model_1000 <- randomForest(hd ~ ., data = data.imputed, ntree = 1000, proximity = TRUE)
```

#### 7. Tuning `mtry` (Variables Considered per Split)
```R
# Loop over different values of mtry (e.g., 1 to 10) and record the final OOB error rate
oob.values <- vector(length = 10)
for (i in 1:10) {
  temp.model <- randomForest(hd ~ ., data = data.imputed, mtry = i, ntree = 1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate), 1]
}
oob.values # Select mtry corresponding to lowest OOB error rate (e.g., mtry = 3)
```

#### 8. MDS Clustering with Proximity Matrix
```R
# Convert proximity matrix into distance matrix
distance.matrix <- dist(1 - model$proximity)

# Run Classical Multi-Dimensional Scaling (MDS)
mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)

# Calculate percentage of variation explained by X and Y axes
mds.var.per <- round(mds.stuff$eig / sum(mds.stuff$eig) * 100, 1)

# Format into data frame and plot with ggplot2
# Displays sample clustering (e.g., healthy samples clustered on one side, unhealthy on the other)
```
