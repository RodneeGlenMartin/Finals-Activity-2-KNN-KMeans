# 🧠 K-Nearest Neighbors & K-Means Clustering

> **Finals Activity 2: Computational Science (CsElec01A)**  
> Step-by-step implementation of KNN and K-Means algorithms in Python, demonstrating foundational understanding of machine learning mechanics.

**Group 14** — Martin · Apal · Bolansoy

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Part 1 — K-Nearest Neighbors (KNN)](#part-1--k-nearest-neighbors-knn)
  - [The Dataset & The New Entry](#the-dataset--the-new-entry)
  - [Step 1: Compute Euclidean Distance](#step-1-compute-euclidean-distance)
  - [Step 2: Sort by Distance](#step-2-sort-by-distance-ascending)
  - [Step 3: Choose K & Vote](#step-3-choose-k--vote-among-nearest-neighbors)
  - [Step 4: Update the Dataset](#step-4-update-the-dataset)
  - [KNN Visualization](#knn-visualization)
- [Part 2 — K-Means Clustering](#part-2--k-means-clustering)
  - [Step 1: Transform to Table](#step-1-transform-the-dataset-into-a-working-table)
  - [Step 2: Initialize Centroids](#step-2-initialize-k-centroids)
  - [Step 3: Compute Distance & Assign Clusters](#step-3-compute-distance--assign-clusters)
  - [Step 4: Recompute Centroids](#step-4-recompute-centroids)
  - [Iteration & Convergence](#iteration--convergence)
  - [Final Clusters](#final-clusters--k-means-in-action)
- [Extended Activity — Diabetes KNN Analysis](#extended-activity--diabetes-knn-analysis)
  - [Data Preprocessing](#data-preprocessing)
  - [KNN Implementation & Results](#knn-implementation--results)
  - [Model Evaluation & Bias-Variance Tradeoff](#model-evaluation--bias-variance-tradeoff)
  - [KNN vs Logistic Regression](#bonus-knn-vs-logistic-regression)
- [Extended Activity — Automobile Dataset](#extended-activity--automobile-dataset)
- [Bias Analysis](#bias-analysis)
- [Technologies Used](#technologies-used)
- [Authors](#authors)

---

## Overview

This project implements two fundamental machine learning algorithms with full step-by-step computation to demonstrate a deep understanding of how they work internally. Both algorithms share the same core distance metric but serve fundamentally different purposes:

| | K-Nearest Neighbors | K-Means Clustering |
|---|---|---|
| **Type** | Supervised Classification | Unsupervised Clustering |
| **Goal** | Classify a new point by majority vote of its neighbors | Group points into K segments by proximity to centroids |
| **Training** | None — memorizes the entire dataset (lazy learner) | Iterative — adjusts centroids until convergence |
| **Core Formula** | $d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}$ | Same Euclidean distance + centroid mean |
| **Output** | A class label for the new point | K cluster assignments for all points |

> **Same Euclidean distance under the hood — but very different goals.**

---

## Project Structure

```
├── README.md
├── knn.py                        # KNN implementation (Customer Tier classification)
├── knn_dataset.csv               # Custom customer dataset (24 entries, 3 tiers)
├── knn_visualization.png         # KNN scatter plot with decision boundary
├── kmeans.py                     # K-Means implementation (Customer Segmentation)
├── kmeans_dataset.csv            # Custom customer dataset (30 entries)
├── kmeans_steps_output.csv       # Full iteration-by-iteration computation log
├── kmeans_visualization.png      # Final cluster visualization
├── Group14_KNN_KMeans.pptx       # Presentation: step-by-step walkthrough
│
└── knn_activity/                 # Extended activities & real-world applications
    ├── autos_ml_algorithms.py    # KNN + K-Means on Automobile dataset
    ├── autos-k-means.csv         # Automobile dataset (horsepower, mpg, price)
    ├── knn_diabetes.py           # KNN on Pima Diabetes dataset
    ├── diabetes-k-nn.csv         # Pima Indians Diabetes dataset (768 samples)
    ├── KNN_Distance_Computation.xlsx  # Manual distance computation spreadsheet
    ├── KNN_Diabetes_Report.docx  # Written report for diabetes KNN analysis
    ├── KNN_Diabetes_Report.pdf   # PDF version of the report
    ├── knn-activity.pdf          # Activity instructions/rubric
    ├── knn_results.png           # Diabetes KNN results visualization
    ├── ConfusionMatrices.png     # Confusion matrix visualizations
    ├── featureCorrelationHeatmap.png   # Feature correlation heatmap
    ├── featuredistributionbyOutcome.png # Feature distribution by outcome
    ├── Autos_KNN_KMeans_Presentation.pptx  # Autos dataset presentation
    ├── Autos_KNN_KMeans_StepByStep.pptx    # Autos step-by-step walkthrough
    └── visuals_autos/            # Generated visualizations for automobile dataset
```

---

## Part 1 — K-Nearest Neighbors (KNN)

> *Classify a new customer by looking at the K most similar customers we already know.*

**File:** [`knn.py`](knn.py)

### The Dataset & The New Entry

Our custom dataset contains **24 customers** with two features and a class label:

| Feature | Description | Range |
|---------|-------------|-------|
| `Annual_Income_k` | Annual income in thousands (\$) | 22–120 |
| `Store_Visits_Per_Month` | Monthly store visit frequency | 1–15 |
| `Customer_Tier` | Class label | Basic / Silver / Gold |

**Distribution:** 8 Basic, 8 Silver, 8 Gold (balanced — intentionally designed to prevent majority-class bias in KNN voting).

**The new customer to classify:**

| Annual Income | Store Visits / Month | Tier |
|:---:|:---:|:---:|
| \$65k | 5 | **?** |

**Goal:** Predict the customer tier using the K nearest neighbors.

---

### Step 1: Compute Euclidean Distance

For every existing customer, we compute the Euclidean distance to the new entry using the formula:

$$d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}$$

Where:
- $X$ = Annual Income ($k)
- $Y$ = Store Visits per month

**Worked example — New Entry vs Customer #10** ($\text{Income} = 65, \text{Visits} = 6$):

$$d_{10} = \sqrt{(65 - 65)^2 + (5 - 6)^2} = \sqrt{0 + 1} = 1.00$$

The new point $(65, 5)$ is just **1.00 unit** from Customer #10. This computation is repeated programmatically for all 24 customers:

```python
# Distance formula: sqrt((X2 - X1)^2 + (Y2 - Y1)^2)
distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

Each step is printed to the console showing the full substitution:

```
d1 (ID 1) = sqrt((65 - 25)^2 + (5 - 2)^2)
   = 40.11
d2 (ID 2) = sqrt((65 - 30)^2 + (5 - 3)^2)
   = 35.06
...
```

---

### Step 2: Sort by Distance (Ascending)

After computing all distances, we **sort the dataset in ascending order** — smallest distance = most similar customer. The top of the sorted list contains our candidates for voting.

| Rank | Customer ID | Income (\$k) | Visits | Tier | Distance |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 10 | 65 | 6 | Silver | 1.00 |
| 2 | 16 | 64 | 7 | Silver | 2.24 |
| 3 | 15 | 68 | 5 | Silver | 3.00 |
| ... | ... | ... | ... | ... | ... |

The highlighted rows (top 3) are the candidates for our majority vote.

---

### Step 3: Choose K & Vote Among Nearest Neighbors

We choose $K = 3$ nearest neighbors:

| Rank | Customer | Distance | Tier |
|:---:|:---:|:---:|:---:|
| #1 | Customer 10 | 1.00 | **Silver** |
| #2 | Customer 16 | 2.24 | **Silver** |
| #3 | Customer 15 | 3.00 | **Silver** |

**Vote Tally:**

| Tier | Votes |
|:---:|:---:|
| Silver | 3 ✅ |
| Basic | 0 |
| Gold | 0 |

$$\text{Predicted Tier} = \arg\max(\text{votes}) = \textbf{Silver}$$

The majority among the 3 nearest neighbors is **Silver** — unanimous in this case.

---

### Step 4: Update the Dataset

After classification, the new customer is inserted into the dataset:

- **Before:** 24 customers
- **After:** 25 customers (new Customer #25 added with tier = Silver)

> *The dataset grows. Next time we classify, this customer also gets a vote.*

---

### KNN Visualization

The scatter plot shows all customers color-coded by tier, with:
- ⭐ **Red star** — the new point $(65, 5)$
- **Dashed lines** — connections to the $K=3$ nearest neighbors
- **Dashed circle** — the $K=3$ decision boundary (radius = distance to the 3rd nearest neighbor)

![KNN Visualization](knn_visualization.png)

---

## Part 2 — K-Means Clustering

> *Group customers into K segments by repeatedly assigning points to the nearest centroid and recomputing the centroid.*

**File:** [`kmeans.py`](kmeans.py)

### Step 1: Transform the Dataset into a Working Table

Our dataset has **30 customers** with two features:

| Feature | Description | Range |
|---------|-------------|-------|
| `Annual_Income_k` | Annual income in thousands (\$) | 15–110 |
| `Spending_Score` | Spending behavior score | 3–95 |

Each customer is mapped to a 2D coordinate: $P_i = (X_i, Y_i)$ where $X$ = Income and $Y$ = Spending.

---

### Step 2: Initialize K Centroids

We choose $K = 5$ starting centers, **randomly selected** from the dataset using `random.seed(42)` for reproducibility:

| Centroid | $X$ (Income) | $Y$ (Spending) |
|:---:|:---:|:---:|
| $C_1$ | 70.0 | 10.0 |
| $C_2$ | 16.0 | 77.0 |
| $C_3$ | 15.0 | 39.0 |
| $C_4$ | 78.0 | 85.0 |
| $C_5$ | 19.0 | 3.0 |

> **Why random initialization?** Hardcoded centroids introduce **initialization bias** — the final clusters could be predetermined by the analyst's choice. Random selection with a fixed seed ensures reproducibility while removing subjective bias.

---

### Step 3: Compute Distance & Assign Clusters

For every point $P_i$, we measure the Euclidean distance to every centroid $C_j$:

$$d(P_i, C_j) = \sqrt{(X_{P_i} - X_{C_j})^2 + (Y_{P_i} - Y_{C_j})^2}$$

The **smallest distance wins** — that point joins that cluster:

$$\text{cluster}(P_i) = \arg\min_j \; d(P_i, C_j)$$

**Example — Point $P_1$ (15.0, 39.0) in Iteration 1:**

| | $C_1$ (70, 10) | $C_2$ (16, 77) | $C_3$ (15, 39) | $C_4$ (78, 85) | $C_5$ (19, 3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $d(P_1, C_j)$ | 62.18 | 38.01 | **0.00** | 78.01 | 36.22 |

$\min = 0.00$ → $P_1$ is assigned to **Cluster 3** (which makes sense — it's the centroid itself).

---

### Step 4: Recompute Centroids

After all 30 points are assigned, each centroid is recalculated as the **mean** of all points in its cluster:

$$C_j^{\text{new}} = \left( \frac{1}{|S_j|} \sum_{P_i \in S_j} X_i, \;\; \frac{1}{|S_j|} \sum_{P_i \in S_j} Y_i \right)$$

Where $S_j$ is the set of points assigned to cluster $j$, and $|S_j|$ is its size.

**After Iteration 1 — New Centroids:**

| Centroid | New $X$ | New $Y$ |
|:---:|:---:|:---:|
| $C_1$ | 78.83 | 20.50 |
| $C_2$ | 17.00 | 80.00 |
| $C_3$ | 36.71 | 47.00 |
| $C_4$ | 76.67 | 73.22 |
| $C_5$ | 17.67 | 5.00 |

> ↻ **Loop:** Now go back to Step 2 with these new centroids and repeat. We stop when **no point changes cluster**.

---

### Iteration & Convergence

The algorithm iterates, recomputing centroids each round until convergence:

| Iteration | $C_1$ | $C_2$ | $C_3$ | $C_4$ | $C_5$ | Points Changed? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | (78.83, 20.50) | (17.00, 80.00) | (36.71, 47.00) | (76.67, 73.22) | (17.67, 5.00) | ✅ Yes |
| 2 | (82.60, 15.60) | (17.00, 80.00) | (43.40, 48.00) | (81.86, 79.00) | (17.67, 5.00) | ✅ Yes |
| 3 | (82.60, 15.60) | (17.00, 80.00) | (47.25, 48.75) | (88.00, 89.60) | (17.67, 5.00) | ✅ Yes |
| **4** | (82.60, 15.60) | (17.00, 80.00) | (47.25, 48.75) | (88.00, 89.60) | (17.67, 5.00) | **❌ No → Converged!** |

> ✓ **Converged at Iteration 4** — No point switched clusters between iterations 3 and 4. The algorithm stops.

---

### Final Clusters — K-Means in Action

The algorithm discovered **5 customer segments**:

| Cluster | Profile | Centroid |
|:---:|:---|:---:|
| 🟢 **Cluster 1** | High income · Low spending | (82.60, 15.60) |
| 🟠 **Cluster 2** | Low income · High spending | (17.00, 80.00) |
| 🟣 **Cluster 3** | Middle income · Middle spending | (47.25, 48.75) |
| 🔵 **Cluster 4** | High income · High spending | (88.00, 89.60) |
| 🔴 **Cluster 5** | Low income · Low spending | (17.67, 5.00) |

![K-Means Visualization](kmeans_visualization.png)

---

## Extended Activity — Diabetes KNN Analysis

**Files:** [`knn_activity/knn_diabetes.py`](knn_activity/knn_diabetes.py) · [`KNN_Diabetes_Report.pdf`](knn_activity/KNN_Diabetes_Report.pdf)  
**Dataset:** Pima Indians Diabetes Dataset — **768 samples**, 8 features, binary outcome (Diabetic / Non-diabetic)

> Full pipeline: data preprocessing, KNN classification, Logistic Regression comparison, and evaluation metrics.

### Data Preprocessing

#### Missing / Zero Value Detection

Several features contain physiologically impossible zeros (e.g., Glucose = 0, BMI = 0) which represent missing data:

| Feature | Zero Count | Percentage | Action |
|---------|:---:|:---:|:---|
| Glucose | 5 | 0.65% | Median imputation |
| BloodPressure | 35 | 4.56% | Median imputation |
| SkinThickness | 227 | 29.56% | Median imputation |
| Insulin | 374 | 48.70% | Median imputation |
| BMI | 11 | 1.43% | Median imputation |
| Pregnancies | 111 | 14.45% | Valid (0 = no prior pregnancies) |

**Why median imputation?** Several features (especially Insulin) are heavily right-skewed. The median is robust to outliers and preserves the central tendency better than the mean. Row removal was rejected because it would eliminate ~50% of the dataset.

#### Feature Scaling: Z-Score Standardization

KNN relies on Euclidean distance, making it **extremely sensitive to feature scale**. Without scaling, Insulin (range 0–846) would completely dominate BMI (range 0–67.1), producing meaningless distances.

**Z-score standardization** transforms each feature to have $\mu = 0$ and $\sigma = 1$:

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the feature mean and $\sigma$ is the standard deviation. This ensures all features contribute equally to the distance computation.

![Feature Distribution by Outcome](knn_activity/featuredistributionbyOutcome.png)

![Feature Correlation Heatmap](knn_activity/featureCorrelationHeatmap.png)

---

### KNN Implementation & Results

#### Train-Test Split

The dataset (768 samples) was split **80/20** with a fixed random seed (42) for reproducibility:
- **Training set:** 614 samples
- **Test set:** 154 samples

#### Distance Computation

One test instance (true label = Non-diabetic) was selected. Euclidean distances were computed against all 614 training samples using the generalized formula:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

For $n = 8$ features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).

**Top 10 Nearest Neighbors:**

| Rank | Train Index | Euclidean Distance | Label |
|:---:|:---:|:---:|:---|
| 1 | 226 | 1.3237 | Diabetic (1) |
| 2 | 521 | 1.3728 | Diabetic (1) |
| 3 | 403 | 1.3952 | Non-diabetic (0) |
| 4 | 430 | 1.4517 | Diabetic (1) |
| 5 | 321 | 1.6715 | Diabetic (1) |
| 6 | 90 | 1.6836 | Non-diabetic (0) |
| 7 | 18 | 1.6851 | Diabetic (1) |
| 8 | 213 | 1.7071 | Non-diabetic (0) |
| 9 | 260 | 1.7184 | Non-diabetic (0) |
| 10 | 412 | 1.7379 | Diabetic (1) |

**K=3 Prediction:** Votes = [Diabetic, Diabetic, Non-diabetic] → Majority = **Diabetic** (True label was Non-diabetic — misclassified in this instance, illustrating KNN's sensitivity to local noise).

#### Results for K = 3, 5, 7

| K | Accuracy | TN | FP | FN | TP |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 70.78% | 74 | 25 | 20 | 35 |
| **5** | **73.38%** | **75** | **24** | **17** | **38** |
| 7 | 71.43% | 74 | 25 | 19 | 36 |

![Confusion Matrices](knn_activity/ConfusionMatrices.png)

---

### Model Evaluation & Bias-Variance Tradeoff

**Best K = 5** achieved the highest accuracy of **73.38%** (113 / 154 correct predictions).

The choice of K represents a fundamental **bias-variance tradeoff**:

| Scenario | K Value | Effect | Result |
|:---|:---:|:---|:---|
| **K too small** | $K=1$ | Memorizes training data; jagged decision boundary | High variance, **overfitting**, poor generalization |
| **K optimal** | $K=5$ | Balances noise tolerance and local sensitivity | Best test accuracy (**73.38%**) |
| **K too large** | $K = N$ | Ignores local structure; always predicts majority class | High bias, **underfitting**, trivial accuracy |

**Why does K=5 work best?**
- $K=3$: Only 3 neighbors vote — noise and outliers have large influence (high variance).
- $K=5$: Considers enough neighbors to be robust to noise while remaining sensitive to local patterns.
- $K=7$: Begins to over-smooth, losing some local discriminative power.

---

### Bonus: KNN vs Logistic Regression

| Algorithm | Accuracy | Type | Key Advantage |
|:---|:---:|:---|:---|
| KNN ($K=3$) | 70.78% | Non-parametric | No training phase; flexible boundary |
| KNN ($K=5$) | 73.38% | Non-parametric | Best KNN result; balanced bias-variance |
| KNN ($K=7$) | 71.43% | Non-parametric | Smoother boundary; more robust to noise |
| **Logistic Regression** | **75.32%** | Parametric | Fast, interpretable, highest accuracy |

![KNN Results](knn_activity/knn_results.png)

**Analysis:** Logistic Regression outperformed all KNN variants by **+1.94 percentage points**. This is consistent with the dataset's characteristics — Glucose, BMI, and Age have approximately linear relationships with the diabetic outcome, which a linear classifier captures efficiently. KNN would be more competitive on datasets with **highly non-linear class boundaries**.

**When KNN works well:**
1. Dataset is small to medium-sized
2. Decision boundary is irregular or non-linear
3. Features are well-scaled and informative
4. Interpretability is not required

**When KNN is not appropriate:**
1. Very large datasets (millions of records) — $O(n)$ prediction cost
2. Many irrelevant or highly correlated features (**curse of dimensionality**)
3. Real-time prediction is required
4. A compact, deployable model is needed

---

## Extended Activity — Automobile Dataset

**File:** [`knn_activity/autos_ml_algorithms.py`](knn_activity/autos_ml_algorithms.py)

Both KNN and K-Means applied to an automobile dataset (3 features: horsepower, highway MPG, price):

| Component | Details |
|:---|:---|
| **Features** | Horsepower, Highway MPG, Price |
| **Class Labels** | Economy ($< \$10{,}000$), Mid-range ($\$10{,}000 – \$20{,}000$), Luxury ($\geq \$20{,}000$) |
| **Scaling** | Min-max normalization to $[0, 1]$ — prevents price from dominating distance |
| **KNN** | Tested across $K = \{1, 3, 5, 7, 9, 11\}$ with train/test split |
| **K-Means** | $K = 3$ clusters with Elbow Method validation (WCSS for $K = 1 \ldots 8$) |

**Min-Max Scaling Formula:**

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Without scaling, price (range: thousands of dollars) would completely dominate horsepower and MPG in the Euclidean distance computation, producing clusters driven solely by price.

---

## Bias Analysis

This project critically evaluates and mitigates algorithmic biases:

| Bias Type | Issue | Mitigation Applied |
|:---|:---|:---|
| **Feature Scaling Bias** | Income (\$k) dominates over Visit Count; Insulin dominates over BMI | Min-max scaling (`autos_ml_algorithms.py`); Z-score standardization (`knn_diabetes.py`) |
| **Centroid Initialization Bias** | Hardcoded centroids predetermine the final clusters | Random initialization with `random.seed(42)` for reproducibility |
| **Confirmation Bias** | Cherry-picking K or initial centroids to match expected output | Multiple K values tested; Elbow Method used for K-Means |
| **Class Imbalance Bias** | Majority class dominates KNN voting | Balanced custom datasets (8 per tier); acknowledged in diabetes analysis (500 vs 268) |
| **Curse of Dimensionality** | Euclidean distance loses discriminative power in high dimensions | Feature selection (3 features for autos, 8 features for diabetes with scaling) |
| **Missing Data Bias** | Zeros treated as valid values skew distributions | Median imputation for physiologically impossible zeros |

---

## How to Run

### Prerequisites

- Python 3.7+
- Core scripts: only `matplotlib` required
- Diabetes script: `numpy`, `matplotlib`, `seaborn`

### Run the Core Algorithms

```bash
# K-Nearest Neighbors — Customer Tier Classification
python knn.py

# K-Means Clustering — Customer Segmentation
python kmeans.py
```

### Run Extended Activities

```bash
cd knn_activity

# Diabetes dataset
python knn_diabetes.py

# Automobile dataset
python autos_ml_algorithms.py
```

---

## Technologies Used

| Technology | Purpose |
|:---|:---|
| **Python 3** | Core implementation language |
| `csv` | Dataset I/O |
| `math` | Euclidean distance: $d = \sqrt{\sum (p_i - q_i)^2}$ |
| `matplotlib` | Visualizations and scatter plots |
| `collections.Counter` | KNN majority voting |
| `random` | Unbiased centroid initialization with seed |
| `numpy` | Array operations (diabetes activity) |

---

## Authors

### Group 14

| Member | Role |
|:---|:---|
| **Rodnee Glen A. Martin** | Group 14 — Member |
| **Renier P. Apal** | Group 14 — Member |
| **Ear Lenser B. Bolansoy** | Group 14 — Member |

---

## License

This project was created for academic purposes as part of a Computational Science (CsElec01A) course (Finals Activity 2).
