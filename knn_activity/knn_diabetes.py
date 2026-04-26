"""
KNN on Pima Indians Diabetes Dataset

The K-Nearest Neighbors algorithm, data preprocessing (median imputation,
Z-score standardization), distance computation, and evaluation metrics
are implemented manually.

Support libraries used:
  - numpy      : array operations & reproducible train-test split
  - matplotlib : plotting
  - seaborn    : heatmap visualization
"""

import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'diabetes-k-nn.csv')

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def euclidean_distance(a, b):
    """Compute Euclidean distance: d = sqrt(sum((a_i - b_i)^2))"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def knn_predict(X_train, y_train, test_point, k):
    """Predict class label using majority vote of k nearest neighbors.
    Returns (predicted_label, sorted_distances_list)."""
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((i, dist, int(y_train[i])))
    distances.sort(key=lambda x: x[1])

    k_nearest = distances[:k]
    votes = {}
    for _, _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    predicted = max(votes, key=votes.get)
    return predicted, distances


def knn_predict_all(X_train, y_train, X_test, k):
    """Predict class labels for all test points."""
    predictions = []
    for point in X_test:
        pred, _ = knn_predict(X_train, y_train, point, k)
        predictions.append(pred)
    return predictions


def confusion_matrix_manual(y_true, y_pred):
    """Compute 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0: tn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        elif t == 1 and p == 1: tp += 1
    return np.array([[tn, fp], [fn, tp]])


def accuracy_manual(y_true, y_pred):
    """Compute accuracy = correct / total."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def classification_report_manual(y_true, y_pred):
    """Generate precision, recall, F1 for each class."""
    labels = sorted(set(y_true) | set(y_pred))
    names = {0: 'Non-diabetic', 1: 'Diabetic'}
    lines = [f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    lines.append("")

    total_support = 0
    weighted_p = weighted_r = weighted_f = 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        lines.append(f"{names.get(label, str(label)):>15} {precision:10.2f} {recall:10.2f} {f1:10.2f} {support:10d}")
        total_support += support
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f += f1 * support

    lines.append("")
    acc = accuracy_manual(y_true, y_pred)
    lines.append(f"{'accuracy':>15} {'':>10} {'':>10} {acc:10.2f} {total_support:10d}")
    lines.append(f"{'weighted avg':>15} {weighted_p/total_support:10.2f} {weighted_r/total_support:10.2f} {weighted_f/total_support:10.2f} {total_support:10d}")
    return "\n".join(lines)


# =============================================================================
# PART 1: DATA UNDERSTANDING
# =============================================================================
print("=" * 60)
print("PART 1: DATA UNDERSTANDING")
print("=" * 60)

# Read CSV with csv module (no pandas)
raw_data = []
with open(DATA_PATH, mode='r') as f:
    for row in csv.DictReader(f):
        raw_data.append({k: float(v) for k, v in row.items()})

n_samples = len(raw_data)
X = np.array([[row[feat] for feat in FEATURES] for row in raw_data])
y = np.array([int(row['Outcome']) for row in raw_data])

print(f"\nDataset shape: {X.shape}")
print(f"\nFirst 5 rows:")
for i in range(5):
    vals = "  ".join([f"{X[i, j]:7.1f}" for j in range(len(FEATURES))])
    print(f"  {vals}  | Outcome={int(y[i])}")

print(f"\nData types: all features float64, target int")

# Descriptive statistics (manual via numpy — no pandas)
print(f"\nDescriptive statistics:")
print(f"  {'Feature':>25} {'mean':>8} {'std':>8} {'min':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'max':>8}")
for j, fname in enumerate(FEATURES + ['Outcome']):
    col = X[:, j] if j < len(FEATURES) else y.astype(float)
    print(f"  {fname:>25} {np.mean(col):8.2f} {np.std(col):8.2f} {np.min(col):8.2f} "
          f"{np.percentile(col,25):8.2f} {np.percentile(col,50):8.2f} "
          f"{np.percentile(col,75):8.2f} {np.max(col):8.2f}")

print(f"\nOutcome distribution:")
print(f"  0 (Non-diabetic): {int(np.sum(y == 0))}")
print(f"  1 (Diabetic):     {int(np.sum(y == 1))}")


# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("PART 2: DATA PREPROCESSING")
print("=" * 60)

# Step 1: Detect zero values (physiologically impossible)
print("\nZero counts per column (physiologically impossible zeros = missing):")
for col in ZERO_COLS:
    j = FEATURES.index(col)
    count = int(np.sum(X[:, j] == 0))
    pct = count / n_samples * 100
    print(f"  {col}: {count} ({pct:.2f}%)")

# Step 2: Median imputation (manual — robust to skewed distributions)
print("\nApplying median imputation...")
for col in ZERO_COLS:
    j = FEATURES.index(col)
    non_zero_vals = X[X[:, j] != 0, j]
    median_val = float(np.median(non_zero_vals))
    X[X[:, j] == 0, j] = median_val
    print(f"  {col}: replaced zeros with median = {median_val:.2f}")

print(f"\nPost-imputation summary:")
print(f"  {'Feature':>25} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
for j, fname in enumerate(FEATURES):
    col = X[:, j]
    print(f"  {fname:>25} {np.mean(col):8.2f} {np.std(col):8.2f} {np.min(col):8.2f} {np.max(col):8.2f}")

# Step 3: Z-score standardization (manual — no StandardScaler)
# z = (x - μ) / σ where μ = mean, σ = population std (ddof=0)
print("\nApplying Z-score standardization...")
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)  # population std (ddof=0)
X_scaled = (X - means) / stds
print(f"  Scaled features (first row): {[round(v, 4) for v in X_scaled[0]]}")

# ── Visualization: Feature distribution by Outcome ──
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
colors = {0: '#2563eb', 1: '#dc2626'}
labels_map = {0: 'Non-diabetic', 1: 'Diabetic'}
for j, fname in enumerate(FEATURES):
    for outcome in [0, 1]:
        subset = X[y == outcome, j]
        axes[j].hist(subset, bins=20, alpha=0.5, color=colors[outcome],
                     label=labels_map[outcome], edgecolor='white')
    axes[j].set_title(fname, fontsize=10, fontweight='bold')
    axes[j].legend(fontsize=7)
plt.suptitle('Feature Distributions by Outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'featuredistributionbyOutcome.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: featuredistributionbyOutcome.png")

# ── Visualization: Feature correlation heatmap ──
all_data = np.column_stack([X, y.reshape(-1, 1)])
all_names = FEATURES + ['Outcome']
corr_matrix = np.corrcoef(all_data.T)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=all_names, yticklabels=all_names, ax=ax,
            center=0, square=True)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'featureCorrelationHeatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: featureCorrelationHeatmap.png")


# =============================================================================
# PART 3: KNN IMPLEMENTATION
# =============================================================================
print("\n" + "=" * 60)
print("PART 3: KNN IMPLEMENTATION")
print("=" * 60)

# Train-test split — reproducible with fixed seed
# Uses numpy RandomState permutation for consistent splits
rng = np.random.RandomState(42)
n_test = int(np.ceil(0.2 * n_samples))  # 154
permutation = rng.permutation(n_samples)
test_idx = permutation[:n_test]
train_idx = permutation[n_test:]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Manual Euclidean Distance Computation ──
print("\n--- Manual Distance Computation (Test Instance #1) ---")
test_instance = X_test[0]
true_label = 'Diabetic' if y_test[0] == 1 else 'Non-diabetic'
print(f"True label: {true_label}")

_, all_distances = knn_predict(X_train, y_train, test_instance, k=3)

print("\nTop 10 Nearest Neighbors:")
print(f"{'Rank':>4} {'Train Idx':>10} {'Distance':>10} {'Label':>15}")
print("-" * 45)
for rank, (idx, dist, label) in enumerate(all_distances[:10], 1):
    lbl = 'Diabetic (1)' if label == 1 else 'Non-diabetic (0)'
    print(f"{rank:>4} {idx:>10} {dist:>10.4f} {lbl:>15}")

k3_votes = [all_distances[i][2] for i in range(3)]
k3_pred = 1 if sum(k3_votes) > 1.5 else 0
print(f"\nK=3 manual prediction: {'Diabetic' if k3_pred == 1 else 'Non-diabetic'} "
      f"(votes: {['D' if v == 1 else 'N' for v in k3_votes]})")

# ── KNN Results for K = 3, 5, 7 ──
print("\n--- KNN Accuracy for K = 3, 5, 7 ---")
results = {}
for k in [3, 5, 7]:
    y_pred = knn_predict_all(X_train, y_train, X_test, k)
    acc = accuracy_manual(y_test, y_pred)
    cm = confusion_matrix_manual(y_test, y_pred)
    results[k] = {'acc': acc, 'pred': y_pred, 'cm': cm}
    print(f"\nK={k}: Accuracy = {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{classification_report_manual(y_test, y_pred)}")


# =============================================================================
# PART 4: MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("PART 4: MODEL EVALUATION")
print("=" * 60)

best_k = max(results, key=lambda k: results[k]['acc'])
print(f"\nBest K: {best_k} with accuracy {results[best_k]['acc']:.4f}")


# =============================================================================
# BONUS: Logistic Regression Comparison (manual implementation)
# =============================================================================
print("\n--- BONUS: Logistic Regression ---")

def sigmoid(z):
    """Sigmoid activation: 1 / (1 + exp(-z))"""
    z = np.clip(z, -500, 500)  # prevent overflow
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X_train, y_train, X_test, lr=0.1, max_iter=1000, C=1.0):
    """Logistic Regression with L2 regularization via gradient descent.
    C = inverse regularization strength."""
    n_samples, n_features = X_train.shape
    lam = 1.0 / C  # regularization parameter

    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):
        z = X_train @ w + b
        h = sigmoid(z)
        error = h - y_train

        # Gradients with L2 regularization
        dw = (1 / n_samples) * (X_train.T @ error) + (lam / n_samples) * w
        db = (1 / n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    # Predict
    z_test = X_test @ w + b
    probs = sigmoid(z_test)
    preds = (probs >= 0.5).astype(int)
    return preds

lr_pred = logistic_regression(X_train, y_train, X_test, lr=0.1, max_iter=1000, C=1.0)
lr_acc = accuracy_manual(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs K
ks = [3, 5, 7]
accs = [results[k]['acc'] for k in ks]
axes[0].plot(ks, accs, 'o-', color='#2563eb', linewidth=2, markersize=8, label='KNN')
axes[0].axhline(y=lr_acc, color='#dc2626', linestyle='--', linewidth=1.5,
                label=f'Logistic Regression ({lr_acc:.3f})')
for k, a in zip(ks, accs):
    axes[0].annotate(f'{a:.3f}', (k, a), textcoords="offset points",
                     xytext=(0, 10), ha='center')
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('KNN Accuracy vs K Value')
axes[0].legend()
axes[0].set_xticks(ks)
axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix for best K
cm_best = results[best_k]['cm']
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Non-diabetic', 'Diabetic'],
            yticklabels=['Non-diabetic', 'Diabetic'])
axes[1].set_title(f'Confusion Matrix (K={best_k})')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'knn_results.png'), dpi=150)
plt.close()
print("\nSaved: knn_results.png")

# ── Confusion matrices for all K values ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [3, 5, 7]):
    cm = results[k]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-diabetic', 'Diabetic'],
                yticklabels=['Non-diabetic', 'Diabetic'])
    ax.set_title(f'K={k} (Acc: {results[k]["acc"]:.2%})')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices for K=3, K=5, K=7', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'ConfusionMatrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ConfusionMatrices.png")

print("\n[OK] All results computed successfully.")
