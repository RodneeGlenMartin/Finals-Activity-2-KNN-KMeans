"""
KNN + K-Means from scratch — applied to the Automobile dataset (autos-k-means.csv).

Features used (3):
    horsepower    — engine power
    highway_mpg   — fuel economy
    price         — sticker price (US$)

Class labels (derived from price for the supervised KNN task):
    Economy   : price <  $10,000
    Mid-range : $10,000 ≤ price < $20,000
    Luxury    : price ≥ $20,000

Both algorithms are implemented in pure Python (no sklearn).
"""

import csv
import random
import math
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

random.seed(42)
np.random.seed(42)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "autos-k-means.csv")
VISUALS_DIR  = os.path.join(BASE_DIR, "visuals")
RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
os.makedirs(VISUALS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PREPARE THE AUTOS DATASET
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ["horsepower", "highway_mpg", "price"]   # 3 numeric features

def load_autos(path):
    """Read the CSV and return X (features), y (price-tier class), raw_prices."""
    X, y, prices, makes = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hp    = float(row["horsepower"])
                mpg   = float(row["highway_mpg"])
                price = float(row["price"])
            except ValueError:
                continue                          # skip malformed rows
            # Derive a 3-class label from price
            if   price < 10000: cls = 0           # Economy
            elif price < 20000: cls = 1           # Mid-range
            else:               cls = 2           # Luxury
            X.append([hp, mpg, price])
            y.append(cls)
            prices.append(price)
            makes.append(row["make"])
    return X, y, prices, makes

X_raw, y_all, prices, makes = load_autos(DATA_PATH)
print(f"Loaded {len(X_raw)} cars from {DATA_PATH}")

CLASS_NAMES = {0: "Economy", 1: "Mid-range", 2: "Luxury"}
counts = [y_all.count(c) for c in range(3)]
print(f"Class balance: Economy={counts[0]}  Mid-range={counts[1]}  Luxury={counts[2]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MIN-MAX SCALING (so each feature lives on [0, 1])
# ─────────────────────────────────────────────────────────────────────────────
# Without scaling, price (≈ thousands) would dominate Euclidean distance and
# horsepower / mpg would barely matter.

def min_max_scale(X):
    n_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(n_features)]
    maxs = [max(row[j] for row in X) for j in range(n_features)]
    spans = [mx - mn for mn, mx in zip(mins, maxs)]
    X_scaled = [
        [(row[j] - mins[j]) / spans[j] for j in range(n_features)]
        for row in X
    ]
    return X_scaled, mins, maxs

X_all, F_MIN, F_MAX = min_max_scale(X_raw)
print(f"Feature mins: {[round(v,2) for v in F_MIN]}")
print(f"Feature maxs: {[round(v,2) for v in F_MAX]}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  K-NEAREST NEIGHBOURS  (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def knn_predict(train_X, train_y, test_point, k=5):
    """Return predicted class label using majority vote of k nearest neighbours."""
    distances = [(euclidean(test_point, tx), ty)
                 for tx, ty in zip(train_X, train_y)]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    votes = {}
    for _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get)

def knn_evaluate(train_X, train_y, test_X, test_y, k=5):
    preds = [knn_predict(train_X, train_y, tp, k) for tp in test_X]
    correct = sum(p == t for p, t in zip(preds, test_y))
    return preds, correct / len(test_y)

def train_test_split(X, y, test_ratio=0.25):
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    tr_i, te_i = indices[:split], indices[split:]
    return ([X[i] for i in tr_i], [y[i] for i in tr_i],
            [X[i] for i in te_i], [y[i] for i in te_i])

tr_X, tr_y, te_X, te_y = train_test_split(X_all, y_all)
knn_preds, knn_acc = knn_evaluate(tr_X, tr_y, te_X, te_y, k=5)
print(f"\n[K-NN] k=5  | train={len(tr_X)}  test={len(te_X)}  accuracy={knn_acc:.2%}")

# Confusion matrix
n_cls = 3
conf = [[0]*n_cls for _ in range(n_cls)]
for true, pred in zip(te_y, knn_preds):
    conf[true][pred] += 1
print("[K-NN] Confusion matrix (rows = actual, cols = predicted):")
for row in conf:
    print("  ", row)

# Try several k values for the report
ks_tried = [1, 3, 5, 7, 9, 11]
k_acc = []
for k in ks_tried:
    _, acc = knn_evaluate(tr_X, tr_y, te_X, te_y, k=k)
    k_acc.append(round(acc * 100, 1))
print(f"[K-NN] Accuracy by k: {dict(zip(ks_tried, k_acc))}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  K-MEANS  (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def kmeans(X, k=3, max_iter=100):
    """K-Means clustering. Returns (centroids, assignments, inertia, history)."""
    centroids = [list(X[i]) for i in random.sample(range(len(X)), k)]
    history   = []

    for iteration in range(max_iter):
        assignments = [
            min(range(k), key=lambda c: euclidean(x, centroids[c]))
            for x in X
        ]
        new_centroids = []
        for c in range(k):
            members = [X[i] for i in range(len(X)) if assignments[i] == c]
            if members:
                new_centroids.append([
                    sum(m[d] for m in members) / len(members)
                    for d in range(len(X[0]))
                ])
            else:
                new_centroids.append(centroids[c])

        history.append({"iteration": iteration + 1,
                        "centroids": [list(c) for c in new_centroids]})

        if new_centroids == centroids:
            print(f"[K-Means] Converged at iteration {iteration + 1}")
            break
        centroids = new_centroids

    inertia = sum(euclidean(X[i], centroids[assignments[i]]) ** 2
                  for i in range(len(X)))
    return centroids, assignments, inertia, history

km_centroids, km_labels, km_inertia, km_history = kmeans(X_all, k=3)
print(f"[K-Means] k=3  | iterations={len(km_history)}  | inertia(WCSS)={km_inertia:.3f}")

# Convert centroids back to original (un-scaled) units for reporting
def unscale(scaled_pt):
    return [scaled_pt[j] * (F_MAX[j] - F_MIN[j]) + F_MIN[j]
            for j in range(len(scaled_pt))]

km_centroids_raw = [unscale(c) for c in km_centroids]
print("[K-Means] Centroids in original units (HP, MPG, Price):")
for i, c in enumerate(km_centroids_raw):
    print(f"  Cluster {i+1}: HP={c[0]:6.1f}  MPG={c[1]:5.1f}  Price=${c[2]:8,.0f}")

# Elbow method: WCSS for k=1..8
wcss = []
for k in range(1, 9):
    _, _, inertia, _ = kmeans(X_all, k=k)
    wcss.append(inertia)
print(f"[Elbow] WCSS per k: {[round(w, 3) for w in wcss]}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATIONS  (saved to ./visuals/)
# ─────────────────────────────────────────────────────────────────────────────
# Auto-themed palette: deep blue (Economy), warm amber (Mid-range), crimson (Luxury)
PALETTE  = ["#1B4965", "#F4A261", "#9D0208"]
DARK_PAL = ["#0D2B3E", "#C77F2A", "#5A0204"]
BG       = "#F8F9FA"

def fig_setup():
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")
    return fig, ax

# Use raw (un-scaled) values for nicer axis labels in plots
HP   = [r[0] for r in X_raw]
MPG  = [r[1] for r in X_raw]
PRC  = [r[2] for r in X_raw]

# ── 5a. Dataset scatter (Horsepower vs Price, coloured by true class)
fig, ax = fig_setup()
for cls in range(3):
    pts = [(HP[i], PRC[i]) for i in range(len(X_raw)) if y_all[i] == cls]
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, c=PALETTE[cls], label=f"{CLASS_NAMES[cls]} (n={counts[cls]})",
               s=70, edgecolors="white", linewidth=0.5, alpha=0.85)
ax.set_xlabel("Horsepower", fontsize=11)
ax.set_ylabel("Price (US$)", fontsize=11)
ax.set_title("Autos Dataset: Horsepower vs Price",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(framealpha=0.9, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "dataset_scatter.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── 5b. K-NN: scatter of test predictions (correct = filled, wrong = ✗)
te_HP    = [unscale(p)[0] for p in te_X]
te_PRICE = [unscale(p)[2] for p in te_X]

fig, ax = fig_setup()
for cls in range(3):
    pts_ok  = [(te_HP[i], te_PRICE[i]) for i in range(len(te_X))
               if te_y[i] == cls and knn_preds[i] == cls]
    pts_err = [(te_HP[i], te_PRICE[i]) for i in range(len(te_X))
               if te_y[i] == cls and knn_preds[i] != cls]
    if pts_ok:
        xs, ys = zip(*pts_ok)
        ax.scatter(xs, ys, c=PALETTE[cls], s=80, edgecolors="white",
                   linewidth=0.5, label=CLASS_NAMES[cls], alpha=0.9)
    if pts_err:
        xs, ys = zip(*pts_err)
        ax.scatter(xs, ys, c=PALETTE[cls], s=160, marker="X",
                   edgecolors="black", linewidth=1.2, alpha=0.95)
ax.set_xlabel("Horsepower", fontsize=11)
ax.set_ylabel("Price (US$)", fontsize=11)
ax.set_title(f"K-NN Predictions (k=5)  |  Accuracy: {knn_acc:.0%}",
             fontsize=13, fontweight="bold", pad=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
err_handle = mpatches.Patch(color="grey", label="✗ = misclassified")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [err_handle],
          labels=ax.get_legend_handles_labels()[1] + ["✗ = misclassified"],
          fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "knn_results.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── 5c. K-NN confusion matrix heat-map
fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(conf, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([CLASS_NAMES[i] for i in range(3)], fontsize=10)
ax.set_yticklabels([CLASS_NAMES[i] for i in range(3)], fontsize=10)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Confusion Matrix (K-NN)",
             fontsize=12, fontweight="bold", pad=8)
for i in range(3):
    for j in range(3):
        ax.text(j, i, conf[i][j], ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if conf[i][j] > 4 else "#333333")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "knn_confusion.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── 5d. K-Means clusters (Horsepower vs Price)
fig, ax = fig_setup()
for cls in range(3):
    pts = [(HP[i], PRC[i]) for i in range(len(X_raw)) if km_labels[i] == cls]
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, c=PALETTE[cls], s=70, edgecolors="white",
               linewidth=0.5, alpha=0.85, label=f"Cluster {cls+1}")
for c, centroid in enumerate(km_centroids_raw):
    ax.scatter(centroid[0], centroid[2], c=DARK_PAL[c], s=300,
               marker="*", edgecolors="black", linewidth=1.2, zorder=5)
ax.set_xlabel("Horsepower", fontsize=11)
ax.set_ylabel("Price (US$)", fontsize=11)
ax.set_title("K-Means Clusters (★ = Centroid)",
             fontsize=13, fontweight="bold", pad=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
ax.legend(framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "kmeans_clusters.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── 5e. Elbow curve
fig, ax = fig_setup()
ks = list(range(1, 9))
ax.plot(ks, wcss, marker="o", color="#1B4965", linewidth=2.5, markersize=8,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(3, color="#9D0208", linestyle="--", linewidth=1.5, label="Optimal k=3")
for k, w in zip(ks, wcss):
    ax.annotate(f"{w:.2f}", (k, w), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color="#555555")
ax.set_xlabel("Number of Clusters (k)", fontsize=11)
ax.set_ylabel("WCSS (Inertia)", fontsize=11)
ax.set_title("Elbow Method — Optimal k", fontsize=13,
             fontweight="bold", pad=10)
ax.set_xticks(ks)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "elbow.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── 5f. Centroid convergence (HP vs Price, first 3 iterations + final)
fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor=BG)
iters_to_show = [0, 1, 2, len(km_history) - 1]
for ax_i, (ax, hist_i) in enumerate(zip(axes, iters_to_show)):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    cents_scaled = km_history[hist_i]["centroids"]
    cents_raw    = [unscale(c) for c in cents_scaled]
    tmp_labels   = [
        min(range(3), key=lambda c: euclidean(x, cents_scaled[c]))
        for x in X_all
    ]
    for cls in range(3):
        pts = [(HP[i], PRC[i]) for i in range(len(X_raw))
               if tmp_labels[i] == cls]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=PALETTE[cls], s=30, alpha=0.6)
    for c, centroid in enumerate(cents_raw):
        ax.scatter(centroid[0], centroid[2], c=DARK_PAL[c], s=240,
                   marker="*", edgecolors="black", linewidth=1.2, zorder=5)
    label = "Final" if ax_i == 3 else f"Iter {hist_i + 1}"
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Horsepower", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
    if ax_i == 0:
        ax.set_ylabel("Price", fontsize=9)
fig.suptitle("K-Means Centroid Convergence",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "kmeans_convergence.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE RESULTS AS JSON
# ─────────────────────────────────────────────────────────────────────────────

results = {
    "dataset": {
        "n_samples": len(X_raw),
        "features": FEATURES,
        "class_counts": {CLASS_NAMES[i]: counts[i] for i in range(3)},
        "feature_mins": [round(v, 2) for v in F_MIN],
        "feature_maxs": [round(v, 2) for v in F_MAX],
    },
    "knn": {
        "k": 5,
        "train_size": len(tr_X),
        "test_size":  len(te_X),
        "accuracy":   round(knn_acc * 100, 1),
        "confusion_matrix": conf,
        "accuracy_by_k": dict(zip(ks_tried, k_acc)),
    },
    "kmeans": {
        "k": 3,
        "n_points": len(X_raw),
        "inertia":  round(km_inertia, 3),
        "iterations": len(km_history),
        "centroids_scaled": [[round(v, 3) for v in c] for c in km_centroids],
        "centroids_raw":    [[round(v, 1) for v in c] for c in km_centroids_raw],
        "wcss": [round(w, 3) for w in wcss],
    },
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nVisuals saved to {VISUALS_DIR}")
print(f"Results JSON saved to {RESULTS_PATH}")
print(json.dumps(results, indent=2))
