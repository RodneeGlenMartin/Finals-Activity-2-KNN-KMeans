import csv
import math
import random
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_C_NAVY  = "#1F3864"
_C_BLUE  = "#2F5496"
_C_GREEN = "#70AD47"
_C_RED   = "#C00000"
_C_ORG   = "#ED7D31"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Columns where 0 is physiologically impossible -> treat as missing
ZERO_MISSING_COLS = [1, 2, 3, 4, 5]   # Glucose, BloodPressure, SkinThickness, Insulin, BMI


# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------
def load_csv(path):
    """Returns (header list, list-of-float-rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [[float(v) for v in row] for row in reader]
    return header, data


# -----------------------------------------------------------------------------
# 2. Median imputation
# -----------------------------------------------------------------------------
def _median(values):
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2.0 if n % 2 == 0 else s[n // 2]


def median_imputation(data):
    """
    Replaces zeros in ZERO_MISSING_COLS with the column median
    (computed on non-zero values).  Returns modified data and the
    dict {col_index: median_value}.
    """
    col_medians = {}
    for col in ZERO_MISSING_COLS:
        valid = [row[col] for row in data if row[col] != 0.0]
        col_medians[col] = _median(valid)

    for row in data:
        for col in ZERO_MISSING_COLS:
            if row[col] == 0.0:
                row[col] = col_medians[col]

    return data, col_medians


# -----------------------------------------------------------------------------
# 3. Z-score standardisation
# -----------------------------------------------------------------------------
def feature_stats(data, n_feat):
    """Returns (means, stds) lists over the first n_feat columns."""
    means, stds = [], []
    for j in range(n_feat):
        vals = [row[j] for row in data]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        means.append(m)
        stds.append(sd if sd > 0.0 else 1.0)
    return means, stds


def zscore_standardize(data, means, stds, n_feat):
    """Returns a new dataset with features z-scored; label column kept as-is."""
    return [
        [(row[j] - means[j]) / stds[j] for j in range(n_feat)] + [row[-1]]
        for row in data
    ]


# -----------------------------------------------------------------------------
# 4. Train / test split
# -----------------------------------------------------------------------------
def train_test_split(data, test_ratio=0.20, seed=42):
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_ratio))
    return shuffled[:cut], shuffled[cut:]


# -----------------------------------------------------------------------------
# 5. Euclidean distance
# -----------------------------------------------------------------------------
def euclidean(a, b, n_feat):
    return math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(n_feat)))


# -----------------------------------------------------------------------------
# 6. KNN predict
# -----------------------------------------------------------------------------
def knn_predict(train, query, k, n_feat):
    distances = sorted(
        ((euclidean(tr, query, n_feat), int(tr[-1])) for tr in train),
        key=lambda x: x[0],
    )
    labels = [lbl for _, lbl in distances[:k]]
    return Counter(labels).most_common(1)[0][0]


# -----------------------------------------------------------------------------
# 7. Evaluate model
# -----------------------------------------------------------------------------
def evaluate(train, test, k, n_feat):
    tp = tn = fp = fn = 0
    for row in test:
        pred   = knn_predict(train, row, k, n_feat)
        actual = int(row[-1])
        if   pred == 1 and actual == 1: tp += 1
        elif pred == 0 and actual == 0: tn += 1
        elif pred == 1 and actual == 0: fp += 1
        else:                           fn += 1
    total = tp + tn + fp + fn
    acc   = (tp + tn) / total
    prec  = tp / (tp + fp)  if (tp + fp) else 0.0
    rec   = tp / (tp + fn)  if (tp + fn) else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(k=k, acc=acc, tp=tp, tn=tn, fp=fp, fn=fn,
                prec=prec, rec=rec, f1=f1)


# -----------------------------------------------------------------------------
# 8. Manual step-by-step Euclidean distance (Part 3 requirement)
# -----------------------------------------------------------------------------
def print_manual_distances(train, test_instance, n_feat=8, n_samples=10):
    """
    For ONE test instance, prints the full step-by-step Euclidean distance
    computation against the first n_samples training records.
    """
    W = 74
    print()
    print("=" * W)
    print("  PART 3 -- MANUAL STEP-BY-STEP EUCLIDEAN DISTANCE COMPUTATION")
    print("=" * W)
    print()
    print("  Selected Test Instance (z-score standardised values):")
    print(f"  {'Feature':<32} {'Standardised Value':>18}  {'Raw Outcome':>11}")
    print("  " + "-" * 64)
    for j, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<32} {test_instance[j]:>18.5f}")
    print(f"  {'True Class Label':<32} {'':>18}  {int(test_instance[-1]):>11}")
    print()
    print("  Formula:  d = sqrt( sum (x_test_i - x_train_i)^2 )")
    print()

    all_dists = []
    for idx in range(n_samples):
        tr = train[idx]
        print("  " + "-" * W)
        print(f"  Training Sample #{idx + 1:02d}  |  Label = {int(tr[-1])}")
        print("  " + "-" * W)
        print(f"  {'Feature':<32} {'Test':>9}  {'Train':>9}  {'Diff':>9}  {'Diff^2':>9}")
        print("  " + "-" * 72)
        sq_diffs = []
        for j, name in enumerate(FEATURE_NAMES):
            diff = test_instance[j] - tr[j]
            sq   = diff ** 2
            sq_diffs.append(sq)
            print(f"  {name:<32} {test_instance[j]:>9.5f}  {tr[j]:>9.5f}"
                  f"  {diff:>9.5f}  {sq:>9.5f}")
        total = sum(sq_diffs)
        dist  = math.sqrt(total)
        all_dists.append((dist, int(tr[-1]), idx + 1))
        print("  " + "-" * 72)
        print(f"  {'Sum of Squared Differences':<32} {'':>9}  {'':>9}  {'':>9}  {total:>9.5f}")
        print(f"  Euclidean Distance  = sqrt({total:.5f})  =  {dist:.5f}")
        print()

    # Sorted summary
    sorted_dists = sorted(all_dists, key=lambda x: x[0])
    print("=" * W)
    print("  SORTED DISTANCES  (ascending -- lowest = nearest neighbour)")
    print("=" * W)
    print(f"  {'Rank':<6} {'Sample #':<10} {'Distance':<14} {'Label'}")
    print("  " + "-" * 42)
    for rank, (d, lbl, snum) in enumerate(sorted_dists, 1):
        print(f"  {rank:<6} #{snum:<9} {d:<14.5f} {'Diabetic (1)' if lbl else 'Non-diabetic (0)'}")

    print()
    print("  K-Neighbour Predictions based on these 10 training samples:")
    for k in [3, 5, 7]:
        neighbors = sorted_dists[:k]
        votes     = [lbl for _, lbl, _ in neighbors]
        pred      = Counter(votes).most_common(1)[0][0]
        print(f"    K={k}:  neighbours = {votes}  ->  Predicted class = {pred} "
              f"({'Diabetic' if pred else 'Non-diabetic'})")
    print()


# -----------------------------------------------------------------------------
# 9. Chart generation
# -----------------------------------------------------------------------------
def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [IMG] Saved -> {os.path.basename(path)}")


def plot_class_distribution(raw, out_dir):
    diabetic     = sum(1 for row in raw if row[-1] == 1.0)
    non_diabetic = len(raw) - diabetic
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        [non_diabetic, diabetic],
        labels=[f"Non-diabetic (0)\n{non_diabetic} records",
                f"Diabetic (1)\n{diabetic} records"],
        autopct="%1.1f%%",
        colors=[_C_BLUE, _C_RED],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(11)
    ax.set_title("Figure 1: Class Distribution\n(768 total records)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_01_class_distribution.png"))


def plot_missing_values(before_stats, out_dir):
    features = [f for f in FEATURE_NAMES if before_stats[f]["zeros"] > 0]
    counts   = [before_stats[f]["zeros"] for f in features]
    colors   = [_C_RED if c > 100 else _C_ORG for c in counts]
    fig, ax  = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(features, counts, color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, padding=5, fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Zero (Missing) Values", fontsize=10)
    ax.set_title("Figure 2: Missing Value Counts per Feature\n(zeros treated as missing)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_02_missing_values.png"))


def plot_euclidean_distances(sorted_dists, out_dir):
    samples = [f"#{snum:02d}" for _, _, snum in sorted_dists]
    dists   = [d   for d, _, _   in sorted_dists]
    labels  = [lbl for _, lbl, _ in sorted_dists]
    colors  = [_C_RED if lbl else _C_BLUE for lbl in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(samples, dists, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xlabel("Training Sample", fontsize=10)
    ax.set_ylabel("Euclidean Distance", fontsize=10)
    ax.set_title("Figure 3: Euclidean Distances from Test Instance\nto 10 Training Samples (sorted ascending)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        mpatches.Patch(color=_C_BLUE, label="Non-diabetic (0)"),
        mpatches.Patch(color=_C_RED,  label="Diabetic (1)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_03_euclidean_distances.png"))


def plot_model_performance(results, out_dir):
    eval_list = [results[k] for k in [3, 5, 7]]
    ks   = [r["k"]        for r in eval_list]
    acc  = [r["acc"]*100  for r in eval_list]
    prec = [r["prec"]*100 for r in eval_list]
    rec  = [r["rec"]*100  for r in eval_list]
    f1   = [r["f1"]*100   for r in eval_list]
    x = list(range(len(ks)))
    w = 0.19
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar([i - 1.5*w for i in x], acc,  width=w, label="Accuracy",
                color=_C_NAVY,  edgecolor="white")
    b2 = ax.bar([i - 0.5*w for i in x], prec, width=w, label="Precision",
                color=_C_BLUE,  edgecolor="white")
    b3 = ax.bar([i + 0.5*w for i in x], rec,  width=w, label="Recall",
                color=_C_GREEN, edgecolor="white")
    b4 = ax.bar([i + 1.5*w for i in x], f1,   width=w, label="F1 Score",
                color=_C_ORG,   edgecolor="white")
    for bars in [b1, b2, b3, b4]:
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in ks], fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Figure 4: KNN Model Performance by K Value\n"
                 "(Accuracy, Precision, Recall, F1 Score)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=75, color="gray", linestyle="--", linewidth=0.9, alpha=0.5, label="_nolegend_")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_04_model_performance.png"))


def plot_confusion_matrices(results, out_dir):
    eval_list = [results[k] for k in [3, 5, 7]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.suptitle("Figure 5: Confusion Matrices for K = 3, 5, 7",
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, r in zip(axes, eval_list):
        cm     = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        mx_val = max(r["tn"], r["tp"], r["fp"], r["fn"])
        ax.imshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=mx_val * 1.2, aspect="auto")
        lbls = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val   = cm[i][j]
                shade = val / (mx_val * 1.2)
                color = "white" if shade > 0.5 else "black"
                ax.text(j, i, f"{lbls[i][j]}\n{val}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: 0", "Pred: 1"], fontsize=9)
        ax.set_yticklabels(["Act: 0", "Act: 1"], fontsize=9)
        ax.set_title(f"K = {r['k']}  |  Acc = {r['acc']*100:.1f}%",
                     fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_05_confusion_matrices.png"))


def plot_accuracy_vs_k(results, out_dir):
    ks  = [3, 5, 7]
    acc = [results[k]["acc"] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(ks, acc, marker="o", markersize=9, linewidth=2.5,
            color=_C_NAVY, markerfacecolor=_C_RED, markeredgecolor="white",
            markeredgewidth=1.5)
    for k, a in zip(ks, acc):
        ax.annotate(f"{a:.2f}%", xy=(k, a), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=_C_NAVY)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K = {k}" for k in ks], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(min(acc) - 5, max(acc) + 8)
    ax.set_title("Figure 6: Accuracy vs K Value", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "chart_06_accuracy_vs_k.png"))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, "diabetes-k-nn.csv")

    W = 74
    print("=" * W)
    print("  KNN FROM SCRATCH  |  Pima Indians Diabetes Dataset")
    print("  GROUP 14: Rodnee Glen A. Martin, Renier P. Apal, Earl Lenser B. Bolansoy")
    print("=" * W)

    # -- Load ------------------------------------------------------------------
    header, raw = load_csv(csv_path)
    n_feat = len(header) - 1
    N      = len(raw)
    print(f"\n[LOAD]  {N} records · {n_feat} features  <-  {os.path.basename(csv_path)}")

    # -- Before-preprocessing summary ------------------------------------------
    print("\n[BEFORE PREPROCESSING]  Raw dataset summary statistics")
    print(f"  {'Feature':<32} {'Min':>8} {'Max':>8} {'Mean':>10} {'#Zeros':>8}")
    print("  " + "-" * 70)
    before_stats = {}
    for j in range(n_feat):
        vals = [row[j] for row in raw]
        mn   = min(vals); mx = max(vals)
        mn_v = sum(vals) / len(vals)
        zeros = sum(1 for v in vals if v == 0.0)
        before_stats[header[j]] = dict(min=mn, max=mx, mean=mn_v, zeros=zeros)
        print(f"  {header[j]:<32} {mn:>8.2f} {mx:>8.2f} {mn_v:>10.4f} {zeros:>8}")

    # -- Median imputation -----------------------------------------------------
    data, col_medians = median_imputation([row[:] for row in raw])
    print("\n[MEDIAN IMPUTATION]  Zeros in physiologically impossible columns replaced")
    for col in ZERO_MISSING_COLS:
        z = before_stats[header[col]]["zeros"]
        print(f"  {header[col]:<32} {z} zeros -> median = {col_medians[col]:.4f}")

    # -- Z-score standardisation -----------------------------------------------
    means, stds = feature_stats(data, n_feat)
    std_data    = zscore_standardize(data, means, stds, n_feat)

    print("\n[Z-SCORE STANDARDISATION]  Parameters computed on full (post-imputation) dataset")
    print(f"  {'Feature':<32} {'Mean':>10} {'Std Dev':>10}")
    print("  " + "-" * 54)
    for j in range(n_feat):
        print(f"  {header[j]:<32} {means[j]:>10.4f} {stds[j]:>10.4f}")

    print("\n[AFTER STANDARDISATION]  Verification (all features ~ mean=0, std=1)")
    print(f"  {'Feature':<32} {'Min':>9} {'Max':>9} {'Mean':>12}")
    print("  " + "-" * 65)
    for j in range(n_feat):
        vals = [row[j] for row in std_data]
        print(f"  {header[j]:<32} {min(vals):>9.4f} {max(vals):>9.4f} {sum(vals)/len(vals):>12.8f}")

    # -- Train / test split ----------------------------------------------------
    train, test = train_test_split(std_data, test_ratio=0.20, seed=42)
    print(f"\n[SPLIT]  Train = {len(train)} samples ({len(train)/N*100:.1f}%)  |  "
          f"Test = {len(test)} samples ({len(test)/N*100:.1f}%)   [seed=42]")

    # -- Manual distance demo --------------------------------------------------
    print_manual_distances(train, test[0], n_feat, n_samples=10)

    # -- Evaluate K = 3, 5, 7 -------------------------------------------------
    print("=" * W)
    print("  PART 4 -- MODEL EVALUATION  (K = 3, 5, 7)")
    print("=" * W)

    results = {}
    for k in [3, 5, 7]:
        r = evaluate(train, test, k, n_feat)
        results[k] = r
        print(f"\n  -- K = {k} --------------------------------------------")
        print(f"     Accuracy  : {r['acc']:.4f}  ({r['acc']*100:.2f}%)")
        print(f"     Precision : {r['prec']:.4f}")
        print(f"     Recall    : {r['rec']:.4f}")
        print(f"     F1 Score  : {r['f1']:.4f}")
        print(f"     Confusion Matrix:")
        print(f"       {'':20} {'Pred 0':>8}   {'Pred 1':>6}")
        print(f"       {'Actual 0':20} {r['tn']:>8}   {r['fp']:>6}")
        print(f"       {'Actual 1':20} {r['fn']:>8}   {r['tp']:>6}")

    best_k = max(results, key=lambda k: results[k]["acc"])
    print(f"\n  DONE  Best K = {best_k}  "
          f"->  Accuracy = {results[best_k]['acc']*100:.2f}%")

    # -- Generate and save charts ----------------------------------------------
    print()
    print("=" * W)
    print("  GENERATING CHARTS")
    print("=" * W)

    # Rebuild sorted_dists from the manual distance section for chart_03
    all_dists = []
    for idx in range(10):
        tr = train[idx]
        d  = euclidean(tr, test[0], n_feat)
        all_dists.append((d, int(tr[-1]), idx + 1))
    sorted_dists = sorted(all_dists, key=lambda x: x[0])

    plot_class_distribution(raw, script_dir)
    plot_missing_values(before_stats, script_dir)
    plot_euclidean_distances(sorted_dists, script_dir)
    plot_model_performance(results, script_dir)
    plot_confusion_matrices(results, script_dir)
    plot_accuracy_vs_k(results, script_dir)

    print()
    print("=" * W)
    print("  Run complete.  6 chart PNGs saved to:", script_dir)
    print("=" * W)


if __name__ == "__main__":
    main()
