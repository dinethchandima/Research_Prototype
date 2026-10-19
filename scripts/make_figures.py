"""Chapter 5 figures from the measured results (results/, labels.csv). Writes PNGs to results/figures/.

Usage:  .venv/Scripts/python scripts/make_figures.py
"""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import ROOT

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
# validated categorical slots (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})
MODELS = ["A_ResNet50", "B_CNN_Transformer", "C_XGBoost"]
NAMES = {"A_ResNet50": "ResNet50\nCNN", "B_CNN_Transformer": "CNN-\nTransformer", "C_XGBoost": "XGBoost\n(selected)",
         "R_angle_rule": "Angle rule\n(reference)"}


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 5.1 evaluation framework
def fig_framework():
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    cols = [("Test cases", BLUE, ["Functionality", "Privacy and compatibility"]),
            ("Quantitative experiments", ORANGE, ["Classification accuracy", "Detection in live use",
                                                  "Performance and timeliness", "Measurement validity"]),
            ("Questionnaires", AQUA, ["Usability (SUS)", "Acceptance and trust"])]
    xs = [0.2, 3.5, 6.8]
    ax.add_patch(FancyBboxPatch((1.6, 5.25), 6.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f2f1ee", ec=MUTED, lw=0.8))
    ax.text(5.0, 5.55, "Artifact evaluated against the objectives and requirements", ha="center", va="center",
            fontsize=8.5, color=INK, weight="bold")
    for (title, col, items), x in zip(cols, xs):
        ax.annotate("", xy=(x + 1.5, 4.75), xytext=(5.0, 5.25),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
        ax.add_patch(FancyBboxPatch((x, 4.15), 3.0, 0.55, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=col, ec="none"))
        ax.text(x + 1.5, 4.43, title, ha="center", va="center", color="white", fontsize=8.5, weight="bold")
        for k, it in enumerate(items):
            y = 3.45 - k * 0.78
            ax.add_patch(FancyBboxPatch((x + 0.15, y - 0.27), 2.7, 0.56, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="white", ec=col, lw=1.0))
            ax.text(x + 1.5, y, it, ha="center", va="center", fontsize=8, color=INK)
    save(fig, "fig5_1_evaluation_framework.png")


# ------------------------------------------------------------------ 5.2 dataset
def fig_dataset():
    rows = list(csv.DictReader(open(ROOT / "labels.csv", encoding="utf-8")))
    cats = ["JumpRope", "JumpingJack"]
    classes = [("correct_form", "Correct form", BLUE), ("knee_valgus", "Knee valgus", ORANGE),
               ("trunk_lean", "Trunk lean", AQUA)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(cats)); w = 0.26
    for k, (key, name, col) in enumerate(classes):
        vals = [sum(r["category"] == c and r["usable"] == "yes" and r["label"] == key for r in rows) for c in cats]
        bars = a.bar(x + (k - 1) * (w + 0.02), vals, w, color=col, label=name)
        bar_labels(a, bars, "{:.0f}")
    a.set_xticks(x, cats); a.set_ylabel("Usable clips"); a.grid(axis="x", visible=False)
    a.set_title("(a) Usable clips by class (n = 126)", fontsize=9, loc="left", color=INK)
    a.set_ylim(0, 52); a.set_yticks([0, 10, 20, 30, 40])
    a.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1, columnspacing=0.8)
    ex = Counter(r["exclude_reason"] for r in rows if r["usable"] == "no" and r["category"] in cats)
    short = {"no clear landing": "No clear landing", "more than one person landing": "More than one person",
             "side view (knees not separately visible)": "Side-on view", "body not fully visible at landing":
             "Body not fully visible", "poor video quality": "Poor video quality"}
    items = sorted(ex.items(), key=lambda kv: kv[1])
    bars = b.barh([short.get(k, k) for k, _ in items], [v for _, v in items], color=BLUE, height=0.6)
    for bar in bars:
        b.annotate(f"{bar.get_width():.0f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                   xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    b.set_xlabel("Clips excluded"); b.grid(axis="y", visible=False)
    b.set_title(f"(b) Reasons for exclusion (n = {sum(ex.values())})", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig5_2_dataset.png")


# ------------------------------------------------------------------ 5.3 model comparison
def fig_models():
    test = json.load(open(ROOT / "results/3class/metrics.json"))["metrics"]
    cv = json.load(open(ROOT / "results/cv_3class/metrics.json"))["summary"]
    keys = MODELS + ["R_angle_rule"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, metric, cvkey, target, title in [
            (axes[0], "accuracy", "pooled_acc", 0.90, "(a) Accuracy"),
            (axes[1], "macro_f1", "pooled_f1", 0.85, "(b) Macro F1-score")]:
        x = np.arange(len(keys)); w = 0.36
        b1 = ax.bar(x - w / 2 - 0.01, [test[k][metric] for k in keys], w, color=BLUE, label="Held-out test set (n = 17)")
        b2 = ax.bar(x + w / 2 + 0.01, [cv[k][cvkey] for k in keys], w, color=ORANGE, label="5-fold CV, pooled (n = 126)")
        for bars in (b1, b2):
            bar_labels(ax, bars)
            bars[-1].set_alpha(0.45)               # reference rule: not one of the three models
        ax.axhline(target, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(len(keys) - 0.5, target + 0.015, f"target {target:.2f}", ha="right", fontsize=7.5, color=INK2)
        ax.set_xticks(x, [NAMES[k] for k in keys], fontsize=7.5)
        ax.set_ylim(0, 1.0); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("Score (3 classes)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig5_3_model_comparison.png")


# ------------------------------------------------------------------ 5.4 confusion matrix (selected model)
def fig_confusion():
    cm = np.array(json.load(open(ROOT / "results/cv_3class/metrics.json"))["summary"]["C_XGBoost"]["pooled_confusion"])
    names = ["Correct form", "Knee valgus", "Trunk lean"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + SEQ)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=8.5,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted by XGBoost"); ax.set_ylabel("Annotator label")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig5_4_confusion_xgboost.png")


# ------------------------------------------------------------------ 5.5 per-fold stability
def fig_folds():
    folds = json.load(open(ROOT / "results/cv_3class/metrics.json"))["per_fold"]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    cols = {"A_ResNet50": BLUE, "B_CNN_Transformer": ORANGE, "C_XGBoost": AQUA}
    for y, k in enumerate(MODELS):
        vals = [f[f"{k}_f1"] for f in folds]
        ax.scatter(vals, [y] * len(vals), s=42, color=cols[k], edgecolor="white", linewidth=1.2, zorder=3)
        m = float(np.mean(vals))
        ax.plot([m, m], [y - 0.25, y + 0.25], color=INK, lw=1.6, zorder=4)
        ax.annotate(f"mean {m:.2f}", (m, y + 0.28), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axvline(0.85, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.845, 2.55, "target 0.85", ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(3), ["ResNet50 CNN", "CNN-Transformer", "XGBoost (selected)"])
    ax.set_ylim(-0.6, 2.8); ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1-score on each test fold (3 classes)"); ax.grid(axis="y", visible=False)
    save(fig, "fig5_5_fold_stability.png")


# ------------------------------------------------------------------ 5.6 size and latency
def fig_deploy():
    m = json.load(open(ROOT / "results/3class/metrics.json"))["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    x = np.arange(3); w = 0.36
    for ax, field, target, title, unit in [
            (axes[0], "size_mb", 50, "(a) Model size", "MB"),
            (axes[1], "latency_ms_cpu1thread", 100, "(b) Latency per landing (CPU, 1 thread)", "ms")]:
        b1 = ax.bar(x - w / 2 - 0.01, [m[k][field] for k in MODELS], w, color=BLUE, label="fp32")
        b2 = ax.bar(x + w / 2 + 0.01, [m[k + "_int8"][field] for k in MODELS], w, color=ORANGE, label="int8 (deployed)")
        for bars in (b1, b2):
            bar_labels(ax, bars, "{:.0f}")
        ax.axhline(target, color=INK2, lw=1, ls=(0, (4, 3)), label="Target (50 MB / 100 ms)")
        ax.set_xticks(x, ["ResNet50\nCNN", "CNN-\nTransformer", "XGBoost\n(selected)"], fontsize=7.5)
        ax.set_ylabel(unit); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].set_ylim(0, 110); axes[1].set_ylim(0, 900)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig5_6_size_latency.png")


if __name__ == "__main__":
    fig_framework(); fig_dataset(); fig_models(); fig_confusion(); fig_folds(); fig_deploy()
