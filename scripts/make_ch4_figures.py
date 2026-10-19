"""Chapter 4 design figures, redrawn to match the implemented artifact (same visual style as the originals).
Writes results/figures/fig4_1 ... fig4_4.

Usage:  .venv/Scripts/python scripts/make_ch4_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from common import ROOT

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE_F = "#1f3b6e", "#e8eef8"
TEAL, TEAL_F = "#2a7f80", "#e3f1f0"
GOLD, GOLD_F = "#b8792a", "#fbf0de"
GREY, INK = "#555555", "#111111"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "savefig.dpi": 300, "savefig.bbox": "tight",
                     "figure.facecolor": "white"})


def box(ax, x, y, w, h, text, ec=NAVY, fc=BLUE_F, fs=9, weight="normal", lw=1.6, color=INK, rad=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rad}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.35)


def arrow(ax, a, b, color=GREY, ls="-", lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ 4.1 architecture
def fig_architecture():
    fig, ax = canvas(9.2, 5.6, 18, 11)
    ax.add_patch(Rectangle((0.3, 3.0), 17.4, 7.7, fill=False, ec=NAVY, lw=1.6, ls=(0, (5, 3))))
    ax.text(0.6, 10.25, "User's web browser (everything runs on the device; no video leaves it)", fontsize=10,
            color=NAVY, weight="bold", va="center")
    w, h, y1, y2 = 3.7, 2.2, 7.1, 3.6
    xs = [0.8, 5.1, 9.4, 13.7]
    top = [("1. Camera input\n(getUserMedia API,\nlive video frames)", GOLD, GOLD_F),
           ("2. Pose extraction\n(MediaPipe BlazePose,\n33 landmarks/frame)", NAVY, BLUE_F),
           ("3. Positioning check +\nlanding detector and\nfeatures (8-frame crops,\nknee and trunk angles)", NAVY, BLUE_F),
           ("4. Inference engine\n(ONNX Runtime Web:\nint8 ResNet50 →\nXGBoost; class +\nconfidence)",
            TEAL, TEAL_F)]
    for x, (t, ec, fc) in zip(xs, top):
        box(ax, x, y1, w, h, t, ec, fc, fs=8.6)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y1 + h / 2), (b, y1 + h / 2))
    bottom = [("7. Session store\n(in-memory landing log;\nCSV export)", GOLD, GOLD_F),
              ("6. Presentation UI\n(skeleton overlay, ready\nindicator, cue banner,\nstatistics, summary)", NAVY, BLUE_F),
              ("5. Feedback engine\n(class → cue rule table;\non-screen + spoken cue;\ncontact, cue and\ntake-off times)",
               TEAL, TEAL_F)]
    bx = [0.8, 5.1, 9.4]
    for x, (t, ec, fc) in zip(bx, bottom):
        box(ax, x, y2, w, h, t, ec, fc, fs=8.6)
    arrow(ax, (xs[3] + w / 2 - 0.4, y1), (bx[2] + w, y2 + h / 2 + 0.3))
    arrow(ax, (bx[2], y2 + h / 2), (bx[1] + w, y2 + h / 2))
    arrow(ax, (bx[1], y2 + h / 2), (bx[0] + w, y2 + h / 2))
    ax.add_patch(Rectangle((0.3, 0.2), 17.4, 2.35, fc=BLUE_F, ec=GREY, lw=1.2))
    ax.text(0.6, 2.2, "Offline model pipeline (Python, run before deployment)", fontsize=9.5, color=GREY,
            weight="bold", va="center")
    ow, oh, oy = 3.9, 1.15, 0.45
    ox = [0.7, 5.0, 9.3, 13.6]
    off = ["UCF101 clips → screening\n+ assisted annotation", "Features: crops, ResNet50,\nangles (126 clips)",
           "Train 3 models → test set\n+ 5-fold cross-validation", "Selected model → ONNX\n(int8) + XGBoost JSON"]
    for x, t in zip(ox, off):
        box(ax, x, oy, ow, oh, t, NAVY, "white", fs=8.2)
    for a, b in zip(ox[:-1], ox[1:]):
        arrow(ax, (a + ow, oy + oh / 2), (b, oy + oh / 2))
    arrow(ax, (16.9, oy + oh), (16.9, y1), color=TEAL, ls=(0, (4, 3)))
    ax.text(16.75, 2.85, "model files", fontsize=8, color=TEAL, va="center", ha="right")
    fig.savefig(OUT / "fig4_1_architecture.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4.2 use case
def stick(ax, x, y, label):
    ax.add_patch(Ellipse((x, y + 1.05), 0.55, 0.55, fill=False, ec=INK, lw=1.2))
    ax.plot([x, x], [y + 0.78, y - 0.2], color=INK, lw=1.2)
    ax.plot([x - 0.55, x + 0.55], [y + 0.45, y + 0.45], color=INK, lw=1.2)
    ax.plot([x, x - 0.4], [y - 0.2, y - 0.9], color=INK, lw=1.2)
    ax.plot([x, x + 0.4], [y - 0.2, y - 0.9], color=INK, lw=1.2)
    ax.text(x, y - 1.25, label, ha="center", fontsize=9)


def fig_use_case():
    fig, ax = canvas(9.2, 6.0, 18, 12)
    ax.add_patch(Rectangle((3.4, 0.4), 11.2, 10.9, fill=False, ec=NAVY, lw=1.6))
    ax.text(9.0, 11.6, "Landing Form Coach (web application)", ha="center", fontsize=10.5, color=NAVY, weight="bold")
    uc = {"pos": (6.4, 9.9, "Check camera position\n(ready indicator)"),
          "start": (6.4, 8.0, "Start camera session"),
          "feed": (6.4, 6.0, "See and hear corrective\nfeedback"),
          "sum": (6.4, 4.0, "View session summary"),
          "set": (6.4, 2.0, "Change settings (model,\nfeedback type, voice)"),
          "detect": (11.4, 8.0, "Detect landing"),
          "classify": (11.4, 10.0, "Classify landing posture\n(knee valgus / trunk lean)"),
          "export": (11.4, 2.6, "Export session log (CSV)")}
    for x, y, t in uc.values():
        ax.add_patch(Ellipse((x, y), 4.2, 1.35, fc=BLUE_F, ec=NAVY, lw=1.2))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.3)
    stick(ax, 1.4, 6.2, "Athlete")
    stick(ax, 16.6, 5.4, "Coach")
    for k in ("pos", "start", "feed", "sum", "set"):
        x, y, _ = uc[k]
        ax.plot([1.95, x - 2.1], [6.65, y], color=GREY, lw=0.9)
    for k in ("feed", "sum", "export"):      # coach: watches feedback, reviews and exports the session
        x, y, _ = uc[k]
        ax.plot([16.05, x + 2.1], [5.85, y], color=GREY, lw=0.9)
    arrow(ax, (8.5, 8.0), (9.3, 8.0), color=NAVY, ls=(0, (4, 3)), lw=1.1)
    ax.text(8.9, 8.35, "«include»", ha="center", fontsize=7.5, color=NAVY)
    arrow(ax, (11.4, 8.68), (11.4, 9.32), color=NAVY, ls=(0, (4, 3)), lw=1.1)
    ax.text(12.6, 9.0, "«include»", ha="center", fontsize=7.5, color=NAVY)
    arrow(ax, (10.0, 3.1), (7.6, 3.55), color=NAVY, ls=(0, (4, 3)), lw=1.1)
    ax.text(9.2, 3.65, "«extend»", ha="center", fontsize=7.5, color=NAVY)
    fig.savefig(OUT / "fig4_2_use_case.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4.3 wireframes
def phone(ax, x, title, label):
    ax.add_patch(FancyBboxPatch((x, 0.9), 4.0, 9.3, boxstyle="round,pad=0.02,rounding_size=0.25", fc="white",
                                ec="#333333", lw=2))
    ax.add_patch(Rectangle((x + 0.1, 9.25), 3.8, 0.8, fc=NAVY, ec="none"))
    ax.text(x + 2.0, 9.65, title, ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.text(x + 2.0, 0.45, label, ha="center", fontsize=9, color=GREY)


def button(ax, x, y, w, t, fc=TEAL, color="white", h=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=color, fontsize=8)


def figure_man(ax, cx, base, color=TEAL):
    ax.add_patch(Ellipse((cx, base + 3.0), 0.45, 0.45, fill=False, ec=color, lw=1.6))
    ax.plot([cx, cx], [base + 2.77, base + 1.5], color=color, lw=1.6)
    ax.plot([cx - 0.6, cx + 0.6], [base + 2.45, base + 2.45], color=color, lw=1.6)
    ax.plot([cx, cx - 0.32, cx - 0.3], [base + 1.5, base + 0.8, base], color=color, lw=1.6)
    ax.plot([cx, cx + 0.32, cx + 0.3], [base + 1.5, base + 0.8, base], color=color, lw=1.6)


def fig_wireframes():
    fig, ax = canvas(9.4, 5.4, 18.4, 10.6)
    # (a) set-up guide
    x = 0.2
    phone(ax, x, "Landing Form Coach", "(a) Set-up guide")
    ax.text(x + 2.0, 8.75, "Get set up", ha="center", fontsize=9, weight="bold")
    for k, t in enumerate(["Camera 3 m away,\nat hip height", "Face the camera,\nwhole body in view",
                           "Land on both feet;\ncue after each landing"]):
        y = 7.6 - k * 1.25
        ax.add_patch(Ellipse((x + 0.55, y + 0.3), 0.38, 0.38, fc=NAVY, ec="none"))
        ax.text(x + 0.55, y + 0.3, str(k + 1), ha="center", va="center", color="white", fontsize=7, weight="bold")
        ax.text(x + 0.9, y + 0.3, t, va="center", fontsize=6.8)
    button(ax, x + 0.4, 3.0, 3.2, "Start camera")
    ax.text(x + 2.0, 2.0, "Video is processed on this\ndevice and never uploaded.", ha="center", fontsize=7,
            color=GREY)
    # (b) live analysis
    x = 4.8
    phone(ax, x, "Live analysis", "(b) Live analysis")
    ax.add_patch(Rectangle((x + 0.3, 3.4), 3.4, 5.6, fc="#f0f0f0", ec="#888888", lw=1))
    ax.add_patch(FancyBboxPatch((x + 0.45, 8.35), 3.1, 0.45, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#333333", ec="none"))
    ax.add_patch(Ellipse((x + 0.68, 8.575), 0.16, 0.16, fc="#2e9e4f", ec="none"))
    ax.text(x + 0.85, 8.575, "Ready: jump when you like", va="center", fontsize=5.6, color="white")
    figure_man(ax, x + 2.0, 4.55)
    ax.add_patch(Rectangle((x + 0.45, 3.55), 3.1, 0.75, fc="#c2410c", ec="none"))
    ax.text(x + 2.0, 3.92, "Knees caving in:\npush your knees out", ha="center", va="center", color="white",
            fontsize=6.2, linespacing=1.2)
    for k, (v, t) in enumerate([("5", "Landings"), ("3", "Good"), ("1", "Knees in"), ("1", "Leaning")]):
        cx = x + 0.75 + k * 0.85
        ax.text(cx, 2.75, v, ha="center", fontsize=9, weight="bold")
        ax.text(cx, 2.35, t, ha="center", fontsize=5.8, color=GREY)
    button(ax, x + 0.4, 1.25, 3.2, "End session", fc=BLUE_F, color=NAVY)
    # (c) settings
    x = 9.4
    phone(ax, x, "Settings", "(c) Settings")
    for k, (lab, val) in enumerate([("Model", "XGBoost (recommended)"), ("Feedback type", "Specific")]):
        y = 8.2 - k * 1.5
        ax.text(x + 0.4, y + 0.45, lab, fontsize=8, weight="bold")
        ax.add_patch(Rectangle((x + 0.4, y - 0.35), 3.2, 0.6, fc="white", ec="#555555", lw=1))
        ax.text(x + 0.55, y - 0.05, val, va="center", fontsize=6.8)
        ax.text(x + 3.42, y - 0.05, "▾", va="center", ha="center", fontsize=8)
    for k, (lab, on) in enumerate([("Voice cues", True), ("Researcher details", False)]):
        y = 4.9 - k * 1.0
        ax.add_patch(FancyBboxPatch((x + 0.4, y - 0.18), 0.75, 0.36, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=TEAL if on else "#cccccc", ec="none"))
        ax.add_patch(Ellipse((x + (1.0 if on else 0.58), y), 0.3, 0.3, fc="white", ec="none"))
        ax.text(x + 1.35, y, lab, va="center", fontsize=7.8)
    ax.text(x + 2.0, 2.4, "Settings are remembered on\nthis device only.", ha="center", fontsize=7, color=GREY)
    # (d) summary
    x = 14.0
    phone(ax, x, "Session summary", "(d) Session summary (sample data)")
    ax.add_patch(Ellipse((x + 1.15, 7.9), 1.3, 1.3, fc="white", ec="#2e9e4f", lw=6))
    ax.text(x + 1.15, 7.9, "60%", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(x + 2.0, 7.9, "3 of 5 landings\nrated good", va="center", fontsize=7.2)
    for k, (lab, v) in enumerate([("Knees caving in", "1"), ("Leaning to the side", "1"),
                                  ("Cues before take-off", "4 of 4"), ("Median landing→cue", "0.42 s")]):
        y = 6.5 - k * 0.55
        ax.text(x + 0.35, y, lab, fontsize=6.8)
        ax.text(x + 3.65, y, v, fontsize=6.8, ha="right", weight="bold")
    ax.add_patch(Rectangle((x + 0.3, 2.85), 3.4, 1.15, fc=BLUE_F, ec="none"))
    ax.text(x + 0.45, 3.42, "Focus next time: push your\nknees out over your toes", va="center", fontsize=6.8)
    button(ax, x + 0.4, 1.4, 3.2, "Download CSV")
    fig.savefig(OUT / "fig4_3_wireframes.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4.4 model processing flow
def fig_models():
    fig, ax = canvas(9.2, 5.0, 18, 10)
    box(ax, 0.2, 3.6, 3.0, 2.8, "Landing window\n(contact → max\nflexion): 8 frames,\nathlete crops\n224 × 224\n"
                                "+ 6 angle features", GOLD, GOLD_F, fs=8)
    rows = [(8.2, "Candidate A – ResNet50 CNN",
             ["Crop at maximum\nknee flexion", "ResNet50, ImageNet-\npretrained (layer 4\n+ head fine-tuned)",
              "Softmax\n(3 classes)"], [NAVY, NAVY, NAVY], [BLUE_F, BLUE_F, BLUE_F]),
            (4.6, "Candidate B – Hybrid CNN-Transformer",
             ["Frozen ResNet50:\n2048 features for\neach of 8 frames", "Transformer encoder\n(2 layers, 4 heads;\ntemporal attention)",
              "Softmax\n(3 classes)"], [NAVY, TEAL, NAVY], [BLUE_F, TEAL_F, BLUE_F]),
            (1.0, "Candidate C – XGBoost on deep-learned features",
             ["Frozen ResNet50 at\nmax flexion → PCA (16)\n+ 6 angle features", "XGBoost\n(200 trees, depth 3)",
              "Class +\nprobability"], [NAVY, TEAL, NAVY], [BLUE_F, TEAL_F, BLUE_F])]
    xs, w, h = [4.2, 8.0, 11.8], 3.3, 1.6
    for y, title, texts, ecs, fcs in rows:
        ax.text(4.2, y + h + 0.3, title, fontsize=9.5, color=NAVY, weight="bold")
        for x, t, ec, fc in zip(xs, texts, ecs, fcs):
            box(ax, x, y, w, h, t, ec, fc, fs=7.8)
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
        arrow(ax, (3.2, 5.0), (xs[0], y + h / 2))
        arrow(ax, (xs[2] + w, y + h / 2), (15.6, 5.0))
    box(ax, 15.6, 3.4, 2.3, 3.2, "Compare:\naccuracy,\nmacro F1,\nlatency, size\n(test set +\n5-fold CV)\n"
                                 "→ XGBoost\nselected", GOLD, GOLD_F, fs=7.6)
    fig.savefig(OUT / "fig4_4_models.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_use_case(); fig_wireframes(); fig_models()
    print("wrote fig4_1 ... fig4_4")
