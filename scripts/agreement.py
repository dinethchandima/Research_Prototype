"""Inter-rater agreement between labels.csv (first annotator) and a second label file (thesis Table 5.5, RQ1).
Cohen's kappa is computed on the clips both raters judged usable; usability and frame agreement are reported too.

Usage:  .venv/Scripts/python scripts/agreement.py [labels_rater2.csv]
"""
import csv
import json
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import ROOT

CLASSES = ["correct_form", "knee_valgus", "trunk_lean"]


def load(path):
    with path.open(newline="", encoding="utf-8") as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def main():
    other = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "labels_rater2.csv")
    a, b = load(ROOT / "labels.csv"), load(other)
    done = [c for c, r in b.items() if r["usable"]]
    both = [c for c in done if a[c]["usable"] == "yes" and b[c]["usable"] == "yes"]
    print(f"{other.name}: {len(done)} of {len(b)} clips decided by {sorted({b[c]['annotator'] for c in done})}")
    print(f"usable agreement: {sum(a[c]['usable'] == b[c]['usable'] for c in done)}/{len(done)} "
          f"(rater 2 excluded {[c for c in done if b[c]['usable'] == 'no']})")
    if len(both) < 2:
        raise SystemExit("not enough clips labelled by both raters yet")
    ya, yb = [a[c]["label"] for c in both], [b[c]["label"] for c in both]
    kappa = cohen_kappa_score(ya, yb, labels=CLASSES)
    agree = float(np.mean([x == y for x, y in zip(ya, yb)]))
    cm = confusion_matrix(ya, yb, labels=CLASSES)
    two = ["correct_form" if y == "correct_form" else "fault" for y in ya], \
          ["correct_form" if y == "correct_form" else "fault" for y in yb]
    kappa2 = cohen_kappa_score(*two)
    dc = [abs(int(a[c]["contact_frame"]) - int(b[c]["contact_frame"])) for c in both]
    df = [abs(int(a[c]["max_flexion_frame"]) - int(b[c]["max_flexion_frame"])) for c in both]
    print(f"labels compared on {len(both)} clips: agreement {agree:.1%}, Cohen's kappa = {kappa:.2f} "
          f"(2 classes, correct vs fault: {kappa2:.2f})")
    print("confusion (rows = annotator 1, cols = annotator 2):", CLASSES)
    print(cm)
    print(f"frames: contact median |diff| {np.median(dc):.0f}, max flexion median |diff| {np.median(df):.0f} frames")
    out = {"file": other.name, "n_decided": len(done), "n_both_usable": len(both), "agreement": agree,
           "kappa_3class": kappa, "kappa_2class": kappa2, "confusion": cm.tolist(), "classes": CLASSES,
           "contact_median_abs_diff": float(np.median(dc)), "flex_median_abs_diff": float(np.median(df))}
    (ROOT / "results" / f"agreement_{other.stem}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
