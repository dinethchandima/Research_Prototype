"""Analyse the live pilot (thesis 5.6.3, Tables 5.8 and 5.9, and timing for Table 5.4).

Inputs:  Live_Pilot_Sheet.xlsx (rater and app results per landing) and the app CSVs live/P01.csv, live/P02.csv, ...
Output:  printed tables and results/live_pilot.json

Usage:  .venv/Scripts/python scripts/live_analysis.py
"""
import csv
import json
import math

import numpy as np
from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

from common import ROOT

CLASSES = ["Correct form", "Knee valgus", "Trunk lean"]
rng = np.random.default_rng(2026)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def kappa_ci(a, b, reps=2000):
    k = cohen_kappa_score(a, b, labels=CLASSES)
    a, b = np.array(a), np.array(b)
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1:
            boots.append(cohen_kappa_score(a[i], b[i], labels=CLASSES))
    return k, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(t):
    p, lo, hi = t
    return "–" if p is None else f"{p:.2f} ({lo:.2f} to {hi:.2f})"


def main():
    ws = load_workbook(ROOT / "Live_Test_Sheet.xlsx", data_only=False)["Landings"]
    rows = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        pid, n, instructed, rater, detection, app, conf, rater2 = r[:8]
        if rater:
            rows.append(dict(pid=pid, n=n, instructed=instructed, rater=rater, detection=detection, app=app,
                             conf=conf, rater2=rater2))
    det = [r for r in rows if r["detection"] == "One landing" and r["app"]]
    print(f"{len(rows)} landings rated, {len(det)} detected once by the app, "
          f"{sum(r['detection'] == 'Missed' for r in rows)} missed, "
          f"{sum(r['detection'] == 'Extra landings' for r in rows)} with extra landings")

    cm = [[sum(1 for r in det if r["rater"] == a and r["app"] == b) for b in CLASSES] for a in CLASSES]
    print("\nTable 5.8 (rows = rater, cols = app):", CLASSES)
    for a, line in zip(CLASSES, cm):
        print(f"  {a:13s} {line} total {sum(line)}")

    fault = lambda x: x != "Correct form"
    res = {
        "detection_knee_valgus": wilson(sum(r["app"] == "Knee valgus" for r in det if r["rater"] == "Knee valgus"),
                                        sum(r["rater"] == "Knee valgus" for r in det)),
        "detection_trunk_lean": wilson(sum(r["app"] == "Trunk lean" for r in det if r["rater"] == "Trunk lean"),
                                       sum(r["rater"] == "Trunk lean" for r in det)),
        "specificity": wilson(sum(r["app"] == "Correct form" for r in det if r["rater"] == "Correct form"),
                              sum(r["rater"] == "Correct form" for r in det)),
        "false_positive_rate": wilson(sum(fault(r["app"]) for r in det if r["rater"] == "Correct form"),
                                      sum(r["rater"] == "Correct form" for r in det)),
        "false_negative_rate": wilson(sum(r["app"] == "Correct form" for r in det if fault(r["rater"])),
                                      sum(fault(r["rater"]) for r in det)),
    }
    print("\nTable 5.9:")
    for k, v in res.items():
        print(f"  {k:24s} {fmt(v)}")
    out = {"n_rated": len(rows), "n_detected": len(det), "confusion": cm, "classes": CLASSES,
           "measures": {k: list(v) for k, v in res.items()}}
    if len(det) >= 2:
        k = kappa_ci([r["rater"] for r in det], [r["app"] for r in det])
        out["kappa_system_vs_rater"] = k
        print(f"  {'kappa, system vs rater':24s} {k[0]:.2f} ({k[1]:.2f} to {k[2]:.2f})")
    two = [r for r in rows if r["rater2"]]
    if len(two) >= 2:
        k2 = kappa_ci([r["rater"] for r in two], [r["rater2"] for r in two])
        out["kappa_rater_vs_rater2"] = k2
        print(f"  {'kappa, rater vs rater 2':24s} {k2[0]:.2f} ({k2[1]:.2f} to {k2[2]:.2f}) on {len(two)} landings")

    print("\nBy participant (app = rater):")
    for pid in sorted({r["pid"] for r in det}):
        d = [r for r in det if r["pid"] == pid]
        print(f"  {pid}: {sum(r['app'] == r['rater'] for r in d)}/{len(d)}")
    for kind in ("Natural", "Instructed"):
        d = [r for r in det if str(r["instructed"]).startswith(kind) and fault(r["rater"])]
        print(f"  faults detected, {kind.lower()} landings: {sum(r['app'] == r['rater'] for r in d)}/{len(d)}")

    cues, before, takeoffs = [], 0, 0
    for f in sorted((ROOT / "live").glob("P*.csv")):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("toCueMs"):
                cues.append(float(r["toCueMs"]))
            if r.get("cueBeforeTakeoff"):
                takeoffs += 1
                before += r["cueBeforeTakeoff"] == "true"
    if cues:
        out["timing"] = {"n": len(cues), "median_s": float(np.median(cues)) / 1000,
                         "p95_s": float(np.percentile(cues, 95)) / 1000, "cue_before_takeoff": [before, takeoffs]}
        print(f"\nTiming ({len(cues)} landings in the CSVs): landing→cue median {np.median(cues) / 1000:.2f} s, "
              f"p95 {np.percentile(cues, 95) / 1000:.2f} s; cue before next take-off {before}/{takeoffs}")
    (ROOT / "results" / "live_pilot.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
