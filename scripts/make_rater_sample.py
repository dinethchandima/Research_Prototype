"""Second-rater sample for inter-rater agreement (thesis Appendix C): 20% of the usable clips, stratified by the
first annotator's label, chosen at random with a fixed seed. Writes labels_rater2.csv with those clips only and
blank annotations, in shuffled order, so the second rater sees neither the first labels nor the class balance.

Usage:  .venv/Scripts/python scripts/make_rater_sample.py
then:   .venv/Scripts/python scripts/label_tool.py --annotator R2 --file labels_rater2.csv --blind
"""
import csv
import random

from common import ROOT

SEED = 2026
FRACTION = 0.20
SRC, OUT = ROOT / "labels.csv", ROOT / "labels_rater2.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT.name} already exists; delete it first if you really want a new sample.")
    with SRC.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    usable = [r for r in rows if r["usable"] == "yes" and r["label"]]
    rng = random.Random(SEED)
    sample = []
    for label in ("correct_form", "knee_valgus", "trunk_lean"):
        group = sorted((r for r in usable if r["label"] == label), key=lambda r: r["clip"])
        sample += rng.sample(group, max(1, round(FRACTION * len(group))))
    rng.shuffle(sample)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sample:
            w.writerow({c: r[c] if c in ("clip", "category", "group") else "" for c in cols})
    print(f"{len(sample)} of {len(usable)} usable clips -> {OUT.name} (seed {SEED})")


if __name__ == "__main__":
    main()
