"""Grouped, stratified 70/15/15 train/val/test split (thesis Section 4).

Whole groups (gXX = same source video) go to one split only, so near-identical
clips never sit in both train and test. Among many random group orders, the one
whose class mix in each split is closest to the overall mix is kept. Every class
with enough clips must appear in val and test.

The split is made on the 3 classes and reused for the 2-class models, so both
are compared on exactly the same clips.

Usage:  .venv/Scripts/python scripts/make_split.py
"""
import csv
import random
from collections import Counter, defaultdict

from ml_common import CLASSES, SEED, SPLIT, labelled_clips

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign(groups, order):
    total = sum(len(v) for v in groups.values())
    out, done = {}, 0
    for g in order:
        frac = done / total
        out[g] = "train" if frac < 0.70 else "val" if frac < 0.85 else "test"
        done += len(groups[g])
    return out


def score(groups, split_of, totals):
    counts = defaultdict(Counter)
    for g, labels in groups.items():
        counts[split_of[g]].update(labels)
    s = 0.0
    for name, ratio in RATIOS.items():
        for cls, n in totals.items():
            s += (counts[name][cls] - ratio * n) ** 2
            if name != "train" and n >= 3 and counts[name][cls] == 0:
                s += 1000  # a class missing from val/test makes its metrics meaningless
    return s


def main():
    rows = labelled_clips()
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r["label"])
    totals = Counter(r["label"] for r in rows)

    rng = random.Random(SEED)
    order, best = list(groups), None
    for _ in range(5000):
        rng.shuffle(order)
        split_of = assign(groups, order)
        s = score(groups, split_of, totals)
        if best is None or s < best[0]:
            best = (s, split_of)
    split_of = best[1]

    with SPLIT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "category", "group", "label", "split"])
        for r in rows:
            w.writerow([r["clip"], r["category"], r["group"], r["label"], split_of[r["group"]]])

    print(f"{len(rows)} labelled clips in {len(groups)} groups -> {SPLIT.name}")
    print(f"{'':<14}" + "".join(f"{c:>14}" for c in CLASSES[3]) + f"{'total':>8}{'groups':>8}")
    for name in RATIOS:
        c = Counter(r["label"] for r in rows if split_of[r["group"]] == name)
        g = sum(1 for v in split_of.values() if v == name)
        print(f"{name:<14}" + "".join(f"{c[k]:>14}" for k in CLASSES[3]) + f"{sum(c.values()):>8}{g:>8}")
    missing = [k for k in CLASSES[3] if totals[k] < 3]
    if missing:
        print(f"WARNING: fewer than 3 clips of {missing}: 3-class test metrics for them are not meaningful.")


if __name__ == "__main__":
    main()
