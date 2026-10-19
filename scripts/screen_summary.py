"""Per-category screening table from suggestions.csv, to decide which UCF101
categories are worth labelling. "No flags" means the helper found nothing wrong;
it is NOT the same as usable. The student still checks every clip.

Usage:  python scripts/screen_summary.py
"""
import csv

from common import CATEGORIES, ROOT


def main():
    with (ROOT / "labels.csv").open(newline="", encoding="utf-8") as f:
        category = {r["clip"]: r["category"] for r in csv.DictReader(f)}
    with (ROOT / "suggestions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    totals = {c: sum(v == c for v in category.values()) for c in CATEGORIES}
    head = f"{'category':<18} {'clips':>5} {'done':>5} {'person':>6} {'1 pers':>6} {'frontal':>7} {'landing':>7} {'no flags':>8}"
    print(head)
    print("-" * len(head))
    grand = [0] * 7
    for c in CATEGORIES:
        rs = [r for r in rows if category.get(r["clip"]) == c]
        counts = [
            totals[c],
            len(rs),
            sum("person often not found" not in r["check"] for r in rs),
            sum("2+ people" not in r["check"] for r in rs),
            sum(r["view_hint"] == "frontal/back" for r in rs),
            sum(r["contact_frame"] != "" and "very short window" not in r["check"] for r in rs),
            sum(r["check"] == "" for r in rs),
        ]
        grand = [g + n for g, n in zip(grand, counts)]
        print(f"{c:<18} " + " ".join(f"{n:>{w}}" for n, w in zip(counts, (5, 5, 6, 6, 7, 7, 8))))
    print("-" * len(head))
    print(f"{'TOTAL':<18} " + " ".join(f"{n:>{w}}" for n, w in zip(grand, (5, 5, 6, 6, 7, 7, 8))))


if __name__ == "__main__":
    main()
