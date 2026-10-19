"""Step 1: build labels.csv from the UCF101 category folders listed in common.py.

Fills in clip, category and group for every category in common.CATEGORIES. All other columns stay empty
for the student to fill in during annotation (thesis Appendix C).

Safe to re-run: rows that already exist in labels.csv keep their
annotations, and only new clips are added.

Usage:  python scripts/build_labels.py
"""
import csv
import re

from common import CATEGORIES, ROOT, category_dir

OUT = ROOT / "labels.csv"
COLUMNS = [
    "clip", "category", "group", "usable", "exclude_reason",
    "contact_frame", "max_flexion_frame", "knee_angle", "trunk_angle",
    "label", "both_flag", "annotator",
]
# e.g. v_HighJump_g01_c03.avi  ->  category=HighJump, group=g01
NAME_RE = re.compile(r"^v_(?P<category>[A-Za-z]+)_(?P<group>g\d+)_c\d+\.avi$", re.IGNORECASE)


def load_existing():
    """Return {clip: row} for rows already in labels.csv, so we never overwrite labels."""
    if not OUT.exists():
        return {}
    with OUT.open(newline="", encoding="utf-8") as f:
        return {row["clip"]: row for row in csv.DictReader(f)}


def main():
    existing = load_existing()
    rows, skipped = [], []

    for category in CATEGORIES:
        folder = category_dir(category)
        for path in sorted(folder.iterdir()):
            if path.name.startswith("._"):  # macOS AppleDouble metadata, not a video
                continue
            m = NAME_RE.match(path.name)
            if not m or m["category"] != category:
                skipped.append(path.name)
                continue
            row = existing.get(path.name) or {c: "" for c in COLUMNS}
            row.update(clip=path.name, category=category, group=f"{category}_{m['group']}")
            rows.append(row)

    # never silently drop a row you have already annotated
    kept = {r["clip"] for r in rows}
    lost = [c for c, r in existing.items() if c not in kept and (r["usable"] or r["label"])]
    if lost:
        raise SystemExit(f"Not writing: {len(lost)} annotated clips would be removed "
                         f"(e.g. {lost[:3]}). Put their category back in common.CATEGORIES.")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {OUT.name} ({len(existing)} already existed)")
    for category in CATEGORIES:
        n = sum(r["category"] == category for r in rows)
        g = len({r["group"] for r in rows if r["category"] == category})
        print(f"  {category:<18} {n:>4} clips in {g} groups")
    if skipped:
        print(f"Skipped {len(skipped)} files with unexpected names: {skipped[:5]}")


if __name__ == "__main__":
    main()
