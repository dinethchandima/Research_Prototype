"""Needs-assessment questionnaire (thesis Appendix A, Section 3.6): descriptive statistics, construct means and
Cronbach's alpha, multi-select counts and open-answer themes. Writes results/needs_survey.json and prints a summary.

Usage:  .venv/Scripts/python scripts/survey_analysis.py
"""
import csv
import json
import re
from collections import Counter

import numpy as np

from common import ROOT

SRC = ROOT / "survays" / "plyometric_survey_responses.csv"
SCALE = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly agree": 5}
CONSTRUCTS = {"Perceived usefulness": ["C1", "C2", "C3"], "Perceived ease of use": ["C4", "C5", "C6"],
              "Trust in functionality": ["C7", "C8", "C9"], "Intention to use": ["C10", "C11", "C12"],
              "Trust in privacy": ["C13", "C14"]}


def alpha(items):
    x = np.array(items, dtype=float).T          # respondents x items
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def code(header):
    m = re.search(r"\[(B2\.\d|C\d+)\b", header) or re.match(r"([A-E]\d+)\.", header)
    return m.group(1) if m else header


def main():
    rows = list(csv.reader(SRC.open(encoding="utf-8-sig")))
    head, data = rows[0], rows[1:]
    col = {code(h): i for i, h in enumerate(head)}
    n = len(data)
    out = {"n": n, "single": {}, "multi": {}, "likert": {}, "constructs": {}, "open": {}}

    def single(c):
        cnt = Counter(r[col[c]] for r in data if r[col[c]].strip())
        out["single"][c] = {"question": head[col[c]], "counts": dict(cnt.most_common()), "n": sum(cnt.values())}
        return cnt

    def multi(c):
        cnt = Counter(o.strip() for r in data for o in r[col[c]].split(",") if o.strip())
        answered = sum(1 for r in data if r[col[c]].strip())
        out["multi"][c] = {"question": head[col[c]], "counts": dict(cnt.most_common()), "n": answered}
        return cnt, answered

    print(f"N = {n}")
    for c in ("A1", "A2", "A3", "A4", "A5", "B1", "B3", "D1", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
              "D12", "D13"):
        cnt = single(c)
        tot = sum(cnt.values())
        print(f"{c}: " + "; ".join(f"{k} {v} ({v / tot:.0%})" for k, v in cnt.most_common()))
    for c in ("B4", "B5", "D2", "D3"):
        cnt, answered = multi(c)
        print(f"{c} (n={answered}, multi-select): " + "; ".join(f"{k} {v} ({v / answered:.0%})" for k, v in cnt.most_common()))

    for c in ["B2.1", "B2.2", "B2.3", "B2.4"] + [f"C{i}" for i in range(1, 15)]:
        v = [SCALE[r[col[c]]] for r in data if r[col[c]] in SCALE]
        agree = sum(x >= 4 for x in v) / len(v)
        out["likert"][c] = {"question": head[col[c]], "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                            "agree_pct": agree, "n": len(v)}
        print(f"{c}: mean {np.mean(v):.2f} (SD {np.std(v, ddof=1):.2f}), agree/strongly agree {agree:.0%}")
    for name, items in CONSTRUCTS.items():
        mat = [[SCALE[r[col[i]]] for r in data] for i in items]
        scores = np.mean(mat, axis=0)
        a = alpha(mat)
        out["constructs"][name] = {"items": items, "mean": float(scores.mean()), "sd": float(scores.std(ddof=1)),
                                   "alpha": float(a)}
        print(f"{name} ({', '.join(items)}): mean {scores.mean():.2f} (SD {scores.std(ddof=1):.2f}), alpha {a:.2f}")
    for c in ("E1", "E2", "E3"):
        cnt = Counter(r[col[c]].strip().lower().rstrip(".") for r in data if r[col[c]].strip())
        out["open"][c] = {"question": head[col[c]], "counts": dict(cnt.most_common()), "n": sum(cnt.values())}
        print(f"{c} (n={sum(cnt.values())}): {cnt.most_common(8)}")
    (ROOT / "results" / "needs_survey.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
