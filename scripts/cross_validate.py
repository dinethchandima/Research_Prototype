"""5-fold grouped, stratified cross-validation of the three models (plus the angle rule).

Each fold holds out about 1/5 of the groups as test; the remaining groups are split
again into train + val (val only picks the best epoch). Every labelled clip is
tested exactly once, so the pooled results use all clips instead of the 17 in the
single test split. Same model code and settings as train_models.py (fp32 only).

Writes results/cv_<n>class/: summary.md, per_fold.csv, pooled_predictions.csv,
pooled confusion matrices.

Usage:  .venv/Scripts/python -u scripts/cross_validate.py --classes 3
"""
import argparse
import csv
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

import train_models as tm
from ml_common import CLASSES, DATA_DIR, RESULTS_DIR, SEED, labelled_clips, to_class

K = 5
MODELS = ["A_ResNet50", "B_CNN_Transformer", "C_XGBoost", "R_angle_rule"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, choices=[2, 3], default=3)
    ap.add_argument("--epochs-a", type=int, default=12)
    ap.add_argument("--epochs-b", type=int, default=150)
    args = ap.parse_args()
    n, names = args.classes, CLASSES[args.classes]
    torch.set_num_threads(8)

    rows = labelled_clips()
    clips = []
    for r in rows:
        d = np.load(DATA_DIR / (r["clip"][:-4] + ".npz"))
        clips.append({"clip": r["clip"], "group": r["group"], "label3": r["label"], "y": to_class(r["label"], n),
                      "crops": d["crops"], "feats": d["feats"], "summary": d["summary"]})
    y_all = np.array([c["y"] for c in clips])
    strat = np.array([CLASSES[3].index(c["label3"]) for c in clips])   # same folds for 2 and 3 classes
    groups = np.array([c["group"] for c in clips])
    print(f"{len(clips)} clips, {len(set(groups))} groups, classes {np.bincount(y_all, minlength=n).tolist()} {names}")

    outer = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
    pooled = {m: np.full(len(clips), -1) for m in MODELS}
    per_fold = []
    for fold, (trval_idx, test_idx) in enumerate(outer.split(np.zeros(len(clips)), strat, groups), 1):
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED + fold)
        tr_i, va_i = next(inner.split(np.zeros(len(trval_idx)), strat[trval_idx], groups[trval_idx]))
        data = {"train": [clips[i] for i in trval_idx[tr_i]], "val": [clips[i] for i in trval_idx[va_i]],
                "test": [clips[i] for i in test_idx]}
        print(f"\n=== Fold {fold}/{K}: train {len(data['train'])}, val {len(data['val'])}, "
              f"test {len(data['test'])} (test classes {np.bincount(y_all[test_idx], minlength=n).tolist()}) ===")
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        print("Training A: ResNet50 ...")
        a = tm.train_resnet(data, n, args.epochs_a)
        print("Training B: CNN-Transformer ...")
        b = tm.train_transformer(data, n, args.epochs_b)
        print("Training C: XGBoost ...")
        pca, xgb = tm.train_xgb(data, n)
        probs = {"A_ResNet50": tm.predict_resnet(a, data["test"]),
                 "B_CNN_Transformer": tm.predict_transformer(b, data["test"]),
                 "C_XGBoost": tm.predict_xgb(pca, xgb, data["test"]),
                 "R_angle_rule": tm.predict_rule(data["test"], n)}
        yt = y_all[test_idx]
        row = {"fold": fold, "n_test": len(test_idx)}
        for m in MODELS:
            p = probs[m].argmax(1)
            pooled[m][test_idx] = p
            row[f"{m}_acc"] = accuracy_score(yt, p)
            row[f"{m}_f1"] = tm.macro_f1(yt, p, n)
            print(f"  {m:<18} accuracy {row[f'{m}_acc']:.3f}  macro F1 {row[f'{m}_f1']:.3f}")
        per_fold.append(row)

    out = RESULTS_DIR / f"cv_{n}class"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0]))
        w.writeheader()
        w.writerows(per_fold)
    with (out / "pooled_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "true"] + MODELS)
        for i, c in enumerate(clips):
            w.writerow([c["clip"], names[y_all[i]]] + [names[pooled[m][i]] for m in MODELS])

    lines = [f"# 5-fold grouped cross-validation ({n} classes)", "",
             f"{len(clips)} clips in {len(set(groups))} groups; every clip is tested once. "
             f"Class counts: {np.bincount(y_all, minlength=n).tolist()} ({', '.join(names)}). fp32 models.", "",
             "| Model | Accuracy mean ± SD over folds | Macro F1 mean ± SD | Pooled accuracy | Pooled macro F1 |",
             "|---|---|---|---|---|"]
    summary = {}
    for m in MODELS:
        acc = np.array([r[f"{m}_acc"] for r in per_fold])
        f1 = np.array([r[f"{m}_f1"] for r in per_fold])
        pa, pf = accuracy_score(y_all, pooled[m]), tm.macro_f1(y_all, pooled[m], n)
        summary[m] = {"acc_mean": acc.mean(), "acc_sd": acc.std(ddof=1), "f1_mean": f1.mean(), "f1_sd": f1.std(ddof=1),
                      "pooled_acc": pa, "pooled_f1": pf,
                      "pooled_confusion": confusion_matrix(y_all, pooled[m], labels=list(range(n))).tolist()}
        lines.append(f"| {m} | {acc.mean():.3f} ± {acc.std(ddof=1):.3f} | {f1.mean():.3f} ± {f1.std(ddof=1):.3f} | "
                     f"{pa:.3f} | {pf:.3f} |")
        tm.save_confusion(np.array(summary[m]["pooled_confusion"]), names, out / f"confusion_{m}.png",
                          f"{m}  CV pooled ({n} classes, n={len(clips)})")
    lines += ["", "R_angle_rule is a reference only; its thresholds came from the pilot clips.", "",
              "## McNemar's test on pooled predictions (exact)", "",
              "| Pair | only 1st correct | only 2nd correct | p |", "|---|---|---|---|"]
    mc = {}
    for i, p in enumerate(MODELS[:3]):
        for q in MODELS[i + 1:3]:
            b_, c_, pv = tm.mcnemar(y_all, pooled[p], pooled[q])
            mc[f"{p} vs {q}"] = {"only_first_correct": b_, "only_second_correct": c_, "p_value": pv}
            lines.append(f"| {p} vs {q} | {b_} | {c_} | {pv:.3f} |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "mcnemar": mc, "per_fold": per_fold},
                                                 indent=2, default=float))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
