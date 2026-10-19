"""Browser-vs-Python parity of the deployed (int8, 3-class) models (thesis Section 5.6.2).

1. Exports the exact model inputs of every labelled clip (8 crops as lossless PNG + angle summary).
2. Runs the deployed ONNX/JSON model files in Python (ONNX Runtime + XGBoost) on those inputs.
3. Opens webapp/test/parity.html in Microsoft Edge (Playwright), which runs the same files through the
   app's own classifier.js with ONNX Runtime Web.
4. Compares the predicted class per clip and writes results/parity.json.

Needs the app server running (start_webapp.bat). Usage:
  .venv/Scripts/python scripts/parity_check.py
"""
import csv
import json
import shutil

import cv2
import numpy as np
import onnxruntime as ort
import xgboost as xgb
from playwright.sync_api import sync_playwright

from ml_common import CLASSES, DATA_DIR, RESULTS_DIR, SPLIT
from common import ROOT

WEB = ROOT / "webapp"
EXPORT = WEB / "test" / "data"
MODELS = WEB / "models"
URL = "http://127.0.0.1:8000/test/parity.html"


def export_inputs():
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)
    with SPLIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    manifest = []
    for n, r in enumerate(rows):
        d = np.load(DATA_DIR / (r["clip"][:-4] + ".npz"))
        folder = EXPORT / f"c{n:03d}"
        folder.mkdir()
        for i, crop in enumerate(d["crops"]):
            cv2.imwrite(str(folder / f"{i}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        manifest.append({"id": folder.name, "clip": r["clip"], "split": r["split"], "label": r["label"],
                         "frames": len(d["crops"]), "summary": [float(v) for v in d["summary"]]})
    (EXPORT / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def load_png_tensor(folder, i):
    rgb = cv2.cvtColor(cv2.imread(str(folder / f"{i}.png")), cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def python_predictions(manifest):
    sess = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    a = sess(MODELS / "3class" / "resnet50_ft.int8.onnx")
    bb = sess(MODELS / "backbone.int8.onnx")
    head = sess(MODELS / "3class" / "cnn_transformer_head.onnx")
    xj = json.loads((MODELS / "3class" / "xgb.json").read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(xj["booster"]).encode()))
    mean, comp = np.array(xj["pca_mean"]), np.array(xj["pca_components"])
    preds = {"A": {}, "B": {}, "C": {}}
    for item in manifest:
        folder = EXPORT / item["id"]
        frames = [load_png_tensor(folder, i) for i in range(item["frames"])]
        preds["A"][item["clip"]] = a.run(None, {"input": frames[-1]})[0][0].tolist()
        feats = np.concatenate([bb.run(None, {"input": f})[0] for f in frames])
        preds["B"][item["clip"]] = head.run(None, {"input": feats[None].astype(np.float32)})[0][0].tolist()
        x = np.concatenate([comp @ (feats[-1] - mean), item["summary"]])[None]
        preds["C"][item["clip"]] = booster.predict(xgb.DMatrix(x)).reshape(-1).tolist()
    return preds


def browser_predictions():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(URL)
        page.wait_for_function("window.__done !== undefined", timeout=20 * 60 * 1000, polling=2000)
        res = page.evaluate("window.__done")
        browser.close()
    if "error" in res:
        raise SystemExit("browser error: " + res["error"])
    return res


def main():
    manifest = export_inputs()
    print(f"exported inputs for {len(manifest)} clips")
    py = python_predictions(manifest)
    print("python predictions done; running the browser (this takes a few minutes)...")
    web = browser_predictions()
    names = CLASSES[3]
    report = {"browser": web.get("userAgent"), "cross_origin_isolated": web.get("isolated"),
              "wasm_threads": web.get("threads"), "models": {}}
    print(f"browser: cross-origin isolated = {web.get('isolated')}, ONNX Runtime threads = {web.get('threads')}")
    for key, label in (("A", "ResNet50 CNN"), ("B", "CNN-Transformer"), ("C", "XGBoost (selected)")):
        out = {}
        for subset in ("test", "all"):
            items = [m for m in manifest if subset == "all" or m["split"] == "test"]
            same = [int(np.argmax(py[key][m["clip"]]) == np.argmax(web[key][m["clip"]]["probs"])) for m in items]
            diff = [float(np.max(np.abs(np.array(py[key][m["clip"]]) - np.array(web[key][m["clip"]]["probs"]))))
                    for m in items]
            out[subset] = {"n": len(items), "agree": int(sum(same)), "agreement_pct": 100 * sum(same) / len(items),
                           "max_prob_diff": max(diff), "mean_prob_diff": float(np.mean(diff))}
        ms = [v["ms"] for v in web[key].values()]
        out["browser_ms_median"] = float(np.median(ms))
        out["browser_ms_p95"] = float(np.percentile(ms, 95))
        report["models"][label] = out
        t, a_ = out["test"], out["all"]
        print(f"{label:<20} test {t['agree']}/{t['n']} ({t['agreement_pct']:.1f}%)  all {a_['agree']}/{a_['n']} "
              f"({a_['agreement_pct']:.1f}%)  max |Δp| {a_['max_prob_diff']:.4f}  "
              f"browser {out['browser_ms_median']:.0f} ms median (headless test run)")
    (RESULTS_DIR / "parity.json").write_text(json.dumps(report, indent=2))
    print(f"saved {RESULTS_DIR / 'parity.json'}  ({names})")


if __name__ == "__main__":
    main()
