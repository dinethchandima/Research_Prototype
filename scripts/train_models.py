"""Train and compare the three models on the same grouped split (thesis Section 4/5).

  A  ResNet50 CNN            fine-tuned (layer4 + head) on the max-flexion crop
  B  Hybrid CNN-Transformer  frozen ResNet50 features of T window frames -> Transformer
  C  XGBoost                 PCA of frozen ResNet50 features at max flexion + angle features
  R  angle rule (reference)  FPPA / trunk thresholds from the pilot; not one of the 3 models

Everything is evaluated on the held-out TEST groups only (val is used to pick
the best epoch). Writes results/<n>class/ (metrics, confusion matrices, McNemar,
predictions, summary.md) and ONNX/JSON models for the web app to webapp/models/.

Usage:
  .venv/Scripts/python scripts/train_models.py --classes 3
  .venv/Scripts/python scripts/train_models.py --classes 2     # correct_form vs fault
"""
import argparse
import copy
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

from extract_features import MEAN, STD
from ml_common import CLASSES, DATA_DIR, EXPORT_DIR, RESULTS_DIR, SEED, SPLIT, T, to_class

RULE_FPPA = 12.5    # pilot: best single FPPA cut for knee_valgus (76% agreement on 21 clips)
RULE_TRUNK = 6.0    # pilot: only trunk_lean clip had 6.7 deg, all others <= 5.8 (n = 1!)
TARGETS = {"accuracy": 0.90, "macro_f1": 0.85, "latency_ms": 100, "size_mb": 50}


# ---------------------------------------------------------------- data

def load(n_classes):
    with SPLIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {"train": [], "val": [], "test": []}
    for r in rows:
        d = np.load(DATA_DIR / (r["clip"][:-4] + ".npz"))
        data[r["split"]].append({"clip": r["clip"], "y": to_class(r["label"], n_classes),
                                 "crops": d["crops"], "feats": d["feats"], "summary": d["summary"]})
    return data


def class_weights(ys, n):
    counts = np.bincount(ys, minlength=n).astype(np.float32)
    w = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(w)


def macro_f1(y, p, n):
    return f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0)


# ---------------------------------------------------------------- A: ResNet50

class Normalised(nn.Module):
    """Takes RGB in 0..1 (what the browser has) and applies ImageNet normalisation."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def to_unit(crops):
    return torch.from_numpy(crops.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)


def augment(x):
    """Flip (valgus / lean can be on either side), small zoom, brightness/contrast."""
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    s = float(torch.empty(1).uniform_(0.85, 1.0))
    size = int(x.shape[-1] * s)
    i, j = np.random.randint(0, x.shape[-1] - size + 1, 2)
    x = nn.functional.interpolate(x[:, :, i:i + size, j:j + size], size=x.shape[-1], mode="bilinear",
                                  align_corners=False)
    x = x * float(torch.empty(1).uniform_(0.8, 1.2)) + float(torch.empty(1).uniform_(-0.1, 0.1))
    return x.clamp(0, 1)


def train_resnet(data, n, epochs):
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("layer4")
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, n))
    model = Normalised(net)

    # train on the last 3 window frames (around max flexion); evaluate on the max-flexion frame
    xs = torch.cat([to_unit(c["crops"][-3:]) for c in data["train"]])
    ys = torch.tensor(np.repeat([c["y"] for c in data["train"]], 3))
    xv = torch.cat([to_unit(c["crops"][-1:]) for c in data["val"]])
    yv = np.array([c["y"] for c in data["val"]])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ys.numpy(), n))
    opt = torch.optim.AdamW([{"params": net.layer4.parameters(), "lr": 1e-4},
                             {"params": net.fc.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        for m in (net.conv1, net.bn1, net.layer1, net.layer2, net.layer3):
            m.eval()  # frozen part keeps ImageNet batch-norm statistics
        perm = torch.randperm(len(xs))
        for k in range(0, len(xs), 16):
            b = perm[k:k + 16]
            opt.zero_grad()
            loss = loss_fn(model(augment(xs[b])), ys[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv).argmax(1).numpy()
        f1 = macro_f1(yv, pv, n)
        print(f"  A epoch {ep + 1}/{epochs}: loss {loss.item():.3f}  val macro F1 {f1:.3f}")
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return model.eval()


def predict_resnet(model, clips):
    with torch.no_grad():
        return model(torch.cat([to_unit(c["crops"][-1:]) for c in clips])).softmax(1).numpy()


# ---------------------------------------------------------------- B: CNN-Transformer

class CNNTransformer(nn.Module):
    """Head on top of per-frame CNN (ResNet50) features: a small Transformer encoder
    over the T frames of the landing window, then a classifier on the [CLS] token."""
    def __init__(self, n, d=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2048, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, T + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n))

    def forward(self, feats):                          # (B, T, 2048)
        x = self.proj(feats)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        return self.head(self.encoder(x)[:, 0])


def train_transformer(data, n, epochs):
    model = CNNTransformer(n)
    xs = torch.tensor(np.stack([c["feats"] for c in data["train"]]))
    ys = torch.tensor([c["y"] for c in data["train"]])
    xv = torch.tensor(np.stack([c["feats"] for c in data["val"]]))
    yv = np.array([c["y"] for c in data["val"]])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ys.numpy(), n))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xs))
        for k in range(0, len(xs), 16):
            b = perm[k:k + 16]
            opt.zero_grad()
            noisy = xs[b] + 0.05 * torch.randn_like(xs[b]) * xs[b].std()
            loss = loss_fn(model(noisy), ys[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f1 = macro_f1(yv, model(xv).argmax(1).numpy(), n)
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()))
        if (ep + 1) % 25 == 0:
            print(f"  B epoch {ep + 1}/{epochs}: loss {loss.item():.3f}  val macro F1 {f1:.3f}  (best {best[0]:.3f})")
    print(f"  B best val macro F1 {best[0]:.3f}")
    model.load_state_dict(best[1])
    return model.eval()


def predict_transformer(model, clips):
    with torch.no_grad():
        return model(torch.tensor(np.stack([c["feats"] for c in clips]))).softmax(1).numpy()


# ---------------------------------------------------------------- C: XGBoost

def xgb_features(pca, clips):
    deep = pca.transform(np.stack([c["feats"][-1] for c in clips]))   # max-flexion frame
    return np.hstack([deep, np.stack([c["summary"] for c in clips])])


def train_xgb(data, n):
    k = min(16, len(data["train"]) - 1)
    pca = PCA(n_components=k, random_state=SEED).fit(np.stack([c["feats"][-1] for c in data["train"]]))
    x, y = xgb_features(pca, data["train"]), np.array([c["y"] for c in data["train"]])
    w = class_weights(y, n).numpy()[y]
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED,
                          objective="multi:softprob" if n > 2 else "binary:logistic",
                          **({"num_class": n} if n > 2 else {}))
    model.fit(x, y, sample_weight=w)
    return pca, model


def predict_xgb(pca, model, clips):
    p = model.predict_proba(xgb_features(pca, clips))
    return p


# ---------------------------------------------------------------- R: angle rule

def predict_rule(clips, n):
    out = []
    for c in clips:
        fppa, trunk = c["summary"][2], c["summary"][3]
        if n == 2:
            cls = 1 if fppa >= RULE_FPPA or trunk >= RULE_TRUNK else 0
        else:
            cls = 2 if trunk >= RULE_TRUNK else 1 if fppa >= RULE_FPPA else 0
        out.append(np.eye(n)[cls])
    return np.array(out)


# ---------------------------------------------------------------- export + latency

def export_onnx(model, example, path, names=("input", "output")):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, str(path), input_names=[names[0]], output_names=[names[1]],
                      opset_version=17, dynamo=False)
    return path.stat().st_size / 1e6


class Softmaxed(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).softmax(-1)


def quantize(src):
    """int8 weights (dynamic quantisation): ~4x smaller file for the browser."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = src.with_name(src.stem + ".int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return dst


def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_latency(path, feeds, runs=30):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1      # closest to single-threaded WebAssembly in a browser
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    for _ in range(3):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------- reporting

def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    b, c = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def save_confusion(cm, names, path, title):
    import cv2
    n, cell, left, top = len(names), 90, 150, 70
    img = np.full((top + n * cell + 60, left + n * cell + 20, 3), 255, np.uint8)
    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "predicted ->", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    mx = max(1, cm.max())
    for i in range(n):
        cv2.putText(img, names[i][:14], (8, top + i * cell + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, names[i][:11], (left + i * cell + 4, top + n * cell + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 40, 40), 1, cv2.LINE_AA)
        for j in range(n):
            v = cm[i, j]
            shade = int(255 - 180 * v / mx)
            colour = (shade, shade - 20 if i == j else shade, shade if i == j else shade)
            x0, y0 = left + j * cell, top + i * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), colour, -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (200, 200, 200), 1)
            cv2.putText(img, str(v), (x0 + cell // 2 - 10, y0 + cell // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255) if v > mx / 2 else (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "rows = annotator label", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (90, 90, 90), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, choices=[2, 3], default=3)
    ap.add_argument("--epochs-a", type=int, default=12)
    ap.add_argument("--epochs-b", type=int, default=150)
    args = ap.parse_args()
    n, names = args.classes, CLASSES[args.classes]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)

    data = load(n)
    for s in data:
        print(f"{s}: {len(data[s])} clips, per class {np.bincount([c['y'] for c in data[s]], minlength=n).tolist()}")
    out_dir = RESULTS_DIR / f"{n}class"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = EXPORT_DIR / f"{n}class"
    test, y = data["test"], np.array([c["y"] for c in data["test"]])

    print("Training A: ResNet50 ...")
    a = train_resnet(data, n, args.epochs_a)
    print("Training B: CNN-Transformer ...")
    b = train_transformer(data, n, args.epochs_b)
    print("Training C: XGBoost ...")
    pca, xgb = train_xgb(data, n)

    probs = {"A_ResNet50": predict_resnet(a, test), "B_CNN_Transformer": predict_transformer(b, test),
             "C_XGBoost": predict_xgb(pca, xgb, test), "R_angle_rule": predict_rule(test, n)}
    preds = {k: v.argmax(1) for k, v in probs.items()}

    # ---- export for the browser + size + latency
    print("Exporting ONNX and timing ...")
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    img1 = torch.rand(1, 3, 224, 224)
    size_backbone = export_onnx(Normalised(backbone).eval(), img1, EXPORT_DIR / "backbone.onnx")
    size_a = export_onnx(Softmaxed(a), img1, model_dir / "resnet50_ft.onnx")
    size_b_head = export_onnx(Softmaxed(b), torch.rand(1, T, 2048), model_dir / "cnn_transformer_head.onnx")
    (model_dir / "xgb.json").write_text(json.dumps({
        "pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
        "booster": json.loads(xgb.get_booster().save_raw("json").decode()),
        "n_classes": n}))
    size_c = (model_dir / "xgb.json").stat().st_size / 1e6

    lat_backbone1 = ort_latency(EXPORT_DIR / "backbone.onnx", {"input": img1.numpy()})
    lat_a = ort_latency(model_dir / "resnet50_ft.onnx", {"input": img1.numpy()})
    lat_b = T * lat_backbone1 + ort_latency(model_dir / "cnn_transformer_head.onnx",
                                            {"input": np.random.rand(1, T, 2048).astype(np.float32)})
    xc = xgb_features(pca, test[:1])
    t0 = time.perf_counter()
    for _ in range(30):
        xgb.predict_proba(xc)
    lat_c = lat_backbone1 + (time.perf_counter() - t0) * 1000 / 30
    size = {"A_ResNet50": size_a, "B_CNN_Transformer": size_backbone + size_b_head,
            "C_XGBoost": size_backbone + size_c, "R_angle_rule": 0.0}
    latency = {"A_ResNet50": lat_a, "B_CNN_Transformer": lat_b, "C_XGBoost": lat_c, "R_angle_rule": 0.0}

    # ---- int8 versions (what the web app loads), re-evaluated on the same test clips
    print("Quantising to int8 and re-evaluating ...")
    qa, qbb = quantize(model_dir / "resnet50_ft.onnx"), quantize(EXPORT_DIR / "backbone.onnx")
    sa, sbb = ort_session(qa), ort_session(qbb)
    probs["A_ResNet50_int8"] = np.concatenate([sa.run(None, {"input": to_unit(c["crops"][-1:]).numpy()})[0]
                                               for c in test])
    qfeats = [np.concatenate([sbb.run(None, {"input": to_unit(c["crops"][i:i + 1]).numpy()})[0]
                              for i in range(T)]) for c in test]
    with torch.no_grad():
        probs["B_CNN_Transformer_int8"] = b(torch.tensor(np.stack(qfeats))).softmax(1).numpy()
    qtest = [dict(c, feats=f) for c, f in zip(test, qfeats)]
    probs["C_XGBoost_int8"] = predict_xgb(pca, xgb, qtest)
    preds = {k: v.argmax(1) for k, v in probs.items()}
    lat_qbb = ort_latency(qbb, {"input": img1.numpy()})
    size_qbb = qbb.stat().st_size / 1e6
    size.update({"A_ResNet50_int8": qa.stat().st_size / 1e6, "B_CNN_Transformer_int8": size_qbb + size_b_head,
                 "C_XGBoost_int8": size_qbb + size_c})
    latency.update({"A_ResNet50_int8": ort_latency(qa, {"input": img1.numpy()}),
                    "B_CNN_Transformer_int8": lat_b - T * lat_backbone1 + T * lat_qbb,
                    "C_XGBoost_int8": lat_c - lat_backbone1 + lat_qbb})

    # ---- metrics
    metrics = {}
    for k, p in preds.items():
        metrics[k] = {
            "accuracy": accuracy_score(y, p), "macro_f1": macro_f1(y, p, n),
            "size_mb": size[k], "latency_ms_cpu1thread": latency[k],
            "report": classification_report(y, p, labels=list(range(n)), target_names=names,
                                            zero_division=0, output_dict=True),
            "confusion": confusion_matrix(y, p, labels=list(range(n))).tolist(),
        }
        save_confusion(np.array(metrics[k]["confusion"]), names, out_dir / f"confusion_{k}.png",
                       f"{k}  ({n} classes, test n={len(y)})")
    pairs = [("A_ResNet50", "B_CNN_Transformer"), ("A_ResNet50", "C_XGBoost"), ("B_CNN_Transformer", "C_XGBoost"),
             ("A_ResNet50_int8", "B_CNN_Transformer_int8"), ("A_ResNet50_int8", "C_XGBoost_int8"),
             ("B_CNN_Transformer_int8", "C_XGBoost_int8")]
    mc = {f"{p} vs {q}": dict(zip(("only_first_correct", "only_second_correct", "p_value"),
                                  mcnemar(y, preds[p], preds[q]))) for p, q in pairs}

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "true"] + list(preds))
        for i, c in enumerate(test):
            w.writerow([c["clip"], names[y[i]]] + [names[preds[k][i]] for k in preds])
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "mcnemar": mc, "classes": names,
                                                      "n": {s: len(v) for s, v in data.items()}}, indent=2))
    (model_dir / "meta.json").write_text(json.dumps({"classes": names, "T": T, "crop": 224}))

    lines = [f"# Model comparison ({n} classes)", "",
             f"Test set: {len(y)} clips from held-out groups "
             f"(train {len(data['train'])}, val {len(data['val'])}). "
             f"Class counts in test: {np.bincount(y, minlength=n).tolist()} ({', '.join(names)}).", "",
             "| Model | Accuracy | Macro F1 | Size (MB) | Latency (ms, CPU 1 thread) |",
             "|---|---|---|---|---|"]
    for k, m in metrics.items():
        lines.append(f"| {k} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['size_mb']:.1f} | "
                     f"{m['latency_ms_cpu1thread']:.0f} |")
    lines += ["", f"Targets: accuracy >= {TARGETS['accuracy']}, macro F1 >= {TARGETS['macro_f1']}, "
                  f"latency <= {TARGETS['latency_ms']} ms (in browser), size <= {TARGETS['size_mb']} MB.",
              "Latency here is ONNX Runtime on this PC's CPU with 1 thread; the browser figure is measured in the web app.",
              "_int8 rows are the quantised models the web app runs, evaluated on the same test clips.",
              "R_angle_rule is a reference only; its thresholds came from the pilot clips, which are part of this data.",
              "", "## McNemar's test (exact, test set)", "", "| Pair | only 1st correct | only 2nd correct | p |",
              "|---|---|---|---|"]
    for k, v in mc.items():
        lines.append(f"| {k} | {v['only_first_correct']} | {v['only_second_correct']} | {v['p_value']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
