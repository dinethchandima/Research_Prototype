// Runs the trained models in the browser with ONNX Runtime Web (global `ort`).
// The int8-quantised files are loaded (about 4x smaller than fp32).
//  A  resnet50_ft.onnx                      max-flexion crop -> class probabilities
//  B  backbone.onnx x T + cnn_transformer   T window crops -> features -> Transformer
//  C  backbone.onnx + xgb.json              max-flexion features -> PCA + angles -> XGBoost

const CROP = 224;

export class Classifier {
  static async load(modelKey, nClasses) {
    const c = new Classifier();
    c.key = modelKey;
    const dir = `models/${nClasses}class/`;
    c.meta = await (await fetch(dir + "meta.json")).json();
    const opts = { executionProviders: ["wasm"] };
    if (modelKey === "A") {
      c.model = await ort.InferenceSession.create(dir + "resnet50_ft.int8.onnx", opts);
    } else {
      c.backbone = await ort.InferenceSession.create("models/backbone.int8.onnx", opts);
      if (modelKey === "B") c.head = await ort.InferenceSession.create(dir + "cnn_transformer_head.onnx", opts);
      else c.xgb = await (await fetch(dir + "xgb.json")).json();
    }
    return c;
  }

  // crops: T canvases (224x224) over the landing window, last = max flexion
  // summary: [peak FPPA L, peak FPPA R, peak FPPA max, peak trunk, mean trunk, window seconds]
  async predict(crops, summary) {
    if (this.key === "A") return this.run(this.model, toTensor([crops[crops.length - 1]]));
    if (this.key === "B") {
      const feats = new Float32Array(crops.length * 2048);
      for (let i = 0; i < crops.length; i++) {
        feats.set(await this.run(this.backbone, toTensor([crops[i]])), i * 2048);
      }
      return this.run(this.head, new ort.Tensor("float32", feats, [1, crops.length, 2048]));
    }
    const f = await this.run(this.backbone, toTensor([crops[crops.length - 1]]));
    return xgbPredict(this.xgb, [...pca(this.xgb, f), ...summary]);
  }

  async run(session, tensor) {
    const out = await session.run({ [session.inputNames[0]]: tensor });
    return Array.from(out[session.outputNames[0]].data);
  }
}

function toTensor(canvases) {
  const data = new Float32Array(canvases.length * 3 * CROP * CROP);
  canvases.forEach((cv, n) => {
    const px = cv.getContext("2d").getImageData(0, 0, CROP, CROP).data;
    const base = n * 3 * CROP * CROP, plane = CROP * CROP;
    for (let i = 0; i < plane; i++) {
      data[base + i] = px[i * 4] / 255;
      data[base + plane + i] = px[i * 4 + 1] / 255;
      data[base + 2 * plane + i] = px[i * 4 + 2] / 255;
    }
  });
  return new ort.Tensor("float32", data, [canvases.length, 3, CROP, CROP]);
}

function pca(m, f) {
  return m.pca_components.map(row => row.reduce((s, w, i) => s + w * (f[i] - m.pca_mean[i]), 0));
}

// Evaluate the XGBoost model from its JSON dump (no extra library needed).
function xgbPredict(m, x) {
  const learner = m.booster.learner;
  const trees = learner.gradient_booster.model.trees;
  const info = learner.gradient_booster.model.tree_info;
  const n = m.n_classes;
  const margins = new Array(n > 2 ? n : 1).fill(0);
  trees.forEach((t, k) => {
    let node = 0;
    while (t.left_children[node] !== -1) {
      // XGBoost compares in 32-bit floats; without fround a value right at a split goes the other way
      const v = Math.fround(x[t.split_indices[node]]);
      node = (Number.isNaN(v) ? t.default_left[node] : v < Math.fround(t.split_conditions[node]))
        ? t.left_children[node] : t.right_children[node];
    }
    margins[n > 2 ? info[k] : 0] += t.base_weights[node];
  });
  // XGBoost 3.x: binary base_score is a probability; multi-class has one margin per class.
  // Checked against Python predict_proba (max difference ~1e-7).
  const base = String(learner.learner_model_param.base_score).replace(/[\[\]]/g, "").split(",").map(parseFloat);
  if (n === 2) {
    const p = 1 / (1 + Math.exp(-(margins[0] + Math.log(base[0] / (1 - base[0])))));
    return [1 - p, p];
  }
  const z = margins.map((v, k) => v + (base[k] ?? 0));
  const mx = Math.max(...z);
  const e = z.map(v => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

// Square crop around the athlete over the window (same as ml_common.person_box), drawn
// onto a black 224x224 canvas.
export function personBox(landmarkList, w, h, pad = 0.2) {
  const pts = [];
  landmarkList.forEach(lm => lm.forEach(p => {
    if ((p.visibility ?? 1) > 0.3) pts.push([p.x * w, p.y * h]);
  }));
  if (pts.length < 4) { const s = Math.max(w, h); return [(w - s) / 2, (h - s) / 2, s]; }
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const side = Math.max(x1 - x0, y1 - y0) * (1 + 2 * pad);
  return [(x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side];
}

export function cropCanvas(src, box) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = CROP;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CROP, CROP);
  const [x0, y0, side] = box, s = CROP / side;
  ctx.drawImage(src, -x0 * s, -y0 * s, src.width * s, src.height * s);
  return cv;
}
