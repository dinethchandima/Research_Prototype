# Landing Form Coach

**Detecting wrong posture during plyometric exercises using machine learning.**
BSc (Hons) Computing project artefact (NB5004CEM) by H.A.D.P.C. Hettiarachchi (COBSCCOMP242P-033),
NIBM / Coventry University.

A web application that watches an athlete through the device camera, detects each jump landing (initial contact to
maximum knee flexion), classifies it as **correct form**, **knee valgus** or **trunk lean**, and gives a corrective
cue on screen and by voice before the next repetition. Everything runs in the browser on the user's device: no video
is uploaded.

---

## 1. Run the web application

Requirements: Windows with Python 3.12, a current Chrome or Edge browser and a webcam. No internet connection is
needed: the browser libraries (ONNX Runtime Web, MediaPipe) are included in `webapp/vendor/`.

```bat
start_webapp.bat
```

This serves `webapp/` at **http://localhost:8000** (cameras work on `localhost` without HTTPS) and opens the browser.
The server sends cross-origin isolation headers, which let the model use several CPU threads.

1. Place the camera about 3 m away at hip height and face it with the whole body in view.
2. Press **Start camera** and allow camera access; wait for the green **Ready** indicator.
3. Jump and land on both feet. A cue appears and is spoken after every landing.
4. **End session** shows a summary; **Download CSV** saves the landing log.

The app always uses the model selected by cross-validation: XGBoost on deep-learned features, 3 classes.
The gear icon opens the settings: voice cues and *Researcher details* (live angles, frames per second, model time, landing-to-cue time and the next
take-off).

## 2. Install the Python environment (to reproduce the results)

```bat
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

The UCF101 videos are **not included** (licence and size). Download UCF101 from the University of Central Florida
(https://www.crcv.ucf.edu/data/UCF101.php) and place the category folders as described in `scripts/common.py`
(`UCF101_DIR`).

## 3. Reproduce the pipeline

| Step | Command | Output |
|---|---|---|
| Build the label file | `python scripts/build_labels.py` | `labels.csv` (clip, category, group) |
| Pose screening and frame suggestions | `python scripts/suggest_frames.py` | `landmarks/`, `suggestions.csv` |
| Screening summary per category | `python scripts/screen_summary.py` | table on screen |
| Pass 1: grid screening (human) | `python scripts/screen_grid.py --annotator XX` | exclusions in `labels.csv`, `labels_grid.csv` |
| Pass 2: labelling (human) | `python scripts/label_tool.py --annotator XX` | labels in `labels.csv` |
| Split, features, training, comparison | `run_pipeline.bat` | `split.csv`, `data/clips/`, `results/2class`, `results/3class`, `webapp/models/` |
| 5-fold grouped cross-validation | `python scripts/cross_validate.py --classes 3` (and `2`) | `results/cv_3class`, `results/cv_2class` |
| Browser vs Python parity (server running) | `python scripts/parity_check.py` | `results/parity.json` |
| Thesis figures | `python scripts/make_figures.py`, `python scripts/make_ch4_figures.py` | `results/figures/` |

Run the commands from the project folder with `.venv\Scripts\python`. Labels are human decisions: the tools only show
the video, the BlazePose skeleton and suggested frames; the annotator accepts or corrects every frame and makes every
label. `label_tool.py --recheck` runs a blind re-check (suggestions hidden), and `--file labels_rater2.csv` creates a
separate file for a second rater.

## 4. Project structure

```
scripts/            Python: labelling tools, landing detector, feature extraction, training, evaluation, figures
webapp/             Browser application (index.html, app.js, detector.js, classifier.js, style.css, serve.py)
webapp/models/      Deployed models: int8 ONNX ResNet50 (feature extractor and fine-tuned), CNN-Transformer head,
                    XGBoost JSON, BlazePose model
webapp/test/        Browser parity test page (used by scripts/parity_check.py)
results/            Metrics, confusion matrices, predictions, cross-validation, parity results, figures
logs/               Training and cross-validation logs
labels.csv          Annotations (usable, exclusion reason, contact and max-flexion frames, angles, label, annotator)
split.csv           Grouped, stratified 70/15/15 split
suggestions.csv     Automatic screening results and suggested frames
```

## 5. Results summary

Dataset: 267 JumpRope and JumpingJack clips screened by eye, 126 usable (61 correct form, 37 knee valgus, 28 trunk
lean), grouped by source video (34 groups).

| Model (3 classes) | Held-out test (n = 17) accuracy / macro F1 | 5-fold CV pooled (n = 126) accuracy / macro F1 |
|---|---|---|
| ResNet50 CNN | 0.647 / 0.627 | 0.437 / 0.389 |
| Hybrid CNN-Transformer | 0.176 / 0.162 | 0.389 / 0.333 |
| **XGBoost on deep-learned features (selected)** | 0.647 / 0.492 | **0.603 / 0.545** |

- XGBoost was significantly better than both other models in cross-validation (exact McNemar test: p = 0.005 and
  p < 0.001). No model reached the targets of 90% accuracy and 0.85 macro F1.
- int8 quantisation reduced each model to about 24–26 MB (target ≤ 50 MB).
- The browser and Python versions of all three deployed models gave identical predictions on all 126 clips.
- The live evaluation with athletes is reported in the thesis.

## 6. Privacy

Video frames are processed only in the browser's memory. The session log stays on the device until the user
downloads it as CSV; only user settings are stored in the browser's local storage. Spoken cues use on-device voices
only.

## Deployment (Vercel)

The repository includes `vercel.json`: Vercel serves the `webapp/` folder as a static site (no build step) with the cross-origin isolation headers that let ONNX Runtime Web use several CPU threads. Import the GitHub repository at vercel.com, keep the default settings and deploy. The camera works only over HTTPS, which Vercel provides.

