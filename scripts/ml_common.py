"""Shared pieces for the model pipeline: which clips are labelled, the class sets,
how the landing window is sampled and how the athlete is cropped."""
import csv

import numpy as np

from common import ROOT

LABELS = ROOT / "labels.csv"
SPLIT = ROOT / "split.csv"
DATA_DIR = ROOT / "data" / "clips"        # per-clip crops + features (extract_features.py)
RESULTS_DIR = ROOT / "results"
EXPORT_DIR = ROOT / "webapp" / "models"   # ONNX files the web app loads

CLASSES = {
    3: ["correct_form", "knee_valgus", "trunk_lean"],
    2: ["correct_form", "fault"],           # knee_valgus + trunk_lean merged
}
T = 8            # frames sampled from contact to max flexion
CROP = 224       # ResNet50 input size
SEED = 42


def to_class(label, n_classes):
    if n_classes == 2:
        return 0 if label == "correct_form" else 1
    return CLASSES[3].index(label)


def labelled_clips():
    """Usable clips with a label and a valid landing window."""
    with LABELS.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["usable"] == "yes" and r["label"] in CLASSES[3]
                and r["contact_frame"] and r["max_flexion_frame"]]
    return [r for r in rows if int(r["max_flexion_frame"]) > int(r["contact_frame"])]


def window_indices(contact, flex, t=T):
    """T frame numbers spread evenly from contact to max flexion (last one = max flexion)."""
    return np.round(np.linspace(contact, flex, t)).astype(int)


def person_box(landmarks, frames, width, height, pad=0.2):
    """Square box around the athlete over the given frames, from BlazePose points,
    in pixels: (x0, y0, side). Falls back to the whole frame if no points."""
    pts = landmarks[frames][:, :, :2].reshape(-1, 2)
    vis = landmarks[frames][:, :, 3].reshape(-1)
    pts = pts[(vis > 0.3) & ~np.isnan(pts).any(axis=1)]
    if len(pts) < 4:
        side = max(width, height)
        return (width - side) // 2, (height - side) // 2, side
    x0, y0 = pts.min(axis=0) * [width, height]
    x1, y1 = pts.max(axis=0) * [width, height]
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - side / 2), int(cy - side / 2), int(side)


def crop(frame, box, size=CROP):
    """Crop a square box (may go past the edges, padded with black) and resize."""
    import cv2
    x0, y0, side = box
    h, w = frame.shape[:2]
    canvas = np.zeros((side, side, 3), frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
