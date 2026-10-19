"""For every labelled clip: sample T frames over the annotated landing window
(contact -> max flexion), crop the athlete, and store
  crops      (T,224,224,3) uint8   input for the fine-tuned ResNet50
  feats      (T,2048) float32      frozen ImageNet ResNet50 features per frame
  angles     (T,3) float32         FPPA left, FPPA right, lateral trunk angle per frame
  summary    (6,) float32          peak FPPA L/R/max, peak + mean trunk, window seconds
in data/clips/<clip>.npz. Clips whose window has not changed are skipped.

Usage:  .venv/Scripts/python scripts/extract_features.py
"""
import cv2
import numpy as np
import torch
import torchvision

from common import clip_path
from ml_common import CROP, DATA_DIR, T, crop, labelled_clips, person_box, window_indices
from suggest_frames import (KEY_POINTS, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, LANDMARK_DIR, R_ANKLE, R_HIP,
                            R_KNEE, R_SHOULDER, VIS_MIN, fppa, lateral_trunk, to_pixels)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def frozen_backbone():
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    return m.eval()


def to_tensor(crops_rgb):
    x = (crops_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def frame_angles(px, visible, t):
    if t >= len(px) or not visible[t]:
        return [np.nan, np.nan, np.nan]
    p = px[t]
    mid_hip = (p[L_HIP] + p[R_HIP]) / 2
    return [fppa(p[L_HIP], p[L_KNEE], p[L_ANKLE], mid_hip), fppa(p[R_HIP], p[R_KNEE], p[R_ANKLE], mid_hip),
            lateral_trunk(mid_hip, (p[L_SHOULDER] + p[R_SHOULDER]) / 2)]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = labelled_clips()
    backbone = frozen_backbone()
    done = 0
    for i, r in enumerate(rows, 1):
        out = DATA_DIR / (r["clip"][:-4] + ".npz")
        c, f = int(r["contact_frame"]), int(r["max_flexion_frame"])
        if out.exists():
            old = np.load(out)
            if int(old["contact"]) == c and int(old["flex"]) == f:
                continue

        d = np.load(LANDMARK_DIR / (r["clip"][:-4] + ".npz"))
        lm, w, h, fps = d["landmarks"], int(d["width"]), int(d["height"]), float(d["fps"])
        cap = cv2.VideoCapture(str(clip_path(r["category"], r["clip"])))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()

        idx = np.clip(window_indices(c, f), 0, len(frames) - 1)
        box = person_box(lm, np.clip(np.arange(c, f + 1), 0, len(lm) - 1), w, h)
        crops = np.stack([cv2.cvtColor(crop(frames[t], box, CROP), cv2.COLOR_BGR2RGB) for t in idx])
        with torch.no_grad():
            feats = backbone(to_tensor(crops)).numpy()

        px = to_pixels(lm, w, h)
        visible = np.nan_to_num(lm[:, KEY_POINTS, 3], nan=0).min(axis=1) >= VIS_MIN
        angles = np.array([frame_angles(px, visible, t) for t in idx], np.float32)
        full = np.array([frame_angles(px, visible, t) for t in range(c, f + 1)], np.float32)
        with np.errstate(all="ignore"):
            summary = np.array([np.nanmax(full[:, 0]), np.nanmax(full[:, 1]), np.nanmax(full[:, :2]),
                                np.nanmax(full[:, 2]), np.nanmean(full[:, 2]), (f - c) / fps], np.float32)

        np.savez_compressed(out, crops=crops, feats=feats.astype(np.float32), angles=angles,
                            summary=np.nan_to_num(summary, nan=0.0), contact=c, flex=f)
        done += 1
        print(f"[{i}/{len(rows)}] {r['clip']}")
    print(f"Extracted {done} clips ({len(rows) - done} unchanged) -> {DATA_DIR}")


if __name__ == "__main__":
    main()
