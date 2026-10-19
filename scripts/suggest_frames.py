"""Step 2: labelling helper. Runs MediaPipe BlazePose on each clip, suggests the
landing window (initial contact -> maximum knee flexion) and computes knee FPPA
and lateral trunk angle over that window.

Everything here is a SUGGESTION to speed up labelling (thesis Appendix C).
The student decides `usable`, `label` and the final frames. This script never
writes `usable`, `exclude_reason`, `label`, `both_flag` or `annotator`.

Outputs
  landmarks/<clip>.npz   cached pose (re-used on re-runs and later for training)
  suggestions.csv        one row per clip: suggested frames, angles, quality hints
  previews/<clip>.mp4    (with --preview) skeleton + landing window drawn on the video

If you have already typed contact_frame and max_flexion_frame into labels.csv,
the angles are computed over YOUR window instead of the suggested one.

Usage (run with the project venv):
  .venv/Scripts/python scripts/suggest_frames.py --limit 30 --preview      # pilot
  .venv/Scripts/python scripts/suggest_frames.py --category LongJump
  .venv/Scripts/python scripts/suggest_frames.py --clips v_HighJump_g01_c01.avi --preview
  .venv/Scripts/python scripts/suggest_frames.py --fill                     # copy into labels.csv
"""
import argparse
import csv
import warnings
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from common import ROOT, clip_path as find_clip

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN means on frames with no person

LABELS = ROOT / "labels.csv"
SUGGESTIONS = ROOT / "suggestions.csv"
LANDMARK_DIR = ROOT / "landmarks"
PREVIEW_DIR = ROOT / "previews"
MODEL = ROOT / "models" / "pose_landmarker_full.task"

# BlazePose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
KEY_POINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

# Landing-detector settings, tuned on the 2026-10-16 blind pilot (22 usable clips):
# contact within 2 frames of the annotator in 18/22 clips (was 2/22 with the first guesses
# SMOOTH 5, FALL 0.020, STOP 0.005). Check again on newly labelled clips before trusting it.
SMOOTH = 5             # moving-average window for the hip signal (max flexion)
ANKLE_SMOOTH = 1       # no smoothing for the ankle signal: smoothing made contact ~3-5 frames late
FALL_SPEED = 0.015     # feet "falling" if moving down faster than this x leg length per frame
STOP_SPEED = 0.010     # feet "stopped" if moving down slower than this x leg length per frame
FLEX_SEARCH_S = 0.6    # look for maximum knee flexion up to this many seconds after contact
VIS_MIN = 0.5          # ignore frames where any key landmark's visibility is below this
SIDE_YAW = 45          # body turned more than this many degrees from the camera -> probably side view
MIN_POSE_PCT = 80      # flag clips where a person is found in fewer frames than this

SUGGESTION_COLUMNS = [
    "clip", "check", "frames", "fps", "pose_found_pct", "max_people", "key_visibility", "body_yaw", "view_hint",
    "landings_found", "contact_frame", "max_flexion_frame", "window_source",
    "fppa_left", "fppa_right", "knee_angle", "trunk_angle",
]


# ---------------------------------------------------------------- pose extraction

def extract_pose(video_path):
    """Run BlazePose on every frame. Returns landmarks (T,33,4) as x,y,z,visibility
    in normalised image coords (NaN where no person found), people count per frame,
    fps, width, height."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,  # 2 so we can flag clips with more than one person
    )
    landmarks, people = [], []
    with vision.PoseLandmarker.create_from_options(options) as detector:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int(i * 1000 / fps))
            frame_lm = np.full((33, 4), np.nan, dtype=np.float32)
            if result.pose_landmarks:
                # keep the biggest person (largest hip-to-ankle height on screen)
                best = max(result.pose_landmarks, key=lambda p: abs(p[L_ANKLE].y - p[L_HIP].y))
                frame_lm[:] = [[p.x, p.y, p.z, p.visibility] for p in best]
            landmarks.append(frame_lm)
            people.append(len(result.pose_landmarks))
            i += 1
    cap.release()
    return np.array(landmarks), np.array(people), fps, width, height


def load_or_extract(clip_path, force):
    cache = LANDMARK_DIR / (clip_path.stem + ".npz")
    if cache.exists() and not force:
        d = np.load(cache)
        return d["landmarks"], d["people"], float(d["fps"]), int(d["width"]), int(d["height"])
    lm, people, fps, w, h = extract_pose(clip_path)
    LANDMARK_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, landmarks=lm, people=people, fps=fps, width=w, height=h)
    return lm, people, fps, w, h


# ---------------------------------------------------------------- signal helpers

def to_pixels(lm, w, h):
    """(T,33,2) pixel coordinates, so angles are not distorted by the frame's aspect ratio."""
    return lm[:, :, :2] * np.array([w, h], dtype=np.float32)


def fill_and_smooth(signal, window=None):
    """Fill NaN gaps by linear interpolation, then moving-average smooth."""
    window = window or SMOOTH
    s = signal.astype(float).copy()
    idx = np.arange(len(s))
    good = ~np.isnan(s)
    if good.sum() < 2:
        return s
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    kernel = np.ones(window) / window
    padded = np.pad(s, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(s)]


# ---------------------------------------------------------------- landing detector

def find_landings(px, fps):
    """Find initial contact and maximum knee flexion from hip and ankle trajectories.
    Image y grows downwards, so 'falling' means y increasing.

    Returns (contact_frame, max_flexion_frame, number_of_landings_found).
    If a clip has several landings (e.g. jump rope), the one after the biggest
    drop is suggested."""
    ankle_y = fill_and_smooth(np.nanmean(px[:, [L_ANKLE, R_ANKLE], 1], axis=1), ANKLE_SMOOTH)
    hip_y = fill_and_smooth(np.nanmean(px[:, [L_HIP, R_HIP], 1], axis=1))
    leg = np.nanmedian(np.abs(np.nanmean(px[:, [L_ANKLE, R_ANKLE], 1], axis=1)
                              - np.nanmean(px[:, [L_HIP, R_HIP], 1], axis=1)))
    if np.isnan(ankle_y).all() or not np.isfinite(leg) or leg <= 0:
        return None, None, 0

    speed = np.gradient(ankle_y) / leg   # leg lengths per frame, + = moving down
    lookback = max(1, int(0.5 * fps))
    candidates = []
    falling = False
    for t in range(1, len(speed)):
        if speed[t] > FALL_SPEED:
            falling = True
        elif falling and speed[t] < STOP_SPEED:
            # feet have stopped moving down -> initial contact
            drop = ankle_y[t] - ankle_y[max(0, t - lookback): t].min()
            candidates.append((drop, t))
            falling = False
    if not candidates:
        return None, None, 0

    _, contact = max(candidates)
    end = min(len(hip_y), contact + max(2, int(FLEX_SEARCH_S * fps)))
    max_flex = contact + int(np.argmax(hip_y[contact:end]))  # lowest hip = most knee flexion
    return contact, max_flex, len(candidates)


# ---------------------------------------------------------------- angles

def fppa(hip, knee, ankle, mid_hip):
    """Knee frontal plane projection angle in degrees (2D).
    0 = hip, knee, ankle in a straight line. Positive = knee pushed towards the
    body midline (valgus), negative = knee pushed outwards (varus)."""
    thigh, shank = knee - hip, ankle - knee
    cos = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    line = ankle - hip
    knee_side = np.sign(line[0] * (knee - hip)[1] - line[1] * (knee - hip)[0])
    midline_side = np.sign(line[0] * (mid_hip - hip)[1] - line[1] * (mid_hip - hip)[0])
    return angle if knee_side == midline_side else -angle


def lateral_trunk(mid_hip, mid_shoulder):
    """Angle in degrees between the trunk (mid-hip -> mid-shoulder) and vertical."""
    v = mid_shoulder - mid_hip
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


def window_angles(px, visible, start, end):
    """Peak knee FPPA per leg and peak lateral trunk angle within [start, end]."""
    left, right, trunk = [], [], []
    for t in range(start, end + 1):
        p = px[t]
        if not visible[t]:
            continue
        mid_hip = (p[L_HIP] + p[R_HIP]) / 2
        mid_sh = (p[L_SHOULDER] + p[R_SHOULDER]) / 2
        left.append(fppa(p[L_HIP], p[L_KNEE], p[L_ANKLE], mid_hip))
        right.append(fppa(p[R_HIP], p[R_KNEE], p[R_ANKLE], mid_hip))
        trunk.append(lateral_trunk(mid_hip, mid_sh))
    if not trunk:
        return None, None, None
    return max(left), max(right), max(trunk)


def body_yaw(lm, frames):
    """How far the body is turned away from the camera, in degrees (0 = facing the
    camera or facing away, 90 = side on). Uses BlazePose's depth (z) estimate of the
    two shoulders; 2D shoulder width alone could not separate side from frontal views."""
    a = lm[frames]
    if not len(a):
        return None
    dx = np.abs(a[:, L_SHOULDER, 0] - a[:, R_SHOULDER, 0])
    dz = np.abs(a[:, L_SHOULDER, 2] - a[:, R_SHOULDER, 2])
    return float(np.median(np.degrees(np.arctan2(dz, dx))))


# ---------------------------------------------------------------- preview video

POSE_LINES = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
              (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
LEFT_SIDE = {L_HIP, L_KNEE, L_ANKLE}  # lines ending here are drawn as the left side


def write_preview(clip_path, px, fps, w, h, contact, max_flex, angles):
    PREVIEW_DIR.mkdir(exist_ok=True)
    out = cv2.VideoWriter(str(PREVIEW_DIR / (clip_path.stem + ".mp4")),
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap = cv2.VideoCapture(str(clip_path))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t < len(px) and not np.isnan(px[t][KEY_POINTS]).any():
            for a, b in POSE_LINES:
                colour = (0, 200, 255) if b in LEFT_SIDE else (255, 200, 0)  # left orange, right blue
                cv2.line(frame, tuple(map(int, px[t][a])), tuple(map(int, px[t][b])), colour, 2)
        text = f"frame {t}"
        if contact is not None and contact <= t <= max_flex:
            text += "  LANDING WINDOW"
            if t == contact:
                text += " (contact)"
            if t == max_flex:
                text += " (max flexion)"
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if angles[0] is not None:
            cv2.putText(frame, f"FPPA L {angles[0]:.0f}  R {angles[1]:.0f}  trunk {angles[2]:.0f}",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(frame)
        t += 1
    cap.release()
    out.release()


# ---------------------------------------------------------------- main

def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def r1(x):
    return "" if x is None else f"{x:.1f}"


def analyse(row, force, preview):
    clip_path = find_clip(row["category"], row["clip"])
    lm, people, fps, w, h = load_or_extract(clip_path, force)
    px = to_pixels(lm, w, h)
    found = ~np.isnan(lm[:, 0, 0])

    contact, max_flex, n_landings = find_landings(px, fps)
    source = "suggested"
    user_c, user_f = as_int(row.get("contact_frame")), as_int(row.get("max_flexion_frame"))
    if user_c is not None and user_f is not None and 0 <= user_c <= user_f < len(px):
        contact, max_flex, source = user_c, user_f, "manual"

    visible = np.nan_to_num(lm[:, KEY_POINTS, 3], nan=0).min(axis=1) >= VIS_MIN
    angles = window_angles(px, visible, contact, max_flex) if contact is not None else (None, None, None)
    vis = lm[:, KEY_POINTS, 3]

    # yaw over the landing window if we have one, else over the whole clip
    in_window = np.zeros(len(lm), bool)
    if contact is not None:
        in_window[contact: max_flex + 1] = True
    yaw = body_yaw(lm, visible & in_window if (visible & in_window).any() else visible)

    check = []
    if 100 * found.mean() < MIN_POSE_PCT:
        check.append("person often not found")
    if people.max(initial=0) > 1:
        check.append("2+ people")
    if yaw is not None and yaw > SIDE_YAW:
        check.append("side view?")
    if contact is None:
        check.append("no landing found")
    elif max_flex - contact < 2:
        check.append("very short window")
    elif angles[0] is None:
        check.append("legs not visible at landing")
    if preview:
        write_preview(clip_path, px, fps, w, h, contact, max_flex, angles)

    knee = None if angles[0] is None else max(angles[0], angles[1])
    return {
        "clip": row["clip"],
        "check": "; ".join(check),
        "frames": len(lm),
        "fps": f"{fps:.2f}",
        "pose_found_pct": f"{100 * found.mean():.0f}" if len(lm) else "0",
        "max_people": int(people.max()) if len(people) else 0,
        "key_visibility": f"{np.nanmean(vis):.2f}" if found.any() else "",
        "body_yaw": r1(yaw),
        "view_hint": "" if yaw is None else ("side?" if yaw > SIDE_YAW else "frontal/back"),
        "landings_found": n_landings,
        "contact_frame": "" if contact is None else contact,
        "max_flexion_frame": "" if max_flex is None else max_flex,
        "window_source": source if contact is not None else "",
        "fppa_left": r1(angles[0]),
        "fppa_right": r1(angles[1]),
        "knee_angle": r1(knee),
        "trunk_angle": r1(angles[2]),
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fill_labels(label_cols, label_rows, results):
    """Copy suggestions into labels.csv. Frames are only filled where empty (your edits
    are kept); angles are refreshed because they are measurements, not judgements."""
    by_clip = {r["clip"]: r for r in results}
    for row in label_rows:
        s = by_clip.get(row["clip"])
        if not s:
            continue
        for col in ("contact_frame", "max_flexion_frame"):
            if not row[col]:
                row[col] = s[col]
        row["knee_angle"], row["trunk_angle"] = s["knee_angle"], s["trunk_angle"]
    write_csv(LABELS, label_cols, label_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", help="only these clip filenames")
    ap.add_argument("--category", help="only clips from this folder")
    ap.add_argument("--limit", type=int, help="stop after this many clips (e.g. 30 for the pilot)")
    ap.add_argument("--preview", action="store_true", help="write annotated preview videos")
    ap.add_argument("--force", action="store_true", help="re-run pose even if landmarks are cached")
    ap.add_argument("--fill", action="store_true", help="copy frames/angles into labels.csv")
    args = ap.parse_args()

    if not LABELS.exists():
        raise SystemExit("labels.csv not found. Run scripts/build_labels.py first.")
    label_cols, label_rows = read_csv(LABELS)
    todo = [r for r in label_rows
            if (not args.clips or r["clip"] in args.clips)
            and (not args.category or r["category"] == args.category)]
    if args.limit:
        todo = todo[: args.limit]

    new = {}
    for i, row in enumerate(todo, 1):
        try:
            new[row["clip"]] = s = analyse(row, args.force, args.preview)
            print(f"[{i}/{len(todo)}] {row['clip']}: contact={s['contact_frame']} "
                  f"flex={s['max_flexion_frame']} knee={s['knee_angle']} trunk={s['trunk_angle']} "
                  f"view={s['view_hint']} check=[{s['check']}]")
        except Exception as e:  # one bad clip should not stop the batch
            print(f"[{i}/{len(todo)}] {row['clip']}: FAILED ({e})")

    # merge with the file as it is now, so another run saving meanwhile is not lost
    existing = {r["clip"]: r for r in read_csv(SUGGESTIONS)[1]} if SUGGESTIONS.exists() else {}
    existing.update(new)
    order = {r["clip"]: n for n, r in enumerate(label_rows)}
    results = sorted(existing.values(), key=lambda r: order.get(r["clip"], 1e9))
    write_csv(SUGGESTIONS, SUGGESTION_COLUMNS, results)
    print(f"Saved {len(results)} rows to {SUGGESTIONS.name}")

    if args.fill:
        fill_labels(label_cols, label_rows, results)
        print(f"Filled frames/angles into {LABELS.name} (usable/label left for you)")


if __name__ == "__main__":
    main()
