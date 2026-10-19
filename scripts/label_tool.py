"""Keyboard labelling tool (thesis Appendix C). Every decision is made by the
annotator; the tool only shows the video, the BlazePose skeleton, the helper's
suggestions and live angles, and saves straight into labels.csv.

Clips you kept in the grid pass (screen_grid.py) are shown first, then clips that
passed screening (no flags). Only unlabelled clips are
queued unless you pass --all. Progress is saved after every clip, so you can quit
any time and carry on later.

Usage (run with the project venv, close labels.csv in Excel first):
  .venv/Scripts/python scripts/label_tool.py --annotator DC
  .venv/Scripts/python scripts/label_tool.py --annotator DC --category JumpRope JumpingJack --limit 30
  .venv/Scripts/python scripts/label_tool.py --annotator DC --clips v_JumpRope_g01_c01.avi --all
  # re-mark only the contact frame (label and max flexion kept) where it is far from the detector
  .venv/Scripts/python scripts/label_tool.py --annotator DC --fix-contact
  # re-check every clip already labelled usable, BLIND: marks start empty, the helper's
  # suggested frames are hidden and G / Y are off, so the frames are your own judgement
  .venv/Scripts/python scripts/label_tool.py --annotator DC --recheck
  # second rater / re-labelling for kappa: a separate file, nobody sees the other's labels
  .venv/Scripts/python scripts/label_tool.py --annotator XY --file labels_rater2.csv --clips ...

Keys
  A / D  or  <- / ->     previous / next frame
  W / S  or  up / down   10 frames forward / back
  SPACE                  play / pause
  G                      go to the suggested contact frame
  C / F                  mark contact / max flexion at this frame
  Y                      accept the suggested contact + max flexion (check them first!)
  R                      reset marks for this clip
  1 / 2 / 3              correct_form / knee_valgus / trunk_lean
  B                      toggle both_flag
  X                      exclude clip (then pick a reason with 1-7)
  ENTER                  save + next clip
  N / P                  skip to next clip / back to previous clip
  K                      show / hide skeleton
  ESC                    quit (everything saved so far is kept)
"""
import argparse
import os

import cv2
import numpy as np

from common import ROOT, clip_path
from suggest_frames import (KEY_POINTS, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, LANDMARK_DIR, LEFT_SIDE,
                            POSE_LINES, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SUGGESTIONS, VIS_MIN,
                            fppa, lateral_trunk, read_csv, to_pixels, window_angles, write_csv)

CLASSES = {"1": "correct_form", "2": "knee_valgus", "3": "trunk_lean"}
EXCLUDE_REASONS = [
    "side view (knees not separately visible)",
    "body not fully visible at landing",
    "more than one person landing",
    "camera cut at landing",
    "no clear landing",
    "poor video quality",
    "other",
]
SCALE = 2          # enlarge the 320x240 UCF101 frames
PANEL_W = 360
WIN = "Labelling tool"
LEFT_KEYS, RIGHT_KEYS = {2424832, ord("a")}, {2555904, ord("d")}
UP_KEYS, DOWN_KEYS = {2490368, ord("w")}, {2621440, ord("s")}

WHITE, GREY, YELLOW, GREEN, RED, CYAN = ((235, 235, 235), (150, 150, 150), (0, 220, 255),
                                         (80, 220, 80), (60, 60, 255), (255, 220, 120))


# ---------------------------------------------------------------- data

def load_clip(row):
    cap = cv2.VideoCapture(str(clip_path(row["category"], row["clip"])))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    lm = px = visible = None
    cache = LANDMARK_DIR / (row["clip"].rsplit(".", 1)[0] + ".npz")
    if cache.exists():
        d = np.load(cache)
        lm = d["landmarks"]
        px = to_pixels(lm, int(d["width"]), int(d["height"]))
        visible = np.nan_to_num(lm[:, KEY_POINTS, 3], nan=0).min(axis=1) >= VIS_MIN
    return frames, fps, lm, px, visible


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def frame_angles(px, visible, t):
    """FPPA left, FPPA right, trunk angle at one frame (or None if legs not visible)."""
    if px is None or t >= len(px) or not visible[t]:
        return None
    p = px[t]
    mid_hip = (p[L_HIP] + p[R_HIP]) / 2
    return (fppa(p[L_HIP], p[L_KNEE], p[L_ANKLE], mid_hip),
            fppa(p[R_HIP], p[R_KNEE], p[R_ANKLE], mid_hip),
            lateral_trunk(mid_hip, (p[L_SHOULDER] + p[R_SHOULDER]) / 2))


# ---------------------------------------------------------------- drawing

def text(img, s, y, colour=WHITE, x=None, scale=0.48):
    cv2.putText(img, s, (x if x is not None else img.shape[1] - PANEL_W + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def render(frames, t, st, px, visible, row, sug, pos, total, status, menu):
    f = frames[t].copy()
    if st["skeleton"] and px is not None and t < len(px):
        for a, b in POSE_LINES:
            pa, pb = px[t][a], px[t][b]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            colour = (0, 200, 255) if b in LEFT_SIDE else (255, 200, 0)
            if not visible[t]:
                colour = GREY
            cv2.line(f, tuple(map(int, pa)), tuple(map(int, pb)), colour, 2)
    f = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    h, w = f.shape[:2]

    c, fl = st["contact"], st["flex"]
    if t in (c, fl):
        cv2.rectangle(f, (0, 0), (w - 1, h - 1), GREEN, 6)
    elif c is not None and fl is not None and c <= t <= fl:
        cv2.rectangle(f, (0, 0), (w - 1, h - 1), GREEN, 2)
    tag = " CONTACT" if t == c else ""
    tag += " MAX FLEXION" if t == fl else ""
    label = f"frame {t}/{len(frames) - 1}{tag}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(f, (6, 6), (16 + tw, 16 + th), (0, 0, 0), -1)
    cv2.putText(f, label, (11, 11 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    canvas = np.full((max(h, 600), w + PANEL_W, 3), 32, np.uint8)
    canvas[:h, :w] = f
    # timeline with marks
    bar_y = h + 18
    if canvas.shape[0] > bar_y + 10:
        cv2.line(canvas, (10, bar_y), (w - 10, bar_y), GREY, 2)
        for frame_no, colour in ((as_int(sug.get("contact_frame")), CYAN), (c, GREEN), (fl, GREEN),
                                 (t, WHITE)):
            if frame_no is not None:
                x = 10 + int((w - 20) * frame_no / max(1, len(frames) - 1))
                cv2.line(canvas, (x, bar_y - 8), (x, bar_y + 8), colour, 3 if colour is WHITE else 2)
        text(canvas, "timeline: white = now, green = your marks, blue = suggested", bar_y + 26, GREY, x=10,
             scale=0.42)

    y = 26
    def line(s, colour=WHITE, step=22):
        nonlocal y
        text(canvas, s, y, colour)
        y += step

    line(row["clip"], CYAN)
    line(f"{row['category']}   clip {pos + 1} of {total}", GREY)
    if sug:
        flags = sug.get("check") or "no flags"
        line(f"screening: {flags}"[:46], YELLOW if sug.get("check") else GREEN)
        line(f"view: {sug.get('view_hint', '?')}   yaw {sug.get('body_yaw', '?')}", GREY)
        line(f"suggested contact {sug.get('contact_frame', '?')}, flex {sug.get('max_flexion_frame', '?')} (G)",
             GREY, 30)
    else:
        line("BLIND MODE: suggestions hidden", YELLOW)
        line("find contact and max flexion yourself", GREY, 52)

    line("YOUR DECISION", CYAN)
    line(f"contact (C): {'-' if c is None else c}     max flex (F): {'-' if fl is None else fl}")
    line(f"label (1/2/3): {st['label'] or '-'}", GREEN if st["label"] else WHITE)
    line(f"both faults (B): {'yes' if st['both'] else 'no'}")
    if c is not None and fl is not None and c <= fl and px is not None:
        l_, r_, tr = window_angles(px, visible, c, fl)
        if tr is not None:
            line(f"window peak: FPPA L {l_:.0f} R {r_:.0f}  trunk {tr:.0f}", GREY)
        else:
            line("window peak: legs not visible", GREY)
    a = frame_angles(px, visible, t)
    line("this frame: " + ("-" if a is None else f"FPPA L {a[0]:.0f} R {a[1]:.0f}  trunk {a[2]:.0f}"),
         GREY, 30)

    if menu:
        line("EXCLUDE - pick a reason:", RED)
        for n, reason in enumerate(EXCLUDE_REASONS, 1):
            line(f" {n}  {reason}", WHITE, 20)
        line("ESC  cancel", GREY)
    else:
        for s in ("A/D <-/->  frame -/+1     W/S  -/+10",
                  "SPACE play/pause" + ("    G go to suggested" if sug else ""),
                  "C contact  F max flex" + ("  Y accept sugg." if sug else ""),
                  "R reset marks",
                  "1 correct  2 valgus  3 trunk lean",
                  "B both faults      X exclude clip",
                  "ENTER save + next   N skip   P back",
                  "K skeleton on/off   ESC quit"):
            line(s, GREY, 20)
    if status:
        y = canvas.shape[0] - 16
        text(canvas, status[:46], y, YELLOW)
    return canvas


def grid_file(path):
    """Where screen_grid.py keeps its pass-1 decisions for this label file."""
    return path.with_name(path.stem + "_grid.csv")


def screening_priority(sug):
    """0 = passed screening, 1 = plausible (frontal, one person), 2 = probably unusable."""
    if sug.get("check") == "":
        return 0
    return 1 if sug.get("view_hint") == "frontal/back" and "2+ people" not in sug.get("check", "") else 2


# ---------------------------------------------------------------- saving

def save(path, cols, rows):
    """Write labels; returns an error message instead of crashing if Excel has the file open."""
    tmp = path.with_suffix(".tmp")
    try:
        write_csv(tmp, cols, rows)
        os.replace(tmp, path)
        return None
    except PermissionError:
        return f"Close {path.name} in Excel, then ENTER"


def apply_label(row, st, px, visible, annotator):
    row.update(usable="yes", exclude_reason="", contact_frame=st["contact"], max_flexion_frame=st["flex"],
               label=st["label"], both_flag="yes" if st["both"] else "no", annotator=annotator,
               knee_angle="", trunk_angle="")
    if px is not None:
        l_, r_, tr = window_angles(px, visible, st["contact"], st["flex"])
        if tr is not None:
            row.update(knee_angle=f"{max(l_, r_):.1f}", trunk_angle=f"{tr:.1f}")


def apply_exclusion(row, reason, annotator):
    row.update(usable="no", exclude_reason=reason, contact_frame="", max_flexion_frame="", knee_angle="",
               trunk_angle="", label="", both_flag="", annotator=annotator)


# ---------------------------------------------------------------- main loop

def label_clip(row, sug, pos, total, args, cols, rows, path):
    """Returns 'next', 'prev' or 'quit'."""
    frames, fps, lm, px, visible = load_clip(row)
    if not frames:
        print(f"Could not read {row['clip']}, skipping")
        return "next"
    st = {"contact": as_int(row["contact_frame"]), "flex": as_int(row["max_flexion_frame"]),
          "label": row["label"], "both": row["both_flag"] == "yes", "skeleton": True}
    if args.recheck:  # start from scratch so old marks are not saved by accident
        st.update(contact=None, flex=None, label="", both=False)
    if args.fix_contact:  # only contact is re-marked; max flexion and label are kept
        st["contact"] = None
    suggested = as_int(sug.get("contact_frame"))
    t = st["contact"] if st["contact"] is not None else max(0, (suggested or 0) - 10)
    t = min(t, len(frames) - 1)
    playing, menu, status = False, False, ""

    while True:
        cv2.imshow(WIN, render(frames, t, st, px, visible, row, sug, pos, total, status, menu))
        key = cv2.waitKeyEx(int(1000 / fps) if playing else 50)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"
        if key == -1:
            if playing:
                t = min(t + 1, len(frames) - 1)
                playing = t < len(frames) - 1
            continue
        if 0 <= key < 256:
            key = ord(chr(key).lower())
        ch = chr(key) if 0 <= key < 256 else ""

        if menu:
            if key == 27:
                menu = False
            elif ch.isdigit() and 1 <= int(ch) <= len(EXCLUDE_REASONS):
                apply_exclusion(row, EXCLUDE_REASONS[int(ch) - 1], args.annotator)
                err = save(path, cols, rows)
                if err:
                    status, menu = err, False
                    continue
                return "next"
            continue

        status = ""
        if key in LEFT_KEYS:
            t, playing = max(0, t - 1), False
        elif key in RIGHT_KEYS:
            t, playing = min(len(frames) - 1, t + 1), False
        elif key in UP_KEYS:
            t = min(len(frames) - 1, t + 10)
        elif key in DOWN_KEYS:
            t = max(0, t - 10)
        elif ch == " ":
            playing = not playing
            if playing and t >= len(frames) - 1:
                t = 0
        elif ch == "g":
            if suggested is not None:
                t, playing = min(suggested, len(frames) - 1), False
            else:
                status = "no suggestion for this clip"
        elif ch == "y" and args.fix_contact:
            status = "mark contact with C (label + F are kept)"
        elif ch in "yg" and ch and (args.recheck or args.blind):
            status = "blind mode: find the frames yourself"
        elif ch == "y":
            sug_flex = as_int(sug.get("max_flexion_frame"))
            if suggested is None or sug_flex is None:
                status = "no suggestion: mark with C and F"
            else:
                st["contact"], st["flex"] = suggested, sug_flex
        elif ch == "c":
            st["contact"] = t
        elif ch == "f":
            st["flex"] = t
        elif ch == "r":
            st.update(contact=None, flex=None, label="", both=False)
        elif ch in CLASSES:
            st["label"] = CLASSES[ch]
        elif ch == "b":
            st["both"] = not st["both"]
        elif ch == "k":
            st["skeleton"] = not st["skeleton"]
        elif ch == "x":
            menu, playing = True, False
        elif key == 13:  # ENTER
            if st["contact"] is None or st["flex"] is None:
                status = "mark contact (C) and max flexion (F)"
            elif st["flex"] <= st["contact"]:
                status = "max flexion must be after contact"
            elif not st["label"]:
                status = "choose a label: 1, 2 or 3"
            else:
                apply_label(row, st, px, visible, args.annotator)
                err = save(path, cols, rows)
                if err:
                    status = err
                    continue
                return "next"
        elif ch == "n":
            return "next"
        elif ch == "p":
            return "prev"
        elif key == 27:
            return "quit"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True, help="your initials, saved with every label")
    ap.add_argument("--file", default="labels.csv", help="label file to write (e.g. labels_rater2.csv)")
    ap.add_argument("--category", nargs="*", help="only these categories")
    ap.add_argument("--clips", nargs="*", help="only these clip filenames")
    ap.add_argument("--limit", type=int, help="at most this many clips this session")
    ap.add_argument("--all", action="store_true", help="also show clips that are already labelled")
    ap.add_argument("--fix-contact", action="store_true",
                    help="re-mark contact only, for usable clips whose contact is >5 frames from the detector")
    ap.add_argument("--blind", action="store_true",
                    help="hide the helper's suggested frames (G and Y off), e.g. for a second rater")
    ap.add_argument("--recheck", action="store_true",
                    help="redo clips already labelled usable: marks start empty, Y key off")
    args = ap.parse_args()

    path = ROOT / args.file
    if not path.exists():  # new rater file: same clips, empty annotations
        cols, base = read_csv(ROOT / "labels.csv")
        write_csv(path, cols, [{c: r[c] if c in ("clip", "category", "group") else "" for c in cols}
                               for r in base])
        print(f"Created {path.name}")
    cols, rows = read_csv(path)
    sugg = {r["clip"]: r for r in read_csv(SUGGESTIONS)[1]} if SUGGESTIONS.exists() else {}

    def far_from_detector(r):
        s = sugg.get(r["clip"], {})
        c = as_int(s.get("contact_frame"))
        return (r["usable"] == "yes" and s.get("window_source") == "suggested" and c is not None
                and abs(as_int(r["contact_frame"]) - c) > 5)
    queue = [r for r in rows
             if (far_from_detector(r) if args.fix_contact else
                 r["usable"] == "yes" if args.recheck else (args.all or not r["usable"]))
             and (not args.category or r["category"] in args.category)
             and (not args.clips or r["clip"] in args.clips)]

    kept = set()
    if grid_file(path).exists():
        kept = {r["clip"] for r in read_csv(grid_file(path))[1] if r["decision"] == "kept"}
    # kept in the grid pass first, then by screening result
    queue.sort(key=lambda r: (r["clip"] not in kept, screening_priority(sugg.get(r["clip"], {}))))
    if args.limit:
        queue = queue[: args.limit]
    if not queue:
        print("Nothing to label with these options.")
        return

    print(f"{len(queue)} clips queued. Labels are saved to {path.name} after every clip.")
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    i = 0
    while 0 <= i < len(queue):
        sug = {} if (args.recheck or args.blind) else sugg.get(queue[i]["clip"], {})  # blind: hide the suggestions
        result = label_clip(queue[i], sug, i, len(queue), args, cols, rows, path)
        if result == "quit":
            break
        i = max(0, i - 1) if result == "prev" else i + 1
    cv2.destroyAllWindows()

    done = sum(1 for r in queue if r["usable"])
    usable = sum(1 for r in queue if r["usable"] == "yes")
    print(f"Done: {done} of {len(queue)} queued clips have a decision ({usable} usable).")


if __name__ == "__main__":
    main()
