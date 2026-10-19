"""Pass 1 of labelling: decide quickly which clips are usable, 12 at a time.

Each cell loops the clip around the suggested landing. Click a clip that breaks
the Appendix C rules to exclude it with the current reason; clips you leave
unmarked are "kept" and go on to pass 2 (label_tool.py), where you mark frames
and choose the label. Every decision is yours; the tool only shows the clips.

Exclusions are saved to the label file (usable = no + reason). Which clips you
kept is saved in <label file>_grid.csv, so the next session carries on where
you stopped.

Usage (close labels.csv in Excel first):
  .venv/Scripts/python scripts/screen_grid.py --annotator DC
  .venv/Scripts/python scripts/screen_grid.py --annotator DC --category JumpRope JumpingJack

Mouse / keys
  click a clip      exclude it with the current reason (click again to undo)
  1 - 7             change the current reason
  ENTER             save this page and go to the next one
  P                 back to the previous page
  K                 show / hide skeleton
  ESC               save this page and quit
"""
import argparse

import cv2
import numpy as np

from common import ROOT, clip_path
from label_tool import (EXCLUDE_REASONS, GREEN, GREY, RED, WHITE, YELLOW, apply_exclusion, as_int,
                        grid_file, save, screening_priority)
from suggest_frames import LANDMARK_DIR, LEFT_SIDE, POSE_LINES, SUGGESTIONS, read_csv, write_csv

COLS, ROWS = 4, 3
CELL_W, CELL_H = 272, 204          # whole window fits a 1366x768 screen
BAR_H = 60
WIN = "Grid screening"
MAX_FRAMES = 40                     # frames looped per cell
GRID_COLUMNS = ["clip", "decision", "annotator"]


def load_cell(row, sug):
    """Frames around the suggested landing (or spread over the whole clip), cell-sized."""
    cap = cv2.VideoCapture(str(clip_path(row["category"], row["clip"])))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return [np.zeros((CELL_H, CELL_W, 3), np.uint8)]

    contact, flex = as_int(sug.get("contact_frame")), as_int(sug.get("max_flexion_frame"))
    if contact is not None:
        start = max(0, contact - 15)
        end = min(len(frames), (flex if flex is not None else contact) + 15)
        idx = list(range(start, end))
    else:
        idx = list(np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int))
    if len(idx) > MAX_FRAMES:
        idx = idx[:: int(np.ceil(len(idx) / MAX_FRAMES))]

    px = None
    cache = LANDMARK_DIR / (row["clip"].rsplit(".", 1)[0] + ".npz")
    if cache.exists():
        d = np.load(cache)
        px = d["landmarks"][:, :, :2] * np.array([CELL_W, CELL_H])

    cells = []
    for i in idx:
        f = cv2.resize(frames[i], (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        skeleton = None
        if px is not None and i < len(px):
            skeleton = px[i]
        cells.append((f, skeleton))
    return cells


def draw_cell(cell, row, sug, excluded, show_skeleton):
    f, skeleton = cell
    f = f.copy()
    if show_skeleton and skeleton is not None:
        for a, b in POSE_LINES:
            pa, pb = skeleton[a], skeleton[b]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                cv2.line(f, tuple(map(int, pa)), tuple(map(int, pb)),
                         (0, 200, 255) if b in LEFT_SIDE else (255, 200, 0), 1, cv2.LINE_AA)
    if excluded:
        f = cv2.addWeighted(f, 0.45, np.full_like(f, (0, 0, 160)), 0.55, 0)
        cv2.putText(f, "EXCLUDED", (10, CELL_H // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        cv2.putText(f, excluded[:38], (10, CELL_H // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1,
                    cv2.LINE_AA)
    flags = sug.get("check") or "no flags"
    cv2.rectangle(f, (0, 0), (CELL_W, 18), (0, 0, 0), -1)
    cv2.putText(f, flags[:44], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                YELLOW if sug.get("check") else GREEN, 1, cv2.LINE_AA)
    name = row["clip"].replace("v_", "").replace(".avi", "")
    cv2.rectangle(f, (0, CELL_H - 18), (CELL_W, CELL_H), (0, 0, 0), -1)
    cv2.putText(f, name, (4, CELL_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(f, (0, 0), (CELL_W - 1, CELL_H - 1), RED if excluded else (60, 60, 60), 2)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True, help="your initials")
    ap.add_argument("--file", default="labels.csv", help="label file (e.g. labels_rater2.csv)")
    ap.add_argument("--category", nargs="*", help="only these categories")
    args = ap.parse_args()

    path = ROOT / args.file
    if not path.exists():
        raise SystemExit(f"{path.name} not found (label_tool.py --file creates a new rater file).")
    cols, rows = read_csv(path)
    sugg = {r["clip"]: r for r in read_csv(SUGGESTIONS)[1]} if SUGGESTIONS.exists() else {}
    gpath = grid_file(path)
    grid = {r["clip"]: r for r in read_csv(gpath)[1]} if gpath.exists() else {}

    queue = [r for r in rows
             if not r["usable"] and r["clip"] not in grid
             and (not args.category or r["category"] in args.category)]
    # likely-usable clips first; within that, by group so similar clips sit together
    queue.sort(key=lambda r: (screening_priority(sugg.get(r["clip"], {})), r["group"], r["clip"]))
    if not queue:
        print("No clips left for the grid pass with these options.")
        return
    per_page = COLS * ROWS
    pages = [queue[i: i + per_page] for i in range(0, len(queue), per_page)]
    print(f"{len(queue)} clips on {len(pages)} pages.")

    state = {"reason": 0, "click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y < ROWS * CELL_H:
            state["click"] = (y // CELL_H) * COLS + x // CELL_W

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    show_skeleton, status, p = True, "", 0

    def save_page(page, excluded):
        """Excluded clips -> label file; the rest -> kept in the grid file."""
        for r in page:
            if r["clip"] in excluded:
                apply_exclusion(r, excluded[r["clip"]], args.annotator)
                grid[r["clip"]] = {"clip": r["clip"], "decision": "excluded", "annotator": args.annotator}
            else:
                if r["usable"] == "no":  # was excluded before, now undone
                    r.update(usable="", exclude_reason="", annotator="")
                grid[r["clip"]] = {"clip": r["clip"], "decision": "kept", "annotator": args.annotator}
        err = save(path, cols, rows)
        if err:
            return err
        write_csv(gpath, GRID_COLUMNS, list(grid.values()))
        return None

    while 0 <= p < len(pages):
        page = pages[p]
        cells = [load_cell(r, sugg.get(r["clip"], {})) for r in page]
        excluded = {r["clip"]: r["exclude_reason"] for r in page if r["usable"] == "no"}
        tick, action = 0, None
        while action is None:
            canvas = np.full((ROWS * CELL_H + BAR_H, COLS * CELL_W, 3), 32, np.uint8)
            for n, (r, cell) in enumerate(zip(page, cells)):
                y0, x0 = (n // COLS) * CELL_H, (n % COLS) * CELL_W
                canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = draw_cell(
                    cell[tick % len(cell)], r, sugg.get(r["clip"], {}), excluded.get(r["clip"]), show_skeleton)
            y = ROWS * CELL_H
            reason = EXCLUDE_REASONS[state["reason"]]
            cv2.putText(canvas, f"Page {p + 1}/{len(pages)}   {len(excluded)} excluded on this page   "
                        f"click = exclude as: {state['reason'] + 1} {reason}",
                        (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, status or "1-7 change reason   ENTER save + next page   P back   "
                        "K skeleton   ESC save + quit   (unmarked clips = kept for labelling)",
                        (10, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW if status else GREY, 1, cv2.LINE_AA)
            cv2.imshow(WIN, canvas)

            key = cv2.waitKeyEx(80)
            tick += 1
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
                break
            if state["click"] is not None:
                n, state["click"] = state["click"], None
                if n < len(page):
                    clip = page[n]["clip"]
                    if clip in excluded:
                        del excluded[clip]
                    else:
                        excluded[clip] = reason
                continue
            if key == -1:
                continue
            ch = chr(key).lower() if 0 <= key < 256 else ""
            if ch.isdigit() and 1 <= int(ch) <= len(EXCLUDE_REASONS):
                state["reason"] = int(ch) - 1
            elif ch == "k":
                show_skeleton = not show_skeleton
            elif key == 13:
                action = "next"
            elif ch == "p":
                action = "prev"
            elif key == 27:
                action = "quit"
            if action:
                err = save_page(page, excluded)
                if err:
                    status, action = err, None
        if action == "quit":
            break
        p = p + 1 if action == "next" else max(0, p - 1)
    cv2.destroyAllWindows()

    kept = sum(g["decision"] == "kept" for g in grid.values())
    gone = sum(g["decision"] == "excluded" for g in grid.values())
    print(f"Grid pass so far: {kept} kept for labelling, {gone} excluded. "
          f"Next: label_tool.py shows the kept clips first.")


if __name__ == "__main__":
    main()
