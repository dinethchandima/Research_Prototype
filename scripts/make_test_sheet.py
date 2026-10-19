"""Functional test recording sheet (thesis Tables 5.2 and 5.3) -> Functional_Test_Sheet.xlsx

Usage:  .venv/Scripts/python scripts/make_test_sheet.py
"""
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from common import ROOT

OUT = ROOT / "Functional_Test_Sheet.xlsx"
F = "Arial"
HEAD = PatternFill("solid", fgColor="1F3B6E")
FILL = PatternFill("solid", fgColor="FFF2CC")     # yellow = fill in
EXAMPLE = PatternFill("solid", fgColor="EDEDED")
thin = Side(style="thin", color="BFBFBF")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")

TESTS = [
    ("TC01", "FR1", "Run start_webapp.bat, open http://localhost:8000, press Start camera, choose Allow.",
     "Live camera view is shown with the skeleton overlay."),
    ("TC02", "FR1", "Block the camera (camera icon in the address bar), reload, press Start camera. "
                    "Afterwards set it back to Allow.",
     "Message 'Camera access was blocked...' with a Try again button; no crash."),
    ("TC03", "FR2", "Stand about 3 m away with the whole body in view.",
     "Skeleton follows the body; chip shows 'Ready: jump when you like'."),
    ("TC04", "FR3", "Do one box jump (or two-foot jump) and land.", "Exactly one landing appears in History."),
    ("TC05", "FR3", "Walk around and stand still for 30 s without jumping.", "No landing is added."),
    ("TC06", "FR4", "Settings: turn on Researcher details. Do one landing.",
     "The History row shows FPPA and trunk angles for the landing."),
    ("TC07", "FR5", "Land with good form (3 trials; record them on the 'Classification trials' sheet).",
     "'Good landing' shown and spoken."),
    ("TC08", "FR5", "Land with mild, deliberate knees-in (3 trials).",
     "'Knees caving in' shown; 'Knees out' spoken."),
    ("TC09", "FR5", "Land with a mild, deliberate sideways trunk lean (3 trials).",
     "'Leaning to the side' shown; 'Chest centred' spoken."),
    ("TC10", "FR6", "Do 3 jumps in a row (ask a friend to watch or film).",
     "Each cue appears before the next take-off."),
    ("TC11", "FR7", "With Researcher details on, do 5 jumps.",
     "5 History rows, each with model ms, landing-to-cue ms and (when detected) next take-off ms."),
    ("TC12", "FR8", "Press End session, then Download CSV.",
     "Session summary appears; the CSV has one row per landing."),
]
BROWSERS = ["Chrome", "Edge", "Firefox", "Safari"]


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=F, bold=True, color="FFFFFF")
        cell.fill = HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BOX


def main():
    wb = Workbook()

    # ---------------------------------------------------------------- How to fill
    ws = wb.active
    ws.title = "How to fill"
    lines = [
        ("Functional test sheet – Landing Form Coach", True),
        ("Thesis Tables 5.2 (test cases) and 5.3 (results by browser).", False),
        ("", False),
        ("1. Start the app with start_webapp.bat; place the camera 3 m away at hip height.", False),
        ("2. On the 'Tests' sheet, do each test and fill only the YELLOW cells: actual result, status, "
         "notes, screenshot file name, tester and date.", False),
        ("3. Status must be Pass, Fail, Partial or Not tested (drop-down). Write what really happened – a "
         "documented Fail is a valid result.", False),
        ("4. TC07–TC09 (classification): do 3 trials each and record them on 'Classification trials'; "
         "then copy the result (e.g. '2 of 3 correct') into the Tests sheet.", False),
        ("5. Safari does not run on Windows: set its rows to 'Not tested' with the note 'no Safari device'.", False),
        ("6. Take one screenshot per test (Win + Shift + S) for Appendix E and write its file name.", False),
        ("7. The 'Summary' sheet fills itself (Table 5.3). Send this file and your session CSV to Claude.", False),
        ("", False),
        ("Grey row on the Tests sheet = an example of how to fill a row. It is not counted in the summary.", False),
        ("Yellow cells = cells to fill in.", False),
    ]
    for i, (t, bold) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=F, bold=bold, size=13 if i == 1 else 10)
    ws["A14"].fill = FILL
    ws.column_dimensions["A"].width = 110

    # ---------------------------------------------------------------- Tests
    ts = wb.create_sheet("Tests")
    cols = ["ID", "FR", "Browser", "What to do", "Expected result", "Actual result", "Status", "Notes",
            "Screenshot file", "Tester", "Date"]
    widths = [7, 6, 10, 46, 40, 40, 12, 26, 18, 10, 12]
    for c, (h, w) in enumerate(zip(cols, widths), 1):
        ts.cell(row=1, column=c, value=h)
        ts.column_dimensions[chr(64 + c)].width = w
    style_header(ts, 1, len(cols))
    example = ["e.g. TC04", "FR3", "Chrome", "Do one box jump and land.", "Exactly one landing appears in History.",
               "One landing appeared, classified Good, cue after about 0.3 s.", "Pass", "",
               "TC04_chrome.png", "DC", "19/10/2026"]
    for c, v in enumerate(example, 1):
        cell = ts.cell(row=2, column=c, value=v)
        cell.fill, cell.font, cell.alignment, cell.border = EXAMPLE, Font(name=F, italic=True, color="595959"), WRAP, BOX
    status_dv = DataValidation(type="list", formula1='"Pass,Fail,Partial,Not tested"', allow_blank=True)
    ts.add_data_validation(status_dv)
    r = 3
    for browser in BROWSERS:
        for tid, fr, what, expected in TESTS:
            row = [tid, fr, browser, what, expected]
            for c, v in enumerate(row, 1):
                cell = ts.cell(row=r, column=c, value=v)
                cell.font, cell.alignment, cell.border = Font(name=F), WRAP, BOX
            for c in range(6, 12):
                cell = ts.cell(row=r, column=c)
                cell.fill, cell.font, cell.alignment, cell.border = FILL, Font(name=F), WRAP, BOX
            if browser == "Safari":
                ts.cell(row=r, column=7, value="Not tested")
                ts.cell(row=r, column=8, value="No Safari device (Windows)")
            status_dv.add(ts.cell(row=r, column=7))
            r += 1
    # TC13 (all browsers) and TC14 (privacy), once each
    extra = [("TC13", "NFR6", "All", "Repeat TC01–TC12 on Chrome, Edge, Firefox and Safari (rows above).",
              "All cases pass on each browser."),
             ("TC14", "NFR5", "Chrome", "F12 → Network tab; start a session and do a few jumps; look at every request.",
              "Only local files are loaded; no request sends images or video; no external servers.")]
    for tid, fr, br, what, expected in extra:
        for c, v in enumerate([tid, fr, br, what, expected], 1):
            cell = ts.cell(row=r, column=c, value=v)
            cell.font, cell.alignment, cell.border = Font(name=F), WRAP, BOX
        for c in range(6, 12):
            cell = ts.cell(row=r, column=c)
            cell.fill, cell.font, cell.alignment, cell.border = FILL, Font(name=F), WRAP, BOX
        status_dv.add(ts.cell(row=r, column=7))
        r += 1
    last = r - 1
    ts.freeze_panes = "D2"
    ts.auto_filter.ref = f"A1:K{last}"

    # ---------------------------------------------------------------- Classification trials
    cs = wb.create_sheet("Classification trials")
    hdr = ["Test", "Browser", "Trial", "Form performed", "App result (title shown)", "Correct?", "Confidence (%)",
           "Notes"]
    for c, (h, w) in enumerate(zip(hdr, [8, 10, 7, 22, 26, 10, 14, 30]), 1):
        cs.cell(row=1, column=c, value=h)
        cs.column_dimensions[chr(64 + c)].width = w
    style_header(cs, 1, len(hdr))
    form = {"TC07": "Good landing", "TC08": "Knees caving in", "TC09": "Leaning to the side"}
    result_dv = DataValidation(type="list", formula1='"Good landing,Knees caving in,Leaning to the side,No landing detected"',
                               allow_blank=True)
    cs.add_data_validation(result_dv)
    r = 2
    for browser in ("Chrome", "Edge"):
        for tid, expected in form.items():
            for trial in (1, 2, 3):
                vals = [tid, browser, trial, expected]
                for c, v in enumerate(vals, 1):
                    cell = cs.cell(row=r, column=c, value=v)
                    cell.font, cell.border = Font(name=F), BOX
                for c in (5, 7, 8):
                    cell = cs.cell(row=r, column=c)
                    cell.fill, cell.font, cell.border = FILL, Font(name=F), BOX
                result_dv.add(cs.cell(row=r, column=5))
                cs.cell(row=r, column=6, value=f'=IF(E{r}="","",IF(E{r}=D{r},"Yes","No"))').font = Font(name=F)
                cs.cell(row=r, column=6).border = BOX
                r += 1
    tr_last = r - 1
    r += 1
    cs.cell(row=r, column=1, value="Correct trials (of those filled in)").font = Font(name=F, bold=True)
    r += 1
    for c, h in enumerate(["Test", "Browser", "Correct", "Filled", "Result text"], 1):
        cs.cell(row=r, column=c, value=h)
    style_header(cs, r, 5)
    r += 1
    for browser in ("Chrome", "Edge"):
        for tid in form:
            cs.cell(row=r, column=1, value=tid).font = Font(name=F)
            cs.cell(row=r, column=2, value=browser).font = Font(name=F)
            cs.cell(row=r, column=3, value=f'=COUNTIFS($A$2:$A${tr_last},A{r},$B$2:$B${tr_last},B{r},'
                                           f'$F$2:$F${tr_last},"Yes")')
            cs.cell(row=r, column=4, value=f'=COUNTIFS($A$2:$A${tr_last},A{r},$B$2:$B${tr_last},B{r},'
                                           f'$E$2:$E${tr_last},"<>")')
            cs.cell(row=r, column=5, value=f'=IF(D{r}=0,"",C{r}&" of "&D{r}&" correct")')
            for c in range(1, 6):
                cs.cell(row=r, column=c).font = Font(name=F)
                cs.cell(row=r, column=c).border = BOX
            r += 1
    cs.freeze_panes = "A2"

    # ---------------------------------------------------------------- Summary (Table 5.3)
    sm = wb.create_sheet("Summary")
    hdr = ["Browser", "Test cases run", "Passed", "Partial", "Failed", "Not tested", "Pass rate (of run)"]
    for c, (h, w) in enumerate(zip(hdr, [12, 15, 10, 10, 10, 12, 18]), 1):
        sm.cell(row=1, column=c, value=h)
        sm.column_dimensions[chr(64 + c)].width = w
    style_header(sm, 1, len(hdr))
    rng_b, rng_s = f"Tests!$C$3:$C${last}", f"Tests!$G$3:$G${last}"
    for i, browser in enumerate(BROWSERS, 2):
        sm.cell(row=i, column=1, value=browser)
        sm.cell(row=i, column=3, value=f'=COUNTIFS({rng_b},A{i},{rng_s},"Pass")')
        sm.cell(row=i, column=4, value=f'=COUNTIFS({rng_b},A{i},{rng_s},"Partial")')
        sm.cell(row=i, column=5, value=f'=COUNTIFS({rng_b},A{i},{rng_s},"Fail")')
        sm.cell(row=i, column=6, value=f'=COUNTIFS({rng_b},A{i},{rng_s},"Not tested")')
        sm.cell(row=i, column=2, value=f"=C{i}+D{i}+E{i}")
        sm.cell(row=i, column=7, value=f'=IF(B{i}=0,"-",C{i}/B{i})')
        sm.cell(row=i, column=7).number_format = "0.0%"
        for c in range(1, 8):
            sm.cell(row=i, column=c).font = Font(name=F)
            sm.cell(row=i, column=c).border = BOX
    n = len(BROWSERS) + 2
    sm.cell(row=n, column=1, value="Total")
    for c, col in zip(range(2, 7), "BCDEF"):
        sm.cell(row=n, column=c, value=f"=SUM({col}2:{col}{n - 1})")
    sm.cell(row=n, column=7, value=f'=IF(B{n}=0,"-",C{n}/B{n})')
    sm.cell(row=n, column=7).number_format = "0.0%"
    for c in range(1, 8):
        sm.cell(row=n, column=c).font = Font(name=F, bold=True)
        sm.cell(row=n, column=c).border = BOX
    sm.cell(row=n + 2, column=1, value="Counts come from the Status column of the Tests sheet (TC01–TC12 per browser; "
                                       "the grey example row is not counted). Pass rate = passed / (passed + partial + "
                                       "failed).").font = Font(name=F, italic=True, size=9)
    wb.save(OUT)
    print("saved", OUT, "| test rows on Tests sheet:", last - 2)


if __name__ == "__main__":
    main()
