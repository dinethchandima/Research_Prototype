"""Live pilot recording sheet (thesis Section 3.7.2, 5.6.3; reduced pilot) -> Live_Pilot_Sheet.xlsx

Usage:  .venv/Scripts/python scripts/make_live_sheet.py
"""
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from common import ROOT

OUT = ROOT / "Live_Test_Sheet.xlsx"
F = "Arial"
HEAD = PatternFill("solid", fgColor="1F3B6E")
FILL = PatternFill("solid", fgColor="FFF2CC")
EXAMPLE = PatternFill("solid", fgColor="EDEDED")
thin = Side(style="thin", color="BFBFBF")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")
LABELS = '"Correct form,Knee valgus,Trunk lean"'
PARTICIPANTS = 15
PLAN = ["Natural box jump", "Natural box jump", "Natural box jump", "Natural depth jump", "Natural depth jump",
        "Natural depth jump", "Instructed knee valgus", "Instructed knee valgus",
        "Instructed trunk lean", "Instructed trunk lean"]


def header(ws, row, names, widths):
    for c, (h, w) in enumerate(zip(names, widths), 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = Font(name=F, bold=True, color="FFFFFF")
        cell.fill, cell.border = HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = w


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Protocol"
    lines = [
        ("Live evaluation – Landing Form Coach (Section 3.7.2: 15 participants × 10 landings)", True),
        ("", False),
        ("BEFORE STARTING (required)", True),
        ("• Ethics approval in place; each participant reads the information sheet and signs consent.", False),
        ("• PAR-Q completed; exclude anyone with a current lower-limb injury or ACL injury history.", False),
        ("• Warm-up; low box or step (20–30 cm) or floor jumps; stop immediately if there is any pain.", False),
        ("", False),
        ("SET-UP", True),
        ("• Laptop camera 3 m in front of the participant at hip height; start_webapp.bat; Ctrl+F5.", False),
        ("• Settings: model XGBoost, feedback type Specific, VOICE CUES OFF (so the rater cannot hear the result).", False),
        ("• The RATER stands where they cannot see the laptop screen. Ideally a coach or physiotherapist.", False),
        ("• The OPERATOR (you) watches the screen and records what the app showed.", False),
        ("", False),
        ("PER PARTICIPANT (10 landings, one at a time, with a pause between them)", True),
        ("• Order: 3 box jumps and 3 depth jumps with natural technique (30 cm box), then 2 with mild instructed knee valgus and 2 with mild instructed trunk "
         "lean (demonstrate first; participants may decline the instructed ones).", False),
        ("• After EACH landing: 1) the rater writes their label in 'Rater label' WITHOUT seeing the app; "
         "2) then the operator writes the app's result and whether the app detected exactly one landing.", False),
        ("• If the app missed the landing or counted extra landings, write that in 'App detection' and leave the "
         "app result empty for a missed landing.", False),
        ("• At the end of each participant: End session → Download CSV, and rename it P01.csv, P02.csv, … ", False),
        ("• Optional: a second person rates independently in 'Rater 2 label' (for coach-vs-rater kappa).", False),
        ("", False),
        ("Yellow cells = fill in. Grey row = example (not counted). Send this file and the CSVs to Claude.", False),
    ]
    for i, (t, bold) in enumerate(lines, 1):
        ws.cell(row=i, column=1, value=t).font = Font(name=F, bold=bold, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 120

    # participants
    ps = wb.create_sheet("Participants")
    header(ps, 1, ["ID", "Consent signed", "PAR-Q passed", "Age group", "Role", "Plyometric experience",
                   "Device used", "Rater (name/role)", "Notes"], [7, 13, 12, 12, 16, 20, 14, 22, 30])
    yn = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    age = DataValidation(type="list", formula1='"18 to 24,25 to 34,35 or older"', allow_blank=True)
    role = DataValidation(type="list", formula1='"Athlete,Coach,Both"', allow_blank=True)
    exp = DataValidation(type="list", formula1='"Less than 1 year,1 to 3 years,4 to 7 years,More than 7 years"',
                         allow_blank=True)
    for dv in (yn, age, role, exp):
        ps.add_data_validation(dv)
    for i in range(PARTICIPANTS):
        r = i + 2
        ps.cell(row=r, column=1, value=f"P{i + 1:02d}").font = Font(name=F)
        ps.cell(row=r, column=7, value="Laptop").font = Font(name=F)
        for c in range(2, 10):
            cell = ps.cell(row=r, column=c)
            cell.border = BOX
            if c != 7:
                cell.fill = FILL
        yn.add(ps.cell(row=r, column=2)); yn.add(ps.cell(row=r, column=3))
        age.add(ps.cell(row=r, column=4)); role.add(ps.cell(row=r, column=5)); exp.add(ps.cell(row=r, column=6))

    # landings
    ls = wb.create_sheet("Landings")
    cols = ["Participant", "Landing #", "Instructed", "Rater label", "App detection", "App result",
            "App confidence (%)", "Rater 2 label (optional)", "Notes", "Rater fault?", "App fault?"]
    header(ls, 1, cols, [11, 9, 22, 15, 16, 15, 12, 16, 28, 10, 10])
    lab = DataValidation(type="list", formula1=LABELS, allow_blank=True)
    det = DataValidation(type="list", formula1='"One landing,Missed,Extra landings"', allow_blank=True)
    for dv in (lab, det):
        ls.add_data_validation(dv)
    example = ["e.g. P01", 1, "Natural", "Correct form", "One landing", "Correct form", 84, "", ""]
    for c, v in enumerate(example, 1):
        cell = ls.cell(row=2, column=c, value=v)
        cell.fill, cell.border = EXAMPLE, BOX
        cell.font = Font(name=F, italic=True, color="595959")
    r = 3
    for p in range(PARTICIPANTS):
        for k, plan in enumerate(PLAN, 1):
            for c, v in enumerate([f"P{p + 1:02d}", k, plan], 1):
                cell = ls.cell(row=r, column=c, value=v)
                cell.font, cell.border = Font(name=F), BOX
            for c in range(4, 10):
                cell = ls.cell(row=r, column=c)
                cell.fill, cell.border, cell.font = FILL, BOX, Font(name=F)
            lab.add(ls.cell(row=r, column=4)); det.add(ls.cell(row=r, column=5))
            lab.add(ls.cell(row=r, column=6)); lab.add(ls.cell(row=r, column=8))
            ls.cell(row=r, column=10, value=f'=IF(D{r}="","",IF(D{r}="Correct form","No","Yes"))')
            ls.cell(row=r, column=11, value=f'=IF(F{r}="","",IF(F{r}="Correct form","No","Yes"))')
            for c in (10, 11):
                ls.cell(row=r, column=c).font, ls.cell(row=r, column=c).border = Font(name=F), BOX
            r += 1
    last = r - 1
    ls.freeze_panes = "D2"

    # quick summary (Claude computes kappa and confidence intervals from the file)
    sm = wb.create_sheet("Summary")
    rng = lambda col: f"Landings!${col}$3:${col}${last}"
    items = [
        ("Landings rated by the rater", f'=COUNTIF({rng("D")},"?*")'),
        ("Landings detected (exactly one)", f'=COUNTIF({rng("E")},"One landing")'),
        ("Missed landings", f'=COUNTIF({rng("E")},"Missed")'),
        ("Extra landings", f'=COUNTIF({rng("E")},"Extra landings")'),
        ("Rater: knee valgus", f'=COUNTIF({rng("D")},"Knee valgus")'),
        ("  … app also said knee valgus", f'=COUNTIFS({rng("D")},"Knee valgus",{rng("F")},"Knee valgus")'),
        ("Rater: trunk lean", f'=COUNTIF({rng("D")},"Trunk lean")'),
        ("  … app also said trunk lean", f'=COUNTIFS({rng("D")},"Trunk lean",{rng("F")},"Trunk lean")'),
        ("Rater: correct form", f'=COUNTIF({rng("D")},"Correct form")'),
        ("  … app also said correct form", f'=COUNTIFS({rng("D")},"Correct form",{rng("F")},"Correct form")'),
        ("Detection rate, knee valgus", "=IF(B6=0,\"-\",B7/B6)"),
        ("Detection rate, trunk lean", "=IF(B8=0,\"-\",B9/B8)"),
        ("Specificity (correct landings not flagged)", "=IF(B10=0,\"-\",B11/B10)"),
    ]
    header(sm, 1, ["Measure", "Value"], [44, 12])
    for i, (name, formula) in enumerate(items, 2):
        sm.cell(row=i, column=1, value=name).font = Font(name=F)
        cell = sm.cell(row=i, column=2, value=formula)
        cell.font, cell.border = Font(name=F), BOX
        if i >= 12:
            cell.number_format = "0.00"
    sm.cell(row=len(items) + 3, column=1,
            value="Kappa, false-positive/negative rates and 95% confidence intervals are computed by "
                  "scripts/live_analysis.py.").font = Font(name=F, italic=True, size=9)
    wb.save(OUT)
    print("saved", OUT, "| landing rows:", last - 2)


if __name__ == "__main__":
    main()
