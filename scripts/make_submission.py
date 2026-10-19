"""Build the artefact submission zip (code, deployed models, annotations, results).

Excluded: UCF101 videos and anything derived from their frames (data/, landmarks/, previews/, webapp/test/data),
fp32 model files the app does not load, the Python environment, thesis drafts and personal notes.

Usage:  .venv/Scripts/python scripts/make_submission.py
"""
import zipfile
from pathlib import Path

from common import ROOT

OUT = ROOT / "LandingFormCoach_artefact.zip"
INCLUDE = ["README.md", "requirements.txt", "run_pipeline.bat", "start_webapp.bat", "labels.csv", "labels_grid.csv",
           "labels_rater2.csv", "Functional_Test_Sheet.xlsx",
           "split.csv", "suggestions.csv", "scripts", "webapp", "results", "logs", "models", "docs"]
SKIP_DIRS = {"__pycache__", "data"}             # webapp/test/data holds exported UCF101 frames


def wanted(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if p.suffix == ".onnx" and ".int8" not in p.name and "cnn_transformer_head" not in p.name:
        return False                            # fp32 ResNet50 files: not loaded by the app
    return p.name not in {".DS_Store", "contact_example.png"} and not p.name.startswith("~$")


def main():
    files = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and wanted(f)]
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, Path("LandingFormCoach") / f.relative_to(ROOT))
    size = OUT.stat().st_size / 1e6
    print(f"{len(files)} files -> {OUT.name} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
