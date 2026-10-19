@echo off
rem Split -> features -> train + compare all models (3-class and 2-class).
rem Run from the project folder after labelling. Results go to results\, web models to webapp\models\.
cd /d "%~dp0scripts"
..\.venv\Scripts\python make_split.py || exit /b 1
..\.venv\Scripts\python extract_features.py || exit /b 1
..\.venv\Scripts\python train_models.py --classes 3 || exit /b 1
..\.venv\Scripts\python train_models.py --classes 2 || exit /b 1
echo.
echo Done. See results\3class\summary.md and results\2class\summary.md
