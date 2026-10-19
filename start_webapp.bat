@echo off
rem Serves the web app on this PC only (no caching). The camera works on localhost without HTTPS.
cd /d "%~dp0webapp"
start "" http://localhost:8000
..\.venv\Scripts\python serve.py
