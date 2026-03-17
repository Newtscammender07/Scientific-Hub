@echo off
echo Starting Scientific Hub Streamlit App...
cd /d "%~dp0"
.\.venv\Scripts\python -m streamlit run app.py
pause
