@echo off
cd /d %~dp0

:: Start FastAPI backend
start cmd /k "call venv\Scripts\activate && uvicorn main:app --reload"

:: Start LLaMA via Ollama
start cmd /k "ollama run llama2"

:: Start Streamlit frontend
start cmd /k "call venv\Scripts\activate && streamlit run app.py"
