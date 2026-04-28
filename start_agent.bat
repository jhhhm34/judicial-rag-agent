@echo off
title Agent Frontend
CALL conda.bat activate agent
cd /d E:\LLM\Langchain-Chatchat
streamlit run app.py
pause