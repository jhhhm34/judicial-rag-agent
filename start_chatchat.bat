@echo off
title Chatchat RAG Service
CALL conda.bat activate chatchat
cd /d E:\LLM\Langchain-Chatchat
chatchat start -a
pause