@echo OFF
if not exist project\venv (
	call install.cmd
)
call project\venv\Scripts\activate
call python -W ignore project\main.py
if exist project\output\result.xml call copy project\output\result.xml Data\output.xml /Y