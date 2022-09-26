@echo OFF
if not exist project\venv (
	call install.cmd
)
if exist Data\input.xml call copy Data\input.xml project\input\well2withoutdips\well2withoutdips.xml /Y
if exist Data\input.las call copy Data\input.las project\input\well2withoutdips\GRwell2withoutdips.las /Y
cd project


call venv\Scripts\activate

python -W ignore main.py
cd ..
if exist project\output\result.xml call copy project\output\result.xml Data\output.xml /Y