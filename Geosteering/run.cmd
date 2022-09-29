if not exist project\venv (
	call install.cmd
)
call project\venv\Scripts\activate
py -W ignore project\main.py --scenario_path="%cd%\Data\input.xml" --offset_path="%cd%\Data\input.las" --result_path="%cd%\Data\output.xml"
call project\venv\Scripts\deactivate
exit 0