set dirData =%cd%\Data\

call project\venv\Scripts\activate
call python project\main.py --scenario_path="%dirData %input.xml" --offset_path="%dirData %input.las" --result_path="%dirData %output.xml"
