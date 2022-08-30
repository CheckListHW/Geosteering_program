call Geosteering\venv\Scripts\activate
python Geosteering\main.py --scenario_path=Data\input.xml --gr_path=Data\input.las --segments_count=100 --delta_deg=10 --st=0.5 --metric=r2 --result_path=Data\output.xml
call Geosteering\venv\Scripts\deactivate