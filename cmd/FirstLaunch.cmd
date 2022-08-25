if exist Geosteering\venv rmdir /Q /S Geosteering\venv
call Python\Python38\python -m venv Geosteering\venv
cd Geosteering\venv\Scripts
call activate
call python.exe -m pip install --upgrade pip
call pip install scikit-fda dtaidistance lasio scipy shapely pandas scikit-learn matplotlib plotly

