if exist Geosteering\venv rmdir /Q /S Geosteering\venv
call Geosteering\Python\python -m venv Geosteering\venv
call Geosteering\venv\Scripts\activate
call Geosteering\venv\Scripts\python.exe -m pip install --upgrade pip
call Geosteering\venv\Scripts\pip install dtaidistance lasio scipy shapely pandas scikit-learn