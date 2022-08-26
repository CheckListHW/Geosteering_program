if exist Geosteering\venv rmdir /Q /S Geosteering\venv
call Python\python -m venv Geosteering\venv
cd Geosteering\venv\Scripts
call activate
call python.exe -m pip install --upgrade pip
call pip install dtaidistance lasio scipy shapely pandas scikit-learn
cd ..
cd ..
cd ..
