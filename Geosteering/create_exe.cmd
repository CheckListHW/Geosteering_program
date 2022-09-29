if not exist project\venv (
	call install.cmd
)
project\venv\Scripts\activate
pip install PyInstaller
pyinstaller --add-data "project\dlls;dlls" project\main.py
move dist\main main /Y