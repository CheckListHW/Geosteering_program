if not exist project\venv (
	call install.cmd
)
if exist main rmdir /s /q main

call project\venv\Scripts\activate
call pip install PyInstaller
call pyinstaller --add-data "project\dlls;dlls" project\main.py

call move dist\main main

if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist main.spec del main.spec

if exist project\venv  rd /s project\venv /Q
if exist project\python rd /s project\python /Q