call C:\Users\KosachevIV\PycharmProjects\Input\venv\Scripts\activate.bat
pyinstaller --noconfirm --onedir --windowed  --add-data "C:/Users/KosachevIV/PycharmProjects/Input/res;res/" --add-data "C:/Users/KosachevIV/PycharmProjects/Input/ui;ui/"  "C:/Users/KosachevIV/PycharmProjects/Input/Input.py" --workpath "C:\Users\KosachevIV\Desktop\Output\build" --specpath "C:\Users\KosachevIV\Desktop\Output\spec" --distpath "C:\Users\KosachevIV\Desktop\Output\dist"
cd C:\Users\KosachevIV\Desktop\Output\dist
tar -acf C:\Users\KosachevIV\YandexDisk\Shared\HW\OilCase\input.zip Input
cd C:\Users
timeout 3
::rd C:\Users\KosachevIV\Desktop\Output /s /q
rd C:\Users\KosachevIV\Desktop\logs /s /q