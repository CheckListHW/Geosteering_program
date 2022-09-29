if not exist project\python (
	curl -L --output master.zip https://github.com/kos4v/python/archive/refs/heads/master.zip
    tar -xf master.zip
    move python-master\python project\python
    del master.zip /s /q
    rmdir python-master /s /q
)


call project\python\python.exe -m venv project\venv
call project\venv\Scripts\Activate
call python.exe -m pip install --upgrade pip
call pip install -r project\requirements.txt