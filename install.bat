
@echo off

setlocal

set "python_url=https://www.python.org/ftp/python/3.10.0/python-3.10.0-embed-amd64.zip"
set "pip_url=https://bootstrap.pypa.io/get-pip.py"
set "python_zip=%~dp0python.zip"
set "python_dir=%~dp0python"
set "scripts_dir=%~dp0python\Scripts"
set "lib_dir=%~dp0python\Lib\site-packages"
set "pip_py=%~dp0get-pip.py"
set "venv_dir=%~dp0env"

REM Set the filename of the Git installer and the download URL
set "GIT_INSTALLER=Git-2.33.0-64-bit.exe"
set "GIT_DOWNLOAD_URL=https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"


REM Check if Git is already installed
git --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Skipping git as it`s installed.
) else (
    REM Download Git installer if not already downloaded
    if not exist "%GIT_INSTALLER%" (
        echo Downloading Git installer...
        powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%GIT_DOWNLOAD_URL%', '%GIT_INSTALLER%')"
        echo Git installer downloaded.
    )

    if exist "%GIT_INSTALLER%" (
    REM Install Git using the installer
    echo Installing Git...
    "%GIT_INSTALLER%"
    echo Git has been installed. )
)

if not exist "%python_zip%" (
    echo Downloading Python 3.10...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%python_url%', '%python_zip%')"
)

if not exist "%python_dir%" (
    echo Extracting Python 3.10...
    powershell -Command "Expand-Archive '%python_zip%' -DestinationPath '%python_dir%'"
)

REM Set environment variable for embedded Python
set "PATH=%python_dir%;%scripts_dir%;%lib_dir%;%PATH%"

if not exist "%python_dir%\Lib\site-packages\pip" (
echo Installing pip...
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%pip_url%', '%pip_py%')"
python "%pip_py%" )

( 
echo python310.zip
echo Lib\site-packages
echo .
) > "%python_dir%\python310._pth"

if not exist "%python_dir%\Lib\site-packages\virtualenv" (
echo Installing virtualenv...
call pip install virtualenv )

if not exist "%venv_dir%" (
echo Creating virtual environment with Python 3.10...
call "%python_dir%\python" -m virtualenv --python="%python_dir%\python.exe" env )

echo Activating virtual environment 
call "%venv_dir%\Scripts\activate"

call python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118 xformers==0.0.21

python -c "import torch; print('Checking if cuda is available:', torch.cuda.is_available(), '\n,Checking xformers install:'); from xformers import ops"

call git clone https://github.com/comfyanonymous/ComfyUI "%~dp0ComfyUI"
@REM call git clone https://github.com/Sxela/ComfyWarp "%~dp0ComfyUI/custom_nodes/ComfyWarp"


call python -m pip install -r "%~dp0ComfyUI/requirements.txt"
@REM call python -m pip install -r "%~dp0ComfyUI/custom_nodes/ComfyWarp/requirements.txt"

call python -m pip install opencv-python scikit-image

call cd ComfyUI
call python main.py

echo Deactivating virtual environment...
deactivate

REM This script downloads the embeddable Python 3.10 zip file from the official website, extracts it to a directory named `python`, creates a virtual environment using the `venv` module, installs `pip` and `virtualenv`, creates another virtual environment using `virtualenv`, and deactivates the virtual environment.

REM Source: Conversation with Bing, 19/07/2023
REM (1) python - Activate virtualenv and run .py script from .bat - Stack Overflow. https://stackoverflow.com/questions/47425520/activate-virtualenv-and-run-py-script-from-bat.
REM (2) VirtualEnv and python-embed - Stack Overflow. https://stackoverflow.com/questions/47754357/virtualenv-and-python-embed.
REM (3) python - Downloading sqlite3 in virtualenv - Stack Overflow. https://stackoverflow.com/questions/45704177/downloading-sqlite3-in-virtualenv.
