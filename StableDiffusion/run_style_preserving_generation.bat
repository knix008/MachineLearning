@echo off
echo Stable Diffusion 3.5 Style-Preserving Generation 시작...
echo.

REM Python 가상환경이 활성화되어 있는지 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다. Python을 먼저 설치해주세요.
    pause
    exit /b 1
)

REM 필요한 패키지 설치 확인
echo 필요한 패키지를 확인하고 설치합니다...
pip install -r requirements.txt

echo.
echo 프로그램을 시작합니다...
echo 웹 브라우저에서 http://localhost:7860 으로 접속하세요.
echo.

python stablediffusion-3.5-style-transfer03.py

pause 