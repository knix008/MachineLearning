@echo off
echo === OpenCV C++ 4배 이미지 확대 빌드 및 실행 ===
echo.

REM 빌드 디렉토리 생성
if not exist build mkdir build
cd build

REM CMake 설정
echo CMake 설정 중...
cmake .. -G "Visual Studio 16 2019" -A x64
if %errorlevel% neq 0 (
    echo CMake 설정 실패!
    pause
    exit /b 1
)

REM 빌드
echo 빌드 중...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo 빌드 실패!
    pause
    exit /b 1
)

echo.
echo 빌드 완료!
echo.

REM 실행
echo 간단한 4배 확대 실행 중...
Release\simple_upscale_4x.exe
if %errorlevel% neq 0 (
    echo 실행 실패!
    pause
    exit /b 1
)

echo.
echo 고급 4배 확대 실행 중...
Release\upscale_4x.exe
if %errorlevel% neq 0 (
    echo 실행 실패!
    pause
    exit /b 1
)

echo.
echo 모든 처리가 완료되었습니다!
pause 