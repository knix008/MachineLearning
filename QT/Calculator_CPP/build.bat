@echo off
echo Building Qt Calculator...

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

REM Configure with CMake (adjust Qt path as needed)
cmake .. -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/6.5.0/msvc2022_64"

REM Build the project
cmake --build . --config Release

echo.
echo Build completed!
echo Run the calculator with: build\Release\calculator.exe
pause
