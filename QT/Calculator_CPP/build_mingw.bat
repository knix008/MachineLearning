@echo off
echo Building Qt Calculator with MinGW...

REM Create build directory if it doesn't exist
if not exist build_mingw mkdir build_mingw
cd build_mingw

REM Configure with CMake for MinGW (adjust Qt path as needed)
cmake .. -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH="C:/Qt/6.5.0/mingw_64"

REM Build the project
mingw32-make

echo.
echo Build completed!
echo Run the calculator with: build_mingw\calculator.exe
pause
