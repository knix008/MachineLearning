@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo [torchmcubes installer for conda]
echo.

set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"

if "%CONDA_PREFIX%"=="" (
    echo [ERROR] Conda environment is not active.
    echo Please run this in Anaconda Prompt after:
    echo   conda activate YOUR_ENV
    exit /b 1
)

echo Using conda env: %CONDA_PREFIX%
echo.

echo [1/5] Checking Python path...
where python
python -c "import sys; print(sys.executable)"
if errorlevel 1 goto :error

echo.
echo [2/5] Upgrading build tools...
python -m pip install --upgrade pip setuptools wheel scikit-build-core cmake ninja
if errorlevel 1 goto :error

echo.
echo [3/6] Installing CPU-only PyTorch...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
python -m pip install --force-reinstall --no-cache-dir torch torchvision --index-url %PYTORCH_INDEX_URL%
if errorlevel 1 goto :error
python -c "import torch; print('torch:', torch.__version__); print('torch.cuda:', torch.version.cuda); print('cmake_prefix:', torch.utils.cmake_prefix_path)"
if errorlevel 1 goto :error

echo.
echo [4/6] Installing pybind11 for CMake...
python -m pip install --upgrade pybind11
if errorlevel 1 goto :error

for /f "usebackq delims=" %%i in (`python -c "import torch, pathlib; print(pathlib.Path(torch.utils.cmake_prefix_path).as_posix())"`) do (
    set "CMAKE_PREFIX_PATH=%%i"
)

if "%CMAKE_PREFIX_PATH%"=="" (
    echo [ERROR] Failed to get torch cmake prefix path.
    goto :error
)

for /f "usebackq delims=" %%i in (`python -c "import pybind11, pathlib; print(pathlib.Path(pybind11.get_cmake_dir()).as_posix())"`) do (
    set "PYBIND11_DIR=%%i"
)
if "%PYBIND11_DIR%"=="" (
    echo [ERROR] Failed to get pybind11 cmake dir.
    goto :error
)

set "Torch_DIR=%CMAKE_PREFIX_PATH%/Torch"
set "CMAKE_ARGS=-DTorch_DIR=%Torch_DIR% -Dpybind11_DIR=%PYBIND11_DIR% -DUSE_CUDA=OFF"

set "CMAKE_GENERATOR=Visual Studio 18 2026"
set "CC=cl"
set "CXX=cl"
set "CUDACXX="
set "CUDAHOSTCXX="
set "CUDA_HOME="
set "CUDA_PATH="

echo CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%
echo Torch_DIR=%Torch_DIR%
echo pybind11_DIR=%PYBIND11_DIR%
echo CMAKE_GENERATOR=%CMAKE_GENERATOR%
echo CC=%CC%
echo CXX=%CXX%
echo CMAKE_ARGS=%CMAKE_ARGS%
echo.

echo [5/6] Installing torchmcubes...
python -m pip install --no-build-isolation --no-cache-dir --force-reinstall git+https://github.com/tatsy/torchmcubes.git
if errorlevel 1 goto :error

echo.
echo [6/6] Testing imports...
python -c "import torch, torchmcubes; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('torchmcubes: OK')"
if errorlevel 1 goto :error

echo.
echo Installation successful.
echo You can now run: python app.py
exit /b 0

:error
echo.
echo Installation failed. Please check messages above.
exit /b 1
