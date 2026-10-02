@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo VCVARS_FAILED
  exit /b 1
)
echo VCVARS_OK
where cl
set CMAKE_GENERATOR=Ninja
set CMAKE_BUILD_PARALLEL_LEVEL=4
cd /d "C:\Users\frddx\Desktop\ChatDFT"
"C:\Users\frddx\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe" -m pip install --no-build-isolation --no-binary=:all: --no-deps "C:\Users\frddx\AppData\Local\Temp\pcdl\pyscf-2.6.2"
echo PIP_EXIT=%errorlevel%
