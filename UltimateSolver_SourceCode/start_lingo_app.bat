@echo off
setlocal

:: Set the path to the isolated environment specifically installed in this folder
set "APP_DIR=%~dp0"
set "ENV_DIR=%APP_DIR%env"

if not exist "%ENV_DIR%\python.exe" (
    echo [错误] 隔离环境 %ENV_DIR% 缺失！
    echo 请先双击运行 install.bat 以安装便携式独立依赖。
    pause
    exit /b 1
)

echo 正在启动 OpenOR Desktop IDE ...
"%ENV_DIR%\python.exe" "%APP_DIR%lingo_ide.py"
