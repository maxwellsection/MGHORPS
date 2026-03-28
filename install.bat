@echo off
setlocal
echo ========================================================
echo   OpenOR Desktop IDE - Portable Environment Installer
echo   跨平台一键安装独立运行环境环境向导 (Windows 示范)
echo ========================================================
echo.

set "APP_DIR=%~dp0"
set "ENV_DIR=%APP_DIR%env"
set "MAMBA_EXE=%APP_DIR%micromamba.exe"

echo [步骤 1] 检查便携包 MicroMamba...
if not exist "%MAMBA_EXE%" (
    echo [下载] 正在无声下载 Micromamba...
    curl -k -L -o "%MAMBA_EXE%" "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"
    if errorlevel 1 (
        echo [错误] 下载失败！请手动查验网络。
        pause
        exit /b 1
    )
)

echo [步骤 2] 构建应用专用且完全隔离的本地虚拟文件夹 (./env) ...
if not exist "%ENV_DIR%" (
    "%MAMBA_EXE%" create -p "%ENV_DIR%" -y -c conda-forge python=3.10 numpy scipy pyside6 cupy pytorch
    echo [✔] 环境构建成功！
) else (
    echo [提示] 检测到本地环境夹 ./env 已存在！
)

echo.
echo ========================================================
echo   安装完毕！
echo   卸载时，您只需要删除 %APP_DIR% 整个文件夹即可，它不会在您的系统任何位置留下残留。
echo   运行测试:
echo   请双击运行 start_lingo_app.bat
echo ========================================================
pause
