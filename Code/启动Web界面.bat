@echo off
chcp 65001 >nul
echo ============================================================
echo MARS-Agent 火眼检测系统 - Streamlit Web版启动器
echo ============================================================
echo.

cd /d "%~dp0web_interface"

rem 优先使用项目隔离环境，保证 LangGraph、YOLO 和 Streamlit 在同一 Python 中。
set "PROJECT_PY=%~dp0..\.venvs\mars-agent\Scripts\python.exe"
if exist "%PROJECT_PY%" (
    set "PYTHON_CMD=%PROJECT_PY%"
) else (
    rem 支持将隔离环境放在仓库同级目录，例如 D:\py_daima\.venvs\mars-agent。
    set "PARENT_PY=%~dp0..\..\.venvs\mars-agent\Scripts\python.exe"
    if exist "%PARENT_PY%" (
        set "PYTHON_CMD=%PARENT_PY%"
    ) else (
        set "PYTHON_CMD=python"
    )
)

echo 正在检查Python环境...
"%PYTHON_CMD%" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python！请先安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python环境检测通过
echo.
echo 正在启动Web服务...
echo 📌 浏览器将自动打开 http://localhost:8501
echo 📌 按 Ctrl+C 停止服务
echo.

"%PYTHON_CMD%" run_web.py

if errorlevel 1 (
    echo.
    echo ❌ 启动失败！请检查：
    echo    1. 是否安装了requirements.txt中的依赖
    echo    2. 权重文件是否存在于 weights/ 目录
    echo    3. 端口8501是否被占用
    pause
)
