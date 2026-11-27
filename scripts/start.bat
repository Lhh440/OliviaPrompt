@echo off
setlocal
REM ============================================================
REM  OliviaPrompt 一键启动（M5）
REM  用法: start.bat [lite|standard]   （默认 lite）
REM  首次运行会自动: 安装本包 -> 下载 GGUF 权重(models/models.json, 断点续传
REM  + sha256 校验) -> 下载 llama-server -> 起服务并打开浏览器
REM  大文件存放位置可用环境变量 OLIVIA_MODELS_DIR 指定（如 E:\OliviaPrompt\models）
REM ============================================================
cd /d "%~dp0.."
set OLIVIA_MODELS_DIR=%OLIVIA_MODELS_DIR%
if "%OLIVIA_MODELS_DIR%"=="" set OLIVIA_MODELS_DIR=%CD%\models

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo [X] 未找到 Python。请先安装 Python 3.10+ 并加入 PATH。
  exit /b 1
)

REM 显存提示（>=12GB 可用 standard）
nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>nul | findstr /R "[0-9]" >nul
if %errorlevel% equ 0 (
  for /f "tokens=1 delims=," %%M in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader') do (
    echo [i] 检测到显卡显存: %%M
  )
)

python scripts\bootstrap.py %1
endlocal
