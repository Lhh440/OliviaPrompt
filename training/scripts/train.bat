@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  OliviaPrompt 训练入口 (M4)
REM  用法:  train.bat [lite|standard] [data|train|export|all]
REM  默认:  train.bat lite all
REM ============================================================

set STAGE=%1
if "%STAGE%"=="" set STAGE=lite
set STEP=%2
if "%STEP%"=="" set STEP=all

if /I "%STAGE%"=="lite" (set CONFIG=lite-4b.yaml) else (set CONFIG=standard-8b.yaml)
if /I "%STAGE%"=="lite" (set GGUF=olivia-lite-qwen3-4b) else (set GGUF=olivia-standard-qwen3-8b)
set QUANT=Q4_K_M
if /I "%STAGE%"=="standard" set QUANT=Q5_K_M

REM ---- 0. ComfyUI 显存检查（方案 v0.1 第十节）----
tasklist 2>nul | findstr /I "ComfyUI" >nul
if !errorlevel! equ 0 (
  echo [X] 检测到 ComfyUI 正在运行，请先关闭它释放显存，再运行本脚本。
  exit /b 1
)

cd /d "%~dp0..\.."

REM ---- 1. 合成 SFT 数据（幂等：已存在则跳过）----
if /I "%STEP%"=="data" goto step_train
if not exist training\data\out\train.jsonl (
  echo [1/3] 合成 SFT 数据 ...
  python training\data\synthesize.py --n 2000 --seed 42 --out training\data\out
  if !errorlevel! neq 0 exit /b 1
) else (
  echo [1/3] 已有 training\data\out\train.jsonl，跳过合成。重新生成请先删除该目录。
)

:step_train
if /I "%STEP%"=="data" goto done

REM ---- 2. 训练 venv + LLaMA-Factory ----
if not exist .venv-train (
  echo [2/3] 创建训练 venv 并安装依赖（首次约 10-20 分钟）...
  python -m venv .venv-train
  call .venv-train\Scripts\activate.bat
  python -m pip install -U pip
  REM 顺序很重要：先装 CUDA 版 torch，否则 pip 会装到 CPU 轮子
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  if !errorlevel! neq 0 exit /b 1
  pip install hqq "llamafactory[torch,metrics]"
  if !errorlevel! neq 0 exit /b 1
) else (
  call .venv-train\Scripts\activate.bat
)

echo [3/3] 训练 %STAGE%（配置: training\configs\%CONFIG%）...
llamafactory-cli train training\configs\%CONFIG%
if !errorlevel! neq 0 exit /b 1

if /I "%STEP%"=="train" goto done

REM ---- 3. 导出 GGUF ----
echo 导出 GGUF（%QUANT%）...
python training\scripts\export_gguf.py --stage %STAGE%
if !errorlevel! neq 0 exit /b 1

:done
echo.
echo ===== 完成 =====
echo 试跑:  .venv-train 之外的普通环境执行
echo   llama-server -m models\%GGUF%-%QUANT:.=_%.gguf --port 8081
echo   olivia-prompt ask "屋顶花园画画的白发女孩" --base-url http://127.0.0.1:8081/v1 --model olivia
echo 评测:
echo   python training\scripts\eval.py --base-url http://127.0.0.1:8081/v1 --model olivia --label %STAGE%
echo 对比基座:
echo   python training\scripts\eval.py --base-url <基座端点> --model Qwen3-4B --baseline --label base
echo   python training\scripts\eval.py --compare training\eval\report_%STAGE%.json training\eval\report_base.json
endlocal
