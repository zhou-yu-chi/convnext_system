@echo off
chcp 65001
title AI 視覺檢測系統 - 離線版

:: 設定解壓縮後的環境路徑 (請依實際情況修改)
set "MY_PYTHON_ENV=C:\MyAI_Env"

:: 檢查 python.exe 是否存在
if not exist "%MY_PYTHON_ENV%\python.exe" (
    echo [錯誤] 找不到 Python 環境！
    echo 請確認您已將環境解壓縮到 %MY_PYTHON_ENV%
    pause
    exit
)

:: 啟動主程式 (直接指定那個環境的 python.exe)
echo [系統] 使用離線環境啟動中...
"%MY_PYTHON_ENV%\python.exe" main.py

if %errorlevel% neq 0 (
    echo.
    echo [錯誤] 程式發生異常。
    pause
)