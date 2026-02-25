# PowerShell Build Script for convnext_system (ASCII-only to prevent encoding errors)

Write-Host "===================================================" -ForegroundColor Green
Write-Host "convnext_system Starting build process..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""

$ISCC_PATH = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if (-not (Test-Path $ISCC_PATH)) {
    Write-Host "ERROR: Inno Setup Compiler (ISCC.exe) not found." -ForegroundColor Red
    exit
}

Write-Host "Cleaning up old build files (build/, dist/, InstallerOutput/)..."
Remove-Item -Path ".\build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".\dist" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".\InstallerOutput" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""

Write-Host "===================================================" -ForegroundColor Yellow
Write-Host "[Step 1/2] Running PyInstaller..." -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Yellow

$specFile = "convnext_system.spec"
Write-Host "Executing PyInstaller with spec file: $specFile"

# 使用你目前的 Conda 環境中的 Python 進行打包
C:\Users\5-005-072\anaconda3\envs\convnext_py3119\python.exe -m PyInstaller --clean $specFile

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: PyInstaller failed to run. Check the red text above." -ForegroundColor Red
    exit
}

Write-Host "PyInstaller finished successfully."
Write-Host ""

Write-Host "===================================================" -ForegroundColor Yellow
Write-Host "[Step 2/2] Running Inno Setup Packager..." -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Yellow

if (-not (Test-Path ".\vcredist_x64.exe")) {
    Write-Host "WARNING: vcredist_x64.exe not found in root folder." -ForegroundColor Yellow
}

$issFile = "setup.iss"
& $ISCC_PATH $issFile

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Inno Setup packaging failed." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "Build successful! Installer created in 'InstallerOutput' folder." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green