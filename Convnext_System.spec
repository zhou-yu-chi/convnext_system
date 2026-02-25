# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# 1. 收集 torch 與 torchvision 的核心資料 (非常重要，否則載入模型會報錯)
torch_datas = collect_data_files('torch')
torchvision_datas = collect_data_files('torchvision')
all_datas = torch_datas + torchvision_datas

block_cipher = None

a = Analysis(
    ['src\\loader.py'], # 注意：如果你的 loader.py 不是在 src 資料夾下，請把原本的 'src\\loader.py' 改成 'loader.py'
    pathex=['src'], # ★ 新增：強迫掃描 Conda 套件庫
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'resources_rc',
        'PySide6', 
        'PySide6.QtCore',
        'PySide6.QtGui',    
        'PySide6.QtWidgets',
        'shiboken6',           
        'torch', 
        'torchvision', 
        'cv2', 
        'numpy', 
        'matplotlib', 
        'sklearn', 
        'PIL',
        'cryptography',
        'control', 
        'ui',
        # ★ 以下是建議補上的，因為你在 loader.py 有用到
        'scipy',
        'matplotlib.backends.backend_qtagg', 
        'sklearn.metrics',
        'sklearn.utils' 
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],                 # ★ 新增：把 splash 物件放進來
    exclude_binaries=True,
    name='convnext_system', # 產生的執行檔名稱
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # 設為 False，這樣執行時才不會跳出黑色的 CMD 視窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'], # 指向你的軟體圖示
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='convnext_system', # 輸出的資料夾名稱
)