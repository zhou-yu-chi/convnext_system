import os
import sys
from pathlib import Path

# ========================================================
# 1. 應用程式核心路徑 (唯讀：通常位於 Program Files)
# ========================================================
if getattr(sys, 'frozen', False):
    # PyInstaller 打包後的執行檔位置
    APP_ROOT = Path(sys.executable).parent.resolve()
else:
    # 開發環境下，這支程式所在的上一層 (視你的資料夾結構而定)
    APP_ROOT = Path(__file__).resolve().parent

# ========================================================
# 2. 外部使用者資料路徑 (可讀寫：位於 Documents/convnext_system)
# ========================================================
USER_DOCUMENTS = Path(os.path.expanduser("~/Documents"))
SYSTEM_DATA_ROOT = USER_DOCUMENTS / "convnext_system"

# 核心資料夾
DATASET_ROOT = SYSTEM_DATA_ROOT / "dataset"
MODELS_ROOT = SYSTEM_DATA_ROOT / "All_Trained_Models"
REPORTS_ROOT = SYSTEM_DATA_ROOT / "validation_reports"

# ========================================================
# 3. 系統隱藏/設定資料路徑 (可讀寫：位於 AppData/Local)
# ========================================================
# 這裡用來放 license.dat，使用者才不會不小心在 Documents 裡刪掉它
HIDDEN_DATA_ROOT = Path(os.getenv('LOCALAPPDATA')) / "convnext_system" / "Settings"
LICENSE_FILE = HIDDEN_DATA_ROOT / "license.dat"

# ========================================================
# 4. 路徑輔助函式
# ========================================================
def get_project_dir(project_name: str) -> Path:
    """取得特定專案的資料夾路徑"""
    return DATASET_ROOT / project_name

def get_model_save_dir(project_name: str) -> Path:
    """取得特定專案的模型儲存路徑"""
    return MODELS_ROOT / project_name

def ensure_all_paths_exist():
    """
    確保所有需要的「使用者可讀寫」資料夾都已建立。
    請在程式剛啟動時呼叫此函式。
    """
    try:
        os.makedirs(DATASET_ROOT, exist_ok=True)
        os.makedirs(MODELS_ROOT, exist_ok=True)
        os.makedirs(REPORTS_ROOT, exist_ok=True)
        os.makedirs(HIDDEN_DATA_ROOT, exist_ok=True)
        print(f"📂 系統資料夾已確保建立於: {SYSTEM_DATA_ROOT}")
    except Exception as e:
        print(f"⚠️ 警告：無法建立應用程式資料夾: {e}")