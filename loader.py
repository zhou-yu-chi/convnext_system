import sys
import os
import traceback
import json
from datetime import datetime
# ========================================================
# 1. 強制引入所有用到的第三方套件
# ========================================================
from cryptography.fernet import Fernet
import PySide6.QtWidgets
import PySide6.QtCore
import PySide6.QtGui
import torch
import torchvision
import PIL
import matplotlib
import matplotlib.pyplot
import numpy
import sklearn
import sklearn.metrics
import sklearn.utils
import scipy
import shutil
import time
import torch.optim
import torch.nn 
import torch.optim
import matplotlib.pyplot
import random
import matplotlib.backends.backend_qtagg

# ========================================================
# 安全設定 (必須與 keygen.py 相同)
# ========================================================
SECRET_KEY = b'UlGsxbokxJGYHQjCrR2Sgqa_RykBFbv57sqV2E5CZUY=' 
LICENSE_FILE = "license.dat" 

# ========================================================
# 授權驗證邏輯
# ========================================================
def check_license_validity():
    """
    檢查授權是否合法。
    """
    app = PySide6.QtWidgets.QApplication.instance()
    if not app:
        app = PySide6.QtWidgets.QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    license_path = os.path.join(base_dir, LICENSE_FILE)
    
    f = Fernet(SECRET_KEY)
    user_key = ""

    if os.path.exists(license_path):
        try:
            with open(license_path, 'r') as file:
                user_key = file.read().strip()
        except:
            pass

    while True:
        if not user_key:
            # 輸入視窗會在這裡跳出，因為我們會在外部先關掉圖片，所以這裡會正常顯示
            input_dialog = PySide6.QtWidgets.QInputDialog()
            # 設定視窗強制置頂，雙重保險
            input_dialog.setWindowFlags(PySide6.QtCore.Qt.WindowStaysOnTopHint | PySide6.QtCore.Qt.Dialog)
            
            text, ok = input_dialog.getText(None, "軟體啟用", 
                                          "請輸入產品授權金鑰 (License Key):", 
                                          PySide6.QtWidgets.QLineEdit.Normal, "")
            if ok and text:
                user_key = text.strip()
            else:
                return False

        try:
            decrypted_data = f.decrypt(user_key.encode('utf-8'))
            license_info = json.loads(decrypted_data.decode('utf-8'))
            
            expire_str = license_info.get("expire_date")
            expire_date = datetime.strptime(expire_str, "%Y-%m-%d")
            
            if datetime.now() > expire_date:
                PySide6.QtWidgets.QMessageBox.critical(None, "授權過期", 
                    f"您的試用期已於 {expire_str} 結束。\n請聯繫供應商更新授權。")
                if os.path.exists(license_path):
                    os.remove(license_path)
                return False
            
            with open(license_path, 'w') as file:
                file.write(user_key)
            
            return True

        except Exception as e:
            PySide6.QtWidgets.QMessageBox.warning(None, "驗證失敗", 
                "無效的金鑰！請確認您輸入的字串是否正確。")
            user_key = ""
            print("test")

# ========================================================
# 2. 定義執行外部程式的邏輯 (修改後)
# ========================================================
def run_external_main():
    # 1. 建立 Qt App
    app = PySide6.QtWidgets.QApplication.instance()
    if not app:
        app = PySide6.QtWidgets.QApplication(sys.argv)

    # ==================================================
    # ★★★ 重點修正：在這裡執行關閉啟動圖 ★★★
    # ==================================================
    try:
        # pyi_splash 是 PyInstaller 打包時才會生成的模組
        # 在 IDE 裡執行會報錯，所以要用 try...except 包起來
        import pyi_splash
        
        # 為了使用者體驗，可以先更新一行文字 (選用)
        pyi_splash.update_text("System Initializing...")
        
        # 檢查圖片是否還活著，如果是，就關掉它！
        if pyi_splash.is_alive():
            pyi_splash.close()
            
    except ImportError:
        # 如果不是打包環境 (例如在 VSCode 執行)，就什麼都不做
        pass
    # ==================================================


    # ★★★ 圖片關閉後，才開始驗證 ★★★
    # 這樣輸入視窗就不會被遮住了
    if not check_license_validity():
        sys.exit(0)

    # ... (以下載入 main.py 的邏輯保持不變) ...
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(base_dir, "main.py")
    
    if not os.path.exists(script_path):
        PySide6.QtWidgets.QMessageBox.critical(None, "錯誤", f"找不到核心程式文件：\n{script_path}")
        return

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    print(f"正在啟動外部程式: {script_path}")

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()

        global_vars = globals().copy()
        global_vars['__file__'] = script_path
        global_vars['__name__'] = '__main__'
        global_vars['__LAUNCHED_BY_LOADER__'] = True 

        exec(code, global_vars)

    except Exception:
        traceback.print_exc()
        os.system("pause")

if __name__ == "__main__":
    run_external_main()