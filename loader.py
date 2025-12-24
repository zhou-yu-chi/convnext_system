import sys
import os
import traceback
import json
from datetime import datetime
# ========================================================
# 1. 強制引入所有用到的第三方套件
# ========================================================
# PyInstaller 很笨，如果這裡沒寫 import，它就不會把這些庫包進 EXE。
# 雖然這個檔案沒用到這些庫，但為了讓 EXE 裡包含它們，必須寫出來。
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

# 注意：不要 import 您的 ui 或 control 資料夾！
# 我們希望那些資料夾保持在外部，這樣才能隨時修改。

# ========================================================
# 安全設定 (必須與 keygen.py 相同)
# ========================================================
SECRET_KEY = b'UlGsxbokxJGYHQjCrR2Sgqa_RykBFbv57sqV2E5CZUY=' 
# 注意：這裡必須填入跟 keygen.py 一模一樣的 Key，否則無法解密

LICENSE_FILE = "license.dat" # 儲存金鑰的檔案名稱

# ========================================================
# 授權驗證邏輯
# ========================================================
def check_license_validity():
    """
    檢查授權是否合法。
    流程：
    1. 找 license.dat
    2. 如果沒有，跳出輸入框讓使用者輸入
    3. 解密驗證日期
    """
    app = PySide6.QtWidgets.QApplication.instance()
    if not app:
        app = PySide6.QtWidgets.QApplication(sys.argv)

    # 取得執行檔目錄
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    license_path = os.path.join(base_dir, LICENSE_FILE)
    
    f = Fernet(SECRET_KEY)
    user_key = ""

    # 1. 嘗試讀取現有的 Key
    if os.path.exists(license_path):
        try:
            with open(license_path, 'r') as file:
                user_key = file.read().strip()
        except:
            pass

    # 2. 如果沒有 Key 或 Key 無效，進入迴圈要求輸入
    while True:
        # 如果尚未取得 Key (檔案不存在或剛啟動)，跳出輸入視窗
        if not user_key:
            input_dialog = PySide6.QtWidgets.QInputDialog()
            text, ok = input_dialog.getText(None, "軟體啟用", 
                                          "請輸入產品授權金鑰 (License Key):", 
                                          PySide6.QtWidgets.QLineEdit.Normal, "")
            if ok and text:
                user_key = text.strip()
            else:
                # 使用者按取消，直接退出程式
                return False

        # 3. 驗證 Key
        try:
            # 解密
            decrypted_data = f.decrypt(user_key.encode('utf-8'))
            license_info = json.loads(decrypted_data.decode('utf-8'))
            
            expire_str = license_info.get("expire_date")
            expire_date = datetime.strptime(expire_str, "%Y-%m-%d")
            
            # 檢查是否過期
            if datetime.now() > expire_date:
                PySide6.QtWidgets.QMessageBox.critical(None, "授權過期", 
                    f"您的試用期已於 {expire_str} 結束。\n請聯繫供應商更新授權。")
                
                # 清除無效的舊檔案
                if os.path.exists(license_path):
                    os.remove(license_path)
                return False
            
            # === 驗證成功 ===
            # 將有效的 Key 存入檔案 (以免下次還要輸入)
            with open(license_path, 'w') as file:
                file.write(user_key)
            
            # 可以在這裡提示剩餘天數 (選用)
            # days_left = (expire_date - datetime.now()).days
            # print(f"授權有效，剩餘 {days_left} 天")
            
            return True

        except Exception as e:
            # 解密失敗或格式錯誤
            PySide6.QtWidgets.QMessageBox.warning(None, "驗證失敗", 
                "無效的金鑰！請確認您輸入的字串是否正確。")
            user_key = "" # 清空 Key，讓迴圈重跑，重新跳出輸入框

# ========================================================
# 2. 定義執行外部程式的邏輯 (修改後)
# ========================================================
def run_external_main():
    # 1. 建立 Qt App (為了讓輸入視窗能運作)
    app = PySide6.QtWidgets.QApplication.instance()
    if not app:
        app = PySide6.QtWidgets.QApplication(sys.argv)

    # ==================================================
    # ★★★ 新增：關閉啟動圖 (Splash Screen) ★★★
    # ==================================================

    # ★★★ 在這裡插入驗證 ★★★
    if not check_license_validity():
        # 驗證失敗，直接結束，不跑後面的 code
        sys.exit(0)

    # ... (以下保持你原本的 code) ...
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(base_dir, "main.py")
    
    if not os.path.exists(script_path):
        app = PySide6.QtWidgets.QApplication(sys.argv)
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

        # ★★★ 新增這行：設定通關密語 ★★★
        global_vars['__LAUNCHED_BY_LOADER__'] = True 

        exec(code, global_vars)

    except Exception:
        traceback.print_exc()
        os.system("pause")

if __name__ == "__main__":
    run_external_main()