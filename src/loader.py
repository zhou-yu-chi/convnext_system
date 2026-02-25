import sys
import os
# 1. 取得 loader.py 所在的絕對路徑 (通常是 _internal/src 或是解壓縮後的暫存區)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得專案根目錄 (上一層)
root_dir = os.path.dirname(current_dir)

# 3. 將這些路徑加入 sys.path，確保能 import 到 control, ui, resources_rc
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import traceback
import json
from datetime import datetime
# ========================================================
# 1. 強制引入所有用到的第三方套件
# ========================================================
from cryptography.fernet import Fernet
from PySide6 import QtWidgets, QtCore, QtGui
import control.path_manager
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
import resources_rc
import multiprocessing

# 安全設定
SECRET_KEY = b'UlGsxbokxJGYHQjCrR2Sgqa_RykBFbv57sqV2E5CZUY=' 
control.path_manager.ensure_all_paths_exist()

# ========================================================
# 背景載入工作區
# ========================================================
class AppLoaderWorker(QtCore.QThread):
    finished_signal = QtCore.Signal(object)
    error_signal = QtCore.Signal(str)

    def run(self):
        try:
            # 這裡 import 重型套件，讓它們在背景載入，不會卡住啟動圖
            # 這些 import 原本在最上面，移到這裡可以讓程式秒開
            import torch
            import torchvision
            import PIL
            import matplotlib
            import matplotlib.pyplot
            import numpy
            import sklearn
            import scipy
            import shutil
            import time
            import random
            
            # 最後載入 main
            import main
            self.finished_signal.emit(main)
        except Exception as e:
            self.error_signal.emit(str(e))

# ========================================================
# 授權驗證邏輯 (修改版：防止圖片閃爍)
# ========================================================
def check_license_validity(splash=None): # ★ 接收 splash 物件
    license_path = str(control.path_manager.LICENSE_FILE)
    f = Fernet(SECRET_KEY)
    user_key = ""

    # 1. 先嘗試靜默驗證 (不隱藏圖片)
    if os.path.exists(license_path):
        try:
            with open(license_path, 'r') as file:
                user_key = file.read().strip()
            
            # 驗證金鑰
            decrypted_data = f.decrypt(user_key.encode('utf-8'))
            license_info = json.loads(decrypted_data.decode('utf-8'))
            expire_str = license_info.get("expire_date")
            expire_date = datetime.strptime(expire_str, "%Y-%m-%d")
            
            if datetime.now() > expire_date:
                # 過期了，才需要進入 UI 流程
                if splash: splash.hide() # ★ 需要跳視窗警告時，才隱藏圖片
                QtWidgets.QMessageBox.critical(None, "授權過期", f"您的試用期已於 {expire_str} 結束。")
                if os.path.exists(license_path): os.remove(license_path)
                return False
            
            # ★ 驗證成功！直接回傳 True，圖片完全不會閃爍
            return True

        except Exception:
            # 檔案損毀或金鑰錯誤，準備進入輸入流程
            pass

    # 2. 需要使用者輸入，這時候才隱藏圖片
    if splash: splash.hide() # ★ 關鍵修改：只有這時候才隱藏

    while True:
        input_dialog = QtWidgets.QInputDialog()
        input_dialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Dialog)
        text, ok = input_dialog.getText(None, "軟體啟用", 
                                      "請輸入產品授權金鑰 (License Key):", 
                                      QtWidgets.QLineEdit.Normal, "")
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
                QtWidgets.QMessageBox.critical(None, "授權過期", f"您的試用期已於 {expire_str} 結束。")
                return False
            
            with open(license_path, 'w') as file:
                file.write(user_key)
            
            # ★ 輸入成功後，把圖片秀回來
            if splash: splash.show()
            return True

        except Exception:
            QtWidgets.QMessageBox.warning(None, "驗證失敗", "無效的金鑰！請確認您輸入的字串是否正確。")

# ========================================================
# 程式進入點
# ========================================================
def run_external_main():
    multiprocessing.freeze_support()
    # 1. 初始化 Application
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    app.setStyle("Fusion")
    font = QtGui.QFont("微軟正黑體", 10)
    app.setFont(font)

    # 2. 使用 Qt 資源路徑 (假設資源標籤是 icons/loading.png)
    # 請檢查您的 .qrc 檔案或詢問同事路徑名稱，這裡參考同事 main.py 的寫法
    logo_resource_path = ":/icons/loading.png" 
    
    # 3. 建立並顯示啟動圖
    splash = None
    pixmap = QtGui.QPixmap(":/icons/logo.png")
    
    if not pixmap.isNull():
        # 如果圖片太大，可以像您同事一樣根據螢幕縮放
        screen = app.primaryScreen()
        screen_size = screen.availableGeometry().size()
        pixmap = pixmap.scaled(screen_size * 0.5, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        splash = QtWidgets.QSplashScreen(pixmap)
        splash.show()
        splash.showMessage("正在驗證授權...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, QtGui.QColor("white"))
    else:
        print("警告：無法從資源檔載入圖片")

    # 4. 授權驗證 (務必加上 if splash 判斷，防止 NoneType 錯誤)
    if not check_license_validity(splash): 
        sys.exit(0)

    if splash: 
        splash.showMessage("正在載入 AI 核心模組...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, QtGui.QColor("white"))
        splash.show() # 確保它是顯示狀態
    
    # 6. 啟動背景 Worker
    loader_thread = AppLoaderWorker()
    
    def on_loaded(main_module):
        try:
            splash.showMessage("啟動主視窗...", QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, QtGui.QColor("white"))
            # 這裡呼叫你的 MainWindow
            real_window = main_module.MainWindow() 
            real_window.show()
            
            if splash:
                splash.finish(real_window)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "啟動失敗", f"主視窗初始化失敗:\n{e}")
            sys.exit(1)

    def on_error(err_msg):
        QtWidgets.QMessageBox.critical(None, "載入錯誤", f"無法載入主程式:\n{err_msg}")
        sys.exit(1)

    loader_thread.finished_signal.connect(on_loaded)
    loader_thread.error_signal.connect(on_error)
    loader_thread.start()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    run_external_main()