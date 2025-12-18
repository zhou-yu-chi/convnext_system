import sys
import os
import traceback

# ========================================================
# 1. 強制引入所有用到的第三方套件
# ========================================================
# PyInstaller 很笨，如果這裡沒寫 import，它就不會把這些庫包進 EXE。
# 雖然這個檔案沒用到這些庫，但為了讓 EXE 裡包含它們，必須寫出來。
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

# 注意：不要 import 您的 ui 或 control 資料夾！
# 我們希望那些資料夾保持在外部，這樣才能隨時修改。

# ========================================================
# 2. 定義執行外部程式的邏輯
# ========================================================
def run_external_main():
    # 取得 EXE 所在的真實目錄
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 設定 main.py 的路徑 (假設它會在 EXE 旁邊)
    script_path = os.path.join(base_dir, "main.py")
    
    # 檢查檔案是否存在
    if not os.path.exists(script_path):
        # 如果找不到 main.py，跳出錯誤視窗
        app = PySide6.QtWidgets.QApplication(sys.argv)
        PySide6.QtWidgets.QMessageBox.critical(None, "錯誤", f"找不到核心程式文件：\n{script_path}")
        return

    # ★★★ 關鍵步驟：把 EXE 所在目錄加入系統路徑 ★★★
    # 這樣 main.py 才能 import 旁邊的 ui 和 control 資料夾
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    print(f"正在啟動外部程式: {script_path}")

    try:
        # 讀取 main.py 的內容
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # 設定執行環境 (讓 main.py 覺得自己是被直接執行的)
        global_vars = globals().copy()
        global_vars['__file__'] = script_path
        global_vars['__name__'] = '__main__'

        # ★★★ 執行 main.py 的程式碼 ★★★
        # 這會使用 EXE 內部封裝好的 PySide6 和 Torch 來跑您外部的程式碼
        exec(code, global_vars)

    except Exception:
        # 如果崩潰，把錯誤訊息印出來並暫停，方便除錯
        traceback.print_exc()
        os.system("pause")

if __name__ == "__main__":
    run_external_main()