import sys
import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QProgressBar, QListWidget, QFileDialog, QMessageBox,
                             QSizePolicy, QFrame, QStackedWidget, QInputDialog) # <--- 加上這個

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPixmap, QIcon,QPalette,QColor

from control.data_handler import DataHandler
from ui.page0_cropping import Page0_Cropping
from ui.page1_labeling import Page1_Labeling
from ui.page2_check import Page2_Check
from ui.page3_training import Page3_Training
from ui.page4_validation import Page4_Verification
from datetime import datetime

# ==========================================
# 新增：歡迎頁面 (Startup Page)
# ==========================================
class StartupPage(QWidget):
    def __init__(self, on_new_click, on_open_click):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)

        # 1. LOGO 或大標題
        lbl_title = QLabel("AI 視覺檢測訓練系統")
        lbl_title.setStyleSheet("font-size: 36px; font-weight: bold; color: #4db6ac; margin-bottom: 20px;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("請選擇您的操作模式")
        lbl_subtitle.setStyleSheet("font-size: 18px; color: #aaaaaa;")
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_subtitle)

        # 2. 兩個大按鈕區域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(40)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 新增專案按鈕
        self.btn_new = QPushButton("✨ 新增專案\n(New Project)")
        self.btn_new.setFixedSize(250, 150)
        self.btn_new.setObjectName("BigButton")
        self.btn_new.clicked.connect(on_new_click)

        # 開啟專案按鈕
        self.btn_open = QPushButton("📂 開啟專案\n(Open Project)")
        self.btn_open.setFixedSize(250, 150)
        self.btn_open.setObjectName("BigButton")
        self.btn_open.clicked.connect(on_open_click)

        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_open)
        layout.addLayout(btn_layout)

        # 3. 版本號
        lbl_ver = QLabel("Version 1.0.0")
        lbl_ver.setStyleSheet("color: #555; margin-top: 50px;")
        lbl_ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_ver)
        self.setLayout(layout)


# ==========================================
# 主視窗 (Main Window) - 包含堆疊邏輯
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_handler = DataHandler()
        self.setWindowTitle("AI 視覺檢測系統 Pro")
        self.resize(1000, 750)

        # === 設定固定的資料集路徑 ===
        # 這裡設定你要的固定路徑
        # === 設定固定的資料集路徑 (相對路徑版) ===

        # 1. 取得目前這支程式 (main.py) 所在的絕對路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 組合路徑：在專案目錄下建立 "dataset" 資料夾
        # 例如：如果你的專案在 C:\Project，這裡就會自動變成 C:\Project\dataset
        self.dataset_root = os.path.join(current_dir, "dataset")

        # 3. (保持原本邏輯) 如果這個資料夾不存在，就自動幫你建起來
        if not os.path.exists(self.dataset_root):
            try:
                os.makedirs(self.dataset_root)
            except Exception as e:
                print(f"無法建立 Dataset 根目錄: {e}")

        # --- 核心架構：使用 StackedWidget ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 1. 建立歡迎頁 (Index 0)
        self.startup_page = StartupPage(self.on_new_project, self.on_open_project)
        self.stacked_widget.addWidget(self.startup_page)

        # 2. 建立工作區 (Index 1)
        self.tabs = QTabWidget()
        # 這裡把 self (MainWindow) 傳進去，解決你之前遇到的 TypeError
        self.page0 = Page0_Cropping(self.data_handler)
        self.page1 = Page1_Labeling(self.data_handler, self)
        self.page2 = Page2_Check(self.data_handler)
        self.page3 = Page3_Training()

        self.page3.set_data_handler(self.data_handler) # <--- 加入這行！
        self.tabs.addTab(self.page3, "3. 模型訓練")
        self.page4 = Page4_Verification(self.data_handler)
        self.tabs.addTab(self.page0, "0. 圖片裁切")
        self.tabs.addTab(self.page1, "1. 照片標註")
        self.tabs.addTab(self.page2, "2. 結果檢查")
        self.tabs.addTab(self.page3, "3. 模型訓練")
        self.tabs.addTab(self.page4, "4. 驗證檢測")
        
        # 右上角關閉專案按鈕
        btn_close_project = QPushButton("❌ 關閉專案")
        btn_close_project.setFixedSize(80, 40)
        btn_close_project.clicked.connect(self.close_project)
        

        corner_container = QWidget()
        corner_layout = QHBoxLayout(corner_container)
        
        # 設定邊距：(左, 上, 右, 下) -> 重點是右邊設 20，讓它往左彈開
        corner_layout.setContentsMargins(0, 0, 20, 0) 
        corner_layout.addWidget(btn_close_project)
        
        # 把這個「有邊距的容器」放到角落，而不是直接放按鈕
        self.tabs.setCornerWidget(corner_container, Qt.Corner.TopRightCorner)

        self.stacked_widget.addWidget(self.tabs)
        self.apply_stylesheet()

    def on_new_project(self):
        """新增專案邏輯：輸入名稱 -> 自動建立資料夾"""
        # 1. 跳出輸入框讓使用者取名
        project_name, ok = QInputDialog.getText(self, "建立新專案", "請輸入專案名稱:")
        
        if ok and project_name:
            # 移除名稱前後空白，避免誤操作
            project_name = project_name.strip()
            if not project_name:
                QMessageBox.warning(self, "錯誤", "專案名稱不能為空！")
                return

            # 2. 組合完整路徑: C:\Users\...\dataset\專案名
            full_path = os.path.join(self.dataset_root, project_name)

            # 3. 檢查是否已經有同名專案
            if os.path.exists(full_path):
                QMessageBox.warning(self, "錯誤", f"專案 '{project_name}' 已經存在！\n請使用「開啟專案」或是換個名字。")
                return

            # 4. 建立資料夾結構
            try:
                os.makedirs(full_path)           # 建立專案資料夾
                os.makedirs(os.path.join(full_path, "OK")) # 建立 OK 資料夾
                os.makedirs(os.path.join(full_path, "NG")) # 建立 NG 資料夾
                
                # 5. 呼叫 DataHandler 設定專案
                # 這裡會是一個空專案，使用者進去後可以用 Page1 的匯入功能加照片
                self.data_handler.create_new_project(full_path)
                
                # 6. 進入工作區
                self.enter_workspace()
                
                # 提示使用者
                QMessageBox.information(self, "成功", f"專案 '{project_name}' 已建立！\n請在第一頁點擊「匯入」按鈕來加入照片。")

            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"建立專案失敗: {str(e)}")

    def on_open_project(self):
        """開啟專案邏輯：鎖定在 dataset 資料夾選擇"""
        # 1. 開啟資料夾選擇視窗，預設路徑設為 dataset_root
        folder = QFileDialog.getExistingDirectory(self, "選擇專案資料夾", self.dataset_root)
        
        if folder:
            # 2. 呼叫 DataHandler 開啟並檢查結構
            success = self.data_handler.open_existing_project(folder)
            if success:
                self.enter_workspace()
            else:
                QMessageBox.warning(self, "錯誤", "這不是一個有效的專案資料夾！\n(資料夾內必須包含 OK 和 NG 子目錄)")

    def enter_workspace(self):
        """進入工作分頁"""
        # 1. 刷新 Page 0 (裁切頁)
        self.page0.refresh_ui()
        
        # 2. 刷新 Page 1 (標註頁)
        self.page1.refresh_ui()
        
        # 3. 刷新 Page 2 (檢查頁)
        self.page2.refresh_ui()
        
        # 4. ★★★ 新增：重置 Page 3 (訓練頁) ★★★
        self.page3.reset_ui()
        
        # 5. ★★★ 新增：重置 Page 4 (驗證頁) ★★★
        self.page4.reset_ui()
        
        # 6. 切換畫面
        self.stacked_widget.setCurrentIndex(1) 
        # 預設跳轉到第 0 頁 (裁切頁) 或您想保留的頁面
        self.tabs.setCurrentIndex(0)

    def close_project(self):
        """返回首頁"""
        reply = QMessageBox.question(self, "關閉專案", "確定要返回主選單嗎？", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.stacked_widget.setCurrentIndex(0) # 切回首頁

    def apply_stylesheet(self):
        style = """
        QMainWindow { background-color: #2b2b2b; }
        QLabel { color: #ffffff; font-family: 'Microsoft JhengHei', Arial; }
        
        /* 歡迎頁大按鈕 */
        QPushButton#BigButton {
            background-color: #3d3d3d;
            border: 2px solid #555;
            border-radius: 15px;
            color: #ddd;
            font-size: 20px;
            font-weight: bold;
        }
        QPushButton#BigButton:hover {
            background-color: #4db6ac;
            color: white;
            border: 2px solid #80cbc4;
        }

        /* 頁面標題 */
        QLabel#PageTitle { font-size: 28px; font-weight: bold; color: #4db6ac; }
        QLabel#ImageArea { background-color: #1e1e1e; border: 2px dashed #555; border-radius: 10px; }

        /* 一般按鈕 */
        QPushButton { background-color: #3d3d3d; color: white; border-radius: 5px; padding: 5px; }
        QPushButton:hover { background-color: #505050; }
        
        /* OK/NG 按鈕 */
        QPushButton#BtnNG { background-color: #e57373; font-weight: bold; font-size: 18px; }
        QPushButton#BtnNG:hover { background-color: #ef5350; }
        QPushButton#BtnOK { background-color: #81c784; font-weight: bold; font-size: 18px; color: #1b5e20; }
        QPushButton#BtnOK:hover { background-color: #66bb6a; }

        /* Tab 樣式 */
        QTabWidget::pane { border: 1px solid #444; background: #2b2b2b; }
        QTabBar::tab { 
            background: #1e1e1e; 
            color: #bbb; 
            
      
            padding: 12px 30px;    /* 上下 12px，左右 30px (原本是 10 20) */
            font-size: 16px;       /* 字體變大 (原本是 14 或預設) */
            min-width: 100px;      /* 設定最小寬度，看起來更氣派 */
            
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        
        QTabBar::tab:selected { 
            background: #4db6ac; 
            color: white; 
            font-weight: bold; 
        }
        """
        self.setStyleSheet(style)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 2. ★★★ 設定「深色系」調色盤 ★★★
    dark_palette = QPalette()
    
    # 背景設為深灰 (不是純黑，純黑太刺眼)
    dark_gray = QColor(53, 53, 53)
    black = QColor(25, 25, 25)
    white = QColor(255, 255, 255)
    
    dark_palette.setColor(QPalette.Window, dark_gray)
    dark_palette.setColor(QPalette.WindowText, white)
    dark_palette.setColor(QPalette.Base, black)             # 輸入框背景
    dark_palette.setColor(QPalette.AlternateBase, dark_gray)
    dark_palette.setColor(QPalette.ToolTipBase, white)
    dark_palette.setColor(QPalette.ToolTipText, white)
    dark_palette.setColor(QPalette.Text, white)             # 一般文字
    dark_palette.setColor(QPalette.Button, dark_gray)       # 按鈕背景
    dark_palette.setColor(QPalette.ButtonText, white)       # 按鈕文字
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218)) # 選取反白顏色: 藍
    dark_palette.setColor(QPalette.HighlightedText, black)
    
    app.setPalette(dark_palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())