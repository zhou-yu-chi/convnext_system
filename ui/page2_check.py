import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QListWidget, QListWidgetItem, QComboBox, 
                             QMessageBox, QSizePolicy, QSplitter, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon

class Page2_Check(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.current_folder = "OK" # 預設先看 OK 資料夾
        self.current_selected_path = None # 目前選到的照片路徑
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 1. 頂部選單：選擇要檢查的資料夾
        top_bar = QHBoxLayout()
        lbl_hint = QLabel("👁️ 檢視模式：")
        lbl_hint.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.combo_folder = QComboBox()
        self.combo_folder.addItems([
            "✅ 檢視 OK 良品資料夾", 
            "❌ 檢視 NG 不良品資料夾", 
            "❓ 檢視 待確認照片 (Unconfirmed)"
        ])
        
        self.combo_folder.setStyleSheet("""
            QComboBox { padding: 5px; font-size: 14px; min-width: 200px; }
        """)
        self.combo_folder.currentIndexChanged.connect(self.on_folder_changed)
        
        self.btn_refresh = QPushButton("🔄 重新整理")
        self.btn_refresh.clicked.connect(self.load_images)

        top_bar.addWidget(lbl_hint)
        top_bar.addWidget(self.combo_folder)
        top_bar.addWidget(self.btn_refresh)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # 2. 中間區域：左右分割 (左邊清單，右邊大圖)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左邊：圖片清單
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 80)) # 設定縮圖大小
        self.list_widget.setStyleSheet("QListWidget { font-size: 14px; }")
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.setFixedWidth(200) # 固定寬度
        
        # 右邊：大圖預覽
        self.image_preview = QLabel("請選擇照片")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setObjectName("ImageArea") # 使用主程式定義的樣式
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_preview.setStyleSheet("background-color: #222; border: 2px dashed #555; border-radius: 8px;")

        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.image_preview)
        splitter.setStretchFactor(0, 1) # 右邊圖片區拿走剩餘空間
        
        layout.addWidget(splitter, 1) # 中間區域佔滿高度

        # 3. 底部按鈕區：移動與刪除
        action_layout = QHBoxLayout()
        
        self.btn_move_ng = QPushButton("❌ 轉移至 NG")
        self.btn_move_ng.setStyleSheet("background-color: #e57373; font-weight: bold;")
        self.btn_move_ng.setMinimumHeight(60)
        self.btn_move_ng.clicked.connect(lambda: self.move_image("NG"))
        
        self.btn_delete = QPushButton("🗑️ 刪除此圖")
        self.btn_delete.setStyleSheet("background-color: #555;")
        self.btn_delete.setMinimumHeight(60)
        self.btn_delete.clicked.connect(self.delete_image)
        
        self.btn_move_ok = QPushButton("⭕ 轉移至 OK")
        self.btn_move_ok.setStyleSheet("background-color: #81c784; color: #1b5e20; font-weight: bold;")
        self.btn_move_ok.setMinimumHeight(60)
        self.btn_move_ok.clicked.connect(lambda: self.move_image("OK"))

        action_layout.addWidget(self.btn_move_ng)
        action_layout.addWidget(self.btn_delete)
        action_layout.addWidget(self.btn_move_ok)
        
        layout.addLayout(action_layout)
        self.setLayout(layout)

        # 初始狀態：隱藏不合理的按鈕 (如果在 OK 資料夾，就不顯示「轉移至 OK」)
        self.update_buttons_state()

    def on_folder_changed(self, index):
        """切換檢視 OK, NG 或 Unconfirmed 資料夾"""
        if index == 0:
            self.current_folder = "OK"
        elif index == 1:
            self.current_folder = "NG"
        else:
            # ★★★ 新增這裡：對應到實體資料夾名稱 ★★★
            self.current_folder = "Unconfirmed" 
            
        self.load_images()
        self.update_buttons_state()

    def update_buttons_state(self):
        """依據目前在哪個資料夾，隱藏不必要的按鈕"""
        
        # 清空預覽 (換資料夾時，先把上一張圖清掉)
        self.image_preview.setText("請選擇照片")
        self.image_preview.setPixmap(QPixmap())
        self.current_selected_path = None

        # ★★★ 修改這裡：按鈕顯示邏輯 ★★★
        if self.current_folder == "OK":
            # 在 OK 資料夾：只能搬去 NG，不能搬去 OK
            self.btn_move_ok.setVisible(False)
            self.btn_move_ng.setVisible(True)
            
        elif self.current_folder == "NG":
            # 在 NG 資料夾：只能搬去 OK，不能搬去 NG
            self.btn_move_ok.setVisible(True)
            self.btn_move_ng.setVisible(False)
            
        elif self.current_folder == "Unconfirmed":
            # 在 待確認 資料夾：兩個按鈕都要顯示，讓使用者決定去向
            self.btn_move_ok.setVisible(True)
            self.btn_move_ng.setVisible(True)

    def load_images(self):
        """讀取資料夾圖片並顯示在清單中"""
        self.list_widget.clear()
        if not self.data_handler.project_path:
            return

        images = self.data_handler.get_images_in_folder(self.current_folder)
        
        for img_path in images:
            file_name = os.path.basename(img_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, img_path) # 把完整路徑藏在 item 裡
            
            # 載入小縮圖 (Optional, 如果怕慢可以先不設 icon)
            item.setIcon(QIcon(img_path))
            
            self.list_widget.addItem(item)

    def on_item_clicked(self, item):
        """點擊清單項目時顯示大圖"""
        path = item.data(Qt.UserRole)
        self.current_selected_path = path
        self.show_image(path)

    def show_image(self, path):
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(self.image_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_preview.setPixmap(scaled)
        else:
            self.image_preview.setText("無法載入圖片")

    def resizeEvent(self, event):
        if self.current_selected_path:
            self.show_image(self.current_selected_path)
        super().resizeEvent(event)

    def move_image(self, target_label):
        """移動照片到另一個資料夾"""
        if not self.current_selected_path:
            QMessageBox.warning(self, "提示", "請先選擇一張照片")
            return

        success = self.data_handler.move_specific_file(self.current_selected_path, target_label)
        if success:
            # 移除清單中的項目
            row = self.list_widget.currentRow()
            self.list_widget.takeItem(row)
            
            # 清空預覽
            self.image_preview.clear()
            self.image_preview.setText("已移動")
            self.current_selected_path = None
        else:
            QMessageBox.warning(self, "錯誤", "移動失敗")

    def delete_image(self):
        """刪除目前選取的照片"""
        if not self.current_selected_path:
            QMessageBox.warning(self, "提示", "請先選擇一張照片")
            return

        reply = QMessageBox.question(self, "確認刪除", "確定要永久刪除這張照片嗎？", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.data_handler.delete_specific_file(self.current_selected_path)
            if success:
                # 移除清單中的項目
                row = self.list_widget.currentRow()
                self.list_widget.takeItem(row)
                
                # 清空預覽
                self.image_preview.clear()
                self.image_preview.setText("已刪除")
                self.current_selected_path = None