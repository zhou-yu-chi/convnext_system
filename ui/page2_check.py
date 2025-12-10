import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QListWidget, QListWidgetItem, QComboBox, 
                             QMessageBox, QSizePolicy, QSplitter, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon, QAction

class Page2_Check(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.current_folder = "OK" 
        self.current_selected_path = None 
        self.init_ui()

    def init_ui(self):
        # 使用與 Page 0 相同的間距設定
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- 1. 頂部工具列 (美化版) ---
        top_bar_container = QFrame()
        top_bar_container.setMaximumHeight(65)
        top_bar_container.setStyleSheet("""
            QFrame {
                background-color: #333; 
                border-radius: 8px; 
                padding: 2px;
            }
        """)
        top_bar = QHBoxLayout(top_bar_container)
        top_bar.setContentsMargins(10, 5, 10, 5)

        lbl_hint = QLabel("👁️ 檢視模式：")
        lbl_hint.setStyleSheet("font-size: 15px; font-weight: bold; color: #ddd; border: none;")
        
        self.combo_folder = QComboBox()
        self.combo_folder.addItems([
            "✅ 檢視 OK 良品資料夾", 
            "❌ 檢視 NG 不良品資料夾", 
            "❓ 檢視 待確認照片 (Unconfirmed)"
        ])
        # 美化下拉選單
        self.combo_folder.setStyleSheet("""
            QComboBox { 
                background-color: #555; color: white; padding: 5px 10px; 
                border-radius: 5px; border: 1px solid #666; font-size: 14px; min-width: 220px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { 
                background-color: #555; color: white; selection-background-color: #00796b; 
            }
        """)
        self.combo_folder.currentIndexChanged.connect(self.on_folder_changed)
        
        self.btn_refresh = QPushButton("🔄 重新整理")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #0277bd; color: white; font-weight: bold; 
                padding: 6px 15px; border-radius: 5px; font-size: 14px;
            }
            QPushButton:hover { background-color: #0288d1; }
        """)
        self.btn_refresh.clicked.connect(self.load_images)

        top_bar.addWidget(lbl_hint)
        top_bar.addWidget(self.combo_folder)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_refresh)
        
        main_layout.addWidget(top_bar_container)

        # --- 2. 中間區域：左右分割 (套用 Page 0 風格) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        # 左側：圖片清單 (完全套用 Page 0 樣式)
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 80)) 
        self.list_widget.setFixedWidth(260) 
        self.list_widget.setSpacing(5)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 5px;
                outline: 0;
            }
            QListWidget::item {
                background-color: #333;
                border-radius: 5px;
                color: #eee;
                padding: 10px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #00796b; 
                border: 1px solid #4db6ac;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #444;
            }
        """)
        # ★ 改用 itemSelectionChanged 才能支援鍵盤上下鍵切換
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 右側：大圖預覽 (深色背景容器)
        right_container = QFrame()
        right_container.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a; /* 極深灰背景 */
                border: 1px solid #444;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)
        
        self.image_preview = QLabel("請選擇照片")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_preview.setStyleSheet("background-color: transparent; color: #666; font-size: 16px;")

        # --- 3. 底部按鈕區 (整合在右側面板下方) ---
        btn_bar = QFrame()
        btn_bar.setStyleSheet("QFrame { background-color: #333; border-top: 1px solid #555; border-radius: 0px; }")
        btn_bar.setMaximumHeight(70)
        action_layout = QHBoxLayout(btn_bar)
        action_layout.setContentsMargins(15, 10, 15, 10)
        action_layout.setSpacing(15)
        
        # 定義按鈕樣式
        btn_style_base = "QPushButton { color: white; font-weight: bold; border-radius: 5px; font-size: 15px; padding: 8px; }"
        
        self.btn_move_ng = QPushButton("❌ 轉至 NG (←)")
        self.btn_move_ng.setMinimumHeight(45)
        self.btn_move_ng.setStyleSheet(btn_style_base + "QPushButton { background-color: #e57373; } QPushButton:hover { background-color: #ef5350; }")
        self.btn_move_ng.clicked.connect(lambda: self.move_image("NG"))
        
        self.btn_delete = QPushButton("🗑️ 刪除 (Del)")
        self.btn_delete.setMinimumHeight(45)
        self.btn_delete.setStyleSheet(btn_style_base + "QPushButton { background-color: #616161; } QPushButton:hover { background-color: #757575; }")
        self.btn_delete.clicked.connect(self.delete_image)
        
        self.btn_move_ok = QPushButton("⭕ 轉至 OK (→)")
        self.btn_move_ok.setMinimumHeight(45)
        self.btn_move_ok.setStyleSheet(btn_style_base + "QPushButton { background-color: #81c784; color: #1b5e20; } QPushButton:hover { background-color: #66bb6a; }")
        self.btn_move_ok.clicked.connect(lambda: self.move_image("OK"))

        action_layout.addWidget(self.btn_move_ng)
        action_layout.addWidget(self.btn_delete)
        action_layout.addWidget(self.btn_move_ok)
        
        right_layout.addWidget(self.image_preview, 1)
        right_layout.addWidget(btn_bar)

        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1) # 右邊佔滿
        
        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)

        # 初始狀態
        self.update_buttons_state()

    # --- 邏輯功能區 ---

    def refresh_ui(self):
        """強制重新整理介面"""
        self.image_preview.clear()
        self.image_preview.setText("請選擇照片")
        self.current_selected_path = None
        self.load_images()
        self.update_buttons_state()

    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)

    def on_folder_changed(self, index):
        if index == 0:
            self.current_folder = "OK"
        elif index == 1:
            self.current_folder = "NG"
        else:
            self.current_folder = "Unconfirmed" 
        self.load_images()
        self.update_buttons_state()

    def update_buttons_state(self):
        self.image_preview.setText("請選擇照片")
        self.image_preview.setPixmap(QPixmap())
        self.current_selected_path = None

        if self.current_folder == "OK":
            self.btn_move_ok.setVisible(False)
            self.btn_move_ng.setVisible(True)
        elif self.current_folder == "NG":
            self.btn_move_ok.setVisible(True)
            self.btn_move_ng.setVisible(False)
        elif self.current_folder == "Unconfirmed":
            self.btn_move_ok.setVisible(True)
            self.btn_move_ng.setVisible(True)

    def load_images(self):
        self.list_widget.clear()
        if not self.data_handler.project_path: return

        images = self.data_handler.get_images_in_folder(self.current_folder)
        
        for img_path in images:
            file_name = os.path.basename(img_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, img_path)
            item.setIcon(QIcon(img_path)) # 顯示縮圖
            self.list_widget.addItem(item)
            
        # 如果有照片，預設選取第一張
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def on_selection_changed(self):
        """當清單選取改變時 (支援滑鼠點擊與鍵盤上下鍵)"""
        items = self.list_widget.selectedItems()
        if items:
            path = items[0].data(Qt.UserRole)
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
        if not self.current_selected_path: return

        success = self.data_handler.move_specific_file(self.current_selected_path, target_label)
        if success:
            self.remove_current_item_and_select_next("已移動")
        else:
            QMessageBox.warning(self, "錯誤", "移動失敗")

    def delete_image(self):
        if not self.current_selected_path: return

        # 這裡為了操作流暢，可以考慮移除確認視窗，或保留 (看您習慣)
        reply = QMessageBox.question(self, "確認刪除", "確定要永久刪除這張照片嗎？", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.data_handler.delete_specific_file(self.current_selected_path)
            if success:
                self.remove_current_item_and_select_next("已刪除")

    def remove_current_item_and_select_next(self, msg):
        """移除目前項目並自動選取下一張"""
        row = self.list_widget.currentRow()
        self.list_widget.takeItem(row)
        
        count = self.list_widget.count()
        if count > 0:
            # 如果刪除的是最後一張，游標往上移
            if row >= count: row = count - 1
            self.list_widget.setCurrentRow(row)
        else:
            self.image_preview.clear()
            self.image_preview.setText(msg)
            self.current_selected_path = None

    # ★★★ 新增：鍵盤快速鍵控制 ★★★
    def keyPressEvent(self, event):
        # 只有在有選取照片時才觸發
        if not self.current_selected_path:
            super().keyPressEvent(event)
            return

        # Delete 鍵 -> 刪除
        if event.key() == Qt.Key_Delete:
            self.delete_image()
            
        # 左鍵 -> 移至 NG (如果按鈕可見)
        elif event.key() == Qt.Key_Left:
            if self.btn_move_ng.isVisible():
                self.move_image("NG")
                
        # 右鍵 -> 移至 OK (如果按鈕可見)
        elif event.key() == Qt.Key_Right:
            if self.btn_move_ok.isVisible():
                self.move_image("OK")
                
        else:
            super().keyPressEvent(event)