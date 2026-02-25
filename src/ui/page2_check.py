import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QListWidget, QListWidgetItem, QComboBox, 
                             QMessageBox, QSizePolicy, QSplitter, QFrame)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImageReader, QImage

# ==========================================
# 0. 後台縮圖載入小精靈 (複製自 Page 0)
# ==========================================
class IconWorker(QThread):
    icon_loaded = Signal(int, QImage)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.is_running = True

    def run(self):
        for i, path in enumerate(self.image_paths):
            if not self.is_running: break
            # 只讀取縮圖，速度極快
            reader = QImageReader(path)
            reader.setScaledSize(QSize(100, 100)) 
            image = reader.read()
            if not image.isNull():
                self.icon_loaded.emit(i, image)
                # 稍微休息一下，讓介面更滑順
                if i % 10 == 0: QThread.msleep(5)

    def stop(self):
        self.is_running = False
        self.wait()

class Page2_Check(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.current_folder = "OK" 
        self.current_selected_path = None
        self.icon_worker = None # 儲存 Worker
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- 1. 頂部工具列 ---
        top_bar_container = QFrame()
        top_bar_container.setMaximumHeight(65)
        top_bar_container.setStyleSheet("QFrame { background-color: #333; border-radius: 8px; padding: 2px; }")
        top_bar = QHBoxLayout(top_bar_container)
        top_bar.setContentsMargins(10, 5, 10, 5)

        lbl_hint = QLabel("👁️ 檢視模式：")
        lbl_hint.setStyleSheet("font-size: 15px; font-weight: bold; color: #ddd; border: none;")
        
        self.combo_folder = QComboBox()
        self.combo_folder.addItems(["✅ 檢視 OK 良品資料夾", "❌ 檢視 NG 不良品資料夾", "❓ 檢視 待確認照片 (Unconfirmed)"])
        self.combo_folder.setStyleSheet("""
            QComboBox { background-color: #555; color: white; padding: 5px 10px; border-radius: 5px; border: 1px solid #666; font-size: 14px; min-width: 220px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #555; color: white; selection-background-color: #00796b; }
        """)
        self.combo_folder.currentIndexChanged.connect(self.on_folder_changed)
        
        self.btn_refresh = QPushButton("🔄 重新整理")
        self.btn_refresh.setStyleSheet("QPushButton { background-color: #0277bd; color: white; font-weight: bold; padding: 6px 15px; border-radius: 5px; font-size: 14px; } QPushButton:hover { background-color: #0288d1; }")
        self.btn_refresh.clicked.connect(self.refresh_ui)

        top_bar.addWidget(lbl_hint)
        top_bar.addWidget(self.combo_folder)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_refresh)
        main_layout.addWidget(top_bar_container)

        # --- 2. 中間區域 ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        # 左側清單
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 80)) 
        self.list_widget.setFixedWidth(260) 
        self.list_widget.setSpacing(5)
        self.list_widget.setStyleSheet("""
            QListWidget { background-color: #2b2b2b; border: 1px solid #444; border-radius: 8px; padding: 5px; outline: 0; }
            QListWidget::item { background-color: #333; border-radius: 5px; color: #eee; padding: 10px; margin-bottom: 2px; }
            QListWidget::item:selected { background-color: #00796b; border: 1px solid #4db6ac; color: white; }
            QListWidget::item:hover { background-color: #444; }
        """)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 右側預覽
        right_container = QFrame()
        right_container.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #444; border-radius: 8px; }")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)
        
        self.image_preview = QLabel("請選擇照片")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_preview.setStyleSheet("background-color: transparent; color: #666; font-size: 16px;")

        # --- 3. 底部按鈕區 ---
        btn_bar = QFrame()
        btn_bar.setStyleSheet("QFrame { background-color: #333; border-top: 1px solid #555; border-radius: 0px; }")
        btn_bar.setMaximumHeight(70)
        action_layout = QHBoxLayout(btn_bar)
        action_layout.setContentsMargins(15, 10, 15, 10)
        action_layout.setSpacing(15)
        
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
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)
        self.update_buttons_state()

    # --- 邏輯功能區 ---

    def refresh_ui(self):
        # 0. 停止舊的 Worker
        if self.icon_worker and self.icon_worker.isRunning():
            self.icon_worker.stop()
            
        self.image_preview.clear()
        self.image_preview.setText("請選擇照片")
        self.current_selected_path = None
        self.load_images()
        self.update_buttons_state()

    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)

    def on_folder_changed(self, index):
        if index == 0: self.current_folder = "OK"
        elif index == 1: self.current_folder = "NG"
        else: self.current_folder = "Unconfirmed" 
        self.refresh_ui() # 這裡直接呼叫 refresh_ui 比較乾淨

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
        
        # 1. 快速建立文字清單
        for img_path in images:
            file_name = os.path.basename(img_path)
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, img_path)
            self.list_widget.addItem(item)
            
        # 2. 啟動後台小精靈載入縮圖
        if images:
            self.icon_worker = IconWorker(images)
            self.icon_worker.icon_loaded.connect(self.on_icon_loaded)
            self.icon_worker.start()
            
            # ★★★ 修改這裡 (解決第一張黑屏的關鍵) ★★★
            # 原本是：self.list_widget.setCurrentRow(0)
            # 改成下面這行：延遲 50 毫秒 (0.05秒) 再選取
            QTimer.singleShot(50, lambda: self.list_widget.setCurrentRow(0))

    def on_icon_loaded(self, row, image):
        item = self.list_widget.item(row)
        if item:
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    def on_selection_changed(self):
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
        if self.data_handler.move_specific_file(self.current_selected_path, target_label):
            self.remove_current_item_and_select_next("已移動")
        else:
            QMessageBox.warning(self, "錯誤", "移動失敗")

    def delete_image(self):
        if not self.current_selected_path: return
        reply = QMessageBox.question(self, "確認刪除", "確定要永久刪除這張照片嗎？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.data_handler.delete_specific_file(self.current_selected_path):
                self.remove_current_item_and_select_next("已刪除")

    def remove_current_item_and_select_next(self, msg):
        row = self.list_widget.currentRow()
        self.list_widget.takeItem(row)
        count = self.list_widget.count()
        if count > 0:
            if row >= count: row = count - 1
            self.list_widget.setCurrentRow(row)
        else:
            self.image_preview.clear()
            self.image_preview.setText(msg)
            self.current_selected_path = None

    def keyPressEvent(self, event):
        if not self.current_selected_path:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Delete:
            self.delete_image()
        elif event.key() == Qt.Key_Left:
            if self.btn_move_ng.isVisible(): self.move_image("NG")
        elif event.key() == Qt.Key_Right:
            if self.btn_move_ok.isVisible(): self.move_image("OK")
        else:
            super().keyPressEvent(event)