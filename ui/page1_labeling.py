import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QListWidget, QListWidgetItem, QSizePolicy, 
                             QSplitter, QFrame)
from PySide6.QtCore import Qt, QSize, QThread, Signal
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
            reader = QImageReader(path)
            reader.setScaledSize(QSize(100, 100)) 
            image = reader.read()
            if not image.isNull():
                self.icon_loaded.emit(i, image)
                if i % 10 == 0: QThread.msleep(5)

    def stop(self):
        self.is_running = False
        self.wait()

class Page1_Labeling(QWidget):
    def __init__(self, data_handler, main_window):
        super().__init__()
        self.main_window = main_window
        self.data_handler = data_handler
        self.current_selected_path = None
        self.icon_worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 頂部資訊列
        top_bar_container = QFrame()
        top_bar_container.setMaximumHeight(65)
        top_bar_container.setStyleSheet("QFrame { background-color: #333; border-radius: 8px; padding: 2px; }")
        top_bar = QHBoxLayout(top_bar_container)
        top_bar.setContentsMargins(15, 5, 15, 5)

        title = QLabel("照片分類標註")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4db6ac;")
        
        self.lbl_project_info = QLabel("尚未載入專案")
        self.lbl_project_info.setStyleSheet("color: #ddd; font-size: 14px;")
        
        self.btn_refresh = QPushButton("🔄 重新整理")
        self.btn_refresh.setStyleSheet("QPushButton { background-color: #0277bd; color: white; font-weight: bold; padding: 6px 15px; border-radius: 5px; font-size: 14px; } QPushButton:hover { background-color: #0288d1; }")
        self.btn_refresh.clicked.connect(self.refresh_ui)

        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_project_info)
        top_bar.addWidget(self.btn_refresh)
        main_layout.addWidget(top_bar_container)

        # 中間區域
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

        # 右側顯示
        right_container = QFrame()
        right_container.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444; border-radius: 8px;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)

        self.image_display = QLabel("準備就緒")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setStyleSheet("background-color: transparent; color: #666; font-size: 16px;")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        # 底部按鈕
        btn_bar = QFrame()
        btn_bar.setStyleSheet("QFrame { background-color: #333; border-top: 1px solid #555; border-radius: 0px; }")
        btn_bar.setMaximumHeight(80)
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(20, 10, 20, 10)
        btn_layout.setSpacing(30)

        btn_style = "QPushButton { color: white; font-weight: bold; border-radius: 8px; font-size: 18px; padding: 10px; }"
        
        self.btn_ng = QPushButton("❌ NG (←)")
        self.btn_ng.setMinimumHeight(50)
        self.btn_ng.setStyleSheet(btn_style + "QPushButton { background-color: #e57373; } QPushButton:hover { background-color: #ef5350; }")
        self.btn_ng.clicked.connect(lambda: self.classify_image("NG"))
        
        self.btn_ok = QPushButton("⭕ OK (→)")
        self.btn_ok.setMinimumHeight(50)
        self.btn_ok.setStyleSheet(btn_style + "QPushButton { background-color: #81c784; color: #1b5e20; } QPushButton:hover { background-color: #66bb6a; }")
        self.btn_ok.clicked.connect(lambda: self.classify_image("OK"))

        btn_layout.addWidget(self.btn_ng)
        btn_layout.addWidget(self.btn_ok)
        
        right_layout.addWidget(self.image_display, 1)
        right_layout.addWidget(btn_bar)

        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # 邏輯處理
    def refresh_ui(self):
        if self.icon_worker and self.icon_worker.isRunning():
            self.icon_worker.stop()

        self.list_widget.clear()
        if not self.data_handler.project_path: return

        images = self.data_handler.scan_roi_images()
        project_name = os.path.basename(self.data_handler.project_path)
        self.lbl_project_info.setText(f"專案: {project_name} | 待分類: {len(images)} 張")

        for path in images:
            filename = os.path.basename(path)
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
        
        if images:
            self.icon_worker = IconWorker(images)
            self.icon_worker.icon_loaded.connect(self.on_icon_loaded)
            self.icon_worker.start()
            
            self.list_widget.setCurrentRow(0)
            self.btn_ok.setEnabled(True)
            self.btn_ng.setEnabled(True)
        else:
            self.image_display.clear()
            self.image_display.setText("🎉 此專案已全部分類完成！")
            self.current_selected_path = None
            self.btn_ok.setEnabled(False)
            self.btn_ng.setEnabled(False)

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
            scaled = pixmap.scaled(self.image_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_display.setPixmap(scaled)

    def classify_image(self, label):
        if not self.current_selected_path: return
        if self.data_handler.move_roi_file_to_result(self.current_selected_path, label):
            row = self.list_widget.currentRow()
            self.list_widget.takeItem(row)
            if self.list_widget.count() > 0:
                if row >= self.list_widget.count(): row = self.list_widget.count() - 1
                self.list_widget.setCurrentRow(row)
                self.lbl_project_info.setText(f"待分類: {self.list_widget.count()} 張")
            else:
                self.refresh_ui()

    # 鍵盤控制
    def keyPressEvent(self, event):
        if not self.current_selected_path:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Left:
            if self.btn_ng.isEnabled(): self.classify_image("NG")
        elif event.key() == Qt.Key_Right:
            if self.btn_ok.isEnabled(): self.classify_image("OK")
        else:
            super().keyPressEvent(event)
            
    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)