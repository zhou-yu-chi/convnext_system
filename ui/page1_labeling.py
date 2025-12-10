import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class Page1_Labeling(QWidget):
    def __init__(self, data_handler, main_window):
        super().__init__()
        self.data_handler = data_handler
        self.current_roi_path = None # 紀錄目前顯示的是哪張圖
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        top_bar = QHBoxLayout()
        self.lbl_info = QLabel("專案資訊")
        self.lbl_info.setStyleSheet("color: #4db6ac; font-weight: bold; font-size: 16px;")
        top_bar.addWidget(self.lbl_info)
        layout.addLayout(top_bar)

        self.image_display = QLabel("準備就緒")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setObjectName("ImageArea") # 沿用 Main 的 CSS
        self.image_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.image_display, 1)

        btn_layout = QHBoxLayout()
        self.btn_ng = QPushButton("❌ NG (不良品)")
        self.btn_ng.setMinimumHeight(60)
        self.btn_ng.setStyleSheet("background-color: #e57373; font-weight: bold; font-size: 18px;")
        self.btn_ng.clicked.connect(lambda: self.classify_image("NG"))
        
        self.btn_ok = QPushButton("⭕ OK (良品)")
        self.btn_ok.setMinimumHeight(60)
        self.btn_ok.setStyleSheet("background-color: #81c784; color: #1b5e20; font-weight: bold; font-size: 18px;")
        self.btn_ok.clicked.connect(lambda: self.classify_image("OK"))

        btn_layout.addWidget(self.btn_ng)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def refresh_ui(self):
        """讀取 ROI 資料夾的狀態"""
        if not self.data_handler.project_path: return

        # 1. 重新掃描 ROI 資料夾
        images = self.data_handler.scan_roi_images()
        count = len(images)
        
        self.lbl_info.setText(f"待分類 (ROI): {count} 張")

        if count > 0:
            # 2. 永遠取第一張 (Queue 模式)
            self.current_roi_path = images[0] 
            self.show_image(self.current_roi_path)
            self.btn_ok.setEnabled(True)
            self.btn_ng.setEnabled(True)
        else:
            self.current_roi_path = None
            self.image_display.clear()
            self.image_display.setText("🎉 ROI 資料夾已清空\n請回到 Page 0 裁切更多照片")
            self.btn_ok.setEnabled(False)
            self.btn_ng.setEnabled(False)

    def show_image(self, path):
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(self.image_display.size(), 
                                 Qt.AspectRatioMode.KeepAspectRatio, 
                                 Qt.TransformationMode.SmoothTransformation)
            self.image_display.setPixmap(scaled)

    def classify_image(self, label):
        """將目前這張 ROI 圖片移到 OK 或 NG"""
        if not self.current_roi_path: return
        
        # 呼叫 DataHandler 移動檔案
        if self.data_handler.move_roi_file_to_result(self.current_roi_path, label):
            # 成功移動後，檔案消失了，重新整理畫面 (會自動載入遞補上來的下一張)
            self.refresh_ui()

    def resizeEvent(self, event):
        if self.current_roi_path:
            self.show_image(self.current_roi_path)
        super().resizeEvent(event)

    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)