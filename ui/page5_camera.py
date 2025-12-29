import cv2
import sys
import datetime # 記得補上這個，存檔命名需要
from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QMessageBox, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap

# ==========================================
# 1. 相機擷取執行緒 (優化版)
# ==========================================
class CameraWorker(QThread):
    frame_received = Signal(QImage) # 傳送畫面給 UI
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = True
        self.cap = None

    def run(self):
        try:
            # 開啟相機 (增加 cv2.CAP_DSHOW 在 Windows 上通常開起速度較快)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                # 如果 DSHOW 失敗，嘗試預設模式
                self.cap = cv2.VideoCapture(self.camera_index)
            
            # 設定解析度 (設為 HD 就夠了，太高會卡)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # OpenCV 是 BGR，要轉成 RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    
                    # 轉換成 QImage
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    
                    # 發送訊號 (copy 一份避免記憶體指針錯誤)
                    self.frame_received.emit(qt_image.copy())
                else:
                    # 如果讀不到畫面，稍微休息避免死迴圈
                    QThread.msleep(100)
                
                # 控制 FPS，30ms 大約是 30 FPS，這讓 UI 有時間喘息
                QThread.msleep(30)

        except Exception as e:
            print(f"相機錯誤: {e}")
        finally:
            if self.cap:
                self.cap.release()

    def stop(self):
        self.is_running = False
        self.wait()

# ==========================================
# 2. 頁面五：相機拍攝 UI (修復無限放大與卡頓)
# ==========================================
class Page5_Camera(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.camera_worker = None
        self.current_frame = None 
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- 頂部工具列 ---
        top_bar_container = QFrame()
        top_bar_container.setMaximumHeight(65)
        top_bar_container.setStyleSheet("QFrame { background-color: #333; border-radius: 8px; padding: 2px; }")
        top_bar = QHBoxLayout(top_bar_container)
        top_bar.setContentsMargins(15, 5, 15, 5)

        lbl_title = QLabel("📷 拍攝照片")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4db6ac;")

        lbl_select = QLabel("選擇鏡頭:")
        lbl_select.setStyleSheet("color: #ddd; font-size: 14px; margin-left: 10px;")

        self.combo_camera = QComboBox()
        self.combo_camera.setStyleSheet("""
            QComboBox { background-color: #555; color: white; padding: 5px; border-radius: 4px; min-width: 150px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #555; color: white; selection-background-color: #00796b; }
        """)
        self.combo_camera.currentIndexChanged.connect(self.start_camera_stream)

        self.btn_scan = QPushButton("🔄 掃描鏡頭")
        self.btn_scan.setStyleSheet("QPushButton { background-color: #0277bd; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #0288d1; }")
        self.btn_scan.clicked.connect(self.scan_cameras)

        top_bar.addWidget(lbl_title)
        top_bar.addWidget(lbl_select)
        top_bar.addWidget(self.combo_camera)
        top_bar.addWidget(self.btn_scan)
        top_bar.addStretch()

        main_layout.addWidget(top_bar_container)

        # --- 中間：影像顯示區 ---
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #1a1a1a; border: 2px solid #444; border-radius: 8px;")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_display = QLabel("等待相機啟動...")
        self.lbl_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_display.setStyleSheet("color: #666; font-size: 16px; background-color: transparent;")
        
        # ★★★ 關鍵設定 1：設定 SizePolicy 為 Ignored ★★★
        # 這告訴 Layout：「不要管圖片多大，你該多大就多大，圖片會自己縮放」
        # 這能有效防止圖片撐大視窗
        self.lbl_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        preview_layout.addWidget(self.lbl_display)
        main_layout.addWidget(preview_frame, 1)

        # --- 底部：拍照按鈕 ---
        btn_bar = QFrame()
        btn_bar.setStyleSheet("background-color: #333; border-radius: 8px;")
        btn_bar.setMaximumHeight(80)
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(20, 10, 20, 10)

        self.btn_capture = QPushButton("📸 拍照存檔 (Space)")
        self.btn_capture.setMinimumHeight(50)
        self.btn_capture.setStyleSheet("""
            QPushButton { background-color: #ef6c00; color: white; font-weight: bold; border-radius: 8px; font-size: 20px; }
            QPushButton:hover { background-color: #f57c00; }
            QPushButton:pressed { background-color: #e65100; }
        """)
        self.btn_capture.clicked.connect(self.take_photo)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_capture, 2)
        btn_layout.addStretch()

        main_layout.addWidget(btn_bar)
        self.setLayout(main_layout)

    # --- 邏輯處理 ---

    def showEvent(self, event):
        """當頁面顯示時，自動重啟相機"""
        # 如果選單是空的，執行完整掃描
        if self.combo_camera.count() == 0:
            self.scan_cameras()
        else:
            # 如果已經有選項，直接重新啟動目前選中的鏡頭
            # 這樣切換回來時就會自動有畫面，不用再按掃描
            self.start_camera_stream(self.combo_camera.currentIndex())
            
        super().showEvent(event)

    def hideEvent(self, event):
        self.stop_worker()
        super().hideEvent(event)

    def scan_cameras(self):
        self.stop_worker()
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        
        available_cams = []
        for i in range(3): # 掃描前3個就好，掃太多會卡
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cams.append(f"Camera {i}")
                cap.release()
        
        if available_cams:
            self.combo_camera.addItems(available_cams)
            self.lbl_display.setText("相機已就緒，請選擇鏡頭")
        else:
            self.lbl_display.setText("⚠️ 未偵測到任何 USB 相機")
            self.combo_camera.addItem("無可用相機")

        self.combo_camera.blockSignals(False)
        
        if available_cams:
            self.start_camera_stream(0)

    def start_camera_stream(self, index):
        self.stop_worker()
        cam_idx = self.combo_camera.currentIndex()
        if cam_idx < 0: return

        self.lbl_display.setText("正在啟動相機...")
        self.camera_worker = CameraWorker(camera_index=cam_idx)
        self.camera_worker.frame_received.connect(self.update_image)
        self.camera_worker.start()

    def stop_worker(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker = None

    def update_image(self, q_img):
        """接收 Thread 傳來的畫面並顯示"""
        self.current_frame = q_img 
        
        # 轉成 Pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # ★★★ 關鍵修改 2：使用 FastTransformation ★★★
        # SmoothTransformation 在動態影片中非常吃資源，改成 FastTransformation 會順暢非常多
        # 視覺上在預覽時差別不大
        scaled_pixmap = pixmap.scaled(
            self.lbl_display.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.FastTransformation 
        )
        self.lbl_display.setPixmap(scaled_pixmap)

    # ★★★ 關鍵修改 3：完全移除 resizeEvent ★★★
    # 這裡原本有的 resizeEvent 函式被刪除了。
    # 因為 Video Stream 每秒更新 30 次，視窗變大時，下一幀畫面進來就會自動填滿。
    # 不需要手動在 resizeEvent 裡更新，那樣會造成無限迴圈。

    def take_photo(self):
        if not self.data_handler.project_path:
            QMessageBox.warning(self, "錯誤", "尚未開啟專案，無法存檔！")
            return

        if self.current_frame is None:
            QMessageBox.warning(self, "錯誤", "目前沒有相機畫面！")
            return

        try:
            # 這裡的 current_frame 依然是高畫質原圖，不受預覽縮放影響，所以存檔畫質會很好
            img_pil = Image.fromqimage(self.current_frame)
            success, msg = self.data_handler.save_camera_photo(img_pil)
            
            if success:
                self.lbl_display.setText(f"✅ 已儲存: {msg}")
                # 閃一下恢復
                QTimer.singleShot(800, lambda: None) 
            else:
                QMessageBox.critical(self, "失敗", f"存檔失敗: {msg}")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理照片時發生錯誤: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.btn_capture.click()
        else:
            super().keyPressEvent(event)