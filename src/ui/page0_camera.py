import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2
import numpy as np
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QComboBox, QMessageBox, QSpinBox, QFrame, QApplication)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap

# ==========================================
# 負責在背景讀取 USB 相機畫面的執行緒
# ==========================================
class CameraThread(QThread):
    update_frame = Signal(QImage)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = True
        self.cap = None
        self.current_frame = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy() # 儲存原始 BGR 畫面供拍照使用
                
                # 將 BGR 轉換為 RGB 供 PySide6 顯示
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                
                self.update_frame.emit(qt_image)
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self.is_running = False
        self.wait()

    def get_current_frame(self):
        return self.current_frame

# ==========================================
# 相機主頁面
# ==========================================
class Page_Camera(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.camera_thread = None
        self.is_continuous_shooting = False
        self.init_ui()

        self.continuous_timer = QTimer(self)
        self.continuous_timer.timeout.connect(self.take_single_photo)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. 頂部工具列 (選擇相機與開關)
        top_bar = QHBoxLayout()
        
        # 啟動 / 關閉相機按鈕
        self.btn_toggle_cam = QPushButton("啟動相機")
        self.btn_toggle_cam.setStyleSheet("background-color: #388e3c; color: white; padding: 5px 15px; font-weight: bold; border-radius: 5px;")
        self.btn_toggle_cam.setCheckable(True)
        self.btn_toggle_cam.clicked.connect(self.toggle_camera_state)

        lbl_cam = QLabel("📷 可用相機:")
        
        self.combo_cam = QComboBox()
        self.combo_cam.addItem("請先點擊掃描...")
        # ★★★ 新增：修改下拉選單的點擊反白顏色，去除刺眼的藍色 ★★★
        self.combo_cam.setStyleSheet("""
            QComboBox { 
                background-color: #444; 
                color: white; 
                padding: 5px; 
                border-radius: 4px; 
            }
            QComboBox QAbstractItemView { 
                background-color: #444; 
                color: white; 
                selection-background-color: #555; /* 將原本預設的藍色改成低調的深灰色 */
            }
        """)
        self.combo_cam.currentIndexChanged.connect(self.on_combo_changed)
        
        self.btn_refresh_cam = QPushButton("🔄 掃描可用相機")
        self.btn_refresh_cam.setStyleSheet("background-color: #555; color: white; padding: 5px 10px; border-radius: 5px;")
        self.btn_refresh_cam.clicked.connect(self.scan_available_cameras)

        top_bar.addWidget(self.btn_toggle_cam)
        top_bar.addSpacing(20)
        top_bar.addWidget(lbl_cam)
        top_bar.addWidget(self.combo_cam)
        top_bar.addWidget(self.btn_refresh_cam)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # 2. 中間相機畫面顯示區
        self.lbl_video = QLabel("請點擊左上角「啟動相機」")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #1a1a1a; border: 2px dashed #555; border-radius: 8px; font-size: 18px; color: #888;")
        self.lbl_video.setMinimumHeight(480)
        layout.addWidget(self.lbl_video, 1)

        # 3. 底部拍照控制列
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("background-color: #333; border-radius: 8px; padding: 5px;")
        bottom_layout = QHBoxLayout(bottom_bar)

        self.btn_single_shot = QPushButton("📸 單張拍照")
        self.btn_single_shot.setStyleSheet("background-color: #0288d1; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_single_shot.clicked.connect(self.take_single_photo)
        self.btn_single_shot.setEnabled(False) # 預設停用，相機開啟才啟用

        lbl_interval = QLabel("連續拍照間隔 (秒):")
        self.combo_interval = QComboBox()
        # 提供常用的秒數選項
        self.combo_interval.addItems(["1", "2", "3", "5", "10", "15", "30", "60"])
        self.combo_interval.setCurrentText("2") # 預設選擇 2 秒
        self.combo_interval.setStyleSheet("""
            QComboBox { 
                background-color: #444; 
                color: white; 
                padding: 5px; 
                border-radius: 4px; 
                min-width: 60px;
            }
            QComboBox QAbstractItemView { 
                background-color: #444; 
                color: white; 
                selection-background-color: #555;
            }
        """)

        self.btn_continuous_shot = QPushButton("▶️ 開始連續拍照")
        self.btn_continuous_shot.setStyleSheet("background-color: #388e3c; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_continuous_shot.setCheckable(True)
        self.btn_continuous_shot.clicked.connect(self.toggle_continuous_shooting)
        self.btn_continuous_shot.setEnabled(False) # 預設停用

        bottom_layout.addWidget(self.btn_single_shot)
        bottom_layout.addStretch()
        bottom_layout.addWidget(lbl_interval)
        bottom_layout.addWidget(self.combo_interval)
        bottom_layout.addWidget(self.btn_continuous_shot)
        layout.addWidget(bottom_bar)

        self.setLayout(layout)
        QTimer.singleShot(100, self.scan_available_cameras)


    # ==========================================
    # ★★★ 新增：離開此分頁時自動觸發隱藏事件 ★★★
    # ==========================================
    def hideEvent(self, event):
        """當使用者切換到其他 Tab 或隱藏此頁面時會自動執行"""
        if self.btn_toggle_cam.isChecked():
            # 取消按鈕選取狀態
            self.btn_toggle_cam.setChecked(False)
            # 呼叫我們原本寫好的切換狀態函式來停止相機
            self.toggle_camera_state()
        super().hideEvent(event)

    # ==========================================
    # ★ 動態掃描相機核心邏輯
    # ==========================================
    def scan_available_cameras(self):
        """掃描系統中可用的相機編號 (OpenCV 會實際去開開看)"""
        self.combo_cam.clear()
        self.combo_cam.addItem("掃描中...")
        self.combo_cam.setEnabled(False)
        self.btn_refresh_cam.setEnabled(False)
        QApplication.processEvents() # 強制刷新 UI 顯示「掃描中」

        available_cams = []
        # 通常掃描 0~3 就夠了，掃太多次會讓程式卡頓很久
        for i in range(4):
            # 嘗試開啟相機 (Windows 建議加上 cv2.CAP_DSHOW 加快掃描速度)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                available_cams.append(i)
                cap.release()

        self.combo_cam.clear()
        if not available_cams:
            self.combo_cam.addItem("❌ 找不到相機")
        else:
            for cam_id in available_cams:
                self.combo_cam.addItem(f"相機 {cam_id}", cam_id) # 顯示文字, 夾帶真實 ID

        self.combo_cam.setEnabled(True)
        self.btn_refresh_cam.setEnabled(True)

    # ==========================================
    # ★ 啟動/關閉相機控制
    # ==========================================
    def toggle_camera_state(self):
        is_pressed = self.btn_toggle_cam.isChecked()
        
        if is_pressed:
            # 準備啟動相機
            if self.combo_cam.count() == 0 or "找" in self.combo_cam.currentText():
                QMessageBox.warning(self, "警告", "目前沒有偵測到可用的相機！")
                self.btn_toggle_cam.setChecked(False)
                return
                
            self.btn_toggle_cam.setText("關閉相機")
            self.btn_toggle_cam.setStyleSheet("background-color: #d32f2f; color: white; padding: 5px 15px; font-weight: bold; border-radius: 5px;")
            self.btn_single_shot.setEnabled(True)
            self.btn_continuous_shot.setEnabled(True)
            self.start_camera()
        else:
            # 關閉相機
            self.btn_toggle_cam.setText("啟動相機")
            self.btn_toggle_cam.setStyleSheet("background-color: #388e3c; color: white; padding: 5px 15px; font-weight: bold; border-radius: 5px;")
            self.btn_single_shot.setEnabled(False)
            self.btn_continuous_shot.setEnabled(False)
            # 如果正在連續拍照，也一併關閉
            if self.is_continuous_shooting:
                self.btn_continuous_shot.setChecked(False)
                self.toggle_continuous_shooting()
            self.stop_camera()

    def on_combo_changed(self):
        # 如果相機正在運作中，切換下拉選單就立刻重啟對應的相機
        if self.btn_toggle_cam.isChecked():
            self.start_camera()

    def start_camera(self):
        """實際啟動相機執行緒"""
        self.stop_camera() # 先確保舊的已關閉
        
        # 取得下拉選單裡夾帶的真實 ID (Data)
        cam_idx = self.combo_cam.currentData() 
        if cam_idx is None: return

        self.camera_thread = CameraThread(camera_index=cam_idx)
        self.camera_thread.update_frame.connect(self.update_image)
        self.camera_thread.start()

    def stop_camera(self):
        """停止相機執行緒"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.lbl_video.clear()
            self.lbl_video.setText("相機已暫停 / 關閉")

    def update_image(self, qt_image):
        """即時更新畫面上 QLabel 的影像"""
        pixmap = QPixmap.fromImage(qt_image)
        # 保持比例縮放以符合畫面大小
        scaled_pixmap = pixmap.scaled(self.lbl_video.width(), self.lbl_video.height(), 
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)

    def take_single_photo(self):
        """執行拍照並存檔"""
        if not self.data_handler.project_path:
            QMessageBox.warning(self, "錯誤", "請先建立或開啟專案後再拍照！")
            if self.btn_continuous_shot.isChecked():
                self.btn_continuous_shot.setChecked(False)
                self.toggle_continuous_shooting()
            return

        if not self.camera_thread or not self.camera_thread.get_current_frame() is not None:
            QMessageBox.warning(self, "錯誤", "無法取得相機畫面！")
            return

        frame = self.camera_thread.get_current_frame()
        
        # 產生不重複檔名：以時間戳為基礎
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"Capture_{timestamp}.jpg"
        
        # 呼叫 DataHandler 取得保證不重複的路徑
        save_path = self.data_handler.generate_unique_path(self.data_handler.project_path, base_filename)
        
        try:
            # 使用 cv2.imencode 來處理可能含有中文路徑的存檔問題
            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            if is_success:
                im_buf_arr.tofile(save_path)
                
                # 通知 DataHandler 重整清單
                self.data_handler.scan_unsorted_images()
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存照片失敗: {str(e)}")

    def toggle_continuous_shooting(self):
        """切換連續拍照狀態"""
        self.is_continuous_shooting = self.btn_continuous_shot.isChecked()
        
        if self.is_continuous_shooting:
            self.btn_continuous_shot.setText("⏹️ 停止連續拍照")
            self.btn_continuous_shot.setStyleSheet("background-color: #d32f2f; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            self.btn_single_shot.setEnabled(False)
            
            
            interval_ms = int(self.combo_interval.currentText()) * 1000
            self.continuous_timer.start(interval_ms)
            
            # 立刻拍第一張
            self.take_single_photo()
        else:
            self.btn_continuous_shot.setText("▶️ 開始連續拍照")
            self.btn_continuous_shot.setStyleSheet("background-color: #388e3c; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            self.btn_single_shot.setEnabled(True)
            self.continuous_timer.stop()