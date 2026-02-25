import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QComboBox, QMessageBox, QFrame, QApplication, QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap

# 沿用你原本的安全設備檢測邏輯
def get_safe_device():
    if not torch.cuda.is_available(): return torch.device('cpu')
    try:
        major, minor = torch.cuda.get_device_capability(0)
        capability_score = major + minor / 10.0
        if capability_score < 3.7: return torch.device('cpu')
        test_conv = nn.Conv2d(1, 1, kernel_size=1).to('cuda')
        test_input = torch.randn(1, 1, 32, 32).to('cuda')
        _ = test_conv(test_input)
        return torch.device('cuda')
    except:
        return torch.device('cpu')

# ==========================================
# 負責在背景「讀取相機 + 執行推論」的執行緒
# ==========================================
class InferenceThread(QThread):
    update_frame = Signal(QImage)
    update_result = Signal(str, float, float) # 標籤, 信心度, 推論耗時(ms)
    log_signal = Signal(str)

    def __init__(self, camera_index, model_path, strict_threshold):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.strict_threshold = strict_threshold
        self.is_running = True
        self.device = get_safe_device()
        self.cap = None

    def run(self):
        try:
            self.log_signal.emit(f"🚀 正在載入模型至 {self.device}...")
            
            # --- 1. 動態重建模型架構 ---
            model = models.convnext_tiny(weights=None)
            num_ftrs = model.classifier[2].in_features
            
            # 讀取 .pth 檔案
            loaded_data = torch.load(self.model_path, map_location=self.device)
            
            # ★★★ 關鍵解包邏輯：判斷這是不是一個「打包過」的字典 ★★★
            if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
                state_dict = loaded_data["model_state_dict"] # 提取真正的權重
                self.log_signal.emit("ℹ️ 偵測到打包模型，已提取 model_state_dict")
            else:
                state_dict = loaded_data # 如果是純權重，就照舊
            
            # 檢查新舊版結構 (Dropout)
            if any("classifier.2.1" in k for k in state_dict.keys()):
                self.log_signal.emit("ℹ️ 偵測到新版模型結構 (含 Dropout)")
                model.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
            else:
                self.log_signal.emit("ℹ️ 偵測到舊版模型結構 (不含 Dropout)")
                model.classifier[2] = nn.Linear(num_ftrs, 2)

            # 載入真正的權重
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.log_signal.emit("✅ 模型載入完成！正在開啟相機...")

            # --- 2. 影像前處理 ---
            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            classes = ['NG', 'OK']

            # --- 3. 開啟相機 ---
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.log_signal.emit("❌ 相機開啟失敗！請確認相機未被佔用。")
                return

            self.log_signal.emit("🟢 即時推論已啟動！")

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret: continue

                start_time = cv2.getTickCount()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- PyTorch 推論 ---
                pil_img = Image.fromarray(rgb_frame)
                input_tensor = val_transforms(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                pred_idx = preds.item()
                confidence = probs[0][pred_idx].item()
                pred_label = classes[pred_idx]

                # 嚴格模式：OK 信心不足強制轉 NG
                if pred_label == 'OK' and confidence < self.strict_threshold:
                    pred_label = 'NG (嚴格模式)'

                # 計算耗時
                end_time = cv2.getTickCount()
                infer_time_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000

                # 發送結果更新 UI
                self.update_result.emit(pred_label, confidence, infer_time_ms)

                # 發送影像給 UI
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                self.update_frame.emit(qt_image)

        except Exception as e:
            self.log_signal.emit(f"❌ 發生錯誤: {str(e)}")
        finally:
            if self.cap: self.cap.release()
            self.log_signal.emit("🔴 即時推論已停止。")

    def stop(self):
        self.is_running = False
        self.wait(2000)

# ==========================================
# 即時推論主頁面 UI
# ==========================================
from PySide6.QtWidgets import QFileDialog

class Page7_RealtimeInference(QWidget):
    def __init__(self):
        super().__init__()
        self.inference_thread = None
        self.model_path = ""
        self.models_root = ""
        self.init_ui()

    def set_models_root(self, path):
        self.models_root = path

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # --- 1. 頂部控制列 ---
        top_panel = QFrame()
        top_panel.setStyleSheet("background-color: #333; border-radius: 8px; padding: 10px;")
        top_layout = QHBoxLayout(top_panel)

        # 載入模型按鈕
        self.btn_load_model = QPushButton("🧠 載入模型 (.pth)")
        self.btn_load_model.setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold; padding: 8px 15px; border-radius: 5px;")
        self.btn_load_model.clicked.connect(self.on_load_model)
        
        self.lbl_model_name = QLabel("尚未載入模型")
        self.lbl_model_name.setStyleSheet("color: #aaa; margin-right: 20px;")

        # 相機選擇
        lbl_cam = QLabel("📷 相機:")
        self.combo_cam = QComboBox()
        self.combo_cam.addItem("點擊掃描...")
        self.combo_cam.setStyleSheet("QComboBox { background-color: #555; color: white; padding: 5px; border-radius: 4px; }")
        
        self.btn_refresh_cam = QPushButton("🔄 掃描")
        self.btn_refresh_cam.setStyleSheet("background-color: #555; color: white; padding: 5px 10px; border-radius: 5px;")
        self.btn_refresh_cam.clicked.connect(self.scan_available_cameras)

        # 嚴格模式門檻
        lbl_strict = QLabel("🛡️ 嚴格門檻(OK):")
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.70)
        self.spin_threshold.setStyleSheet("QDoubleSpinBox { background-color: #555; color: white; padding: 5px; border-radius: 3px; }")

        # 啟動推論按鈕
        self.btn_toggle_infer = QPushButton("▶️ 開始即時推論")
        self.btn_toggle_infer.setStyleSheet("background-color: #388e3c; color: white; padding: 8px 20px; font-weight: bold; border-radius: 5px; font-size: 14px;")
        self.btn_toggle_infer.setCheckable(True)
        self.btn_toggle_infer.setEnabled(False) # 沒載入模型前不給按
        self.btn_toggle_infer.clicked.connect(self.toggle_inference)

        top_layout.addWidget(self.btn_load_model)
        top_layout.addWidget(self.lbl_model_name)
        top_layout.addWidget(lbl_cam)
        top_layout.addWidget(self.combo_cam)
        top_layout.addWidget(self.btn_refresh_cam)
        top_layout.addSpacing(20)
        top_layout.addWidget(lbl_strict)
        top_layout.addWidget(self.spin_threshold)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_toggle_infer)

        layout.addWidget(top_panel)

        # --- 2. 中間畫面區 ---
        middle_layout = QHBoxLayout()
        
        # 影像顯示區
        self.lbl_video = QLabel("等待啟動相機...")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #1a1a1a; border: 2px dashed #555; border-radius: 8px; font-size: 18px; color: #888;")
        self.lbl_video.setMinimumSize(640, 480)
        
        # 右側狀態區
        status_panel = QFrame()
        status_panel.setFixedWidth(250)
        status_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; border: 1px solid #444;")
        status_layout = QVBoxLayout(status_panel)

        lbl_status_title = QLabel("即時判定結果")
        lbl_status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #aaa; border: none;")

        self.lbl_result = QLabel("--")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 60px; font-weight: bold; color: #555; border: none; margin: 20px 0px;")

        self.lbl_conf = QLabel("信心度: --%")
        self.lbl_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_conf.setStyleSheet("font-size: 16px; color: #cfcfcf; border: none;")

        self.lbl_fps = QLabel("推論延遲: -- ms")
        self.lbl_fps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_fps.setStyleSheet("font-size: 14px; color: #888; border: none; margin-top: 10px;")

        status_layout.addWidget(lbl_status_title)
        status_layout.addWidget(self.lbl_result)
        status_layout.addWidget(self.lbl_conf)
        status_layout.addWidget(self.lbl_fps)
        status_layout.addStretch()

        middle_layout.addWidget(self.lbl_video, 1)
        middle_layout.addWidget(status_panel)
        
        layout.addLayout(middle_layout, 1)
        self.setLayout(layout)

        # 啟動時掃描相機
        QTimer.singleShot(100, self.scan_available_cameras)

    def on_load_model(self):
        start_path = self.models_root if self.models_root else ""
        
        path, _ = QFileDialog.getOpenFileName(self, "選擇要部署的模型", start_path, "PyTorch Model (*.pth)")
        
        if path:
            self.model_path = path
            self.lbl_model_name.setText(os.path.basename(path))
            self.btn_toggle_infer.setEnabled(True)
            self.lbl_model_name.setStyleSheet("color: #4db6ac; font-weight: bold; margin-right: 20px;")

    def scan_available_cameras(self):
        self.combo_cam.clear()
        self.combo_cam.addItem("掃描中...")
        QApplication.processEvents()

        available_cams = []
        for i in range(4):
            # ★★★ 修正：掃描也要加 CAP_DSHOW
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                available_cams.append(i)
                cap.release()

        self.combo_cam.clear()
        if not available_cams:
            self.combo_cam.addItem("❌ 找不到相機")
        else:
            for cam_id in available_cams:
                self.combo_cam.addItem(f"相機 {cam_id}", cam_id)

    def toggle_inference(self):
        is_pressed = self.btn_toggle_infer.isChecked()
        
        if is_pressed:
            if self.combo_cam.count() == 0 or "找" in self.combo_cam.currentText():
                QMessageBox.warning(self, "警告", "請先確認相機已連接！")
                self.btn_toggle_infer.setChecked(False)
                return

            self.btn_toggle_infer.setText("⏹️ 停止推論")
            self.btn_toggle_infer.setStyleSheet("background-color: #d32f2f; color: white; padding: 8px 20px; font-weight: bold; border-radius: 5px; font-size: 14px;")
            self.btn_load_model.setEnabled(False)
            self.spin_threshold.setEnabled(False)
            
            cam_idx = self.combo_cam.currentData()
            thresh = self.spin_threshold.value()

            self.inference_thread = InferenceThread(cam_idx, self.model_path, thresh)
            self.inference_thread.update_frame.connect(self.update_video)
            self.inference_thread.update_result.connect(self.update_ui_result)
            
            # ★★★ 修正：接上 log_signal，不然出錯了會完全沒畫面也沒報錯！
            self.inference_thread.log_signal.connect(lambda msg: print(f"[Page7 Log] {msg}"))
            
            self.inference_thread.start()
        else:
            self.stop_inference()

    def stop_inference(self):
        # ★★★ 修正 4：完整釋放執行緒與強制按鈕狀態回歸 ★★★
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None # 徹底清空，避免殘留
            
        self.btn_toggle_infer.setChecked(False) # 強制將按鈕設為「未點擊」狀態
        self.btn_toggle_infer.setText("▶️ 開始即時推論")
        self.btn_toggle_infer.setStyleSheet("background-color: #388e3c; color: white; padding: 8px 20px; font-weight: bold; border-radius: 5px; font-size: 14px;")
        
        self.btn_load_model.setEnabled(True)
        self.spin_threshold.setEnabled(True)
        
        self.lbl_video.clear()
        self.lbl_video.setText("推論已暫停")
        self.lbl_result.setText("--")
        self.lbl_result.setStyleSheet("font-size: 60px; font-weight: bold; color: #555; border: none; margin: 20px 0px;")
        self.lbl_conf.setText("信心度: --%")
        self.lbl_fps.setText("推論延遲: -- ms")

    def update_video(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.lbl_video.width(), self.lbl_video.height(), 
                               Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_video.setPixmap(scaled)

    def update_ui_result(self, label, conf, infer_time):
        self.lbl_result.setText(label)
        self.lbl_conf.setText(f"信心度: {conf:.1%}")
        self.lbl_fps.setText(f"推論延遲: {infer_time:.1f} ms")

        # 改變顏色提示
        if "NG" in label:
            self.lbl_result.setStyleSheet("font-size: 50px; font-weight: bold; color: #e57373; border: none; margin: 20px 0px;")
        else:
            self.lbl_result.setStyleSheet("font-size: 60px; font-weight: bold; color: #81c784; border: none; margin: 20px 0px;")

    def hideEvent(self, event):
        """切換到其他頁面時，自動關閉推論，釋放相機與 GPU 資源"""
        if self.btn_toggle_infer.isChecked():
            self.btn_toggle_infer.setChecked(False)
            self.stop_inference()
        super().hideEvent(event)