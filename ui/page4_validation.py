import os
import datetime
import shutil  # <--- 1. 新增：引入 shutil 用來複製檔案
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox, QProgressBar, 
                             QTextEdit, QFrame, QDoubleSpinBox, QCheckBox, QGroupBox) # <--- 新增 QDoubleSpinBox, QCheckBox, QGroupBox
from PySide6.QtCore import Qt, QThread, Signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# ==========================================
# 1. 後台驗證執行緒 (修改了邏輯)
# ==========================================
def get_safe_device():
    # 1. 基礎檢查：是否有顯卡
    if not torch.cuda.is_available():
        print("ℹ️ 未偵測到 GPU，使用 CPU")
        return torch.device('cpu')

    try:
        # 2. ★★★ 進階檢查：算力版本 (Compute Capability) ★★★
        # 您的錯誤訊息明確指出：PyTorch 需要至少 3.7，但 K4000 只有 3.0
        major, minor = torch.cuda.get_device_capability(0)
        capability_score = major + minor / 10.0
        
        print(f"🔎 偵測到 GPU 算力版本: {capability_score} (Major: {major}, Minor: {minor})")
        
        # 設定最低門檻 (根據您的報錯，設為 3.7)
        if capability_score < 3.7:
            print(f"⚠️ GPU 算力過低 ({capability_score} < 3.7)。PyTorch 已不支援此顯卡。")
            print("🔄 強制切換回 CPU 模式，以避免崩潰。")
            return torch.device('cpu')

        # 3. ★★★ 實戰測試：執行一次卷積運算 (觸發 cuDNN) ★★★
        # 之前的 torch.zeros 只是搬運，這裡我們要真的「算」一次
        # 如果 cuDNN 不支援，這行就會直接報錯，被 except 抓到
        test_conv = nn.Conv2d(1, 1, kernel_size=1).to('cuda')
        test_input = torch.randn(1, 1, 32, 32).to('cuda')
        _ = test_conv(test_input)
        
        print(f"✅ GPU ({torch.cuda.get_device_name(0)}) 檢測與運算測試通過，將使用 CUDA 加速")
        return torch.device('cuda')

    except Exception as e:
        print(f"⚠️ GPU 存在但無法通過運算測試 (可能是驅動或架構問題): {e}")
        print("🔄 自動切換回 CPU 模式")
        return torch.device('cpu')
    
class VerificationWorker(QThread):
    progress_signal = Signal(int, int)
    log_signal = Signal(str)
    finished_signal = Signal(list)

    # 2. 修改：多接收一個 unconfirmed_dir 參數
    def __init__(self, model_path, image_paths, device_str, unconfirmed_dir, strict_mode, strict_threshold):
        super().__init__()
        self.model_path = model_path
        self.image_paths = image_paths
        self.device = get_safe_device()
        self.unconfirmed_dir = unconfirmed_dir
        self.strict_mode = strict_mode           # 新增
        self.strict_threshold = strict_threshold # 新增
        self.is_running = True



    def run(self):
        results = []
        try:
            # =========================================================
            # ★★★ 新增：驗證開始前，強制清空舊的 Unconfirmed 資料夾 ★★★
            # =========================================================
            if self.unconfirmed_dir:
                if os.path.exists(self.unconfirmed_dir):
                    self.log_signal.emit("🧹 正在清空舊的待確認區照片...")
                    try:
                        # 遞迴刪除整個資料夾
                        shutil.rmtree(self.unconfirmed_dir)
                        # 刪完後馬上重建一個空的
                        os.makedirs(self.unconfirmed_dir)
                    except Exception as e:
                        self.log_signal.emit(f"⚠️ 清空資料夾失敗: {e}")
                else:
                    # 如果原本不存在，就直接建立
                    os.makedirs(self.unconfirmed_dir)

            self.log_signal.emit(f"🚀 正在載入模型: {os.path.basename(self.model_path)}...")
            
            # --- 重建模型 ---
            # --- 重建模型 (智慧判斷結構) ---
            model = models.convnext_tiny(weights=None)
            num_ftrs = model.classifier[2].in_features
            
            # 1. 先讀取權重檔，看看裡面的結構長怎樣
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # 2. 檢查權重檔是否包含 'classifier.2.1' (這是 Dropout 版的特徵)
            has_dropout_layer = any("classifier.2.1" in k for k in state_dict.keys())
            
            if has_dropout_layer:
                self.log_signal.emit("ℹ️ 偵測到新版模型結構 (含 Dropout)")
                model.classifier[2] = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 2)
                )
            else:
                self.log_signal.emit("ℹ️ 偵測到舊版模型結構 (不含 Dropout)")
                model.classifier[2] = nn.Linear(num_ftrs, 2)

            # 3. 載入權重
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.log_signal.emit("✅ 模型載入成功！開始推論...")

            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            classes = ['NG', 'OK'] 
            total = len(self.image_paths)
            
            saved_count = 0 # 記錄存了幾張到 Unconfirmed

            for i, img_path in enumerate(self.image_paths):
                if not self.is_running: break

                try:
                    # 取得真實標籤 (Ground Truth)
                    parent_folder = os.path.basename(os.path.dirname(img_path))
                    true_label = parent_folder if parent_folder in ['OK', 'NG'] else None

                    # 讀取與推論
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = val_transforms(image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        
                    pred_idx = preds.item()
                    confidence = probs[0][pred_idx].item()
                    pred_label = classes[pred_idx]

                    # ==================================================
                    # ★★★ 新增邏輯：嚴格模式 (Strict Mode) ★★★
                    # 如果預測是 OK，但信心度不夠高，強制轉為 NG
                    # ==================================================
                    is_forced_ng = False
                    if self.strict_mode and pred_label == 'OK' and confidence < self.strict_threshold:
                        pred_label = 'NG' # 強制改判
                        is_forced_ng = True
                        # 注意：這裡雖然改了 label，但 confidence 分數我們通常維持原樣，
                        # 或者你可以選擇是否要標記這個 confidence 已經不代表 NG 的信心了
                    
                    # ==================================================
                    # ★★★ 修正後的邏輯：先判斷對錯，再加註信心警語 ★★★
                    # ==================================================
                    
                    status = ""
                    is_wrong = False
                    
                    # 1. 先判斷對錯 (這裡的 pred_label 已經可能是被強制改過的)
                    if true_label:
                        if true_label == pred_label:
                            status = "✅ 正確"
                        else:
                            status = "❌ 錯誤"
                            is_wrong = True
                    
                    # 補上被強制改判的註記
                    if is_forced_ng:
                        status += f" (🛡️ 嚴格模式: OK信心<{self.strict_threshold} 強制轉NG)"

                    # 2. 檢查信心度 (原本的警語邏輯，保留)
                    # 這裡我們可以保留，用來提示這張圖本身就很模稜兩可
                    is_unsure = False
                    if confidence < 0.80:  # 這是原本的「信心不足」門檻
                        status += " (⚠️ 信心不足)"
                        is_unsure = True

                    if (is_wrong or is_unsure) and self.unconfirmed_dir:
                        try:
                            file_name = os.path.basename(img_path)
                            
                            # =================================================
                            # ★★★ 修改開始：手動實作不重複存檔邏輯 ★★★
                            # 因為 Worker 這裡沒有 data_handler 物件，我們直接寫一個簡單的 while 檢查
                            # =================================================
                            name, ext = os.path.splitext(file_name)
                            dst_path = os.path.join(self.unconfirmed_dir, file_name)
                            
                            # 如果檔案已存在，加上時間戳記
                            counter = 1
                            while os.path.exists(dst_path):
                                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                                new_name = f"{name}_{timestamp}_{counter}{ext}"
                                dst_path = os.path.join(self.unconfirmed_dir, new_name)
                                counter += 1
                            shutil.copy2(img_path, dst_path)
                            status += " (已存至待確認區)"
                            saved_count += 1
                        except Exception as e:
                            print(f"複製失敗: {e}")

                    result_item = {
                        "file_name": os.path.basename(img_path),
                        "path": img_path,
                        "true_label": true_label,
                        "prediction": pred_label,
                        "confidence": confidence
                    }
                    results.append(result_item)
                    
                    # ==================================================

                    self.log_signal.emit(f"[{i+1}/{total}] {os.path.basename(img_path)} -> {pred_label} ({confidence:.1%}) {status}")
                    self.progress_signal.emit(i + 1, total)

                except Exception as e:
                    self.log_signal.emit(f"❌ 讀取失敗 {os.path.basename(img_path)}: {e}")

            # 結束時提示
            if saved_count > 0:
                self.log_signal.emit(f"\n⚠️ 共有 {saved_count} 張預測錯誤的照片已複製到 'Unconfirmed' 資料夾。\n請前往 [Page 2 結果檢查] 進行人工複判。")

            self.finished_signal.emit(results)

        except Exception as e:
            self.log_signal.emit(f"❌ 嚴重錯誤: {str(e)}")
            self.finished_signal.emit([])

    def stop(self):
        self.is_running = False


# ==========================================
# 2. 頁面四 UI (修改 init 接收 data_handler)
# ==========================================
class Page4_Verification(QWidget):
    verification_complete = Signal(list)
    # 3. 修改：__init__ 接收 data_handler
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler # 存下來
        self.image_paths = []
        self.model_path = ""
        self.worker = None
        self.init_ui()

    def reset_ui(self):
        """重置介面狀態：清空路徑、Log、準確率歸零"""
        # 1. 清空變數
        self.image_paths = []
        self.model_path = ""
        
        # 2. 清空 Log 與進度條
        self.txt_output.clear()  # <--- ★★★ 修正這裡：變數名稱是 txt_output ★★★
        self.progress_bar.setValue(0)
        
        # 3. 重置準確率顯示
        if hasattr(self, 'lbl_acc') and self.lbl_acc:
             lbl_val = self.lbl_acc.layout().itemAt(1).widget()
             lbl_val.setText("--%")
             
        # 4. 重置按鈕狀態
        self.btn_start.setEnabled(False)
        self.btn_export_model.setEnabled(False)
        self.btn_load_images.setEnabled(True)
        self.btn_load_model.setEnabled(True)

    def init_ui(self):
        # ... (這裡的介面程式碼完全不用動，維持您原本的樣子即可) ...
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        lbl_title = QLabel("步驟 4: 模型驗證")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4db6ac;")
        main_layout.addWidget(lbl_title)

        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #333; border-radius: 10px; padding: 10px;")
        
        # 改用 QVBoxLayout 讓控制面板可以放兩排東西
        panel_layout = QVBoxLayout(control_panel) 
        
        # 第一排：按鈕群
        btns_layout = QHBoxLayout()
        
        self.btn_load_images = QPushButton("📂 匯入驗證資料夾")
        self.btn_load_images.setStyleSheet(self.get_btn_style("#0277bd"))
        self.btn_load_images.clicked.connect(self.on_load_images)
        
        self.btn_load_model = QPushButton("🧠 選擇模型 (.pth)")
        self.btn_load_model.setStyleSheet(self.get_btn_style("#ef6c00"))
        self.btn_load_model.clicked.connect(self.on_load_model)

        self.btn_start = QPushButton("🚀 開始驗證")
        self.btn_start.setStyleSheet(self.get_btn_style("#00796b"))
        self.btn_start.clicked.connect(self.on_start_verification)
        self.btn_start.setEnabled(False)

        self.btn_export_model = QPushButton("💾 模型匯出")
        self.btn_export_model.setStyleSheet(self.get_btn_style("#7b1fa2"))
        self.btn_export_model.clicked.connect(self.on_export_model)
        self.btn_export_model.setEnabled(False)

        btns_layout.addWidget(self.btn_load_images)
        btns_layout.addWidget(self.btn_load_model)
        btns_layout.addWidget(self.btn_start)
        btns_layout.addWidget(self.btn_export_model)
        
        # 第二排：嚴格模式設定 (新增的部分)
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 5, 0, 0) # 上方留點空隙
        
        # 1. 勾選框
        self.chk_strict_mode = QCheckBox("🛡️ 啟用嚴格過濾模式 (Strict Mode)")
        self.chk_strict_mode.setStyleSheet("color: white; font-weight: bold;")
        self.chk_strict_mode.setToolTip("若勾選，當預測為 OK 但信心度低於門檻時，將強制判為 NG，以避免漏檢。")
        self.chk_strict_mode.setChecked(True) # 預設關閉，讓使用者自己開
        
        # 2. 說明文字
        lbl_strict = QLabel("OK 判定門檻:")
        lbl_strict.setStyleSheet("color: #cfcfcf;")
        
        # 3. 數字調整框
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.50) # 預設您想要的 0.7
        self.spin_threshold.setStyleSheet("""
            QDoubleSpinBox { background-color: #555; color: white; padding: 5px; border-radius: 3px; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 20px; }
        """)
        
        settings_layout.addWidget(self.chk_strict_mode)
        settings_layout.addStretch() # 把後面的東西推到右邊
        settings_layout.addWidget(lbl_strict)
        settings_layout.addWidget(self.spin_threshold)

        # 將兩排加入面板
        panel_layout.addLayout(btns_layout)
        panel_layout.addLayout(settings_layout) # 加入第二排
        
        main_layout.addWidget(control_panel)

        

        self.lbl_acc = self.create_metric_card("🏆 整體準確率 (Accuracy)")
        main_layout.addWidget(self.lbl_acc)

        lbl_formula = QLabel("💡 計算方式： ( 預測正確的照片數 / 總照片數 ) × 100%")
        lbl_formula.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_formula.setStyleSheet("color: #aaa; font-size: 14px; font-style: italic; margin-bottom: 10px;")
        main_layout.addWidget(lbl_formula)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; height: 25px; color: white; }
            QProgressBar::chunk { background-color: #4db6ac; width: 20px; }
        """)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setPlaceholderText("請匯入包含 OK/NG 的資料夾以開始驗證...")
        self.txt_output.setStyleSheet("""
            QTextEdit { background-color: #1e1e1e; color: #cfcfcf; font-family: Consolas; font-size: 13px; border: 1px solid #555; }
        """)
        main_layout.addWidget(self.txt_output)
        self.setLayout(main_layout)

    def create_metric_card(self, title):
        container = QFrame()
        container.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; border: 1px solid #444;")
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #aaa; font-size: 16px;")
        lbl_value = QLabel("--%")
        lbl_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_value.setStyleSheet("color: #4db6ac; font-size: 48px; font-weight: bold;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        return container

    def update_metric_display(self, acc):
        if hasattr(self, 'lbl_acc') and self.lbl_acc:
             lbl_val = self.lbl_acc.layout().itemAt(1).widget()
             lbl_val.setText(f"{acc:.2%}")

    def get_btn_style(self, color):
        return f"""
            QPushButton {{ background-color: {color}; color: white; font-weight: bold; border-radius: 5px; padding: 10px; font-size: 14px; }}
            QPushButton:hover {{ filter: brightness(1.1); }}
            QPushButton:disabled {{ background-color: #555; color: #aaa; }}
        """

    def on_load_images(self):
        # 1. 計算專案根目錄下的 dataset 路徑
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) # .../ui
        root_dir = os.path.dirname(current_file_dir)                # .../ (專案根目錄)
        start_dir = os.path.join(root_dir, "dataset")                 # 設定預設開啟 dataset 資料夾
        
        # 如果 dataset 資料夾還沒建立，就預設開啟根目錄
        if not os.path.exists(start_dir):
            start_dir = root_dir

        # 2. 將 start_dir 傳入 getExistingDirectory (第三個參數)
        folder = QFileDialog.getExistingDirectory(self, "選擇驗證資料夾", start_dir)
        
        if folder:
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_paths = []
            
            # 使用 os.walk 遞迴搜尋資料夾
            for root, dirs, files in os.walk(folder):
                
                # 過濾掉不需要的資料夾 (維持之前的邏輯)
                if "dataset_split" in dirs:
                    dirs.remove("dataset_split") 
                
                if "ROI" in dirs:
                    dirs.remove("ROI")
                    
                if "Unconfirmed" in dirs:
                    dirs.remove("Unconfirmed")

                for f in files:
                    if f.lower().endswith(valid_exts):
                        self.image_paths.append(os.path.join(root, f))
                        
            self.txt_output.append(f"📂 已載入 {len(self.image_paths)} 張圖片")
            self.check_ready()
        
    def on_load_model(self):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_file_dir)
        models_dir = os.path.join(root_dir, "All_Trained_Models")
        start_path = models_dir if os.path.exists(models_dir) else root_dir
        
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型檔案", start_path, "PyTorch Model (*.pth)")
        
        if path:
            self.model_path = path
            self.txt_output.append(f"🧠 已選擇模型: {os.path.basename(path)}")
            self.check_ready()
            
            # ★★★ 新增這行：啟用匯出按鈕 ★★★
            self.btn_export_model.setEnabled(True)
    def check_ready(self):
        if self.image_paths and self.model_path:
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(False)

    def on_start_verification(self):
        self.btn_start.setEnabled(False)
        self.btn_load_images.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        self.chk_strict_mode.setEnabled(False) # 鎖定設定
        self.spin_threshold.setEnabled(False)
        
        self.txt_output.clear()
        self.progress_bar.setValue(0)
        self.update_metric_display(0)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        unconfirmed_path = None
        if self.data_handler and self.data_handler.project_path:
            unconfirmed_path = os.path.join(self.data_handler.project_path, "Unconfirmed")

        # 取得 UI 設定值
        use_strict = self.chk_strict_mode.isChecked()
        strict_thresh = self.spin_threshold.value()

        self.txt_output.append(f"🚀 開始驗證... (Device: {device})")
        if use_strict:
            self.txt_output.append(f"🛡️ 嚴格模式已啟用：信心度 < {strict_thresh:.2f} 的 OK 將被視為 NG")

        # 傳入 Worker (參數變多了)
        self.worker = VerificationWorker(
            self.model_path, 
            self.image_paths, 
            device, 
            unconfirmed_path,
            use_strict,       # 傳入
            strict_thresh     # 傳入
        )
        
        self.worker.log_signal.connect(self.txt_output.append)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_verification_finished)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_verification_finished(self, results):
        self.btn_start.setEnabled(True)
        self.btn_load_images.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        self.chk_strict_mode.setEnabled(True) # 解鎖
        self.spin_threshold.setEnabled(True)  # 解鎖
        
        if not results:
            QMessageBox.warning(self, "結束", "無結果")
            return

        valid_results = [r for r in results if r['true_label'] is not None]
        summary = ""
        
        if len(valid_results) > 0:
            # 轉換標籤為數字 (假設 NG=0, OK=1，這要看你的 classes 定義)
            # 這裡我們用字串比對比較保險
            y_true_str = [r['true_label'] for r in valid_results]
            y_pred_str = [r['prediction'] for r in valid_results]
            
            # 將字串標籤轉為 0(NG) 和 1(OK) 以便計算
            # 定義：NG是正樣本(我們在乎的)，設為 1；OK 設為 0
            # 注意：sklearn 的 pos_label 預設是 1
            y_true = [1 if x == 'NG' else 0 for x in y_true_str]
            y_pred = [1 if x == 'NG' else 0 for x in y_pred_str]
            
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # 混淆矩陣
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            # tn=OK判OK, fp=OK判NG(誤殺), fn=NG判OK(漏檢), tp=NG判NG(抓對)

            correct_count = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == p)
            total_count = len(y_true)

            self.update_metric_display(acc)
            
            summary = (
                f"\n=== 📊 驗證結果詳細報告 ===\n"
                f"總照片數    : {total_count} 張\n"
                f"準確率 (Acc): {acc:.2%}\n"
                f"--------------------------\n"
                f"🎯 關鍵指標 (針對 NG):\n"
                f"  ★ 檢出率 (Recall)   : {recall:.2%} (越高越好，代表沒漏抓)\n"
                f"  ★ 精確率 (Precision): {precision:.2%} (越高代表誤殺少)\n"
                f"  ★ F1-Score          : {f1:.4f}\n"
                f"--------------------------\n"
                f"🔍 混淆矩陣分析:\n"
                f"  ✅ 正確 OK : {tn} 張\n"
                f"  ✅ 抓到 NG : {tp} 張\n"
                f"  ❌ 誤殺 OK : {fp} 張 (OK 被判成 NG)\n"
                f"  💀 漏檢 NG : {fn} 張 (最危險！NG 被判成 OK)\n"
            )
        else:
            summary += "\n⚠️ 警告: 無法計算準確率，因為圖片不在 OK/NG 資料夾內。\n"

        self.txt_output.append(summary)
        self.save_report(results, summary)
        self.verification_complete.emit(results)

    def on_export_model(self):
        """匯出目前選擇的模型檔案"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "錯誤", "尚未選擇模型或原檔案不存在！")
            return

        # 預設檔名使用原本的檔名
        default_name = os.path.basename(self.model_path)
        
        # 跳出「另存新檔」視窗
        save_path, _ = QFileDialog.getSaveFileName(self, "匯出模型", default_name, "PyTorch Model (*.pth)")
        
        if save_path:
            try:
                # 複製檔案 (需要 import shutil，我們之前在檔案最上面已經加過了)
                shutil.copy2(self.model_path, save_path)
                QMessageBox.information(self, "成功", f"模型已成功匯出至：\n{save_path}")
                self.txt_output.append(f"💾 模型已匯出: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"匯出失敗: {str(e)}")
                

    def save_report(self, results, summary):
        # ... (這裡的 save_report 維持您之前改好的樣子，不用動) ...
        try:
            report_dir = "validation_reports"
            if not os.path.exists(report_dir): os.makedirs(report_dir)
            today = datetime.datetime.now().strftime("%Y%m%d")
            idx = 1
            while True:
                filename = f"{today}_Test{idx}.txt"
                full_path = os.path.join(report_dir, filename)
                if not os.path.exists(full_path): break
                idx += 1
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(f"測試時間: {datetime.datetime.now()}\n")
                f.write(f"模型路徑: {self.model_path}\n")
                f.write(summary)
                f.write("\n=== 詳細清單 ===\n")
                f.write(f"{'檔名':<30} | {'真實':<6} | {'預測':<6} | {'信心度':<8} | {'結果':<4}\n")
                f.write("-" * 80 + "\n")
                for r in results:
                    true_s = r['true_label'] if r['true_label'] else "?"
                    mark = "✅" if r['true_label'] == r['prediction'] else "❌"
                    if r['true_label'] is None: mark = "-"
                    f.write(f"{r['file_name']:<30} | {true_s:<6} | {r['prediction']:<6} | {r['confidence']:.4f}   | {mark}\n")
            
            self.txt_output.append(f"📁 報告已儲存: {full_path}")
            acc_text = self.lbl_acc.layout().itemAt(1).widget().text()
            QMessageBox.information(self, "完成", f"驗證完成！報告已儲存。\n\n整體準確率: {acc_text}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"存檔失敗: {e}")