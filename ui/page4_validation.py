import os
import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 新增：計算指標用的庫
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# PySide6 UI 元件
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox, QProgressBar, 
                             QTextEdit, QFrame, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal

# ==========================================
# 1. 後台驗證執行緒
# ==========================================
class VerificationWorker(QThread):
    progress_signal = Signal(int, int)
    log_signal = Signal(str)
    finished_signal = Signal(list)

    def __init__(self, model_path, image_paths, device_str):
        super().__init__()
        self.model_path = model_path
        self.image_paths = image_paths
        self.device = torch.device(device_str)
        self.is_running = True

    def run(self):
        results = []
        try:
            self.log_signal.emit(f"🚀 正在載入模型: {os.path.basename(self.model_path)}...")
            
            # --- 重建模型 ---
            model = models.convnext_tiny(weights=None)
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(num_ftrs, 2)
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.log_signal.emit("✅ 模型載入成功！開始推論...")

            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 確保類別順序與訓練時一致 (通常 ImageFolder 是照字母排: NG=0, OK=1)
            classes = ['NG', 'OK'] 
            
            total = len(self.image_paths)
            for i, img_path in enumerate(self.image_paths):
                if not self.is_running: break

                try:
                    # 嘗試從父資料夾取得 "正確答案" (Ground Truth)
                    parent_folder = os.path.basename(os.path.dirname(img_path))
                    true_label = parent_folder if parent_folder in ['OK', 'NG'] else None

                    # 讀取圖片與推論
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = val_transforms(image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        
                    pred_idx = preds.item()
                    confidence = probs[0][pred_idx].item()
                    pred_label = classes[pred_idx]

                    result_item = {
                        "file_name": os.path.basename(img_path),
                        "path": img_path,
                        "true_label": true_label, # 真實標籤 (可能是 None)
                        "prediction": pred_label, # 預測結果
                        "confidence": confidence
                    }
                    results.append(result_item)
                    
                    # Log 顯示
                    status = ""
                    if true_label:
                        is_correct = "✅" if true_label == pred_label else "❌"
                        status = f"| 真實: {true_label} {is_correct}"
                    
                    self.log_signal.emit(f"[{i+1}/{total}] {result_item['file_name']} -> {pred_label} ({confidence:.1%}) {status}")
                    self.progress_signal.emit(i + 1, total)

                except Exception as e:
                    self.log_signal.emit(f"❌ 讀取失敗 {os.path.basename(img_path)}: {e}")

            self.finished_signal.emit(results)

        except Exception as e:
            self.log_signal.emit(f"❌ 嚴重錯誤: {str(e)}")
            self.finished_signal.emit([])

    def stop(self):
        self.is_running = False


# ==========================================
# 2. 頁面四 UI
# ==========================================
class Page4_Verification(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.model_path = ""
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- 標題區 ---
        lbl_title = QLabel("步驟 4: 模型驗證")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4db6ac;")
        main_layout.addWidget(lbl_title)

        # --- 頂部：控制面板 ---
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #333; border-radius: 10px; padding: 10px;")
        control_layout = QHBoxLayout(control_panel)

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

        control_layout.addWidget(self.btn_load_images)
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_start)
        main_layout.addWidget(control_panel)

        # --- 中間：只顯示準確率與公式 ---
        # 1. 準確率卡片
        self.lbl_acc = self.create_metric_card("🏆 整體準確率 (Accuracy)")
        main_layout.addWidget(self.lbl_acc)

        # 2. 公式說明文字 (新增這段)
        lbl_formula = QLabel("💡 計算方式： ( 預測正確的照片數 / 總照片數 ) × 100%")
        lbl_formula.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_formula.setStyleSheet("color: #aaa; font-size: 14px; font-style: italic; margin-bottom: 10px;")
        main_layout.addWidget(lbl_formula)

        # --- 底部：進度條與 Log ---
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
        """建立漂亮的指標顯示卡片"""
        container = QFrame()
        container.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; border: 1px solid #444;")
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #aaa; font-size: 12px;")
        
        lbl_value = QLabel("--%")
        lbl_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_value.setStyleSheet("color: #4db6ac; font-size: 24px; font-weight: bold;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        return container

    def update_metric_display(self, acc):
        """更新儀表板數字"""
        # 我們現在只剩下 lbl_acc 這個元件，所以只更新它
        # 取得容器內的數值 Label (它是 layout 裡的第 2 個元件，index 1)
        if hasattr(self, 'lbl_acc') and self.lbl_acc:
             lbl_val = self.lbl_acc.layout().itemAt(1).widget()
             lbl_val.setText(f"{acc:.2%}")

    def get_btn_style(self, color):
        return f"""
            QPushButton {{ background-color: {color}; color: white; font-weight: bold; border-radius: 5px; padding: 10px; font-size: 14px; }}
            QPushButton:hover {{ filter: brightness(1.1); }}
            QPushButton:disabled {{ background-color: #555; color: #aaa; }}
        """

    # --- 邏輯功能 ---

    def on_load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇驗證資料夾 (建議包含 OK/NG 子資料夾)")
        if folder:
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_paths = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(valid_exts):
                        self.image_paths.append(os.path.join(root, f))
            
            self.txt_output.append(f"📂 已載入 {len(self.image_paths)} 張圖片 (來源: {os.path.basename(folder)})")
            self.check_ready()
        
    def on_load_model(self):
        """選擇 .pth 模型檔 (預設開啟 All_Trained_Models 資料夾)"""
        
        # 1. 計算預設路徑：從 ui 資料夾往上一層 -> 進入 All_Trained_Models
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) # ui 資料夾
        root_dir = os.path.dirname(current_file_dir)                 # 專案根目錄
        models_dir = os.path.join(root_dir, "All_Trained_Models")      # 目標資料夾
        
        # 如果這個資料夾還沒被建立過 (例如還沒訓練過)，就預設開在根目錄，避免程式報錯
        start_path = models_dir if os.path.exists(models_dir) else root_dir

        # 2. 開啟檔案選擇視窗 (第三個參數就是起始路徑)
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型檔案", start_path, "PyTorch Model (*.pth)")
        
        if path:
            self.model_path = path
            # 顯示檔名就好，不用顯示長長的路徑
            self.txt_output.append(f"🧠 已選擇模型: {os.path.basename(path)}")
            self.check_ready()

    def check_ready(self):
        if self.image_paths and self.model_path:
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(False)

    def on_start_verification(self):
        self.btn_start.setEnabled(False)
        self.btn_load_images.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        self.txt_output.clear()
        self.progress_bar.setValue(0)
        
        # ★★★ 修正這裡 ★★★
        # 原本是 self.update_metric_display(0,0,0,0)
        # 因為現在只剩準確率，所以只要傳一個 0 進去歸零就好
        self.update_metric_display(0)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.txt_output.append(f"🚀 開始驗證... (Device: {device})")

        self.worker = VerificationWorker(self.model_path, self.image_paths, device)
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
        
        if not results:
            QMessageBox.warning(self, "結束", "無結果")
            return

        # --- 計算指標 ---
        valid_results = [r for r in results if r['true_label'] is not None]
        
        summary = ""
        
        if len(valid_results) > 0:
            y_true = [r['true_label'] for r in valid_results]
            y_pred = [r['prediction'] for r in valid_results]
            
            # 1. 計算準確率
            acc = accuracy_score(y_true, y_pred)
            
            # 2. 計算答對幾題
            correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            total_count = len(y_true)

            # 更新 UI 數字
            self.update_metric_display(acc)
            
            # 產生報告文字
            summary = (
                f"\n=== 📊 驗證結果摘要 ===\n"
                f"總照片數    : {total_count} 張\n"
                f"預測正確    : {correct_count} 張\n"
                f"預測錯誤    : {total_count - correct_count} 張\n"
                f"--------------------------\n"
                f"準確率 (Accuracy) : {acc:.2%}  (即 {correct_count} ÷ {total_count})\n"
            )
        else:
            summary += "\n⚠️ 警告: 無法計算準確率，因為圖片不在 OK/NG 資料夾內，無法得知正確答案。\n"

        self.txt_output.append(summary)
        self.save_report(results, summary)

    def save_report(self, results, summary):
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
            
            # ★★★ 修正這裡 ★★★
            # 移除 lbl_f1，改為單純顯示完成，或是顯示目前的準確率
            acc_text = self.lbl_acc.layout().itemAt(1).widget().text()
            QMessageBox.information(self, "完成", f"驗證完成！報告已儲存。\n\n整體準確率: {acc_text}")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"存檔失敗: {e}")