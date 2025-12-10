import os
import shutil
import random
import time
import traceback

# PySide6 UI 元件
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QProgressBar, QTextEdit, QMessageBox, QGroupBox, 
                             QFormLayout, QFrame, QAbstractSpinBox) # <--- 新增 QAbstractSpinBox
from PySide6.QtCore import Qt, QThread, Signal, QObject

# 繪圖相關 (Matplotlib 嵌入 PySide6)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# PyTorch 相關
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import datetime
# ==========================================
# 1. 後台工作執行緒 (避免介面卡死)
# ==========================================
class TrainingWorker(QThread):
    # 定義訊號：用來跟主介面溝通
    log_signal = Signal(str)            # 傳送文字 Log
    progress_signal = Signal(int, int)  # 傳送進度 (目前, 總共)
    metric_signal = Signal(dict)        # 傳送訓練數據 (Loss, Acc) 用於繪圖
    finished_signal = Signal(bool, str) # 完成訊號 (是否成功, 訊息)

    def __init__(self, project_path, params):
        super().__init__()
        self.project_path = project_path
        self.params = params
        self.is_running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            self.log_signal.emit(f"🚀 初始化訓練程序... (使用裝置: {self.device})")
            
            # --- 步驟 1: 資料準備 (切割 Dataset) ---
            dataset_dir = os.path.join(self.project_path, "dataset_split")
            if not self.prepare_data(dataset_dir):
                self.finished_signal.emit(False, "資料準備失敗，請檢查原始照片是否足夠。")
                return

            # --- 步驟 2: 載入資料 ---
            dataloaders, dataset_sizes = self.get_dataloaders(dataset_dir)
            
            # --- 步驟 3: 建立模型 ---
            self.log_signal.emit("🧠 正在載入 ConvNeXt 模型 (這可能需要一點時間)...")
            model = models.convnext_tiny(weights='DEFAULT')
            
            # 修改最後一層全連接層 (配合我們的 2 個類別: OK, NG)
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(num_ftrs, 2)
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.params['lr'])

            best_acc = 0.0
            epochs = self.params['epochs']
            
            # 建立模型儲存資料夾
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. 往上一層走，回到主程式根目錄 (假設 main.py 在這)
            root_dir = os.path.dirname(current_file_dir)
            
            # 3. 指定一個統一的資料夾名稱，例如 "All_Trained_Models"
            save_dir = os.path.join(root_dir, "All_Trained_Models")
            
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            # 產生這次訓練專用的檔名 (包含日期時間)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 建議檔名也可以加上專案名稱，比較好分辨是訓練哪個專案的模型
            project_name = os.path.basename(self.project_path)
            model_filename = f"best_{project_name}_{timestamp}.pth"
            
            save_path = os.path.join(save_dir, model_filename)
            
            self.log_signal.emit(f"💾 本次訓練模型將儲存至根目錄: {save_path}")
            # ★★★ 修改重點結束 ★★★

            for epoch in range(epochs):
                if not self.is_running: break # 允許中途停止

                self.log_signal.emit(f"\nEpoch {epoch+1}/{epochs} 開始...")
                
                epoch_metrics = {'epoch': epoch + 1}

                for phase in ['train', 'val']:
                    # ... (中間訓練邏輯完全不用動) ...
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in dataloaders[phase]:
                        if not self.is_running: break
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    
                    if not self.is_running: break

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    
                    # 記錄數據 (保持您修正過的小寫 Key)
                    prefix = "train" if phase == 'train' else "val" 
                    self.log_signal.emit(f"  - {prefix.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
                    
                    epoch_metrics[f'{prefix}_loss'] = epoch_loss
                    epoch_metrics[f'{prefix}_acc'] = epoch_acc.item()

                    # ★★★ 儲存最佳模型 (只看驗證集) ★★★
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # 這裡會覆蓋「本次訓練」的檔案，確保留下來的是這一次跑最好的那個 Epoch
                        torch.save(model.state_dict(), save_path)
                        self.log_signal.emit(f"  🌟 發現新高分 ({epoch_acc:.2%})！已更新模型檔案。")

                # 發送繪圖數據
                self.metric_signal.emit(epoch_metrics)
                self.progress_signal.emit(epoch + 1, epochs)

            if self.is_running:
                self.finished_signal.emit(True, f"訓練完成！最佳準確率: {best_acc:.2%}")
            else:
                self.finished_signal.emit(False, "訓練已手動停止。")

        except Exception as e:
            err_msg = traceback.format_exc()
            self.finished_signal.emit(False, f"發生未預期的錯誤:\n{e}")
            print(err_msg)

    def stop(self):
        self.is_running = False

    def prepare_data(self, target_dir):
        """自動切割資料集：從 OK/NG 複製到 dataset_split/train 與 val"""
        split_ratio = self.params['split_ratio']
        
        # 來源：專案根目錄下的 OK 與 NG
        src_ok = os.path.join(self.project_path, "OK")
        src_ng = os.path.join(self.project_path, "NG")
        
        if not os.path.exists(src_ok) or not os.path.exists(src_ng):
            return False

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir) # 清空舊的
            
        self.log_signal.emit(f"正在重新切割資料集 (比例 {split_ratio})...")
        
        for class_name, src_path in [('OK', src_ok), ('NG', src_ng)]:
            images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg','.png','.bmp'))]
            random.shuffle(images)
            
            if not images: continue
            
            split_point = int(len(images) * split_ratio)
            # 確保至少有一張圖在 val
            if split_point >= len(images): split_point = len(images) - 1
            if split_point < 0: split_point = 0

            train_imgs = images[:split_point]
            val_imgs = images[split_point:]
            
            for phase, img_list in [('train', train_imgs), ('val', val_imgs)]:
                dst_folder = os.path.join(target_dir, phase, class_name)
                os.makedirs(dst_folder, exist_ok=True)
                for img in img_list:
                    shutil.copy(os.path.join(src_path, img), os.path.join(dst_folder, img))
                    
        return True

    def get_dataloaders(self, dataset_dir):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.params['batch_size'], shuffle=True)
                       for x in ['train', 'val']}
                       
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes


# ==========================================
# 2. 頁面三：訓練介面 UI
# ==========================================
class Page3_Training(QWidget):
    def __init__(self):
        super().__init__()
        self.data_handler = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        # --- 左側：設定面板 (30%) ---
        left_panel = QFrame()
        left_panel.setStyleSheet(".QFrame { background-color: #333; border-radius: 10px; }")
        left_layout = QVBoxLayout(left_panel)
        
        lbl_title = QLabel("⚙️ 訓練參數設定")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4db6ac; margin-bottom: 10px;")
        left_layout.addWidget(lbl_title)

        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        # 定義通用的 ComboBox 樣式 (解決透明問題)
        # ★★★ 修正點 2：新增背景色與 ItemView 樣式 ★★★
        combo_style = """
            QComboBox { 
                background-color: #555; 
                color: white; 
                padding: 5px; 
                border: 1px solid #777; 
                border-radius: 4px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { 
                background-color: #555; 
                color: white; 
                selection-background-color: #00796b; 
            }
        """

        # 1. Epochs (訓練輪數)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(10)
        self.spin_epochs.setButtonSymbols(QAbstractSpinBox.NoButtons) # 隱藏上下小箭頭看起來比較現代
        self.spin_epochs.setStyleSheet("padding: 5px; background-color: #555; color: white; border: 1px solid #666; border-radius: 4px;")
        
        # ★ 修改這裡：改用 add_param_row 加入
        self.add_param_row(
            form_layout, 
            "訓練輪數 (Epochs)", 
            self.spin_epochs, 
            "模型完整看過一次所有照片稱為 1 個 Epoch。\n次數越多模型學得越久，但也可能導致過度擬合 (Overfitting)。",
            "50~100 (註解:訓練的回合數)"
        )

        # 2. Batch Size (批次大小)
        self.combo_batch = QComboBox()
        self.combo_batch.addItems(["8", "16", "32", "64"])
        self.combo_batch.setCurrentText("16")
        self.combo_batch.setStyleSheet("""
            QComboBox { background-color: #555; color: white; padding: 5px; border: 1px solid #666; border-radius: 4px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #555; color: white; selection-background-color: #00796b; }
        """)
        
        # ★ 修改這裡
        self.add_param_row(
            form_layout, 
            "批次大小 (Batch)", 
            self.combo_batch, 
            "每次訓練時，同時塞入多少張照片給模型看。\n數值越大吃越多顯卡記憶體，但訓練速度較快。",
            "16 或 32 (註解:一次輸入給模型的訓練樣本數量)"
        )

        # 3. Learning Rate (學習率)
        self.combo_lr = QComboBox()
        self.combo_lr.addItem("0.001 (快 - 可能不穩)", 0.001)
        self.combo_lr.addItem("0.0001 (中 - 推薦)", 0.0001)
        self.combo_lr.addItem("0.00001 (慢 - 精細)", 0.00001)
        self.combo_lr.setCurrentIndex(1)
        self.combo_lr.setStyleSheet(self.combo_batch.styleSheet()) # 沿用上面的樣式
        
        # ★ 修改這裡
        self.add_param_row(
            form_layout, 
            "學習率 (LR)", 
            self.combo_lr, 
            "模型修正錯誤的步伐大小。\n設太大會學不會(震盪)，設太小會學很慢。",
            "0.0001 (註解:學習速度與精細度的平衡)"
        )

        # 4. Split Ratio (訓練集比例)
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.5, 0.95)
        self.spin_ratio.setSingleStep(0.1)
        self.spin_ratio.setValue(0.8)
        self.spin_ratio.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin_ratio.setStyleSheet(self.spin_epochs.styleSheet()) # 沿用上面的樣式
        
        # ★ 修改這裡
        self.add_param_row(
            form_layout, 
            "訓練集比例 (Split)", 
            self.spin_ratio, 
            "將多少比例的照片切分出來用於「訓練」，剩餘的用於「驗證」。\n0.8 代表 80% 訓練，20% 驗證。",
            "0.8 (註解:即 8:2 分配)"
        )
        # ★★★ 修正點 1：移除按鈕 (NoButtons) ★★★
        self.spin_ratio.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin_ratio.setStyleSheet("padding: 5px; background-color: #555; color: white; border: 1px solid #777; border-radius: 4px;")
        form_layout.addRow("訓練集比例:", self.spin_ratio)

        left_layout.addLayout(form_layout)
        left_layout.addStretch()

        self.btn_start = QPushButton("🚀 開始訓練")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #00796b; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #004d40; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.btn_start.clicked.connect(self.on_start_clicked)
        left_layout.addWidget(self.btn_start)

        # --- 右側：監控面板 (70%) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 圖表區
        self.figure = Figure(figsize=(5, 3), dpi=100, facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setup_chart_initial()
        right_layout.addWidget(self.canvas, 2)

        # 2. 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; height: 25px; color: white; }
            QProgressBar::chunk { background-color: #4db6ac; width: 20px; }
        """)
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        # 3. Log 輸出區
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #1e1e1e; color: #cfcfcf; font-family: Consolas; font-size: 12px; border: 1px solid #555;")
        self.txt_log.setPlaceholderText("等待訓練開始...")
        right_layout.addWidget(self.txt_log, 1)

        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 7)
        self.setLayout(main_layout)
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def setup_chart_initial(self):
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.set_title("Training Metrics", color='white')
        self.ax.set_xlabel("Epochs", color='white')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.figure.tight_layout()

    def update_chart(self, metrics):
        # 因為 Key 已經在 Worker 統一改為小寫，這裡就能正確抓到值了
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['val_loss'].append(metrics.get('val_loss', 0))
        self.history['train_acc'].append(metrics.get('train_acc', 0))
        self.history['val_acc'].append(metrics.get('val_acc', 0))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        self.ax.plot(epochs, self.history['train_loss'], label='Train Loss', color='#ef5350')
        self.ax.plot(epochs, self.history['val_loss'], label='Val Loss', color='#ffca28')
        self.ax.plot(epochs, self.history['train_acc'], label='Train Acc', color='#42a5f5')
        self.ax.plot(epochs, self.history['val_acc'], label='Val Acc', color='#66bb6a')
        
        self.ax.legend(loc='upper right', facecolor='#333', labelcolor='white')
        self.ax.set_title("Training Metrics", color='white')
        self.canvas.draw()

    def on_start_clicked(self):
        if not self.data_handler or not self.data_handler.project_path:
            QMessageBox.warning(self, "錯誤", "尚未載入專案，請先建立或開啟專案！")
            return

        self.btn_start.setEnabled(False)
        self.btn_start.setText("⏳ 訓練中...")
        self.txt_log.clear()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.setup_chart_initial()

        params = {
            'epochs': self.spin_epochs.value(),
            'batch_size': int(self.combo_batch.currentText()),
            'lr': self.combo_lr.currentData(),
            'split_ratio': self.spin_ratio.value()
        }

        self.worker = TrainingWorker(self.data_handler.project_path, params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.metric_signal.connect(self.update_chart)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def append_log(self, text):
        self.txt_log.append(text)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_training_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 開始訓練")
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "中斷", message)
        self.worker = None

    def set_data_handler(self, handler):
        self.data_handler = handler
    
    # 將此函式加入 class Page3_Training 裡面
    def add_param_row(self, layout, label_text, widget, description, recommend_val):
        """建立一個帶有說明與建議值的參數列"""
        
        # 1. 設定 Tooltip (滑鼠移上去看到的詳細解釋)
        # 設定黑底白字的樣式
        widget.setToolTip(f"{label_text}\n\n說明：{description}\n建議值：{recommend_val}")
        widget.setStyleSheet(widget.styleSheet() + "QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #555; }")

        # 2. 建立右側容器 (垂直排列：上面是輸入框，下面是建議文字)
        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(2) # 讓文字緊貼輸入框

        # 加入輸入框
        v_layout.addWidget(widget)

        # 加入建議文字 (灰色斜體小字)ㄇ
        lbl_tip = QLabel(f"<font color='#888888' size='3'> 建議: {recommend_val}</font>")
        lbl_tip.setStyleSheet("font-family: 'Microsoft JhengHei'; font-style: italic;")
        v_layout.addWidget(lbl_tip)

        # 3. 設定左側標籤樣式 (加粗)
        lbl_title = QLabel(label_text)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #ddd;")

        # 4. 加到表單佈局中
        layout.addRow(lbl_title, container)

        