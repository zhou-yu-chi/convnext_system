import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QListWidget, QListWidgetItem, QComboBox, 
                             QSplitter, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImageReader, QImage

# ==========================================
# 重用 IconWorker (確保列表滑動流暢)
# ==========================================
class IconWorker(QThread):
    icon_loaded = Signal(int, QImage)

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list # 這裡接收的是包含路徑的字典列表
        self.is_running = True

    def run(self):
        for i, item in enumerate(self.data_list):
            if not self.is_running: break
            path = item.get('path', '')
            if not os.path.exists(path): continue
            
            reader = QImageReader(path)
            reader.setScaledSize(QSize(100, 100)) 
            image = reader.read()
            if not image.isNull():
                self.icon_loaded.emit(i, image)
                if i % 10 == 0: QThread.msleep(5)

    def stop(self):
        self.is_running = False
        self.wait()

class Page6_ResultReview(QWidget):
    def __init__(self):
        super().__init__()
        self.all_results = []       # 儲存 Page 4 傳過來的所有結果
        self.current_filtered = []  # 目前下拉選單篩選後的結果
        self.current_selected_path = None
        self.icon_worker = None
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

        lbl_filter = QLabel("🔍 篩選條件：")
        lbl_filter.setStyleSheet("font-size: 15px; font-weight: bold; color: #ddd; border: none;")
        
        # 四個選項
        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "✅ 判定 OK 且 正確 (True Positive)",
            "✅ 判定 NG 且 正確 (True Negative)",
            "⚠️ 判定 OK 但 錯誤 (漏檢 / Leakage)",
            "⚠️ 判定 NG 但 錯誤 (誤殺 / Overkill)"
        ])
        self.combo_filter.setStyleSheet("""
            QComboBox { background-color: #555; color: white; padding: 5px 10px; border-radius: 5px; border: 1px solid #666; font-size: 14px; min-width: 250px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #555; color: white; selection-background-color: #00796b; }
        """)
        self.combo_filter.currentIndexChanged.connect(self.filter_results)

        self.lbl_count = QLabel("數量: 0")
        self.lbl_count.setStyleSheet("color: #4db6ac; font-weight: bold; font-size: 14px; border: none; margin-left: 15px;")

        top_bar.addWidget(lbl_filter)
        top_bar.addWidget(self.combo_filter)
        top_bar.addWidget(self.lbl_count)
        top_bar.addStretch()
        
        main_layout.addWidget(top_bar_container)

        # --- 2. 中間顯示區 (左清單 | 右預覽) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        # 左側清單
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 80)) 
        self.list_widget.setFixedWidth(280) 
        self.list_widget.setSpacing(5)
        self.list_widget.setStyleSheet("""
            QListWidget { background-color: #2b2b2b; border: 1px solid #444; border-radius: 8px; padding: 5px; outline: 0; }
            QListWidget::item { background-color: #333; border-radius: 5px; color: #eee; padding: 10px; margin-bottom: 2px; }
            QListWidget::item:selected { background-color: #00796b; border: 1px solid #4db6ac; color: white; }
            QListWidget::item:hover { background-color: #444; }
        """)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 右側預覽容器
        right_container = QFrame()
        right_container.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #444; border-radius: 8px; }")
        right_layout = QVBoxLayout(right_container)
        
        self.image_preview = QLabel("等待驗證結果...")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_preview.setStyleSheet("background-color: transparent; color: #666; font-size: 16px;")
        
        # 圖片下方資訊
        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("color: #aaa; font-size: 14px; padding: 10px; border: none;")
        self.lbl_info.setMaximumHeight(50)

        right_layout.addWidget(self.image_preview, 1)
        right_layout.addWidget(self.lbl_info)

        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # ==========================================
    # ★★★ 核心功能：接收 Page 4 資料 ★★★
    # ==========================================
    def update_results(self, results):
        """
        當 Page 4 驗證結束時呼叫此函式。
        results: List[dict] -> [{'file_name':..., 'path':..., 'true_label':..., 'prediction':..., 'confidence':...}]
        """
        # 1. 清除舊資料
        self.clear_ui()
        self.all_results = results
        
        # 2. 自動執行一次篩選 (預設顯示第一類)
        self.filter_results()
        
        # 3. 提示
        if not results:
            self.image_preview.setText("本次驗證無資料")

    def clear_ui(self):
        """清除頁面所有顯示內容"""
        if self.icon_worker and self.icon_worker.isRunning():
            self.icon_worker.stop()
        
        self.list_widget.clear()
        self.image_preview.clear()
        self.image_preview.setText("請從左側選擇照片")
        self.lbl_info.setText("")
        self.all_results = []
        self.current_filtered = []
        self.lbl_count.setText("數量: 0")

    def filter_results(self):
        """根據下拉選單篩選 results"""
        if not self.all_results: return

        # 先停止舊的載入線程
        if self.icon_worker and self.icon_worker.isRunning():
            self.icon_worker.stop()
        
        self.list_widget.clear()
        self.current_filtered = []
        
        idx = self.combo_filter.currentIndex()
        
        # 篩選邏輯
        for r in self.all_results:
            true_lbl = r.get('true_label')
            pred_lbl = r.get('prediction')
            
            # 如果沒有真實標籤 (例如不在 OK/NG 資料夾的照片)，直接跳過
            if not true_lbl: continue

            is_match = False
            
            if idx == 0:   # Model OK & Correct
                if pred_lbl == 'OK' and true_lbl == 'OK': is_match = True
            elif idx == 1: # Model NG & Correct
                if pred_lbl == 'NG' and true_lbl == 'NG': is_match = True
            elif idx == 2: # Model OK but Wrong (真實是 NG)
                if pred_lbl == 'OK' and true_lbl == 'NG': is_match = True
            elif idx == 3: # Model NG but Wrong (真實是 OK)
                if pred_lbl == 'NG' and true_lbl == 'OK': is_match = True
            
            if is_match:
                self.current_filtered.append(r)

        # 更新介面清單
        self.lbl_count.setText(f"數量: {len(self.current_filtered)}")
        
        if not self.current_filtered:
            self.image_preview.setText("此分類下沒有照片")
            return

        # 填入 ListWidget
        for item_data in self.current_filtered:
            name = item_data.get('file_name', 'Unknown')
            conf = item_data.get('confidence', 0.0)
            
            # 顯示檔名與信心度
            display_text = f"{name}\n信心度: {conf:.1%}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, item_data) # 把整包資料存進去
            self.list_widget.addItem(item)

        # 啟動縮圖載入
        self.icon_worker = IconWorker(self.current_filtered)
        self.icon_worker.icon_loaded.connect(self.on_icon_loaded)
        self.icon_worker.start()
        
        # 自動選取第一項
        QTimer.singleShot(50, lambda: self.list_widget.setCurrentRow(0))

    def on_icon_loaded(self, row, image):
        item = self.list_widget.item(row)
        if item:
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if items:
            data = items[0].data(Qt.UserRole)
            self.show_image_detail(data)

    def show_image_detail(self, data):
        path = data.get('path')
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            # 顯示圖片
            scaled = pixmap.scaled(self.image_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_preview.setPixmap(scaled)
            
            # 顯示詳細資訊
            t = data.get('true_label')
            p = data.get('prediction')
            c = data.get('confidence')
            self.lbl_info.setText(f"📂 檔名: {os.path.basename(path)}  |  真實: {t}  |  預測: {p}  |  信心度: {c:.2%}")
        else:
            self.image_preview.setText("圖片讀取失敗")
            self.lbl_info.setText("")

    def resizeEvent(self, event):
        # 視窗縮放時，重新繪製圖片以適應大小
        items = self.list_widget.selectedItems()
        if items:
            data = items[0].data(Qt.UserRole)
            self.show_image_detail(data)
        super().resizeEvent(event)