import os
from PIL import Image 
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QMessageBox, QFileDialog, QListWidget, 
                             QListWidgetItem, QSplitter, QSizePolicy, QFrame, 
                             QProgressDialog, QApplication)
from PySide6.QtCore import Qt, QRect, QPoint, QSize, QThread, Signal # <--- 引入 QThread, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QIcon, QImage, QImageReader # <--- 引入 QImageReader

# ==========================================
# 0. 新增：後台縮圖載入小精靈 (解決 400 張照片卡頓的關鍵)
# ==========================================
class IconWorker(QThread):
    # 定義訊號：(第幾行, 圖片物件)
    icon_loaded = Signal(int, QImage)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.is_running = True

    def run(self):
        # 在後台一張一張讀取
        for i, path in enumerate(self.image_paths):
            if not self.is_running: break
            
            # 使用 QImageReader 只讀取縮圖，速度比讀整張圖快 10 倍以上！
            reader = QImageReader(path)
            # 設定讀取時就直接縮小到 100x100 (節省記憶體與時間)
            reader.setScaledSize(QSize(100, 100)) 
            image = reader.read()
            
            if not image.isNull():
                self.icon_loaded.emit(i, image) # 通知主程式：這張圖好了
                
                # 每處理 10 張稍微休息一下，讓介面更滑順 (可選)
                if i % 10 == 0:
                    QThread.msleep(5)

    def stop(self):
        self.is_running = False
        self.wait()

# ==========================================
# 1. 增強版 Label (功能維持不變)
# ==========================================
class CroppableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent) 
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_crop_rect_original = None 

    def set_image(self, image_path):
        # 這裡讀大圖只讀一張，所以用 QPixmap 沒問題
        self.original_pixmap = QPixmap(image_path)
        self.update_display()
        if self.last_crop_rect_original:
            self.restore_crop_box()
        else:
            self.start_point = None
            self.end_point = None
        self.update()

    def update_display(self):
        if not self.original_pixmap: return
        w_limit = self.width()
        h_limit = self.height()
        self.scaled_pixmap = self.original_pixmap.scaled(w_limit, h_limit, 
                                                       Qt.AspectRatioMode.KeepAspectRatio, 
                                                       Qt.TransformationMode.SmoothTransformation)
        self.scale_factor = self.original_pixmap.width() / self.scaled_pixmap.width()
        self.offset_x = (self.width() - self.scaled_pixmap.width()) // 2
        self.offset_y = (self.height() - self.scaled_pixmap.height()) // 2
        if self.last_crop_rect_original:
            self.restore_crop_box()
        self.update()

    def restore_crop_box(self):
        if not self.last_crop_rect_original or not self.scale_factor: return
        rx1, ry1, rx2, ry2 = self.last_crop_rect_original
        sx1 = int(rx1 / self.scale_factor) + self.offset_x
        sy1 = int(ry1 / self.scale_factor) + self.offset_y
        sx2 = int(rx2 / self.scale_factor) + self.offset_x
        sy2 = int(ry2 / self.scale_factor) + self.offset_y
        self.start_point = QPoint(sx1, sy1)
        self.end_point = QPoint(sx2, sy2)

    def paintEvent(self, event):
        if not self.scaled_pixmap:
            painter = QPainter(self)
            painter.setPen(QColor(100, 100, 100))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            text = "請匯入照片並從左側清單選擇"
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(text)
            text_h = fm.height()
            painter.drawText((self.width() - text_w) // 2, (self.height() - text_h) // 2, text)
            return
            
        painter = QPainter(self)
        painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)
        
        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            pen = QPen(QColor(255, 50, 50), 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(255, 0, 0, 30))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            self.end_point = event.position().toPoint()
            self.update()
            self.last_crop_rect_original = self.get_crop_rect_original()

    def get_crop_rect_original(self):
        if not self.start_point or not self.end_point:
            return None
        screen_rect = QRect(self.start_point, self.end_point).normalized()
        x = screen_rect.x() - self.offset_x
        y = screen_rect.y() - self.offset_y
        w = screen_rect.width()
        h = screen_rect.height()
        if x < 0: x = 0
        if y < 0: y = 0
        real_x = int(x * self.scale_factor)
        real_y = int(y * self.scale_factor)
        real_w = int(w * self.scale_factor)
        real_h = int(h * self.scale_factor)
        if real_w <= 0 or real_h <= 0: return None
        return (real_x, real_y, real_x + real_w, real_y + real_h)
    
    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def clear_canvas(self):
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.start_point = None
        self.end_point = None
        self.last_crop_rect_original = None
        self.clear()
        self.update()

# ==========================================
# 2. 主頁面
# ==========================================
class Page0_Cropping(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.current_image_path = None
        self.icon_worker = None # 儲存 Worker 的變數
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(10)
        
        # --- 1. 頂部工具列 ---
        top_bar_container = QFrame()
        top_bar_container.setMaximumHeight(65) 
        top_bar_container.setStyleSheet("""
            QFrame {
                background-color: #333; 
                border-radius: 8px; 
                padding: 2px;
            }
        """)
        top_bar = QHBoxLayout(top_bar_container)
        top_bar.setContentsMargins(10, 5, 10, 5)

        self.btn_import = QPushButton(" 📥 匯入照片")
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #0277bd; color: white; font-weight: bold; 
                padding: 6px 15px; border-radius: 5px; font-size: 14px;
            }
            QPushButton:hover { background-color: #0288d1; }
        """)
        self.btn_import.clicked.connect(self.on_import_clicked)
        
        self.lbl_info = QLabel("等待匯入...")
        self.lbl_info.setStyleSheet("color: #ddd; margin-left: 15px; font-size: 14px; font-weight: bold;")
        
        top_bar.addWidget(self.btn_import)
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()
        
        main_layout.addWidget(top_bar_container)

        # --- 2. 中間區域 ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        # 左側清單
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(70, 70)) 
        self.list_widget.setFixedWidth(240)
        self.list_widget.setSpacing(3)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 5px;
                outline: 0;
            }
            QListWidget::item {
                background-color: #333;
                border-radius: 5px;
                color: #eee;
                padding: 8px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #00796b; 
                border: 1px solid #4db6ac;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #444;
            }
        """)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 右側畫布容器
        right_container = QFrame()
        right_container.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #444;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = CroppableLabel()
        self.image_label.setStyleSheet("background-color: transparent;") 
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 底部按鈕控制列
        btn_bar = QFrame()
        btn_bar.setStyleSheet("QFrame { background-color: #333; border-top: 1px solid #555; border-radius: 0px; }")
        btn_bar.setMaximumHeight(60) 
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(10, 8, 10, 8)

        self.btn_batch = QPushButton("⚡ 一鍵裁切全部 (Batch)")
        self.btn_batch.setMinimumHeight(40)
        self.btn_batch.setStyleSheet("""
            QPushButton {
                background-color: #7b1fa2; color: white; border-radius: 5px; 
                padding: 5px 15px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #9c27b0; }
        """)
        self.btn_batch.clicked.connect(self.apply_batch_crop) 
        
        self.btn_crop = QPushButton("✂️ 單張裁切 (Enter)")
        self.btn_crop.setMinimumHeight(40)
        self.btn_crop.setStyleSheet("""
            QPushButton {
                background-color: #ef6c00; color: white; font-weight: bold; 
                border-radius: 5px; padding: 5px 20px; font-size: 16px;
            }
            QPushButton:hover { background-color: #ff9800; }
        """)
        self.btn_crop.clicked.connect(self.apply_crop)

        btn_layout.addWidget(self.btn_batch)
        btn_layout.addStretch() 
        btn_layout.addWidget(self.btn_crop)
        
        right_layout.addWidget(self.image_label, 1) 
        right_layout.addWidget(btn_bar)
        
        splitter.addWidget(self.list_widget)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1) 
        
        main_layout.addWidget(splitter, 1) 
        self.setLayout(main_layout)

    # --- 邏輯處理 ---

    def on_import_clicked(self):
        if not self.data_handler.project_path:
            # 如果還沒建立專案，提示一下
            QMessageBox.warning(self, "提示", "請先建立或開啟一個專案！")
            return

        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if folder:
            # 1. 先取得要匯入的檔案清單 (這一步很快)
            files_to_import = self.data_handler.get_import_list(folder)
            total = len(files_to_import)
            
            if total == 0:
                QMessageBox.information(self, "提示", "該資料夾內沒有圖片！")
                return

            # 2. 建立進度條
            progress = QProgressDialog("正在匯入照片中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal) # 鎖定視窗避免亂按
            progress.setMinimumDuration(0) # 確保進度條立刻顯示
            progress.setValue(0)

            count = 0
            
            # 3. 開始迴圈搬運
            for i, filename in enumerate(files_to_import):
                if progress.wasCanceled():
                    break
                
                source_path = os.path.join(folder, filename)
                
                # 呼叫剛剛寫好的單張複製功能
                if self.data_handler.copy_file_to_project(source_path):
                    count += 1
                
                # 更新進度條
                progress.setValue(i + 1)
                
                # ★ 關鍵：讓介面喘口氣，處理繪圖事件，這樣才不會「白屏」或「轉圈圈」
                QApplication.processEvents()
            
            progress.close()

            # 4. 全部搬完後，重新掃描並刷新畫面
            self.data_handler.scan_unsorted_images()
            self.refresh_ui()
            
            QMessageBox.information(self, "完成", f"成功匯入 {count} 張照片！")

    def refresh_ui(self):
        # 0. 如果有正在跑的 Worker，先停掉，避免衝突
        if self.icon_worker and self.icon_worker.isRunning():
            self.icon_worker.stop()

        self.list_widget.clear()
        images = self.data_handler.scan_unsorted_images()
        self.lbl_info.setText(f"待處理: {len(images)} 張")
        
        # 1. 快速建立「只有文字」的清單 (這個步驟超快，400 張也只要 0.01 秒)
        for path in images:
            filename = os.path.basename(path)
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, path)
            # ★ 注意：這裡先不設 Icon，這樣畫面會瞬間出來
            self.list_widget.addItem(item)

        # 2. 啟動後台小精靈去載入縮圖
        if images:
            self.icon_worker = IconWorker(images)
            self.icon_worker.icon_loaded.connect(self.on_icon_loaded) # 接上訊號
            self.icon_worker.start()

            # 選取第一張
            self.list_widget.setCurrentRow(0)
            self.btn_crop.setEnabled(True)
            self.btn_batch.setEnabled(True)
        else:
            self.image_label.clear_canvas()
            self.current_image_path = None
            self.btn_crop.setEnabled(False)
            self.btn_batch.setEnabled(False)

    # ★ 小精靈回報訊號時執行的函式
    def on_icon_loaded(self, row, image):
        # 找到對應的那一行
        item = self.list_widget.item(row)
        if item:
            # 把讀好的圖片轉成 Icon 放上去
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    def on_item_clicked(self, item):
        path = item.data(Qt.UserRole)
        self.load_image(path)

    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if items:
            path = items[0].data(Qt.UserRole)
            self.load_image(path)

    def load_image(self, path):
        if path and os.path.exists(path):
            self.current_image_path = path
            self.image_label.set_image(path)
        else:
            self.image_label.clear_canvas()

    def apply_crop(self):
        if not self.current_image_path: return
        crop_box = self.image_label.get_crop_rect_original()
        if not crop_box:
            QMessageBox.warning(self, "錯誤", "請先畫框！")
            return
        try:
            img = Image.open(self.current_image_path)
            cropped_img = img.crop(crop_box)
            success = self.data_handler.save_crop_to_roi(cropped_img, self.current_image_path)
            if success:
                self.move_to_next()
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def apply_batch_crop(self):
        crop_rect = self.image_label.last_crop_rect_original
        if not crop_rect:
            QMessageBox.warning(self, "無法執行", "請先選擇一張照片並畫好紅框！")
            return
            
        images = self.data_handler.scan_unsorted_images()
        total = len(images)
        if total == 0: return

        reply = QMessageBox.question(self, "確認批次裁切", 
                                   f"確定要自動裁切剩餘的 {total} 張照片嗎？",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # 停止 icon worker，避免批次處理時它還在後台讀檔案，造成搶資源
            if self.icon_worker and self.icon_worker.isRunning():
                self.icon_worker.stop()

            progress = QProgressDialog("正在批次裁切中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            
            success_count = 0
            
            for i, img_path in enumerate(images):
                if progress.wasCanceled():
                    break
                try:
                    img = Image.open(img_path)
                    cropped_img = img.crop(crop_rect)
                    if self.data_handler.save_crop_to_roi(cropped_img, img_path):
                        success_count += 1
                except Exception as e:
                    print(f"Skipped {img_path}: {e}")
                
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            progress.close()
            QMessageBox.information(self, "完成", f"成功裁切: {success_count} 張")
            self.refresh_ui()

    def move_to_next(self):
        current_row = self.list_widget.currentRow()
        self.list_widget.takeItem(current_row)
        
        count = self.list_widget.count()
        self.lbl_info.setText(f"待處理: {count} 張")
        
        if count > 0:
            if current_row >= count:
                current_row = count - 1
            self.list_widget.setCurrentRow(current_row)
        else:
            self.refresh_ui()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.btn_crop.isEnabled():
                self.apply_crop()
        else:
            super().keyPressEvent(event)
            
    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)