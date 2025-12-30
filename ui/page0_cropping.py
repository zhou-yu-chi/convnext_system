import os
from PIL import Image 
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QMessageBox, QFileDialog, QListWidget, 
                             QListWidgetItem, QSplitter, QSizePolicy, QFrame, 
                             QProgressDialog, QApplication, QComboBox)
from PySide6.QtCore import Qt, QRect, QPoint, QSize, QThread, Signal 
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QIcon, QImage, QImageReader

# ==========================================
# 0. 後台縮圖載入小精靈 (維持不變)
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

# ==========================================
# 1. 增強版 Label (支援多重裁切框)
# ==========================================
class CroppableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent) 
        self.start_point = None
        self.end_point = None
        self.is_drawing = False         
        self.is_moving_box = False      
        self.move_offset = QPoint(0,0)  

        self.setCursor(Qt.CursorShape.CrossCursor)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 儲存多個框的列表，每個元素為 (x, y, w, h)
        self.current_rois = [] 
        
        self.mode = "free" 

    def set_mode(self, mode, params=None):
        self.mode = mode
        self.current_rois = [] # 清空當前框
        
        # params 若傳入，格式統一為 list of tuples: [(x,y,w,h), (x,y,w,h)...]
        if params:
            # 如果是單一 tuple (x,y,w,h) 或 (w,h)，轉成 list
            if isinstance(params, tuple):
                self.current_rois = [params]
            elif isinstance(params, list):
                self.current_rois = params
        
        self.update_display()

    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.update_display()

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
        
        # 如果是固定模式，重置框的位置 (不支援記憶上次位置，避免切換圖片時框跑掉)
        # 對於多框模式，我們直接使用傳入的座標，不需要額外計算
        self.update()

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
        
        # 繪製所有框
        rects_to_draw = []
        
        if self.mode == "free":
            if self.start_point and self.end_point:
                rects_to_draw.append(QRect(self.start_point, self.end_point).normalized())
        else:
            # 固定模式 (單框或多框)
            for roi in self.current_rois:
                # 解析 ROI (相容不同長度的 tuple)
                if len(roi) == 4:
                    rx, ry, rw, rh = roi
                else: # 只有寬高
                    continue

                # 轉換為螢幕座標
                sx = int(round(rx / self.scale_factor)) + self.offset_x
                sy = int(round(ry / self.scale_factor)) + self.offset_y
                sw = int(round(rw / self.scale_factor))
                sh = int(round(rh / self.scale_factor))
                
                # 如果正在移動，加上位移量 (Apply Offset)
                if self.is_moving_box:
                     # 簡單實作：移動時所有框一起動，這裡是算出移動後的左上角
                     # 因為 start_point 是滑鼠點下去時所有框的基準點，這裡簡化處理：
                     # 我們實際上是修改 self.current_rois 的值比較好，但在 paintEvent 不改值
                     # 改用 mouseMove 動態更新 current_rois
                     pass

                rects_to_draw.append(QRect(sx, sy, sw, sh))

        # 實際畫出
        for i, rect in enumerate(rects_to_draw):
            pen = QPen(QColor(255, 50, 50), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(255, 0, 0, 30))

            # 顯示編號與尺寸
            w_real = int(rect.width() * self.scale_factor)
            h_real = int(rect.height() * self.scale_factor)
            text = f"#{i+1}: {w_real}x{h_real}"
            
            painter.setPen(QColor(255, 255, 255))
            text_pos = rect.topLeft()
            text_pos.setY(text_pos.y() - 5)
            if text_pos.y() < 20: text_pos.setY(rect.bottom() + 15)

            painter.drawText(text_pos, text)

    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            
            if self.mode == "free":
                self.is_drawing = True
                self.start_point = pos
                self.end_point = pos
            else:
                # 固定模式：點擊開始移動 (Group Move)
                self.is_moving_box = True
                self.start_point = pos # 記錄滑鼠起始點
                
            self.update()

    def mouseMoveEvent(self, event):
        if not self.original_pixmap: return
        pos = event.position().toPoint()

        if self.mode == "free" and self.is_drawing:
            self.end_point = pos
            self.update()
            
        elif self.mode != "free" and self.is_moving_box:
            # 計算滑鼠位移量
            dx = (pos.x() - self.start_point.x()) * self.scale_factor
            dy = (pos.y() - self.start_point.y()) * self.scale_factor
            
            # 更新所有 ROI 的真實座標
            new_rois = []
            for roi in self.current_rois:
                if len(roi) == 4:
                    rx, ry, rw, rh = roi
                    new_rois.append((rx + dx, ry + dy, rw, rh))
            
            self.current_rois = new_rois
            self.start_point = pos # 更新基準點
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            self.is_moving_box = False
            self.update()

    def get_crop_rects(self):
        """回傳所有裁切框的真實座標列表 [(x,y,x2,y2), ...]"""
        results = []
        
        # 1. 自由模式
        if self.mode == "free":
            if self.start_point and self.end_point:
                # 計算螢幕上的框
                screen_rect = QRect(self.start_point, self.end_point).normalized()
                # 限制範圍
                img_rect = QRect(self.offset_x, self.offset_y, self.scaled_pixmap.width(), self.scaled_pixmap.height())
                intersect = screen_rect.intersected(img_rect)
                
                x = (intersect.x() - self.offset_x) * self.scale_factor
                y = (intersect.y() - self.offset_y) * self.scale_factor
                w = intersect.width() * self.scale_factor
                h = intersect.height() * self.scale_factor
                
                if w > 0 and h > 0:
                    results.append((int(x), int(y), int(x+w), int(y+h)))
        
        # 2. 固定模式 (包含多框)
        else:
            for roi in self.current_rois:
                if len(roi) == 4:
                    rx, ry, rw, rh = roi
                    # 邊界檢查
                    if rx < 0: rx = 0
                    if ry < 0: ry = 0
                    if rx + rw > self.original_pixmap.width(): rx = self.original_pixmap.width() - rw
                    if ry + rh > self.original_pixmap.height(): ry = self.original_pixmap.height() - rh
                    
                    results.append((int(rx), int(ry), int(rx+rw), int(ry+rh)))
                    
        return results
    
    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def clear_canvas(self):
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.start_point = None
        self.end_point = None
        self.current_rois = []
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
        self.icon_worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(10)
        
        # 工具列 1
        top_bar_container = QFrame()
        top_bar_container.setStyleSheet("QFrame { background-color: #333; border-radius: 8px; padding: 2px; }")
        top_bar = QHBoxLayout(top_bar_container)
        
        self.btn_import = QPushButton(" 📥 匯入照片")
        self.btn_import.setStyleSheet("QPushButton { background-color: #0277bd; color: white; padding: 5px 15px; border-radius: 5px; font-weight: bold; }")
        self.btn_import.clicked.connect(self.on_import_clicked)
        
        self.lbl_info = QLabel("等待匯入...")
        self.lbl_info.setStyleSheet("color: #ddd; margin-left: 10px;")
        
        top_bar.addWidget(self.btn_import)
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()
        
        # 工具列 2：模式選擇
        mode_bar_container = QFrame()
        mode_bar_container.setStyleSheet("QFrame { background-color: #2b2b2b; border-radius: 8px; border: 1px solid #444; }")
        mode_bar = QHBoxLayout(mode_bar_container)
        mode_bar.setContentsMargins(10, 5, 10, 5)

        lbl_mode = QLabel("裁切模式:")
        lbl_mode.setStyleSheet("color: #fff; font-weight: bold; border: none;")
        
        self.combo_mode = QComboBox()
        # 加入新的 3-1 和 3-3 選項
        self.combo_mode.addItems([
            "✏️ 自定義自由框 (Free)", 
            "合金 2-3", 
            "合金 2-5", 
            "紙片 2-3",
            "紙片 2-6",
            "3-1 (雙裁切)",   # 新增
            "3-3 (雙裁切)"    # 新增
        ])
        self.combo_mode.setStyleSheet("""
            QComboBox { background-color: #444; color: white; padding: 5px; border-radius: 4px; min-width: 150px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #444; color: white; selection-background-color: #00796b; }
        """)
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)

        lbl_hint = QLabel("(拖曳畫面可移動框)")
        lbl_hint.setStyleSheet("color: #888; font-style: italic; border: none; font-size: 12px;")

        mode_bar.addWidget(lbl_mode)
        mode_bar.addWidget(self.combo_mode)
        mode_bar.addWidget(lbl_hint)
        mode_bar.addStretch()

        main_layout.addWidget(top_bar_container)
        main_layout.addWidget(mode_bar_container)

        # 分割視窗
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(70, 70)) 
        self.list_widget.setFixedWidth(240)
        self.list_widget.setStyleSheet("QListWidget { background-color: #2b2b2b; border: 1px solid #444; border-radius: 8px; color: #eee; }")
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        right_container = QFrame()
        right_container.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = CroppableLabel()
        self.image_label.setStyleSheet("background-color: transparent;") 
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        btn_bar = QFrame()
        btn_bar.setStyleSheet("background-color: #333; border-top: 1px solid #555; border-radius: 0px;")
        btn_bar.setMaximumHeight(60) 
        btn_layout = QHBoxLayout(btn_bar)
        
        self.btn_batch = QPushButton("⚡ 一鍵裁切全部")
        self.btn_batch.setStyleSheet("background-color: #7b1fa2; color: white; border-radius: 5px; padding: 5px 15px; font-weight: bold;")
        self.btn_batch.clicked.connect(self.apply_batch_crop) 
        
        self.btn_crop = QPushButton("✂️ 裁切 (Enter)")
        self.btn_crop.setStyleSheet("background-color: #ef6c00; color: white; border-radius: 5px; padding: 5px 20px; font-weight: bold;")
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

    def on_mode_changed(self, index):
        """處理下拉選單切換，設定對應的座標 (x, y, w, h)"""
        if index == 0:
            self.image_label.set_mode("free")
        elif index == 1: # 合金 2-3
            self.image_label.set_mode("fixed", (200, 130, 260, 240))
        elif index == 2: # 合金 2-5
            self.image_label.set_mode("fixed", (480, 400, 330, 420))
        elif index == 3: # 紙片 2-3
            self.image_label.set_mode("fixed", (350, 170, 500, 625))
        elif index == 4: # 紙片 2-6
            self.image_label.set_mode("fixed", (410, 445, 350, 390))
        elif index == 5: # 3-1 (雙裁切)
            # 傳入 List 包含兩個 tuple
            self.image_label.set_mode("fixed", [
                (250, 570, 180, 140), # Crop 1
                (680, 410, 160, 130)  # Crop 2
            ])
        elif index == 6: # 3-3 (雙裁切)
            self.image_label.set_mode("fixed", [
                (340, 580, 170, 130), # Crop 1
                (790, 700, 190, 140)  # Crop 2
            ])

    def on_import_clicked(self):
        if not self.data_handler.project_path:
            QMessageBox.warning(self, "提示", "請先建立或開啟一個專案！")
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if folder:
            files_to_import = self.data_handler.get_import_list(folder)
            total = len(files_to_import)
            if total == 0:
                QMessageBox.information(self, "提示", "該資料夾內沒有圖片！")
                return
            
            # 檢查重複
            duplicates_count = 0
            for filename in files_to_import:
                dest_path = os.path.join(self.data_handler.project_path, filename)
                if os.path.exists(dest_path): duplicates_count += 1
            
            should_rename_all = False
            if duplicates_count > 0:
                reply = QMessageBox.question(self, "發現重複檔案", f"偵測到 {duplicates_count} 張照片檔名重複！\n是否自動改名並匯入？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes: should_rename_all = True
            
            progress = QProgressDialog("正在匯入照片中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            
            count = 0
            for i, filename in enumerate(files_to_import):
                if progress.wasCanceled(): break
                source_path = os.path.join(folder, filename)
                result = self.data_handler.copy_file_to_project(source_path, rename_if_exists=should_rename_all)
                if result is True: count += 1
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            progress.close()
            self.data_handler.scan_unsorted_images()
            self.refresh_ui()
            QMessageBox.information(self, "完成", f"成功匯入 {count} 張照片！")

    def refresh_ui(self):
        if self.icon_worker and self.icon_worker.isRunning(): self.icon_worker.stop()
        self.list_widget.clear()
        images = self.data_handler.scan_unsorted_images()
        self.lbl_info.setText(f"待處理: {len(images)} 張")
        
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
            self.btn_crop.setEnabled(True)
            self.btn_batch.setEnabled(True)
        else:
            self.image_label.clear_canvas()
            self.current_image_path = None
            self.btn_crop.setEnabled(False)
            self.btn_batch.setEnabled(False)

    def on_icon_loaded(self, row, image):
        item = self.list_widget.item(row)
        if item: item.setIcon(QIcon(QPixmap.fromImage(image)))

    def on_item_clicked(self, item):
        path = item.data(Qt.UserRole)
        self.load_image(path)

    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if items: self.load_image(items[0].data(Qt.UserRole))

    def load_image(self, path):
        if path and os.path.exists(path):
            self.current_image_path = path
            self.image_label.set_image(path)
        else:
            self.image_label.clear_canvas()

    def apply_crop(self):
        if not self.current_image_path: return
        crop_rects = self.image_label.get_crop_rects() # 取得所有框
        if not crop_rects:
            QMessageBox.warning(self, "錯誤", "請先畫框！")
            return
        
        try:
            img = Image.open(self.current_image_path)
            success_count = 0
            
            # === 多框處理邏輯 ===
            # DataHandler.save_crop_to_roi 會刪除原始檔案
            # 所以如果有兩個以上的框，前 N-1 個要手動存檔，最後一個才呼叫 DataHandler
            
            roi_folder = os.path.join(self.data_handler.project_path, "ROI")
            base_name = os.path.basename(self.current_image_path)
            
            for i, rect in enumerate(crop_rects):
                cropped_img = img.crop(rect)
                
                # 如果這是最後一張裁切，呼叫 DataHandler (會觸發刪除原檔)
                if i == len(crop_rects) - 1:
                    if self.data_handler.save_crop_to_roi(cropped_img, self.current_image_path):
                        success_count += 1
                else:
                    # 如果不是最後一張，我們手動儲存到 ROI 資料夾，避免原檔被刪除
                    # 產生唯一檔名，例如 image_1.jpg
                    name_part, ext = os.path.splitext(base_name)
                    new_name = f"{name_part}_{i+1}{ext}"
                    save_path = self.data_handler.generate_unique_path(roi_folder, new_name)
                    cropped_img.save(save_path)
                    success_count += 1
            
            if success_count > 0:
                self.move_to_next()
                
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def apply_batch_crop(self):
        # 取得當前設定的所有框
        crop_rects = self.image_label.get_crop_rects()
        if not crop_rects:
            QMessageBox.warning(self, "無法執行", "請先選擇一張照片並畫好框！")
            return
            
        images = self.data_handler.scan_unsorted_images()
        total = len(images)
        if total == 0: return
        
        reply = QMessageBox.question(self, "確認批次裁切", 
                                   f"確定要依據目前的設定 (共 {len(crop_rects)} 個框)\n自動裁切剩餘的 {total} 張照片嗎？", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.icon_worker and self.icon_worker.isRunning(): self.icon_worker.stop()
            
            progress = QProgressDialog("正在批次裁切中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            
            roi_folder = os.path.join(self.data_handler.project_path, "ROI")
            
            for idx, img_path in enumerate(images):
                if progress.wasCanceled(): break
                try:
                    img = Image.open(img_path)
                    base_name = os.path.basename(img_path)
                    
                    for i, rect in enumerate(crop_rects):
                        cropped_img = img.crop(rect)
                        
                        # 邏輯同上：最後一張才刪除原始圖
                        if i == len(crop_rects) - 1:
                            self.data_handler.save_crop_to_roi(cropped_img, img_path)
                        else:
                            name_part, ext = os.path.splitext(base_name)
                            new_name = f"{name_part}_{i+1}{ext}"
                            save_path = self.data_handler.generate_unique_path(roi_folder, new_name)
                            cropped_img.save(save_path)
                            
                except: pass
                
                progress.setValue(idx + 1)
                QApplication.processEvents()
                
            progress.close()
            # 批次完成後重新掃描
            self.data_handler.scan_roi_images() 
            self.refresh_ui()

    def move_to_next(self):
        current_row = self.list_widget.currentRow()
        self.list_widget.takeItem(current_row)
        count = self.list_widget.count()
        self.lbl_info.setText(f"待處理: {count} 張")
        if count > 0:
            if current_row >= count: current_row = count - 1
            self.list_widget.setCurrentRow(current_row)
        else:
            self.refresh_ui()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.btn_crop.isEnabled(): self.apply_crop()
        else:
            super().keyPressEvent(event)
            
    def showEvent(self, event):
        self.refresh_ui()
        super().showEvent(event)