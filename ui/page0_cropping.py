import os
from PIL import Image 
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QMessageBox, QFileDialog, QListWidget, 
                             QListWidgetItem, QSplitter, QSizePolicy, QFrame, 
                             QProgressDialog, QApplication, QComboBox, QGroupBox, QRadioButton)
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
# 1. 增強版 Label (核心邏輯升級)
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
        self.last_crop_rect_original = None 

        self.mode = "free" 
        self.fixed_size = (100, 100) 

    def set_mode(self, mode, size=None):
        self.mode = mode
        if size:
            self.fixed_size = size
        
        # 如果有圖片且模式是 fixed
        if self.original_pixmap and mode == "fixed":
            # ★ 新增邏輯：如果是 (x, y, w, h) 四個參數，直接定位
            if len(self.fixed_size) == 4:
                self.apply_fixed_pos()
            else:
                # 舊邏輯：只有寬高，放在畫面中間等待移動
                cx = self.width() // 2
                cy = self.height() // 2
                w_screen = int(round(self.fixed_size[0] / self.scale_factor))
                h_screen = int(round(self.fixed_size[1] / self.scale_factor))
                self.start_point = QPoint(cx - w_screen//2, cy - h_screen//2)
                self.end_point = QPoint(cx + w_screen//2, cy + h_screen//2)
                self.update()

    # ★ 新增這個函式：負責把原本圖片的 xy 轉換成螢幕上的 xy
    def apply_fixed_pos(self):
        if len(self.fixed_size) != 4: return
        
        # 解包 (x, y, w, h)
        real_x, real_y, real_w, real_h = self.fixed_size
        
        # 換算成螢幕座標
        sx = int(round(real_x / self.scale_factor)) + self.offset_x
        sy = int(round(real_y / self.scale_factor)) + self.offset_y
        sw = int(round(real_w / self.scale_factor))
        sh = int(round(real_h / self.scale_factor))
        
        self.start_point = QPoint(sx, sy)
        self.end_point = QPoint(sx + sw, sy + sh)
        self.last_crop_rect_original = self.get_crop_rect_original()
        self.update()

    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.update_display()
        
        if self.last_crop_rect_original:
            self.restore_crop_box()
        else:
            if self.mode == "fixed":
                # 如果是新圖片且在固定模式，直接初始化一個標準框
                self.set_mode(self.mode, self.fixed_size)
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
        
        # ★ 修改這裡：優先判斷是否為「固定座標模式」
        if self.mode == "fixed" and len(self.fixed_size) == 4:
            self.apply_fixed_pos()
        elif self.last_crop_rect_original:
            self.restore_crop_box()
        self.update()

    def restore_crop_box(self):
        if not self.last_crop_rect_original or not self.scale_factor: return
        rx1, ry1, rx2, ry2 = self.last_crop_rect_original
        
        # ★ 修正2：還原座標時使用 round，避免切換圖片時框越變越小或越大
        sx1 = int(round(rx1 / self.scale_factor)) + self.offset_x
        sy1 = int(round(ry1 / self.scale_factor)) + self.offset_y
        sx2 = int(round(rx2 / self.scale_factor)) + self.offset_x
        sy2 = int(round(ry2 / self.scale_factor)) + self.offset_y
        
        self.start_point = QPoint(sx1, sy1)
        self.end_point = QPoint(sx2, sy2)

    def paintEvent(self, event):
        if not self.scaled_pixmap:
            # ... (保持原本的文字顯示邏輯) ...
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
            pen = QPen(QColor(255, 50, 50), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(255, 0, 0, 30))

            # 顯示尺寸文字
            original_rect = self.get_crop_rect_original()
            if original_rect:
                rx1, ry1, rx2, ry2 = original_rect
                w = rx2 - rx1
                h = ry2 - ry1
                text = f"{w} x {h}"
                
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(painter.font())
                text_pos = rect.topLeft()
                text_pos.setY(text_pos.y() - 5)
                if text_pos.y() < 20:
                    text_pos = rect.bottomLeft()
                    text_pos.setY(text_pos.y() + 15)
                
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(text)
                th = fm.height()
                painter.fillRect(text_pos.x(), text_pos.y() - th + 5, tw + 6, th, QColor(0, 0, 0, 180))
                painter.drawText(text_pos, text)

    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            
            if self.mode == "free":
                self.is_drawing = True
                self.start_point = pos
                self.end_point = pos
                
            elif self.mode == "fixed":
                # ★★★ 修正這裡：判斷是 4 個參數 (x,y,w,h) 還是 2 個參數 (w,h) ★★★
                if len(self.fixed_size) == 4:
                    # 如果是 4 個參數，寬高在 index 2 和 3
                    fw, fh = self.fixed_size[2], self.fixed_size[3]
                else:
                    # 如果是 2 個參數，寬高在 index 0 和 1
                    fw, fh = self.fixed_size[0], self.fixed_size[1]

                w_screen = int(round(fw / self.scale_factor))
                h_screen = int(round(fh / self.scale_factor))
                
                # 設定框框中心點為滑鼠點擊處
                self.start_point = QPoint(pos.x() - w_screen//2, pos.y() - h_screen//2)
                self.end_point = QPoint(pos.x() + w_screen//2, pos.y() + h_screen//2)
                
                self.is_moving_box = True
                self.move_offset = QPoint(w_screen//2, h_screen//2)
                
            self.update()

    def mouseMoveEvent(self, event):
        if not self.original_pixmap: return
        pos = event.position().toPoint()

        if self.mode == "free" and self.is_drawing:
            self.end_point = pos
            self.update()
            
        elif self.mode == "fixed" and self.is_moving_box:
            # ★ 修改：同樣判斷寬高來源
            if len(self.fixed_size) == 4:
                fw, fh = self.fixed_size[2], self.fixed_size[3]
            else:
                fw, fh = self.fixed_size[0], self.fixed_size[1]

            w_screen = int(round(fw / self.scale_factor))
            h_screen = int(round(fh / self.scale_factor))
            
            new_start = pos - self.move_offset
            self.start_point = new_start
            self.end_point = QPoint(new_start.x() + w_screen, new_start.y() + h_screen)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            self.is_moving_box = False
            if self.mode == "free":
                self.end_point = event.position().toPoint()
            self.update()
            self.last_crop_rect_original = self.get_crop_rect_original()

    def get_crop_rect_original(self):
        """計算原始圖片上的裁切座標 (回傳真實尺寸)"""
        if not self.start_point or not self.end_point:
            return None
            
        # 1. 計算螢幕上的框
        screen_rect = QRect(self.start_point, self.end_point).normalized()
        
        # 2. 限制在圖片顯示範圍內
        img_rect = QRect(self.offset_x, self.offset_y, self.scaled_pixmap.width(), self.scaled_pixmap.height())
        intersect_rect = screen_rect.intersected(img_rect)
        
        # 轉換為相對於圖片的座標
        x = intersect_rect.x() - self.offset_x
        y = intersect_rect.y() - self.offset_y
        w = intersect_rect.width()
        h = intersect_rect.height()
        
        if w <= 0 or h <= 0: return None
        
        # ★ 修正5：座標轉換使用 round 四捨五入
        real_x = int(round(x * self.scale_factor))
        real_y = int(round(y * self.scale_factor))
        
        if self.mode == "fixed":
            # ★ 修改：強制鎖定寬高
            if len(self.fixed_size) == 4:
                real_w, real_h = self.fixed_size[2], self.fixed_size[3]
            else:
                real_w, real_h = self.fixed_size[0], self.fixed_size[1]
            
            # 邊界檢查 (保持原本邏輯)
            if real_x + real_w > self.original_pixmap.width():
                real_x = self.original_pixmap.width() - real_w
            if real_y + real_h > self.original_pixmap.height():
                real_y = self.original_pixmap.height() - real_h
            if real_x < 0: real_x = 0
            if real_y < 0: real_y = 0
            
            return (real_x, real_y, real_x + real_w, real_y + real_h)
        else:
            # Free mode
            w = intersect_rect.width()
            h = intersect_rect.height()
            real_w = int(round(w * self.scale_factor))
            real_h = int(round(h * self.scale_factor))
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
# 2. 主頁面 (加入工具列控制項)
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
        
        # --- 1. 頂部工具列 (第一排) ---
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
        
        # --- ★★★ 新增：裁切模式工具列 (第二排) ★★★ ---
        mode_bar_container = QFrame()
        mode_bar_container.setStyleSheet("QFrame { background-color: #2b2b2b; border-radius: 8px; border: 1px solid #444; }")
        mode_bar = QHBoxLayout(mode_bar_container)
        mode_bar.setContentsMargins(10, 5, 10, 5)

        lbl_mode = QLabel("裁切模式:")
        lbl_mode.setStyleSheet("color: #fff; font-weight: bold; border: none;")
        
        # 模式下拉選單
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["✏️ 自定義自由框 (Free)", "合金 2-3", "合金 2-5", "紙 2-3","紙2-6"])
        self.combo_mode.setStyleSheet("""
            QComboBox { background-color: #444; color: white; padding: 5px; border-radius: 4px; min-width: 150px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #444; color: white; selection-background-color: #00796b; }
        """)
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)

        lbl_hint = QLabel("(固定模式下：點擊畫面可放置框，拖曳可移動框)")
        lbl_hint.setStyleSheet("color: #888; font-style: italic; border: none; font-size: 12px;")

        mode_bar.addWidget(lbl_mode)
        mode_bar.addWidget(self.combo_mode)
        mode_bar.addWidget(lbl_hint)
        mode_bar.addStretch()

        main_layout.addWidget(top_bar_container)
        main_layout.addWidget(mode_bar_container) # 加入第二排

        # --- 2. 中間區域 ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        # 左側清單
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(70, 70)) 
        self.list_widget.setFixedWidth(240)
        self.list_widget.setStyleSheet("QListWidget { background-color: #2b2b2b; border: 1px solid #444; border-radius: 8px; color: #eee; }")
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 右側畫布
        right_container = QFrame()
        right_container.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = CroppableLabel()
        self.image_label.setStyleSheet("background-color: transparent;") 
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 底部按鈕
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
        """處理下拉選單切換"""
        if index == 0:
            self.image_label.set_mode("free")
            
        elif index == 1:
            # ★★★ 修改這裡：傳入 (x, y, w, h) ★★★
            # 範例：x=300, y=100, 寬=110, 高=100
            self.image_label.set_mode("fixed", (250, 180, 160, 140))
            
        elif index == 2:
            # 範例：x=0, y=0, 寬=200, 高=200 (從左上角開始)
            self.image_label.set_mode("fixed", (530, 450, 230, 320))
            
        elif index == 3:
            # 舊的寫法 (只有寬高) 依然支援，會變成需點擊放置
            self.image_label.set_mode("fixed", (400, 220,400,525))
        elif index == 4:
            # 舊的寫法 (只有寬高) 依然支援，會變成需點擊放置
            self.image_label.set_mode("fixed", (460, 495,250,290))
    
    def on_import_clicked(self):
        # (請複製您原本的 on_import_clicked 代碼)
        super().on_import_clicked() if hasattr(super(), 'on_import_clicked') else None # 僅示意，請貼上原代碼

    # 這裡我把原本的 refresh_ui 等函式簡寫，您直接用原本的即可
    # 唯一要注意的是，如果您原本是用 self.btn_crop.clicked.connect... 綁定
    # 記得確認上面的 init_ui 已經綁定好了
    
    # 為了讓您方便複製，我把剩下的關鍵函式貼上：

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
            duplicates_count = 0
            for filename in files_to_import:
                dest_path = os.path.join(self.data_handler.project_path, filename)
                if os.path.exists(dest_path): duplicates_count += 1
            should_rename_all = False
            if duplicates_count > 0:
                reply = QMessageBox.question(self, "發現重複檔案", f"偵測到 {duplicates_count} 張照片檔名重複！\n是否自動改名並匯入？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes: should_rename_all = True
                else: should_rename_all = False
            
            progress = QProgressDialog("正在匯入照片中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
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
        crop_box = self.image_label.get_crop_rect_original()
        if not crop_box:
            QMessageBox.warning(self, "錯誤", "請先畫框！")
            return
        try:
            img = Image.open(self.current_image_path)
            cropped_img = img.crop(crop_box)
            success = self.data_handler.save_crop_to_roi(cropped_img, self.current_image_path)
            if success: self.move_to_next()
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
        reply = QMessageBox.question(self, "確認批次裁切", f"確定要自動裁切剩餘的 {total} 張照片嗎？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.icon_worker and self.icon_worker.isRunning(): self.icon_worker.stop()
            progress = QProgressDialog("正在批次裁切中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            success_count = 0
            for i, img_path in enumerate(images):
                if progress.wasCanceled(): break
                try:
                    img = Image.open(img_path)
                    cropped_img = img.crop(crop_rect)
                    if self.data_handler.save_crop_to_roi(cropped_img, img_path): success_count += 1
                except: pass
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