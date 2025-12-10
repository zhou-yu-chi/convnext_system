import os
from PIL import Image 
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QMessageBox, QFileDialog, QListWidget, 
                             QListWidgetItem, QSplitter, QSizePolicy, QFrame, 
                             QProgressDialog, QApplication) # 增加 QProgressDialog
from PySide6.QtCore import Qt, QRect, QPoint, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QIcon

# ==========================================
# 1. 增強版 Label：支援記憶裁切框 (功能不變)
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
# 2. 主頁面：調整版面比例 + 新增批次功能
# ==========================================
class Page0_Cropping(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler
        self.current_image_path = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) 
        main_layout.setSpacing(10)
        
        # --- 1. 頂部工具列 (壓縮高度) ---
        top_bar_container = QFrame()
        # 強制設定最大高度，讓它不會佔用太多空間
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

        # --- 2. 中間區域 (放大比例) ---
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
        btn_bar.setMaximumHeight(60) # 限制高度
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
        self.btn_batch.clicked.connect(self.apply_batch_crop) # 綁定新功能
        
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
        
        # 這裡設定 stretch=1，確保 splitter 佔據剩餘所有垂直空間
        main_layout.addWidget(splitter, 1) 
        self.setLayout(main_layout)

    # --- 邏輯處理 ---

    def on_import_clicked(self):
        if not self.data_handler.project_path: return
        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if folder:
            count = self.data_handler.import_images_from_folder(folder)
            if count > 0:
                self.refresh_ui()

    def refresh_ui(self):
        self.list_widget.clear()
        images = self.data_handler.scan_unsorted_images()
        self.lbl_info.setText(f"待處理: {len(images)} 張")
        
        for path in images:
            filename = os.path.basename(path)
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, path) 
            item.setIcon(QIcon(path))
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            self.btn_crop.setEnabled(True)
            self.btn_batch.setEnabled(True)
        else:
            self.image_label.clear_canvas()
            self.current_image_path = None
            self.btn_crop.setEnabled(False)
            self.btn_batch.setEnabled(False)

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
        # 1. 檢查是否有記憶的裁切框
        crop_rect = self.image_label.last_crop_rect_original
        if not crop_rect:
            QMessageBox.warning(self, "無法執行", "請先選擇一張照片並畫好紅框，\n程式才能知道要用多大的範圍去裁切其他照片！")
            return
            
        # 2. 取得所有待處理圖片
        images = self.data_handler.scan_unsorted_images()
        total = len(images)
        if total == 0: return

        reply = QMessageBox.question(self, "確認批次裁切", 
                                   f"確定要使用目前的紅框設定，\n自動裁切剩餘的 {total} 張照片嗎？\n(這將會直接存入 ROI 並刪除原檔)",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # 建立進度條
            progress = QProgressDialog("正在批次裁切中...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            
            success_count = 0
            
            # 3. 執行迴圈
            # 因為 data_handler.save_crop_to_roi 會刪除檔案，所以我們對 images 副本進行操作是安全的
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
                # 讓介面保持回應
                QApplication.processEvents()
            
            progress.close()
            
            # 4. 完成後刷新
            QMessageBox.information(self, "完成", f"批次處理結束！\n成功裁切: {success_count} 張")
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