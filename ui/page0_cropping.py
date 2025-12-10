import os
from PIL import Image #PILLOW庫用於裁切
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QMessageBox, QFileDialog) #視窗介面
from PySide6.QtCore import Qt, QRect  #處裡座標
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor  #繪圖相關

#繼承QLabel以實現可裁切功能
class CroppableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent) 
        self.start_point = None  #滑鼠起點
        self.end_point = None  #滑鼠終點
        self.is_drawing = False  #是否正在拖曳滑鼠
        self.setCursor(Qt.CursorShape.CrossCursor)  #設定游標為十字準心
        self.original_pixmap = None  #原始圖片
        self.scaled_pixmap = None  #縮放後圖片
        self.scale_factor = 1.0  #縮放比例
        self.offset_x = 0   #左右留白
        self.offset_y = 0  #上下留白
    
    #載入圖片
    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)  #讀取圖片
        self.update_display()  #計算縮放
        self.start_point = None #重置裁切框
        self.end_point = None 
        self.update() #重繪

    #計算放置與治中
    def update_display(self):
        if not self.original_pixmap: return  #如果沒圖就跳過

        #取得目前視窗大小
        w_limit = self.width()
        h_limit = self.height()

        #把原圖等比例放大
        self.scaled_pixmap = self.original_pixmap.scaled(w_limit, h_limit, 
                                                       Qt.AspectRatioMode.KeepAspectRatio, 
                                                       Qt.TransformationMode.SmoothTransformation)
        #計算縮放比例與留白
        self.scale_factor = self.original_pixmap.width() / self.scaled_pixmap.width()
        self.offset_x = (self.width() - self.scaled_pixmap.width()) // 2
        self.offset_y = (self.height() - self.scaled_pixmap.height()) // 2
        self.update()

    # 這是 Qt 系統自動呼叫的「繪圖函式」
    def paintEvent(self, event):
        # 如果沒有縮圖 (scaled_pixmap 是 None)，就直接離開，不要畫任何東西
        if not self.scaled_pixmap:
            super().paintEvent(event) # 畫背景文字 (例如: "無待處理照片")
            return
            
        painter = QPainter(self)
        painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)
        
        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

    # 處理滑鼠事件
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:# 如果按左鍵
            self.is_drawing = True
            self.start_point = event.position().toPoint()# 記錄起點
            self.end_point = self.start_point
            self.update()# 重畫 (顯示紅點)

    def mouseMoveEvent(self, event):
        if self.is_drawing:# 如果正在拖曳
            self.end_point = event.position().toPoint()# 更新終點
            self.update()# 重畫 (顯示動態框框)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:# 放開左鍵
            self.is_drawing = False
            self.end_point = event.position().toPoint()# 確認最終終點
            self.update()

    #座標轉換：取得裁切框在原始圖片的座標
    def get_crop_rect_original(self):
        if not self.start_point or not self.end_point:
            return None
        # 1. 取得螢幕上紅框的座標
        screen_rect = QRect(self.start_point, self.end_point).normalized()
        # 2. 扣掉留白 (Offset)，算出相對於「圖片左上角」的座標
        x = screen_rect.x() - self.offset_x
        y = screen_rect.y() - self.offset_y
        w = screen_rect.width()
        h = screen_rect.height()
        # 防止座標變成負數 (例如畫到留白處)
        if x < 0: x = 0
        if y < 0: y = 0
        # 3. 乘上倍率 (還原回原始解析度)
        real_x = int(x * self.scale_factor)
        real_y = int(y * self.scale_factor)
        real_w = int(w * self.scale_factor)
        real_h = int(h * self.scale_factor)
        if real_w <= 0 or real_h <= 0: return None
        # 回傳給 Pillow 裁切用的座標 (左, 上, 右, 下)
        return (real_x, real_y, real_x + real_w, real_y + real_h)
    
    #如果視窗大小改變，更新顯示
    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def clear_canvas(self):
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.start_point = None
        self.end_point = None
        self.clear()   # 清除 QLabel 的文字或圖片
        self.update()  # 強制觸發 paintEvent 重畫 (會變成空白)

class Page0_Cropping(QWidget):
    def __init__(self, data_handler):
        super().__init__()
        self.data_handler = data_handler # 這是我們的「管家」，負責搬檔案
        self.current_image_path = None # 紀錄現在螢幕上是哪張圖的路徑
        self.init_ui()  # 呼叫介面排版

    def init_ui(self):
        layout = QVBoxLayout() # 垂直排列
        
        # --- 頂部區塊 ---
        top_bar = QHBoxLayout()  # 水平排列
        self.btn_import = QPushButton("📥 匯入照片 (至根目錄)")
        self.btn_import.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold; padding: 8px;")
        self.btn_import.clicked.connect(self.on_import_clicked) # 綁定按鈕功能
        top_bar.addWidget(self.btn_import)
        
        self.lbl_info = QLabel("等待匯入...")
        self.lbl_info.setStyleSheet("color: #aaa; margin-left: 10px;")
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()  # 塞一個彈簧，把按鈕擠到左邊
        layout.addLayout(top_bar)

        # --- 中間區塊：放入剛剛寫好的畫布 ---
        self.image_label = CroppableLabel()
        self.image_label.setStyleSheet("border: 2px dashed #555; background-color: #222;")
        layout.addWidget(self.image_label, 1)

        # --- 底部區塊：操作按鈕 ---
        btn_layout = QHBoxLayout()
        self.btn_skip = QPushButton("⏭️ 不裁切 (直接存入 ROI)")
        self.btn_skip.setMinimumHeight(50)
        self.btn_skip.clicked.connect(self.skip_image)
        
        self.btn_crop = QPushButton("✂️ 裁切並存入 ROI")
        self.btn_crop.setStyleSheet("background-color: #ef6c00; font-weight: bold; font-size: 16px;")
        self.btn_crop.setMinimumHeight(50)
        self.btn_crop.clicked.connect(self.apply_crop)

        btn_layout.addWidget(self.btn_skip)
        btn_layout.addWidget(self.btn_crop)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    # 當按下「匯入」
    def on_import_clicked(self):
        if not self.data_handler.project_path: return # 如果沒開專案就不理
        
        # 跳出選擇資料夾視窗
        folder = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if folder:
            # 叫管家把照片複製進來
            count = self.data_handler.import_images_from_folder(folder)
            if count > 0:
                self.refresh_ui() # 有新照片了，刷新畫面

    # ★ 刷新畫面：永遠只拿第一張 (Queue 模式)
    def refresh_ui(self):
        # 1. 叫管家去根目錄掃描看看還有幾張圖
        images = self.data_handler.scan_unsorted_images()
        
        count = len(images)
        self.lbl_info.setText(f"待裁切: {count} 張")

        if count > 0:
            # 2. 取出清單中的第 0 個 (也就是排隊的第一張)
            path = images[0]
            self.current_image_path = path
            # 3. 顯示在畫布上
            self.image_label.set_image(path)
            # 4. 啟用按鈕
            self.btn_crop.setEnabled(True)
            self.btn_skip.setEnabled(True)
        else:
            # 如果清單是空的
            self.current_image_path = None
            
            # 1. 呼叫剛剛寫的清除功能
            self.image_label.clear_canvas() 
            
            # 2. 顯示提示文字
            self.image_label.setText("🎉 已無待處理照片\n(請點擊上方標籤前往 [1. 照片標註])")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # 文字置中
            
            # 3. 鎖定按鈕
            self.btn_crop.setEnabled(False)
            self.btn_skip.setEnabled(False)

    # 當按下「裁切」
    def apply_crop(self):
        if not self.current_image_path: return
        
        # 1. 問畫布：現在紅框的真實座標是多少？
        crop_box = self.image_label.get_crop_rect_original()
        if not crop_box:
            QMessageBox.warning(self, "錯誤", "請先畫框！")
            return
            
        try:
            # 2. 用 Pillow 打開原始大圖
            img = Image.open(self.current_image_path)
            # 3. 喀嚓！剪下去
            cropped_img = img.crop(crop_box)
            
            # 4. 叫管家做事：把剪好的存去 ROI，把舊的刪掉
            success = self.data_handler.save_crop_to_roi(cropped_img, self.current_image_path)
            
            # 5. 如果成功，重新刷新
            if success:
                self.refresh_ui() 
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    # 當按下「跳過」
    def skip_image(self):
        if not self.current_image_path: return
        # 直接叫管家把這張圖搬去 ROI，不做任何修改
        if self.data_handler.skip_to_roi(self.current_image_path):
            self.refresh_ui() # 搬走後，刷新畫面讀下一張

    # 當這個頁面顯示出來時 (例如切換分頁)
    def showEvent(self, event):
        self.refresh_ui() # 確保畫面是最新的
        super().showEvent(event)