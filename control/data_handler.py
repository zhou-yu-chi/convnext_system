import os
import shutil
import time
import datetime

class DataHandler:
    def __init__(self):
        self.all_images = []       
        self.roi_images = []       
        self.current_index = 0     
        self.project_path = ""

    # ========================================================
    # ★★★ 核心救星：產生絕對不重複的檔名 ★★★
    # ========================================================
    def generate_unique_path(self, target_folder, filename):
        """
        傳入目標資料夾與檔名，回傳一個保證不重複的新路徑。
        策略：如果重複，就加上 [時間戳_流水號]。
        """
        name, ext = os.path.splitext(filename)
        destination = os.path.join(target_folder, filename)
        
        # 如果檔案不存在，直接回傳原路徑
        if not os.path.exists(destination):
            return destination
        
        # 如果檔案已存在，開始改名迴圈
        counter = 1
        while True:
            # 格式：原檔名_年月日時分秒_流水號.副檔名
            # 例如：image1_20231223103055_1.jpg
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_name = f"{name}_{timestamp}_{counter}{ext}"
            new_destination = os.path.join(target_folder, new_name)
            
            if not os.path.exists(new_destination):
                return new_destination
            
            counter += 1

    # ========================================================

    def get_import_list(self, source_folder):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        if not os.path.exists(source_folder): return []
        files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
        return files

    def copy_file_to_project(self, source_path, rename_if_exists=False):
        """
        複製檔案到專案根目錄 (Page 0 匯入用)
        """
        if not self.project_path: return False
        
        try:
            file_name = os.path.basename(source_path)
            # 預設目標路徑
            destination = os.path.join(self.project_path, file_name)

            # 1. 檢查是否重複
            if os.path.exists(destination):
                # 如果 UI 沒說要改名，就回傳訊號讓 UI 去問使用者
                if not rename_if_exists:
                    return "DUPLICATE"
                
                # 如果 UI 說「全部改名」，就呼叫我們的無敵改名函式
                destination = self.generate_unique_path(self.project_path, file_name)

            # 2. 執行複製
            shutil.copy2(source_path, destination)
            return True

        except Exception as e:
            print(f"複製失敗 {source_path}: {e}")
            return False
        
    def create_new_project(self, folder_path):
        self.project_path = folder_path
        # 確保 Unconfirmed 資料夾也被建立
        for name in ["OK", "NG", "ROI", "Unconfirmed"]:
            path = os.path.join(folder_path, name)
            if not os.path.exists(path):
                os.makedirs(path)
        return self.scan_unsorted_images()

    def open_existing_project(self, folder_path):
        self.project_path = folder_path
        if not os.path.exists(os.path.join(folder_path, "OK")) or \
           not os.path.exists(os.path.join(folder_path, "NG")):
            return False
            
        # 補建可能缺少的資料夾
        for name in ["ROI", "Unconfirmed"]:
            path = os.path.join(folder_path, name)
            if not os.path.exists(path):
                os.makedirs(path)

        self.scan_unsorted_images()
        return True

    def scan_unsorted_images(self):
        self.all_images = []
        self.current_index = 0 
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        if not os.path.exists(self.project_path): return []

        for file in os.listdir(self.project_path):
            full_path = os.path.join(self.project_path, file)
            if os.path.isfile(full_path) and file.lower().endswith(valid_extensions):
                self.all_images.append(full_path)
        return self.all_images

    def get_current_image(self):
        if 0 <= self.current_index < len(self.all_images):
            return self.all_images[self.current_index]
        return None

    # --- Page 0 專用 ---
    def save_crop_to_roi(self, pil_image, original_path):
        if not self.project_path: return False
        
        file_name = os.path.basename(original_path)
        roi_dir = os.path.join(self.project_path, "ROI")
        
        # ★★★ 使用 generate_unique_path 防止 ROI 裡面也有重複檔名
        roi_path = self.generate_unique_path(roi_dir, file_name)
        
        try:
            pil_image.save(roi_path)
            os.remove(original_path)
            self.scan_unsorted_images()
            return True
        except Exception as e:
            print(f"ROI 存檔失敗: {e}")
            return False

    # --- Page 1 專用 ---
    def scan_roi_images(self):
        self.roi_images = []
        roi_dir = os.path.join(self.project_path, "ROI")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        if os.path.exists(roi_dir):
            for file in os.listdir(roi_dir):
                full_path = os.path.join(roi_dir, file)
                if os.path.isfile(full_path) and file.lower().endswith(valid_extensions):
                    self.roi_images.append(full_path)
        return self.roi_images

    def move_roi_file_to_result(self, file_path, label):
        """Page 1: 將圖片從 ROI 移動到 OK 或 NG"""
        if not self.project_path: return False
        
        file_name = os.path.basename(file_path)
        target_dir = os.path.join(self.project_path, label)
        
        # ★★★ 關鍵：移動前先檢查 OK/NG 資料夾有沒有重複的，有就改名
        target_path = self.generate_unique_path(target_dir, file_name)
        
        try:
            shutil.move(file_path, target_path)
            self.scan_roi_images()
            return True
        except Exception as e:
            print(f"分類移動失敗: {e}")
            return False

    # --- Page 2 專用 ---
    def get_images_in_folder(self, folder_name):
        target_dir = os.path.join(self.project_path, folder_name)
        if not os.path.exists(target_dir): return []
        images = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for f in os.listdir(target_dir):
            fp = os.path.join(target_dir, f)
            if os.path.isfile(fp) and f.lower().endswith(valid_exts):
                images.append(fp)
        return images
    
    def move_specific_file(self, file_path, target_label):
        """Page 2: 在 OK/NG/Unconfirmed 之間移動"""
        if not self.project_path: return False

        file_name = os.path.basename(file_path)
        target_dir = os.path.join(self.project_path, target_label)
        
        # ★★★ 關鍵：移動前確保不覆蓋現有檔案
        target_path = self.generate_unique_path(target_dir, file_name)

        try:
            shutil.move(file_path, target_path)
            return True
        except Exception as e:
            print(f"移動失敗: {e}")
            return False
    
    def delete_specific_file(self, path):
        try:
            os.remove(path)
            return True
        except:
            return False