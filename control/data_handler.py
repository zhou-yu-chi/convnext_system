import os
import shutil

class DataHandler:
    def __init__(self):
        self.all_images = []       # 給 Page 0 用：根目錄下的原始圖片
        self.roi_images = []       # 給 Page 1 用：ROI 資料夾下的圖片
        self.current_index = 0     # Page 0 的指標
        self.project_path = ""

    def create_new_project(self, folder_path):
        """新增專案：建立 OK, NG, ROI 三個資料夾"""
        self.project_path = folder_path
        
        for name in ["OK", "NG", "ROI" ,"unconfirmed"]:
            path = os.path.join(folder_path, name)
            if not os.path.exists(path):
                os.makedirs(path)
            
        return self.scan_unsorted_images()

    def open_existing_project(self, folder_path):
        """開啟專案"""
        self.project_path = folder_path
        # 檢查基本結構 (至少要有 OK/NG，ROI 沒有就補建)
        if not os.path.exists(os.path.join(folder_path, "OK")) or \
           not os.path.exists(os.path.join(folder_path, "NG")):
            return False
            
        roi_path = os.path.join(folder_path, "ROI")
        unconfirmed_path = os.path.join(folder_path, "unconfirmed")
        if not os.path.exists(unconfirmed_path):
            os.makedirs(unconfirmed_path)
        if not os.path.exists(roi_path):
            os.makedirs(roi_path)

        self.scan_unsorted_images()
        return True

    def import_images_from_folder(self, source_folder):
        """匯入圖片到「專案根目錄」(給 Page 0 用)"""
        if not self.project_path: return 0
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        count = 0
        for file in os.listdir(source_folder):
            if file.lower().endswith(valid_extensions):
                shutil.copy2(os.path.join(source_folder, file), 
                             os.path.join(self.project_path, file))
                count += 1
        self.scan_unsorted_images()
        return count

    def scan_unsorted_images(self):
        """掃描根目錄 (Page 0 來源)"""
        self.all_images = []
        # 重置 Index，確保重新整理後從頭看
        self.current_index = 0 
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        if not os.path.exists(self.project_path): return []

        for file in os.listdir(self.project_path):
            full_path = os.path.join(self.project_path, file)
            # 排除掉資料夾，只抓檔案
            if os.path.isfile(full_path) and file.lower().endswith(valid_extensions):
                self.all_images.append(full_path)
        return self.all_images

    def get_current_image(self):
        """取得 Page 0 目前要處理的那張圖"""
        if 0 <= self.current_index < len(self.all_images):
            return self.all_images[self.current_index]
        return None

    # --- Page 0 專用功能 ---
    def save_crop_to_roi(self, pil_image, original_path):
        """將 PIL 圖片存入 ROI，並刪除原始檔"""
        if not self.project_path: return False
        
        file_name = os.path.basename(original_path)
        roi_path = os.path.join(self.project_path, "ROI", file_name)
        
        try:
            # 1. 儲存裁切後的圖到 ROI
            pil_image.save(roi_path)
            
            # 2. 刪除根目錄的原始圖
            os.remove(original_path)
            
            # 3. 重新掃描根目錄 (因為少了一張圖)
            self.scan_unsorted_images()
            return True
        except Exception as e:
            print(f"ROI 存檔失敗: {e}")
            return False

    def skip_to_roi(self, original_path):
        """不裁切，直接移動到 ROI"""
        if not self.project_path: return False
        file_name = os.path.basename(original_path)
        roi_path = os.path.join(self.project_path, "ROI", file_name)
        
        try:
            shutil.move(original_path, roi_path)
            self.scan_unsorted_images()
            return True
        except Exception as e:
            print(f"移動失敗: {e}")
            return False

    # --- Page 1 專用功能 ---
    def scan_roi_images(self):
        """掃描 ROI 資料夾 (Page 1 來源)"""
        self.roi_images = []
        roi_dir = os.path.join(self.project_path, "ROI")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        if os.path.exists(roi_dir):
            for file in os.listdir(roi_dir):
                full_path = os.path.join(roi_dir, file)
                if os.path.isfile(full_path) and file.lower().endswith(valid_extensions):
                    self.roi_images.append(full_path)
        return self.roi_images

    def get_first_roi_image(self):
        """Page 1 永遠只拿第一張 (Queue 模式)"""
        if self.roi_images:
            return self.roi_images[0]
        return None

    def move_roi_file_to_result(self, file_path, label):
        """將圖片從 ROI 移動到 OK 或 NG"""
        if not self.project_path: return False
        
        file_name = os.path.basename(file_path)
        target_path = os.path.join(self.project_path, label, file_name)
        
        try:
            shutil.move(file_path, target_path)
            # 移完後，重新掃描 ROI，列表會少一張，下一張自動遞補
            self.scan_roi_images()
            return True
        except Exception as e:
            print(f"分類移動失敗: {e}")
            return False

    # --- Page 2 專用 ---
    def get_images_in_folder(self, folder_name):
        # 維持原本邏輯
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
        # 維持原本邏輯 (給 Page 2 使用)
        file_name = os.path.basename(file_path)
        target_path = os.path.join(self.project_path, target_label, file_name)
        try:
            shutil.move(file_path, target_path)
            return True
        except:
            return False
    
    def delete_specific_file(self, path):
        try:
            os.remove(path)
            return True
        except:
            return False