# file: keygen.py
import json
import base64
from datetime import datetime
from cryptography.fernet import Fernet

# ==========================================
# 這是你的私鑰，絕對不能外洩，必須與 loader.py 裡的一模一樣
# 你可以使用 Fernet.generate_key() 生成一次，然後固定下來
# ==========================================
SECRET_KEY = b'UlGsxbokxJGYHQjCrR2Sgqa_RykBFbv57sqV2E5CZUY='
# 第一次執行時，你可以先 print(Fernet.generate_key()) 拿到一串 key，然後填入上面

def generate_license_key(expiration_date_str):
    """
    產生包含截止日期的金鑰
    expiration_date_str: "YYYY-MM-DD"
    """
    f = Fernet(SECRET_KEY)
    
    # 建立授權資訊
    data = {
        "expire_date": expiration_date_str,
        "is_valid": True
        # 未來如果要做綁定電腦 ID，可以加在這裡
    }
    
    # 轉成 JSON string 再編碼
    json_data = json.dumps(data).encode('utf-8')
    
    # 加密
    encrypted_token = f.encrypt(json_data)
    
    return encrypted_token.decode('utf-8')

if __name__ == "__main__":
    # 設定你想要給客戶的期限
    expire_date = "2025-12-23"  # 範例：給兩個月
    
    key = generate_license_key(expire_date)
    print(f"=== 授權資訊 ===")
    print(f"截止日期: {expire_date}")
    print(f"給客戶的金鑰 (請複製下面這一長串):")
    print("-" * 60)
    print(key)
    print("-" * 60)