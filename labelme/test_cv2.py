import cv2
import numpy as np

# 讀取圖片 (假設圖片名稱為 'receipts.jpg')
# image_path = '1744859920162350802-041189eb-3987-4b7a-9d0a-fae92a2d9894-20250418-Naomi.jpg'
image_path = r"C:\Users\chester\Desktop\labelme\20250815\1746614255050050757-fff4d00a-0877-4e4f-bae8-032607bc3e75-20250505- FIA_1.jpg" # 請替換成你的圖片路徑
original_image = cv2.imread(image_path)

# 1. 影像預處理
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

# 使用自適應二值化，對光線不均勻的場景效果更好
# THRESH_BINARY_INV 表示將物件變為白色，背景變為黑色
binary_image = cv2.adaptiveThreshold(
    blurred_image, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 11, 2
)

# 2. 輪廓偵測
# cv2.RETR_EXTERNAL 只偵測最外層的輪廓
contours, hierarchy = cv2.findContours(
    binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"在圖片 '{image_path}' 中找到了 {len(contours)} 個輪廓。")

# 用於為分割出來的圖片命名
receipt_count = 0
img_height, img_width, _ = original_image.shape

# 3. 輪廓篩選與迭代
for contour in contours:
    # 4. 根據面積篩選
    if cv2.contourArea(contour) > 5000:
        receipt_count += 1
        
        # 取得輪廓的邊界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 為了美觀，可以在裁切時留下一點邊距
        padding = 10
        
        # 【關鍵修改】: 檢查並修正邊界，防止超出範圍
        y_start = max(0, y - padding)
        y_end = min(img_height, y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img_width, x + w + padding)

        # 5. 使用修正後的安全座標來裁切
        cropped_receipt = original_image[y_start:y_end, x_start:x_end]

        # 【新增保護】: 再次確認裁切後的圖片不是空的才儲存
        if cropped_receipt.size > 0:
            # 儲存裁切下來的圖片
            output_filename = f"receipt_{receipt_count}.png"
            cv2.imwrite(output_filename, cropped_receipt)
            print(f"已儲存分割後的發票: {output_filename}")
        else:
            print(f"警告：第 {receipt_count} 個輪廓裁切後為空，已跳過。")

        # (可選) 在原圖上畫出邊界框以供調試
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# (可選) 顯示標記了邊界的原始圖片
# cv2.imshow('Detected Receipts', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
