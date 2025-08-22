import os
import cv2
from ultralytics import YOLO
import numpy as np  # 建議加入

# 載入 YOLO 模型
model = YOLO(r"C:\Users\chester\Desktop\labelme\invoice_detection_project_yolov11\run_with_safe_check_yolov11\weights\best.pt")

# 輸入與輸出資料夾
input_dir = r"C:\Users\chester\Desktop\labelme\20250815_single_class_yolov8\images\val"
output_dir = "./invoice_cut_output_v11"

# 建立輸出資料夾（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍歷所有 JPG 檔案
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        print(f"正在處理: {image_path}")
        
        # 執行模型推論
        results = model(image_path, conf=0.1,iou=0.5)

        # 繪製結果
        annotated_frame = results[0].plot()

        # 儲存結果圖片
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated_frame)
        print(f"儲存完成：{output_path}")

print("所有圖片處理完成！")