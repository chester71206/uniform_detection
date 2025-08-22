import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'  # <--- 加入這一行！

# import sys
# sys.stdout = open("uniform_cut_v1.txt", "w", encoding="utf-8", buffering=1)  # 即時寫入 + UTF-8 無 BOM

# # 如果你有 stderr 輸出錯誤資訊，也加這行
# sys.stderr = sys.stdout
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from ultralytics import YOLO
import yaml

MODEL_TO_FINETUNE = "yolo11x.pt" 


def sync_labels_and_images(image_dir, label_dir):
    """
    同步圖片和標籤資料夾，確保每個標籤都有對應的圖片，反之亦然。
    """
    print("--- 開始同步圖片與標籤資料 ---")
    
    # 獲取所有圖片和標籤的基礎名稱 (不含副檔名)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith('.txt')}
    
    # 1. 找出有標籤但沒有對應圖片的檔案，並刪除標籤
    labels_to_delete = label_files - image_files
    if labels_to_delete:
        print(f"發現 {len(labels_to_delete)} 個多餘的標籤檔，將進行刪除...")
        for basename in labels_to_delete:
            label_path = os.path.join(label_dir, f"{basename}.txt")
            os.remove(label_path)
            print(f"  已刪除: {label_path}")
                    
    print("--- 同步完成 ---")




if __name__ == '__main__':
    # ------------------ 設定 ------------------
    # 資料集根目錄 (與 prelabel_generator.py 中設定的相同)
    DATASET_ROOT = r"C:\Users\chester\Desktop\labelme\2025_merged_0815_0814_yolov8"
    
    # YOLO 的 YAML 設定檔路徑
    YAML_PATH = os.path.join(DATASET_ROOT, r"C:\Users\chester\Desktop\labelme\20250815_single_class_yolov8\data.yaml")
    
    # 你的類別名稱
    CLASS_NAMES = ['invoice']
    
    # 訓練參數
    EPOCHS = 100
    BATCH_SIZE = 4 # 根據你的顯卡記憶體調整
    IMG_SIZE = 640
    WORKERS = 16
    AMP = True 
    CACHE = True
    PROJECT_NAME = 'invoice_detection_project_yolov11'
    RUN_NAME = 'run_with_safe_check_yolov11'
    # ------------------------------------------


    # 步驟 2: 同步訓練資料夾
    # 你可以為 train 和 val 分別同步
    train_image_dir = os.path.join(DATASET_ROOT, 'images', 'train')
    train_label_dir = os.path.join(DATASET_ROOT, 'labels', 'train')
    # 驗證集路徑
    val_image_dir = os.path.join(DATASET_ROOT, 'images', 'val')
    val_label_dir = os.path.join(DATASET_ROOT, 'labels', 'val')
    sync_labels_and_images(train_image_dir, train_label_dir)
    sync_labels_and_images(val_image_dir, val_label_dir)
    
    # 步驟 3: 載入模型並開始訓練
    print("\n--- 開始 YOLOv11 微調訓練 ---")
    model = YOLO(MODEL_TO_FINETUNE)
    
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,     # <--- 使用新設定
        amp=AMP,             # <--- 使用新設定
        cache=CACHE,         # <--- 使用新設定
        project=PROJECT_NAME,
        name=RUN_NAME,
        device=0,
        #resume=True,
        save_period=5
    )
    
    print("\n--- 訓練完成 ---")
    print(f"最佳模型儲存於: {results.save_dir}/weights/best.pt")