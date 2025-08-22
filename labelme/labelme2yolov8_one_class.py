import os
import json
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
import yaml

# <<< 修改點 1: 讓函式接受一個目錄列表 (dirs) 而不是單一目錄 (dir)
def convert_labelme_to_yolo(labelme_dirs, output_dir, class_names, train_split_ratio=0.9):
    """
    將 LabelMe 格式的資料夾轉換為 YOLOv8 格式 (版本3)。
    - 修正了圖片路徑處理的 bug，使其更加穩健。
    - 保留了只處理有效標註圖片的功能。
    - 增加了更清晰的診斷訊息。
    - [新功能] 支援從多個來源資料夾讀取資料。
    """
    # --- 1. 建立目錄結構 ---
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

    # --- 2. 建立類別映射 ---
    class_name_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"Class mapping: {class_name_to_id}")

    # --- 3. 獲取檔案並分割 ---
    # <<< 修改點 2: 從所有指定的目錄中遞迴搜尋 JSON 檔案
    json_files = []
    for dir_path in labelme_dirs:
        # 使用 extend 將找到的檔案列表加入總列表中
        json_files.extend(glob.glob(os.path.join(dir_path, '**', '*.json'), recursive=True))

    if not json_files:
        print(f"Error: No .json files found recursively in the provided directories: {labelme_dirs}")
        return
        
    train_files, val_files = train_test_split(json_files, train_size=train_split_ratio, random_state=42)
    print(f"Total files found across all directories: {len(json_files)}")
    print(f"Splitting into -> Train files: {len(train_files)}, Val files: {len(val_files)}")

    # --- 4. 處理檔案轉換 (這部分邏輯完全不用變) ---
    for split_name, split_files in [('train', train_files), ('val', val_files)]:
        files_processed_with_labels = 0
        for json_file in tqdm(split_files, desc=f"Processing {split_name} set"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            yolo_annotations = []
            for shape in data['shapes']:
                label = shape['label']
                if label not in class_name_to_id:
                    continue
                
                class_id = class_name_to_id[label]
                points = shape['points']
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])

                box_width = x2 - x1
                box_height = y2 - y1
                x_center = x1 + box_width / 2
                y_center = y1 + box_height / 2

                norm_x_center = x_center / img_width
                norm_y_center = y_center / img_height
                norm_width = box_width / img_width
                norm_height = box_height / img_height

                yolo_annotations.append(
                    f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

            if yolo_annotations:
                json_dir = os.path.dirname(json_file)
                image_path_in_json = data['imagePath']
                original_image_path = os.path.join(json_dir, image_path_in_json)

                # 這個備用路徑的檢查邏輯在這裡非常重要，因為 JSON 和圖片可能不在同一個根目錄
                if not os.path.exists(original_image_path):
                    # 遍歷所有可能的根目錄來尋找圖片
                    found = False
                    for root_dir in labelme_dirs:
                        potential_path = os.path.join(root_dir, os.path.basename(image_path_in_json))
                        if os.path.exists(potential_path):
                            original_image_path = potential_path
                            found = True
                            break
                    if not found:
                        print(f"\n[ERROR] Image not found for {os.path.basename(json_file)}. Skipping this file.")
                        continue

                base_filename = os.path.basename(original_image_path)
                filename_no_ext = os.path.splitext(base_filename)[0]

                output_image_path = os.path.join(output_dir, 'images', split_name, base_filename)
                output_label_path = os.path.join(output_dir, 'labels', split_name, f"{filename_no_ext}.txt")
                
                shutil.copy(original_image_path, output_image_path)
                
                with open(output_label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_annotations))
                
                files_processed_with_labels += 1
        print(f"  > Successfully processed {files_processed_with_labels} images with labels for the {split_name} set.")


    # --- 5. 建立 data.yaml 檔案 (這部分邏輯完全不用變) ---
    yaml_data = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    
    print(f"\nConversion complete! YOLOv8 dataset created at: {output_dir}")
    print(f"YAML configuration file created at: {yaml_path}")


# --- 如何使用 ---
if __name__ == '__main__':
    # <<< 修改點 3: 將單一路徑改為路徑列表
    # 1. LabelMe 專案的所有根目錄
    labelme_project_dirs = [
        r'C:\Users\chester\Desktop\labelme\20250815',
        r'C:\Users\chester\Desktop\labelme\20250814'
    ]
    
    # 2. YOLOv8 資料集的輸出目錄
    yolo_output_dir = './2025_merged_0815_0814_yolov8'
    
    # 3. 確認您的類別名稱完全匹配 (大小寫、無多餘空格)
    my_class_names = ['invoice']
    
    # <<< 修改點 4: 傳入的是路徑列表
    # 執行新版的轉換函式
    convert_labelme_to_yolo(labelme_project_dirs, yolo_output_dir, class_names=my_class_names)