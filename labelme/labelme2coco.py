import os
import json
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def convert_labelme_to_coco(labelme_dir, output_dir, train_split_ratio=0.8, class_name_to_id=None):
    """
    將 LabelMe 格式的資料夾轉換為 COCO 格式。

    :param labelme_dir: 包含 LabelMe JSON 檔案和對應圖片的資料夾。
    :param output_dir: 儲存 COCO 格式資料的輸出目錄。
    :param train_split_ratio: 訓練集所佔的比例。
    :param class_name_to_id: (可選) 類別名稱到 ID 的映射字典，例如 {'seal': 1}。
                             如果為 None，將自動從資料中生成。
    """
    # --- 1. 建立輸出目錄結構 ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # --- 2. 獲取所有 LabelMe JSON 檔案 ---
    label_files = glob.glob(os.path.join(labelme_dir, '*.json'))
    
    # --- 3. 自動生成類別或使用提供的類別 ---
    if class_name_to_id is None:
        print("Auto-detecting classes...")
        all_labels = set()
        for label_file in tqdm(label_files, desc="Detecting Classes"):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    all_labels.add(shape['label'])
        
        class_names = sorted(list(all_labels))
        class_name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    
    print(f"Class mapping: {class_name_to_id}")

    # --- 4. 分割訓練集和驗證集 ---
    train_files, val_files = train_test_split(label_files, train_size=train_split_ratio, random_state=42)
    print(f"Total files: {len(label_files)}, Train files: {len(train_files)}, Val files: {len(val_files)}")

    # --- 5. 遍歷處理訓練集和驗證集 ---
    for split_name, split_files in [('train', train_files), ('val', val_files)]:
        coco_output = {
            "info": {"description": f"Converted from LabelMe to COCO - {split_name}"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 填充 categories
        for class_name, class_id in class_name_to_id.items():
            coco_output['categories'].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "none"
            })

        image_id_counter = 1
        annotation_id_counter = 1
        
        # 遍歷該分割集的所有 JSON 檔案
        for label_file in tqdm(split_files, desc=f"Processing {split_name} set"):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 拷貝圖片到對應的 train/val 資料夾
            image_path = os.path.join(labelme_dir, data['imagePath'])
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found, skipping: {image_path}")
                continue
            
            output_image_path = os.path.join(output_dir, split_name, data['imagePath'])
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            import shutil
            shutil.copy(image_path, output_image_path)

            # 添加 image 資訊
            image_info = {
                "id": image_id_counter,
                "file_name": data['imagePath'],
                "width": data['imageWidth'],
                "height": data['imageHeight']
            }
            coco_output['images'].append(image_info)
            
            # 添加 annotation 資訊
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    class_name = shape['label']
                    if class_name not in class_name_to_id:
                        print(f"Warning: Class '{class_name}' not in mapping. Skipping.")
                        continue
                    
                    class_id = class_name_to_id[class_name]
                    
                    points = shape['points']
                    x1 = min(points[0][0], points[1][0])
                    y1 = min(points[0][1], points[1][1])
                    x2 = max(points[0][0], points[1][0])
                    y2 = max(points[0][1], points[1][1])
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    annotation_info = {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": class_id,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": [] # 對於矩形框，這個可以是空的
                    }
                    coco_output['annotations'].append(annotation_info)
                    annotation_id_counter += 1
            
            image_id_counter += 1
        
        # --- 6. 儲存 COCO JSON 檔案 ---
        output_json_path = os.path.join(output_dir, 'annotations', f'instances_{split_name}.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)
        print(f"Successfully created {output_json_path}")


if __name__ == '__main__':
    # --- 使用範例 ---
    # 你的 LabelMe 專案資料夾路徑，裡面應該同時包含 .jpg 和 .json 檔案
    labelme_project_dir = r'C:\Users\chester\Desktop\commeet\發票資料\tw_double_uniform'
    
    # 你想生成的 COCO 格式資料集的路徑
    coco_output_dir = './new_coco_dataset'
    
    # 你的類別和 ID 的對應關係。
    # LabelMe 標註時，label 欄位寫的是 'seal'，我們希望在 COCO 中它的 category_id 是 1
    # 如果你的 label 欄位寫的是 '0'，那就改成 {'0': 1}
    class_mapping = {
        '0': 1
    }
    
    # 開始轉換
    # 如果你不想預先定義 class_mapping，可以傳入 None，腳本會自動掃描所有類別
    # convert_labelme_to_coco(labelme_project_dir, coco_output_dir, class_name_to_id=None)
    convert_labelme_to_coco(labelme_project_dir, coco_output_dir, train_split_ratio=0.8, class_name_to_id=class_mapping)