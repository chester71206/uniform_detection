import os
import json
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

def convert_labelme_to_coco_force_single_class(labelme_dir, output_dir, train_split_ratio=0.8, final_class_name="item", final_class_id=1):
    """
    將 LabelMe 格式強制轉換為單一類別的 COCO 格式。
    忽略 LabelMe JSON 中的原始標籤，將所有標註都賦予同一個指定的 class_id。

    :param labelme_dir: 包含 LabelMe JSON 檔案和對應圖片的資料夾。
    :param output_dir: 儲存 COCO 格式資料的輸出目錄。
    :param train_split_ratio: 訓練集所佔的比例。
    :param final_class_name: 在 COCO 中為這個單一類別指定的名稱。
    :param final_class_id: 為這個單一類別指定的 ID (強烈建議為 1)。
    """
    # --- 1. 建立輸出目錄結構 ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # --- 2. 獲取所有 LabelMe JSON 檔案 ---
    label_files = glob.glob(os.path.join(labelme_dir, '*.json'))
    
    # --- 3. 分割訓練集和驗證集 ---
    train_files, val_files = train_test_split(label_files, train_size=train_split_ratio, random_state=42)
    print(f"Total files: {len(label_files)}, Train files: {len(train_files)}, Val files: {len(val_files)}")

    # --- 4. 準備 COCO categories 區塊 (因為只有一個類別，所以很簡單) ---
    coco_categories = [{
        "id": final_class_id,
        "name": final_class_name,
        "supercategory": "none"
    }]
    print(f"Forcing all annotations into a single class: ID={final_class_id}, Name='{final_class_name}'")

    # --- 5. 遍歷處理訓練集和驗證集 ---
    for split_name, split_files in [('train', train_files), ('val', val_files)]:
        coco_output = {
            "info": {"description": f"Converted from LabelMe to COCO - {split_name}"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": coco_categories # 直接使用上面定義好的單一類別
        }
      
        image_id_counter = 1
        annotation_id_counter = 1
        
        for label_file in tqdm(split_files, desc=f"Processing {split_name} set"):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_path = os.path.join(labelme_dir, data['imagePath'])
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found, skipping: {image_path}")
                continue
            
            output_image_path = os.path.join(output_dir, split_name, data['imagePath'])
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            shutil.copy(image_path, output_image_path)

            image_info = {
                "id": image_id_counter,
                "file_name": data['imagePath'],
                "width": data['imageWidth'],
                "height": data['imageHeight']
            }
            coco_output['images'].append(image_info)
            
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    # --- 無腦作法的核心修改點 ---
                    # 不再讀取 shape['label'] 或使用 class_mapping
                    # 直接將 category_id 寫死為我們想要的 final_class_id
                    
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
                        "category_id": final_class_id, # <--- 核心修改
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    coco_output['annotations'].append(annotation_info)
                    annotation_id_counter += 1
            
            image_id_counter += 1
      
        # --- 6. 儲存 COCO JSON 檔案 ---
        output_json_path = os.path.join(output_dir, 'annotations', f'instances_{split_name}.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=4)
        print(f"Successfully created {output_json_path}")

# --- 如何使用 ---
if __name__ == '__main__':
    labelme_project_dir = r'C:\Users\chester\Desktop\labelme\20250815'
    coco_output_dir = './20250815_single_class_paddleDetection'
    
    # 不需要 class_mapping 了
    # 直接呼叫新函式
    convert_labelme_to_coco_force_single_class(
        labelme_project_dir, 
        coco_output_dir, 
        final_class_name="invoice",  # 給你的類別取一個有意義的名稱
        final_class_id=1            # 指定 ID 為 1
    )