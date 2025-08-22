import os
import json
import glob
from tqdm import tqdm

def clean_labelme_labels(labelme_dir, unified_label="item"):
    """
    遍歷指定資料夾中的所有 LabelMe JSON 檔案，
    並將其中所有的 shape's label 統一為指定的名稱。

    :param labelme_dir: 包含 LabelMe JSON 檔案的資料夾。
    :param unified_label: 你希望統一使用的標籤名稱。
    """
    json_files = glob.glob(os.path.join(labelme_dir, '*.json'))
    if not json_files:
        print(f"Warning: No .json files found in '{labelme_dir}'")
        return

    print(f"Found {len(json_files)} json files. Unifying all labels to '{unified_label}'...")
    
    modified_count = 0
    for json_file in tqdm(json_files, desc="Cleaning labels"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_modified = False
            for shape in data['shapes']:
                if shape['label'] != unified_label:
                    shape['label'] = unified_label
                    is_modified = True
            
            if is_modified:
                modified_count += 1
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    print(f"\nDone! Unified labels for {modified_count} files.")

# --- 如何使用 ---
if __name__ == '__main__':
    # 你的 LabelMe 專案資料夾路徑
    my_labelme_project_dir = r'C:\Users\chester\Desktop\labelme\20250815'
    
    # 執行清理
    clean_labelme_labels(my_labelme_project_dir, unified_label="invoice")