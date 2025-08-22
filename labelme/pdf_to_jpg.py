import os
from pdf2image import convert_from_path
import shutil  # <--- 1. 匯入 shutil 模組

# --- 設定路徑 ---
pdf_folder = r"C:\Users\chester\Downloads\20250815-20250815T024303Z-1-001\20250815"
output_folder = r"./20250815"
# 確保最外層的輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# --- 使用 os.walk() 遍歷所有子資料夾 ---
for root, dirs, files in os.walk(pdf_folder):
    # 計算對應的輸出子資料夾路徑
    rel_path = os.path.relpath(root, pdf_folder)
    # 建立對應的輸出子資料夾
    out_dir = os.path.join(output_folder, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    # --- 處理目前資料夾中的所有檔案 ---
    for filename in files:
        # 建立完整的來源檔案路徑
        source_path = os.path.join(root, filename)

        if filename.lower().endswith(".pdf"):
            # 取得不含副檔名的檔名
            base_name = os.path.splitext(filename)[0]

            print(f"正在處理 PDF: {source_path}")

            try:
                # 執行 PDF 轉換
                images = convert_from_path(source_path, fmt='jpg', dpi=200, grayscale=True)

                if len(images) == 1:
                    output_path = os.path.join(out_dir, f"{base_name}.jpg")
                    images[0].save(output_path, 'JPEG')
                    print(f" -> 已儲存: {output_path}")
                else:
                    for i, img in enumerate(images, start=1):
                        output_path = os.path.join(out_dir, f"{base_name}_{i}.jpg")
                        img.save(output_path, "JPEG")
                        print(f" -> 已儲存: {output_path}")

            except Exception as e:
                print(f"處理檔案 {source_path} 時發生錯誤: {e}")

        # <--- 2. 新增 else 區塊來處理非 PDF 檔案 ---
        else:
            # 建立完整的目標檔案路徑
            destination_path = os.path.join(out_dir, filename)
            
            print(f"正在複製檔案: {filename}")
            try:
                # 3. 使用 shutil.copy2 來複製檔案
                shutil.copy2(source_path, destination_path)
                print(f" -> 已複製到: {destination_path}")
            except Exception as e:
                print(f"複製檔案 {source_path} 時發生錯誤: {e}")


print("\n所有 PDF 轉換及檔案複製完成。") # <--- 修改完成訊息