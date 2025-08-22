import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_yolo_results(csv_path='results.csv', save_path='training_results.jpg'):
    """
    從 YOLOv8 的 results.csv 檔案讀取數據並繪製訓練圖表。

    Args:
        csv_path (str): results.csv 檔案的路徑。
        save_path (str): 儲存圖表的圖片路徑。
    """
    # --- 1. 檢查檔案是否存在 ---
    if not os.path.exists(csv_path):
        print(f"❌ 錯誤：找不到 CSV 檔案於 '{csv_path}'")
        return

    print(f"✅ 正在讀取數據從 '{csv_path}'...")
    # --- 2. 讀取並清理 CSV 數據 ---
    # YOLOv8 的 CSV 欄位名稱可能包含前導空格，需要清理
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # --- 3. 設定繪圖風格 ---
    sns.set_theme(style="whitegrid")
    # 創建一個 2x2 的子圖佈局，並設定整張圖片的大小
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('YOLOv8 Training Performance', fontsize=20, weight='bold')

    # --- 4. 繪製各個子圖 ---

    # 子圖 1: Box Loss (邊界框損失)
    axs[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='tab:blue', marker='o', linestyle='--')
    axs[0, 0].plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss', color='tab:orange', marker='o')
    axs[0, 0].set_title('Box Loss over Epochs', fontsize=14)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # 子圖 2: Class Loss (分類損失)
    axs[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='tab:blue', marker='o', linestyle='--')
    axs[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Validation Class Loss', color='tab:orange', marker='o')
    axs[0, 1].set_title('Class Loss over Epochs', fontsize=14)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # 子圖 3: DFL Loss (分佈焦距損失)
    axs[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', color='tab:blue', marker='o', linestyle='--')
    axs[1, 0].plot(df['epoch'], df['val/dfl_loss'], label='Validation DFL Loss', color='tab:orange', marker='o')
    axs[1, 0].set_title('DFL Loss over Epochs', fontsize=14)
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()

    # 子圖 4: mAP Metrics (模型評估指標)
    axs[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.50', color='tab:green', marker='^')
    axs[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.50-0.95', color='tab:red', marker='^')
    axs[1, 1].set_title('mAP Score over Epochs', fontsize=14)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('mAP Score')
    axs[1, 1].legend()

    # --- 5. 調整佈局並儲存圖片 ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局以容納主標題
    plt.savefig(save_path, dpi=300) # dpi=300 確保圖片品質清晰
    
    print(f"🎉 圖表已成功儲存至 '{save_path}'")
    
    # 選擇性地顯示圖表
    # plt.show()


if __name__ == '__main__':
    # 假設您的 CSV 檔案在同一個資料夾，檔名為 results.csv
    # 如果檔案在別處，請修改這裡的路徑
    path_to_csv = r'C:\Users\chester\Desktop\labelme\invoice_detection_project_yolov11\run_with_safe_check_yolov11\results.csv' 
    plot_yolo_results(csv_path=path_to_csv,save_path="./invoice_cut_yolo_v11.jpg")