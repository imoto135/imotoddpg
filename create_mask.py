import cv2
import numpy as np
import os

def generate_masks(image_path: str, stain_mask_output_path: str, text_mask_output_path: str):
    """
    古文書の画像から「シミのマスク」と「文字のマスク」を生成する関数。

    Args:
        image_path (str): 入力となる古文書画像のパス。
        stain_mask_output_path (str): 生成したシミマスクを保存するパス。
        text_mask_output_path (str): 生成した文字マスクを保存するパス。
    """
    print(f"画像を読み込みます: {image_path}")
    # 1. 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("エラー: 画像が読み込めませんでした。")
        return

    # ==========================================================
    # Part 1: シミのマスクを生成 (既存のロジック)
    # ==========================================================
    print("シミのマスクを生成中...")
    # チューニングが必要なHSVパラメータ
    lower_hsv = np.array([5, 40, 40])
    upper_hsv = np.array([25, 255, 255])
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    stain_mask_raw = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(stain_mask_raw, cv2.MORPH_OPEN, kernel, iterations=2)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # シミを黒(0), それ以外を白(255)とするマスク
    final_stain_mask = cv2.bitwise_not(closed_mask)

    # フォルダが存在しない場合は作成
    stain_dir = os.path.dirname(stain_mask_output_path)
    os.makedirs(stain_dir, exist_ok=True)
    cv2.imwrite(stain_mask_output_path, final_stain_mask)
    print(f"シミのマスクを保存しました: {stain_mask_output_path}")

    # ==========================================================
    # Part 2: 文字のマスクを生成 (新しいロジック)
    # ==========================================================
    print("文字のマスクを生成中...")
    # 1. 画像をグレースケールに変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 適応的二値化を適用して文字を抽出
    # blockSize: 閾値を計算するための近傍領域のサイズ（奇数である必要あり）
    # C: 計算された閾値から引く定数。値を大きくすると黒い部分が減る（文字が細くなる）
    block_size = 5
    C = 10
    final_text_mask = cv2.adaptiveThreshold(gray_image, 
                                            255, # 最大値
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 閾値の計算方法
                                            cv2.THRESH_BINARY_INV, # 白黒反転（文字を白にする）
                                            block_size, 
                                            C)

    # フォルダが存在しない場合は作成
    text_dir = os.path.dirname(text_mask_output_path)
    os.makedirs(text_dir, exist_ok=True)
    cv2.imwrite(text_mask_output_path, final_text_mask)
    print(f"文字のマスクを保存しました: {text_mask_output_path}")


if __name__ == '__main__':
    # --- ここを編集してください ---
    INPUT_IMAGE_PATH = 'makura1pretreat/200017197_00009/page_resize.png'
    # INPUT_IMAGE_PATH = r"C:\Users\yuuit\myproject\imotoddpg\200017197_00008.jpg"
    OUTPUT_DIR = "mask_output"
    # -------------------------

    # --- ファイル名を自動生成 ---
    base_filename = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    
    # シミマスクの出力パス
    output_stain_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_stain_mask.png")
    
    # 文字マスクの出力パス
    output_text_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_text_mask.png")
    
    # --- 関数呼び出し ---
    generate_masks(INPUT_IMAGE_PATH, output_stain_mask_path, output_text_mask_path)