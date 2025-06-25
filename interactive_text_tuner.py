import cv2
import numpy as np

def nothing(x):
    pass

# --- 編集 ---
IMAGE_PATH = "/home/ihpc/SSD/imoto/imotoddpg/makura/image/200017197_00008.jpg" # 調整したい画像のパス
# -----------

# 画像読み込み
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("エラー: 画像が読み込めませんでした。")
    exit()

# 文字抽出はグレースケール画像で行う
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ウィンドウ作成
cv2.namedWindow('Text Mask Preview')

# トラックバーを作成
# blockSizeは3以上の奇数である必要があるため、トラックバーの値(0,1,2,...)を (3,5,7,...)に変換する
# Cは微調整用の定数
cv2.createTrackbar('blockSize', 'Text Mask Preview', 5, 25, nothing) # 5*2+3=13が初期値
cv2.createTrackbar('C', 'Text Mask Preview', 5, 20, nothing)

print("スライダーを調整して最適なblockSizeとCの範囲を見つけてください。")
print("Escキーを押すと終了します。")

while True:
    # トラックバーから現在の値を取得
    block_size_track = cv2.getTrackbarPos('blockSize', 'Text Mask Preview')
    c_val = cv2.getTrackbarPos('C', 'Text Mask Preview')

    # トラックバーの値を3以上の奇数に変換 (e.g., 0 -> 3, 1 -> 5, 2 -> 7, ...)
    actual_block_size = block_size_track * 2 + 3

    # 適応的二値化を適用
    text_mask = cv2.adaptiveThreshold(gray_image, 
                                      255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      actual_block_size, 
                                      c_val)

    # プレビューウィンドウに結果を表示
    cv2.imshow('Text Mask Preview', text_mask)
    cv2.imshow('Original Image', image)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Escキーで終了
        break

cv2.destroyAllWindows()
print("\n調整が完了しました。")
print("以下の値をcreate_mask.pyにコピーしてください。")
print(f"block_size = {actual_block_size}")
print(f"C = {c_val}")