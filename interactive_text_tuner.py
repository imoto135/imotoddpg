import cv2
import numpy as np

def nothing(x):
    pass

# --- 編集 ---
# IMAGE_PATH = "/home/ihpc/SSD/imoto/imotoddpg/makura/image/200017197_00008.jpg"
IMAGE_PATH = r"C:\Users\yuuit\myproject\imotoddpg\mask_output\200017197_00008_text_mask.png" # 調整したい画像のパス

# 表示するウィンドウの最大の「高さ」をピクセル単位で指定
# お使いのモニターの解像度に合わせて調整してください (例: FullHDなら1080, ノートPCなら768など)
MAX_DISPLAY_HEIGHT = 800
# -----------

# 画像読み込み
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("エラー: 画像が読み込めませんでした。")
    exit()

# --- ▼▼▼ ここから修正・追加 ▼▼▼ ---

# 元の画像の高さを取得
original_height = image.shape[0]

# もし画像の高さが最大表示高さを超えていたら、比率を保ったまま縮小する
if original_height > MAX_DISPLAY_HEIGHT:
    # 縮小率を計算
    scale_ratio = MAX_DISPLAY_HEIGHT / original_height
    # cv2.resizeで画像を縮小
    image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
    print(f"画像が大きいため、{int(scale_ratio*100)}%に縮小して表示します。")

# --- ▲▲▲ ここまで修正・追加 ▲▲▲ ---


# 文字抽出はグレースケール画像で行う
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ウィンドウ作成
cv2.namedWindow('Text Mask Preview')
cv2.namedWindow('Original Image')
cv2.moveWindow('Original Image', 50, 50)
cv2.moveWindow('Text Mask Preview', 50 + image.shape[1] + 20, 50)

# トラックバーを作成
cv2.createTrackbar('blockSize', 'Text Mask Preview', 5, 25, nothing)
cv2.createTrackbar('C', 'Text Mask Preview', 5, 20, nothing)

print("スライダーを調整して最適なblockSizeとCの範囲を見つけてください。")
print("Escキーを押すと終了します。")

while True:
    block_size_track = cv2.getTrackbarPos('blockSize', 'Text Mask Preview')
    c_val = cv2.getTrackbarPos('C', 'Text Mask Preview')
    actual_block_size = block_size_track * 2 + 3
    
    text_mask = cv2.adaptiveThreshold(gray_image, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      actual_block_size, c_val)
    
    cv2.imshow('Text Mask Preview', text_mask)
    cv2.imshow('Original Image', image)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print("\n調整が完了しました。")
print("以下の値をcreate_mask.pyにコピーしてください。")
print(f"block_size = {actual_block_size}")
print(f"C = {c_val}")