import cv2
import numpy as np

def nothing(x):
    pass

# --- 編集 ---
IMAGE_PATH = "/home/ihpc/SSD/imoto/imotoddpg/makura/image/200017197_00008.jpg" # 調整したい画像のパス
# -----------

# 画像読み込み
image = cv2.imread(IMAGE_PATH)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ウィンドウ作成
cv2.namedWindow('Mask Preview')
cv2.createTrackbar('H_min', 'Mask Preview', 10, 179, nothing)
cv2.createTrackbar('H_max', 'Mask Preview', 25, 179, nothing)
cv2.createTrackbar('S_min', 'Mask Preview', 40, 255, nothing)
cv2.createTrackbar('S_max', 'Mask Preview', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Mask Preview', 40, 255, nothing)
cv2.createTrackbar('V_max', 'Mask Preview', 255, 255, nothing)

print("スライダーを調整して最適なHSV範囲を見つけてください。")
print("Escキーを押すと終了します。")

while True:
    h_min = cv2.getTrackbarPos('H_min', 'Mask Preview')
    h_max = cv2.getTrackbarPos('H_max', 'Mask Preview')
    s_min = cv2.getTrackbarPos('S_min', 'Mask Preview')
    s_max = cv2.getTrackbarPos('S_max', 'Mask Preview')
    v_min = cv2.getTrackbarPos('V_min', 'Mask Preview')
    v_max = cv2.getTrackbarPos('V_max', 'Mask Preview')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv_image, lower, upper)
    
    # プレビュー用に元の画像とマスクを重ねる
    preview = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Mask Preview', preview)
    cv2.imshow('Original Image', image)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Escキーで終了
        break

cv2.destroyAllWindows()
print(f"\n最終的なHSV範囲:\nlower = np.array([{h_min}, {s_min}, {v_min}])\nupper = np.array([{h_max}, {s_max}, {v_max}])")