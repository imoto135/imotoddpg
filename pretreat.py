#resize and crop images into patches
# 必要なライブラリをインポート
import os
from PIL import Image
import glob  # ★ globライブラリをインポート

class ImageProcessor:
    def crop(self, input_image_path, output_path, crop_file_color, crop_file_binary, threshold=128):
        """
        一枚の画像を読み込み、リサイズ、二値化した後、
        カラー版と二値版のパッチ画像に分割して保存する。
        """
        # --- 1. 出力フォルダの準備 ---
        # フォルダ作成は呼び出し元で行うため、ここではシンプルに
        os.makedirs(crop_file_color, exist_ok=True)
        os.makedirs(crop_file_binary, exist_ok=True)

        # --- 2. 画像の読み込みとリサイズ ---
        try:
            img = Image.open(input_image_path)
        except FileNotFoundError:
            print(f"エラー: 入力ファイルが見つかりません: {input_image_path}")
            return
        except Exception as e:
            print(f"エラー: {input_image_path} を開く際に問題が発生しました: {e}")
            return

        # 固定サイズにリサイズ
        resized_size = (704, 704)
        img_resized = img.resize(resized_size, Image.Resampling.LANCZOS)
        img_resized.save(os.path.join(output_path, "page_resize.png"))
        print(f"  - リサイズ済みのカラー画像を保存しました: {os.path.join(output_path, 'page_resize.png')}")

        # --- 3. 二値化処理 ---
        img_binary = img_resized.convert("L").point(lambda p: 0 if p < threshold else 255, '1')
        img_binary.save(os.path.join(output_path, "page_resize_binary.png"))
        print(f"  - リサイズ済みの二値化画像を保存しました: {os.path.join(output_path, 'page_resize_binary.png')}")


        # --- 4. パッチ画像の分割（クロップ） ---
        patch_size_x = resized_size[0] // 11  # -> 64
        patch_size_y = resized_size[1] // 11  # -> 64
        
        patch_count = 0
        for y in range(0, resized_size[1], patch_size_y):
            for x in range(0, resized_size[0], patch_size_x):
                box = (x, y, x + patch_size_x, y + patch_size_y)
                
                color_region = img_resized.crop(box)
                color_region.save(os.path.join(crop_file_color, f"{x}_{y}.png"))
                
                binary_region = img_binary.crop(box)
                binary_region.save(os.path.join(crop_file_binary, f"{x}_{y}.png"))
                patch_count += 1
        
        print(f"  - {patch_count}個のパッチを生成しました。")

# --- 使用例 ---
if __name__ == '__main__':
    # ★ ワイルドカードを含む入力「パターン」
    input_pattern = '/home/ihpc/SSD/imoto/imotoddpg/makura2/density_5e-04_intensity_0.50/no_text/*.jpg'
    # ★ 全ての出力のベースとなる親フォルダ
    output_base_dir = '/home/ihpc/SSD/imoto/imotoddpg/makura2pretreat/density_5e-04_intensity_0.50/no_text'

    # ★ globを使って、パターンに一致するファイルのリストを取得
    file_list = glob.glob(input_pattern)

    # ★ ファイルが見つからなかった場合のガード処理
    if not file_list:
        print(f"エラー: パターンに一致するファイルが見つかりません: {input_pattern}")
    else:
        print(f"{len(file_list)}個のファイルが見つかりました。一括処理を開始します...\n")
        
        # クラスのインスタンスを作成
        processor = ImageProcessor()

        # ★ 取得したファイルリストを一つずつループ処理
        for input_file in file_list:
            # 入力ファイル名から拡張子を除いた名前を取得 (例: image01.jpg -> image01)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            print(f"--- 処理中: {base_name} ---")

            # ★ ファイルごとに出力先フォルダを動的に生成
            # これにより、各画像のパッチが混ざらないようにする
            output_path_for_file = os.path.join(output_base_dir, base_name)
            crop_file_color = os.path.join(output_path_for_file, 'color_patches')
            crop_file_binary = os.path.join(output_path_for_file, 'binary_patches')
            
            # フォルダがなければ作成
            os.makedirs(output_path_for_file, exist_ok=True)
            
            # クラスのメソッドを呼び出して処理を実行
            processor.crop(input_file, output_path_for_file, crop_file_color, crop_file_binary, threshold=128)

        print("\nすべての一括処理が完了しました。")