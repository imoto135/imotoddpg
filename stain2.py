import os
import numpy as np
import glob
import scipy.ndimage as ndi
from skimage import io

# グレースケール画像から文字マスクを作成（暗い部分を文字とみなす）
def get_text_mask(img, threshold=150):
	"""画像から文字領域のマスクを生成する"""
	gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
	mask = (gray < threshold).astype(np.uint8)[:, :, None]	# 文字領域（暗）→ 1
	return mask

# ランダムなブロブ状の汚れマスク生成（RGB画像）
def generate_stain_mask(H, W, color=None, density=3e-4, size=30, roughness=2.0, seed=None):
	"""リアルな汚れのマスクをアルゴリズムで生成する"""
	if seed is not None:
		np.random.seed(seed)

	numblobs = int(density * H * W)
	mask = np.zeros((H, W), 'i')
	for _ in range(numblobs):
		mask[np.random.randint(0, H), np.random.randint(0, W)] = 1

	dt = ndi.distance_transform_edt(1 - mask)
	mask = (dt < size).astype(np.float32)
	mask = ndi.gaussian_filter(mask, size / (2 * roughness))
	mask -= mask.min()
	mask /= mask.max()

	noise = np.random.rand(H, W)
	noise = ndi.gaussian_filter(noise, size / (2 * roughness))
	noise -= noise.min()
	noise /= noise.max()

	blob = (mask * noise > 0.5).astype(np.float32)

	if color is None:
		r = np.random.randint(170, 200)
		g = np.random.randint(110, 150)
		b = np.random.randint(90, 120)
		color = np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

	stain_mask = (blob[:, :, None] * color).astype(np.uint8)
	return stain_mask, color

# 汚れ適用処理
def apply_stain(img, stain_mask, text_mask=None, avoid_text=True, intensity=1.0):
	"""画像に汚れマスクを合成する"""
	img = img.astype(np.float32)
	stain = stain_mask.astype(np.float32)

	if avoid_text and text_mask is not None:
		# 文字を避けたブレンド
		bg_mask = (1 - text_mask).astype(np.float32)
		stain_area = np.mean(stain, axis=2, keepdims=True) > 10
		blend_mask = bg_mask * stain_area
		result = img * (1 - blend_mask * intensity) + stain * (blend_mask * intensity)
		result = np.clip(result, 0, 255).astype(np.uint8)
	else:
		# 文字含めて全体に汚れをブレンド
		stain_area = np.mean(stain, axis=2, keepdims=True) > 10
		result = img * (1 - stain_area * intensity) + stain * (stain_area * intensity)
		result = np.clip(result, 0, 255).astype(np.uint8)
	return result

# [追加] 汚れ色をseedから再現性をもってランダム生成する関数
def generate_fixed_color(seed):
	"""seed値に基づいて再現性のあるランダムな色を生成する"""
	np.random.seed(seed)
	r = np.random.randint(170, 200)
	g = np.random.randint(110, 150)
	b = np.random.randint(90, 120)
	return np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

# メインの処理関数
def process_images_with_multiple_params(input_pattern, output_base_dir, density=3e-4, intensities=[0.5], seed=42):
	"""指定されたパターンに一致する全画像に、複数のパラメータで汚れを付与し、保存する"""
	# --- 1. 画像ファイルのリストを取得 ---
	files = glob.glob(input_pattern)
	if not files:
		print(f"指定されたパスに画像が見つかりません: {input_pattern}")
		return
	
	print(f"合計 {len(files)} 個のファイルが見つかりました。")

	# --- 2. [変更点] 複数のintensityでループ処理 ---
	for intensity in intensities:
		print(f"--- Intensity = {intensity} の処理を開始 ---")

		# --- 3. [変更点] パラメータごとにフォルダを動的に作成 ---
		param_str = f"density_{density:.0e}_intensity_{intensity:.2f}"
		output_dir = os.path.join(output_base_dir, param_str)
		
		with_text_path = os.path.join(output_dir, "with_text")
		no_text_path = os.path.join(output_dir, "no_text")
		stain_mask_path = os.path.join(output_dir, "stain_masks")
		text_mask_path = os.path.join(output_dir, "text_masks")

		os.makedirs(with_text_path, exist_ok=True)
		os.makedirs(no_text_path, exist_ok=True)
		os.makedirs(stain_mask_path, exist_ok=True)
		os.makedirs(text_mask_path, exist_ok=True)

		# --- 4. 各画像をループ処理 ---
		for i, fname in enumerate(sorted(files)):
			try:
				img = io.imread(fname)
				H, W, _ = img.shape
				base_fname = os.path.basename(fname)
				
				# --- 5. [変更点] 画像ごとに再現性のあるランダム色を生成 ---
				file_seed = seed + i
				fixed_color = generate_fixed_color(file_seed)

				# 汚れマスクと文字マスクを生成
				stain_mask, _ = generate_stain_mask(H, W, color=fixed_color, density=density, seed=file_seed)
				text_mask = get_text_mask(img)

				# マスクを画像として保存
				io.imsave(os.path.join(stain_mask_path, base_fname), stain_mask)
				text_mask_to_save = (np.squeeze(text_mask) * 255).astype(np.uint8)
				io.imsave(os.path.join(text_mask_path, base_fname), text_mask_to_save)

				# 汚れを合成した画像を2種類作成し、保存
				img_with_text = apply_stain(img, stain_mask, avoid_text=False, intensity=intensity)
				img_no_text = apply_stain(img, stain_mask, text_mask, avoid_text=True, intensity=intensity)

				io.imsave(os.path.join(with_text_path, base_fname), img_with_text)
				io.imsave(os.path.join(no_text_path, base_fname), img_no_text)
			except Exception as e:
				print(f"ファイル {fname} の処理中にエラーが発生しました: {e}")
		
		print(f"[Intensity={intensity}] {len(files)}枚の処理が完了しました。")

	print(f"--- 全ての処理が完了しました ---")
	print(f"出力先ベースフォルダ: {output_base_dir}")


# --- スクリプトの実行部分 ---
if __name__ == '__main__':
	# 実行時のパラメータ設定
	process_images_with_multiple_params(
		input_pattern="/home/ihpc/anaconda3/envs/imoto/imotoddpg/makura/image/*.jpg",
		output_base_dir="/home/ihpc/anaconda3/envs/imoto/imotoddpg/makura2",
		density=5e-4,
		intensities=[0.3, 0.5, 0.8, 1.0], # 複数の汚れの濃さをリストで指定
		seed=12345
	)