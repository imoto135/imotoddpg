import torch
import numpy as np
from PIL import Image

class StainRemovalOperator:
    """
    シミ除去タスクのためのOperatorクラス。
    平坦化されたテンソルを受け取り、内部で4D画像に復元して処理する。
    """
    def __init__(self, device, image_size, channels, mask_path='masks/stain_mask.png'):
        """
        コンストラクタ

        Args:
            device: 使用するデバイス (e.g., 'cuda')
            image_size (int): 画像のサイズ (e.g., 64)
            channels (int): 画像のチャンネル数 (e.g., 3)
            mask_path (str): シミの領域を示すマスク画像のパス
        """
        self.device = device
        self.image_size = image_size
        self.channels = channels
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        
        # シミの部分を0, それ以外を1とする
        self.mask = (mask > 0.5).float().to(device)

    def _reshape_and_apply_mask(self, x_flat):
        # x_flatは (バッチサイズ, チャンネル*高さ*幅) の形状
        batch_size = x_flat.shape[0]
        
        # 1. 4D画像テンソルに形状を戻す
        x_4d = x_flat.view(batch_size, self.channels, self.image_size, self.image_size)
        
        # 2. マスクを適用する (maskは[H,W]なので、B,Cにブロードキャストされる)
        masked_x_4d = x_4d * self.mask.unsqueeze(0).unsqueeze(0)
        
        # 3. 再び平坦化して返す
        return masked_x_4d.view(batch_size, -1)

    def A(self, x):
        """順問題: y = Ax"""
        return self._reshape_and_apply_mask(x)

    def At(self, y):
        """随伴作用素: x = A^T(y)"""
        return self._reshape_and_apply_mask(y)

    def A_pinv(self, y, alpha=0.0):
        """擬似逆作用素"""
        return self._reshape_and_apply_mask(y)

    def A_pinv_add_eta(self, y, eta):
        """DDPGのコードが要求する可能性のあるメソッド"""
        return self._reshape_and_apply_mask(y)