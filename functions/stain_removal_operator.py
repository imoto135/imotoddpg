# functions/stain_removal_operator.py

import torch
import numpy as np
from PIL import Image

class StainRemovalOperator:
    """
    シミ除去タスクのためのOperatorクラス。
    シミの領域をマスクとして扱い、インペインティング問題として解く。
    """
    def __init__(self, device, image_size, mask_path='masks/stain_mask.png'):
        """
        コンストラクタ

        Args:
            device: 使用するデバイス (e.g., 'cuda')
            image_size (int): 画像のサイズ (e.g., 256)
            mask_path (str): シミの領域を示すマスク画像のパス
        """
        self.device = device
        
        # マスク画像を読み込み、前処理を行う
        mask = Image.open(mask_path).convert('L') # グレースケールで読み込み
        mask = mask.resize((image_size, image_size), Image.Resampling.LANCZOS)
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        
        # シミの部分を0, それ以外を1とする。
        # 画像処理ソフトでシミを黒(0), 背景を白(255)で作成することを想定。
        self.mask = (mask > 0.5).float().to(device)
        self.mask_flat = self.mask.view(-1) # 1次元に展開したマスク

    def A(self, x):
        """
        順問題 (Forward Operator): y = Ax
        クリーンな画像xから劣化画像yを生成する。
        ここでは、マスクを乗算してシミ領域の情報を欠損させる。
        """
        # (バッチサイズ, チャンネル数, 高さ, 幅) の次元を合わせる
        return x * self.mask.unsqueeze(0).unsqueeze(0)

    def A_T(self, y):
        """
        随伴作用素 (Adjoint Operator): x = A^T(y)
        マスクを乗算するだけの単純な作用素の場合、A_TはAと同じになる。
        """
        return y * self.mask.unsqueeze(0).unsqueeze(0)

    def A_pinv(self, y, alpha=0.0):
        """
        擬似逆作用素 (Pseudo-inverse Operator)
        基本的にはA_Tと同じだが、安定性のための正則化項alphaを加えることもできる。
        今回は単純な実装とする。
        """
        return y * self.mask.unsqueeze(0).unsqueeze(0)

    # DDPG/DDRMのコードが `A_pinv_add_eta` のような別の名前を要求している場合、
    # それに合わせてメソッド名や引数を調整する必要があるかもしれません。
    # 例えば、ユーザーのコードベースには以下のようなメソッドが存在する可能性があります。
    def A_pinv_add_eta(self, y, eta):
        """
        DDPGのコードが要求する可能性のあるメソッド。
        etaはノイズのレベルに応じた正則化項。
        """
        # 最も単純な実装は、etaを無視してA_pinvと同じ処理をすることです。
        # より高度な実装では、etaを使って計算を安定させます。
        return self.A_pinv(y)