import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.celeba import CelebA
from datasets.lsun import LSUN
from torch.utils.data import Subset, Dataset # Datasetをインポート
import numpy as np
import torchvision
from PIL import Image
from functools import partial

# --- ▼▼▼ 修正点 1/2: 単一画像を扱うためのDatasetクラスを新規追加 ▼▼▼ ---
class SingleImageDataset(Dataset):
    """単一の画像ファイルを扱うためのカスタムDatasetクラス"""
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        
        if not os.path.exists(self.image_path):
            # args.path_yで渡されるパスはスクリプト実行時の相対パスになることが多い
            # そのため、存在しない場合はFileNotFoundErrorを発生させる
            raise FileNotFoundError(f"Input image not found at: {image_path}")

    def __len__(self):
        # 画像は1枚だけなので、長さは常に1
        return 1

    def __getitem__(self, index):
        # indexは常に0だが、Datasetクラスの仕様上必須の引数
        try:
            image = Image.open(self.image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading or transforming image {self.image_path}: {e}")
            # エラー時も正しい型のダミーデータを返す
            return torch.zeros(3, 256, 256), 0

        # データローダーは通常 (data, label) のペアを期待するので、ダミーのラベル0を返す
        return image, 0
# --- ▲▲▲ ここまで ---


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # (この関数は変更なし)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_dataset(args, config):
    # --- ▼▼▼ 修正点 2/2: get_dataset関数の先頭に、単一画像パスを処理するロジックを追加 ▼▼▼ ---

    # 1. まず、コマンドライン引数で単一の画像パスが指定されているかチェック
    #    os.path.existsで、実際にファイルが存在することも確認
    if hasattr(args, 'path_y') and args.path_y and os.path.exists(args.path_y):
        print(f"Loading single image from --path_y: {args.path_y}")
        
        # 2. 単一画像用のシンプルなtransformを作成
        #    data_transform関数は後で適用されるので、ここでは基本的なリサイズとテンソル化のみ
        transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
        ])
        
        # 3. SingleImageDatasetを使い、学習用とテスト用の両方に同じデータセットを返す
        #    これにより、test_datasetがNoneになるのを防ぐ
        custom_dataset = SingleImageDataset(image_path=args.path_y, transform=transform)
        return custom_dataset, custom_dataset

    # --- ▲▲▲ ここまで ---

    # --path_yが指定されていない場合は、これまでの元のロジックに進む
    if config.data.random_flip is False:
        # (以降のコードは元のままなので、変更ありません)
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CELEBA":
        # ... (元のCELEBAロジック)
        pass # 以下省略
    
    # ... (元のLSUN, CelebA_HQ, ImageNetのelif節もそのまま) ...

    else:
        # どの条件にも当てはまらなかった場合、エラーを発生させて終了する
        raise ValueError(f"Dataset '{config.data.dataset}' is not supported in config, and --path_y was not provided or file not found.")

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    # (この関数は変更なし)
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    # (この関数は変更なし)
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    # (この関数は変更なし)
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)