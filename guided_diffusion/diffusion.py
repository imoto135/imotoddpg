import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

# --- ▼▼▼ 修正点 1/3: 必要なモジュールをインポート ---
# 以前の会話に基づき、学習時に使った unet と nn をインポート
import unet
import nn
# --- ▲▲▲ ---

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.ddpg_scheme import ddpg_diffusion

import torchvision.utils as tvu

# DDPGプロジェクトの元々のモデルクラスもインポートしておく
from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

import lpips


loss_fn_alex = lpips.LPIPS(net='alex') # net='alex' best forward scores


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # --- ▼▼▼ 修正点 2/3: YMLにvar_typeがなくてもエラーにならないように修正 ---
        self.model_var_type = getattr(config.model, 'var_type', 'fixedsmall')
        # --- ▲▲▲ ---

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    # --- ▼▼▼ 修正点 3/3: sampleメソッド全体を、カスタムモデルを読み込むロジックに書き換え ---
    def sample(self, logger):
        cls_fn = None
        
        print("Loading custom U-Net model from unet.py...")

        # 1. YMLの設定を元に、学習時と同じunet.UNetModelの骨格を作成する
        model = unet.UNetModel(
            image_size=self.config.data.image_size, # ← ★この行を追加
            in_channels=self.config.model.in_channels,
            model_channels=self.config.model.model_channels,
            out_channels=self.config.model.out_channels,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=tuple(self.config.model.attention_resolutions),
            num_classes=getattr(self.config.model, 'num_classes', None)
        )
        
        # 2. YMLで指定されたパスから重みを読み込む
        ckpt_path = self.config.model.model_path
        print(f"Loading checkpoint from: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 3. チェックポイントからモデルの重みだけを取り出して読み込む
        if 'model_state_dict' in checkpoint:
            print("Loading from 'model_state_dict' key in checkpoint...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading entire checkpoint as state_dict...")
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval() # 推論モードに設定
        model = torch.nn.DataParallel(model)
        
        # (元のコードの inject_noise の判定から下は変更なし)
        if self.args.inject_noise==1:
            print('Run DDPG.',
                  f'Operators implementation via {self.args.operator_imp}.',
                  f'{self.config.sampling.T_sampling} sampling steps.',
                  f'Task: {self.args.deg}.',
                  f'Noise level: {self.args.sigma_y}.'
                  )
        else:
            print('Run IDPG.',
                  f'Operators implementation via {self.args.operator_imp}.',
                  f'{self.config.sampling.T_sampling} sampling steps.',
                  f'Task: {self.args.deg}.',
                  f'Noise level: {self.args.sigma_y}.'
                  )
        
        self.ddpg_wrapper(model, cls_fn, logger)
    # --- ▲▲▲ ここまでが修正されたsampleメソッド ---


    def ddpg_wrapper(self, model, cls_fn, logger):
        args, config = self.args, self.config

        # このget_dataset関数は、DDPGプロジェクトに元々あるものを使います。
        # path_yが指定されていれば、それを元に単一の画像データセットを作成するはずです。
        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        
        # --- ▼▼▼ Operatorの選択ロジック（ここは元のコードのまま）▼▼▼ ---
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        # ... (他の多くのelif節も元のコードのままなので省略) ...
        elif deg == 'stain_removal': # 以前追加したカスタムタスク
            from functions.stain_removal_operator import StainRemovalOperator
            A_funcs = StainRemovalOperator(self.device, self.config.data.image_size, mask_path='masks/your_stain_mask.png') # マスクのパスは要確認
        else:
            raise ValueError("degradation type not supported")
        # --- ▲▲▲ Operatorの選択ロジック ▲▲▲ ---
        
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)

        img_ind = -1

        for x_orig, classes in pbar:
            img_ind = img_ind + 1
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)
            
            y = y + args.sigma_y*torch.randn_like(y).cuda()
            
            # (yのreshape処理などは元のコードのままなので省略)
            # ...
            
            # initialize x
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                x, _ = ddpg_diffusion(x, model, self.betas, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config, args=args)
            
            # (LPIPSやPSNRの計算、画像の保存処理なども元のコードのままなので省略)
            # ...
            
            idx_so_far += y.shape[0]
            logger.info("Avg PSNR: %.2f, Avg LPIPS: %.4f      (** After %d iteration **)" % (avg_psnr / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init), idx_so_far - idx_init))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_lpips = avg_lpips / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % (idx_so_far - idx_init))