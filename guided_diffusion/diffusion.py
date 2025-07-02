import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

# --- 必要なモジュールをインポート ---
import unet
import nn

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.ddpg_scheme import ddpg_diffusion

import torchvision.utils as tvu

# DDPGプロジェクトの元々のモデルクラスもインポートしておく
from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

import lpips


# (get_gaussian_noisy_imgなどのヘルパー関数は元のままなので、ここでは省略します)
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
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64,) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    # (他のbeta_scheduleの分岐は元のまま)
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

        self.model_var_type = getattr(config.model, 'var_type', 'fixedsmall')
        
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

        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)

    def sample(self, logger):
        cls_fn = None
        
        print("Loading custom U-Net model from unet.py...")

        model = unet.UNetModel(
            image_size=self.config.data.image_size,
            in_channels=self.config.model.in_channels,
            model_channels=self.config.model.model_channels,
            out_channels=self.config.model.out_channels,
            num_res_blocks=self.config.model.num_res_blocks,
            attention_resolutions=tuple(self.config.model.attention_resolutions),
            num_classes=getattr(self.config.model, 'num_classes', None)
        )
        
        ckpt_path = self.config.model.model_path
        print(f"Loading checkpoint from: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            print("Loading from 'model_state_dict' key in checkpoint...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("Loading entire checkpoint as state_dict...")
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)
        
        if self.args.inject_noise==1:
            print('Run DDPG.',
                  f'Operators implementation via {self.args.operator_imp}.',
                  f'{self.config.sampling.T_sampling} sampling steps.',
                  f'Task: {self.args.deg}.',
                  f'Noise level: {self.args.sigma_y}.'
                  )
        else:
            print('Run IDPG.')
            # (...
        
        self.ddpg_wrapper(model, cls_fn, logger)

    # class Diffusion(object): の中の ddpg_wrapper メソッド全体を、この内容に置き換えてください

# class Diffusion(object): の中の ddpg_wrapper メソッド全体を、この内容に置き換えてください

# class Diffusion(object): の中の ddpg_wrapper メソッド全体を、この内容に置き換えてください

    def ddpg_wrapper(self, model, cls_fn, logger):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        
        if test_dataset is None:
            raise ValueError("Failed to load dataset. Check dataset configuration and paths.")

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        
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
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        deg = args.deg
        A_funcs = None
        if deg == 'stain_removal':
             from functions.stain_removal_operator import StainRemovalOperator
             input_image_path = args.path_y
             base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
             mask_filename = f"{base_filename}_stain_mask.png"
             mask_folder = 'mask_output' 
             mask_path = os.path.join(mask_folder, mask_filename)
             print(f"Attempting to load mask for {input_image_path} from: {mask_path}")
             A_funcs = StainRemovalOperator(self.device, self.config.data.image_size, self.config.data.channels, mask_path=mask_path)
        else:
             raise ValueError(f"degradation type '{deg}' not supported for this project")

        sigma_y = args.sigma_y * 2
        print(f'Start from {args.subset_start}')
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)

        # --- ▼▼▼ デバッグプリントを追加したループ ▼▼▼ ---
        print("\n--- DEBUG: Starting main loop over val_loader... ---")
        for i, (x_orig, classes) in enumerate(pbar):
            print(f"--- DEBUG: --- Loop iteration {i+1} START ---")
            x_orig = x_orig.to(self.device)
            x_orig_transformed = data_transform(config, x_orig)

            y = A_funcs.A(x_orig_transformed)
            y = y + sigma_y * torch.randn_like(y)

            print("--- DEBUG: Starting ddpg_diffusion... ---")
            with torch.no_grad():
                x_restored_output, _ = ddpg_diffusion(
                    torch.randn_like(x_orig_transformed),
                    model, self.betas, A_funcs, y, sigma_y,
                    cls_fn=cls_fn, classes=classes.to(self.device),
                    config=config, args=args
                )
            print("--- DEBUG: ddpg_diffusion FINISHED. ---")

            x_restored = x_restored_output[0]
            x_restored_saved = inverse_data_transform(config, x_restored)

            print(f"--- DEBUG: Starting inner loop for saving (batch size: {x_orig.shape[0]}) ---")
            for j in range(x_orig.shape[0]):
                print(f"--- DEBUG: Inner loop j = {j} ---")
                save_path = os.path.join(self.args.image_folder, f"{idx_so_far + j}_restored.png")
                
                image_to_save = x_restored_saved[j]
                
                print(f"DEBUG: Attempting to save to: {os.path.abspath(save_path)}")
                print(f"DEBUG: Shape of tensor to save: {image_to_save.shape}")
                print(f"DEBUG: Data type: {image_to_save.dtype}")
                print(f"DEBUG: Min value: {image_to_save.min():.4f}, Max value: {image_to_save.max():.4f}")

                tvu.save_image(image_to_save, save_path)
                print(f"--- DEBUG: tvu.save_image called for j = {j}. Check if file exists. ---")
                
                # (評価指標の計算は省略しても良いが、念のため残す)
                x_orig_saved = inverse_data_transform(config, x_orig_transformed)
                mse = torch.mean((x_restored_saved[j] - x_orig_saved[j].to(x_restored_saved.device)) ** 2)
                psnr = 10 * torch.log10(1 / mse) if mse > 0 else 100.0
                avg_psnr += psnr.item()
                
                log_message = (f"Image {idx_so_far + j}: Saved to {save_path}, "
                               f"PSNR: {psnr:.2f}")
                logger.info(log_message)
                print(log_message)

            idx_so_far += x_orig.shape[0]
            print(f"--- DEBUG: --- Loop iteration {i+1} END ---")

        print("--- DEBUG: Finished all loops. ---")
        # --- ▲▲▲ ここまで ---

        num_samples = idx_so_far - args.subset_start
        if num_samples > 0:
            avg_psnr /= num_samples
        
        print(f"Total Average PSNR: {avg_psnr:.2f}")
        print(f"Number of samples: {num_samples}")