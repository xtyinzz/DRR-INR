"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
from args import parse_args
import yaml
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from nerfacc_scripts.radiance_fields.ngp import NGPRadianceField
from configs.configs import Config

from nerfacc_scripts.datasets.nerf_synthetic import SubjectLoader

from nerfacc_scripts.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def run(args, config):
    device = args.device
    set_random_seed(args.random_seed)



    # training parameters
    max_steps = args.maxsteps
    init_batch_size = 1024
    target_sample_batch_size = 1 << 16
    weight_decay = (
        1e-5 if args.nerfacc_scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3

    train_dataset = SubjectLoader(
        subject_id=args.nerfacc_scene,
        root_fp=args.nerfacc_data_root,
        split=args.nerfacc_train_split,
        num_rays=init_batch_size,
        device=device,
    )

    test_dataset = SubjectLoader(
        subject_id=args.nerfacc_scene,
        root_fp=args.nerfacc_data_root,
        split="test",
        num_rays=None,
        device=device,
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)
    print('Estimator', estimator.aabbs)
    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    # radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
    # if config.config['model']['type'] == 'NGPRadianceField':
    #     radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
    # else:
    radiance_field = config.get_model().to(device)

    # if hasattr(radiance_field, 'encoder_kwargs') and radiance_field.encoder_kwargs.get('if_refine', False):
    #     refiner_params = set(radiance_field.mlp_base[0].sp.parameters())
    #     other_params = set(radiance_field.parameters()) - refiner_params
    #     optimizer = torch.optim.Adam([
    #         {'params': list(refiner_params), 'lr': 1e-4, 'weight_decay': weight_decay},
    #         {'params': list(other_params), 'lr': 1e-2, 'weight_decay': weight_decay}
    #     ], eps=1e-15)
    #     print(f'Optimizer for DRR-NeRF are initialized')
    # else:
    optimizer = torch.optim.Adam(
        radiance_field.parameters(),
        lr=1e-2,
        eps=1e-15,
        weight_decay=weight_decay,
    )
    
    # ---- Verification Step ----
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group['params'])
    
    model_params = set(radiance_field.parameters())
    
    unassigned_params = model_params - optimizer_params
    if unassigned_params:
        print("WARNING: The following parameters are not assigned to the optimizer:")
        for param in unassigned_params:
            # This is a simplified way to identify the parameter.
            # For a more precise location, you'd need to traverse the model.
            print(f"  - Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    # ---- End Verification ----

    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
            ),
        ]
    )
    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        if step % args.reportsteps == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

        if step > 0 and step % max_steps == 0:
            # evaluation
            radiance_field.eval()
            estimator.eval()
            psnrs = []
            lpips = []
            with torch.no_grad():
                if hasattr(radiance_field, 'encoder_kwargs') and radiance_field.mlp_base[0].if_refine:
                    radiance_field.mlp_base[0].refine_spatial()
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    # rgb, acc, depth, _ = render_image_with_occgrid_test(
                    #     1024,
                    #     # scene
                    #     radiance_field,
                    #     estimator,
                    #     rays,
                    #     # rendering options
                    #     near_plane=near_plane,
                    #     render_step_size=render_step_size,
                    #     render_bkgd=render_bkgd,
                    # )
                    rgb, acc, depth, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        rays,
                        # rendering options
                        near_plane=near_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    lpips.append(lpips_fn(rgb, pixels).item())
                    # if i == 0:
                    #     imageio.imwrite(
                    #         "rgb_test.png",
                    #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                    #     )
                    #     imageio.imwrite(
                    #         "rgb_error.png",
                    #         (
                    #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                    #         ).astype(np.uint8),
                    #     )
            psnr_avg = sum(psnrs) / len(psnrs)
            lpips_avg = sum(lpips) / len(lpips)
            print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    print(os.getcwd())
    with open(args.config, 'r') as f:
        config_list = yaml.load(f, yaml.FullLoader)

    args.config_id = args.config_id[0]
    print(f"Running config_id {args.config_id}")
    config = Config(config_list[args.config_id])
    config.set_args(args)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run(args, config)
