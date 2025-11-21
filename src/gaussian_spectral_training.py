"""
What this script does:
- Loads a COLMAP-style dataset (your VGGT → COLMAP output) using gsplat's
  Parser/Dataset wrapper.
- Initializes 3D Gaussians from SfM points (or random) with SH colors,
  using the same k-NN based scale heuristic as the original trainer.
- Trains the Gaussians with:
    * L1 + SSIM image loss
    * Optional depth loss (disparity space)
    * Opacity & scale regularization
    * Densification/pruning via DefaultStrategy (from gsplat.strategy),
      which is the main quality driver in the 3DGS paper.
- Saves a final .pt checkpoint and .ply mesh of the learned Gaussians.

Removed features (does NOT affect render quality):
- Viewer / GUI (viser, nerfview, GsplatViewer)
- TensorBoard logging and JSON stats
- PNG compression utilities
- LPIPS / PSNR / SSIM evaluation loops
- Multi-GPU / DDP and CLI routing
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from spectral_loss import spectral_anisotropy_loss


import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from fused_ssim import fused_ssim
from torch import Tensor
from typing_extensions import Literal

from datasets.colmap import Dataset, Parser
from utils import knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------

@dataclass
class Config:
    """
    High-level configuration for training.

    Changes needed when running training:
    - data_dir: path to COLMAP (VGGT) scene
    - result_dir: where checkpoints / renders go
    - max_steps: number of optimization steps
    """
    # Data / IO
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Normalize the world space
    normalize_world_space: bool = True
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Training schedule
    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0
    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Rendering / camera
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Use random background for training to discourage transparency
    random_bkgd: bool = False


    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Learning rates
    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Loss weights
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Densification / pruning strategy
    strategy: object = field(default_factory=lambda: DefaultStrategy(verbose=True))

    # For custom spectral loss
    use_spectral_loss: bool = False
    spectral_lambda: float = 0.01  # weight for custom spectral loss

    def adjust_steps(self, factor: float):
        """
        Scale all step-based hyperparameters and strategy schedule.
        This keeps densification timing consistent when changing max_steps.
        """
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)


# Gaussian initialization 

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    """
    Initialize 3D Gaussian parameters and their optimizers.

    - If init_type="sfm", uses SfM points / colors from the COLMAP parser.
      This is the standard, best-quality initialization.
    - If init_type="random", samples points in a bounding box.

    Returns:
        splats: ParameterDict containing means, scales, quats, opacities, SH/feature params
        optimizers: dict of separate optimizers for each parameter group
    """

    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize GS size as average distance to 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute GSs across ranks (still works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Learning rate scaling with batch size
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


# Core training logic

def rasterize_splats(
    splats: torch.nn.ParameterDict,
    cfg: Config,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    image_ids: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Render the current Gaussian set from given camera(s) using gsplat's rasterizer.

    - Exp on log-scales
    - Sigmoid on logit-opacities
    - SH-based colors 
    """
    means = splats["means"]                       # [N, 3]
    quats = splats["quats"]                       # [N, 4] (normalized in CUDA)
    scales = torch.exp(splats["scales"])          # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N]

    # Concatenate SH band 0 and higher bands
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]

    rasterize_mode = "antialiased" if cfg.antialiased else "classic"

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,                                    # [C, 3, 3]
        width=width,
        height=height,
        packed=cfg.packed,
        absgrad=cfg.strategy.absgrad if isinstance(cfg.strategy, DefaultStrategy) else False,
        sparse_grad=cfg.sparse_grad,
        rasterize_mode=rasterize_mode,
        distributed=False,
        camera_model=cfg.camera_model,
    )

    if masks is not None:
        render_colors[~masks] = 0

    return render_colors, render_alphas, info


def train(cfg: Config):
    """
    Full training loop:
    - Loads dataset
    - Initializes splats
    - Runs optimization with densification strategy and losses
    """
    set_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directories
    os.makedirs(cfg.result_dir, exist_ok=True)
    ckpt_dir = Path(cfg.result_dir) / "ckpts"
    ply_dir = Path(cfg.result_dir) / "ply"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Dump config for reproducibility
    with open(Path(cfg.result_dir) / "cfg.yml", "w") as f:
        yaml.dump(vars(cfg), f)

    # Data: use gsplat's COLMAP parser and dataset wrapper
    #Parser reads the data from data dir (VGGT): camera, poses, points, images
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
    )
    trainset = Dataset(
        parser,
        split="train",
        patch_size=cfg.patch_size,
        load_depths=cfg.depth_loss,
    )
    #used in depth loss
    scene_scale = parser.scene_scale * 1.1 * cfg.global_scale
    print("Scene scale:", scene_scale)

    # Model: initialize Gaussians + optimizers, then setup strategy
    
    splats, optimizers = create_splats_with_optimizers(
        parser,
        init_type=cfg.init_type,
        init_num_pts=cfg.init_num_pts,
        init_extent=cfg.init_extent,
        init_opacity=cfg.init_opa,
        init_scale=cfg.init_scale,
        means_lr=cfg.means_lr,
        scales_lr=cfg.scales_lr,
        opacities_lr=cfg.opacities_lr,
        quats_lr=cfg.quats_lr,
        sh0_lr=cfg.sh0_lr,
        shN_lr=cfg.shN_lr,
        scene_scale=scene_scale,
        sh_degree=cfg.sh_degree,
        sparse_grad=cfg.sparse_grad,
        visible_adam=False,
        batch_size=cfg.batch_size,
        feature_dim=None,
        device=device,
        world_rank=0,
        world_size=1,
    )
    print("Model initialized. Number of GS:", splats["means"].shape[0])

    # Densification strategy sanity check + initial state
    cfg.strategy.check_sanity(splats, optimizers)
    strategy_state = cfg.strategy.initialize_state(scene_scale=scene_scale)

    # Optimizer schedules (same as original for means)
    max_steps = cfg.max_steps
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"],
            gamma=0.01 ** (1.0 / max_steps),
        )
    ]

    # DataLoader
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    # Main optimization loop
    global_tic = time.time()
    pbar = tqdm.tqdm(range(max_steps))
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)     # [1, 4, 4]
        Ks = data["K"].to(device)                       # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0       # [1, H, W, 3]
        image_ids = data["image_id"].to(device)         # [1]
        masks = data["mask"].to(device) if "mask" in data else None

        if cfg.depth_loss:
            points = data["points"].to(device)          # [1, M, 2]
            depths_gt = data["depths"].to(device)       # [1, M]

        height, width = pixels.shape[1:3]

        # SH degree schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # Render from current splats
        renders, alphas, info = rasterize_splats(
            splats=splats,
            cfg=cfg,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            image_ids=image_ids,
            masks=masks,
        )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        # Strategy pre-backward hook (densification / stats)
        cfg.strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        # Loss computation
        #Compares ground truth rgb colors with predicted rgb colors
        #per-pixel brightness error
        l1loss = F.l1_loss(colors, pixels)

        #SSIM expects channels first thats why permute is used
        #Structural Similarity Index between x and y : [0,1]
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2),
            pixels.permute(0, 3, 1, 2),
            padding="valid",
        )
        #SSim helps to ignore small color shifts
        #Mix L1 and SSIM using cfg.ssim_lambda:
        #    - cfg.ssim_lambda = 0 → pure L1
        #    - cfg.ssim_lambda = 1 → pure SSIM
        photo_loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        # 4) Spectral anisotropy regularization (our custom cost from Spectral-GS)
        #    Idea:
        #      - splats["scales"] stores log-scales for each Gaussian along x,y,z.
        #      - We interpret exp(scales) as axis lengths s1, s2, s3.
        #      - spectral_anisotropy_loss:
        #           * computes spectral entropy from (s1^2, s2^2, s3^2)
        #           * returns a penalty in [0,1] that is HIGH for needle-like GSs
        #             and LOW for isotropic GSs.
        #
        #    We keep two tensors:
        #      - spectral_loss : scalar penalty we add to the main loss
        #      - mean_entropy  : average normalized entropy, just for logging
        device = colors.device
        spectral_loss = torch.tensor(0.0, device=device)
        mean_entropy = torch.tensor(0.0, device=device)

        if cfg.spectral_lambda > 0.0:
            # splats["scales"] is (N, 3) log-scale per Gaussian
            scales_log = splats["scales"]

            # spectral_anisotropy_loss returns:
            #   loss_scalar: mean(1 - H_norm) over Gaussians
            #   H_mean:      mean(H_norm) in [0,1], for monitoring
            spectral_loss, mean_entropy = spectral_anisotropy_loss(scales_log)

            # 5) Total loss = photometric term + λ * spectral regularization
            loss = photo_loss + cfg.spectral_lambda * spectral_loss
        else:
            # If spectral regularization is disabled (λ = 0),
            # we just use the photometric loss.
            loss = photo_loss

        if cfg.depth_loss and depths is not None:
            # Sample depths at sparse points and compare disparity
            pts_norm = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )  # [-1, 1]
            grid = pts_norm.unsqueeze(2)  # [1, M, 1, 2]
            depths_samp = F.grid_sample(
                depths.permute(0, 3, 1, 2),
                grid,
                align_corners=True,
            )  # [1, 1, M, 1]
            depths_samp = depths_samp.squeeze(3).squeeze(1)  # [1, M]

            disp = torch.where(
                depths_samp > 0.0,
                1.0 / depths_samp,
                torch.zeros_like(depths_samp),
            )
            disp_gt = 1.0 / depths_gt
            depthloss = F.l1_loss(disp, disp_gt) * scene_scale
            loss += depthloss * cfg.depth_lambda

        # Optional spot to plug your custom Spectral-GS loss:
        # if cfg.use_spectral_loss:
        #     from spectral_loss import spectral_loss
        #     spec = spectral_loss(colors, pixels, info)
        #     loss += cfg.spectral_lambda * spec

        # Regularizers
        if cfg.opacity_reg > 0.0:
            loss += cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if cfg.scale_reg > 0.0:
            loss += cfg.scale_reg * torch.exp(splats["scales"]).mean()

        # Backward + optimizer + strategy post-backward
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        loss.backward()

        for opt in optimizers.values():
            opt.step()
        for sched in schedulers:
            sched.step()

        # Strategy post-backward hook (densification, pruning)
        if isinstance(cfg.strategy, DefaultStrategy):
            cfg.strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            cfg.strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0],
            )

        # Progress bar description
        pbar.set_description(f"step={step} loss={loss.item():.4f} photo={photo_loss.item():.4f}"
                             f"spec={spectral_loss.item():.4f} H={mean_entropy.item():.3f} sh={sh_degree_to_use}")

        # Save checkpoints / PLY occasionally
        if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
            ckpt_path = ckpt_dir / f"train_step{step:04d}.pt"
            torch.save({"step": step, "splats": splats.state_dict()}, ckpt_path)
            print(f"[CKPT] Saved to {ckpt_path}")

        if step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1:
            ply_path = ply_dir / f"train_step{step:04d}.ply"
            export_splats(
                path=str(ply_path),
                means=splats["means"].detach().cpu(),
                quats=splats["quats"].detach().cpu(),
                scales=torch.exp(splats["scales"]).detach().cpu(),
                opacities=torch.sigmoid(splats["opacities"]).detach().cpu(),
                shs=torch.cat([splats["sh0"], splats["shN"]], dim=1).detach().cpu(),
            )
            print(f"[PLY] Saved to {ply_path}")

    print("Training finished in", time.time() - global_tic, "seconds.")


# CLI entrypoint

def main():
    # Only one config preset: "default". You can extend if you want.
    configs = {
        "default": (
            "Standard 3DGS training with densification, single GPU.",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    train(cfg)


if __name__ == "__main__":
    main()
