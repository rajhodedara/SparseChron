"""Configuration management utilities.

This module provides configuration classes for the SparseChron project,
including training hyperparameters, hardware settings, and scene configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for training the SparseChron model.

    Attributes:
        scene_dir: Directory containing the scene data.
        output_dir: Directory to save outputs and checkpoints.
        max_iterations: Maximum number of training iterations.
        gradient_accumulation_steps: Number of micro-batches per optimizer step.
        debug_single_batch: Restrict training to a single image (smoke test).
        seed: Global random seed for reproducibility.
        lr_position: Learning rate for Gaussian positions.
        lr_sh: Learning rate for spherical harmonics.
        lr_opacity: Learning rate for opacity.
        lr_scale: Learning rate for scale.
        lr_rotation: Learning rate for rotation.
        lr_deformation: Learning rate for deformation network.
        lr_final_factor: Exponential LR decay target; LRs decay to
            lr * lr_final_factor by max_iterations (1.0 disables decay).
        lambda_depth: Weight for depth loss.
        lambda_deform_reg: Weight for deformation regularization loss.
        lambda_temporal_smoothness: Weight for temporal smoothness loss.
        temporal_sample_size: Number of dynamic Gaussians sampled per iteration
            for the temporal smoothness loss.
        ssim_weight: Weight for SSIM in the combined L1+SSIM loss.
        densify_from_iter: Iteration to start densification.
        densify_until_iter: Iteration to stop densification.
        densify_interval: Interval between densification steps.
        prune_interval: Interval between pruning steps.
        max_gaussians: Maximum number of Gaussians to keep in memory.
        densify_grad_threshold: Positional-gradient threshold for densification
            (3DGS-style; 2e-4 matches the reference implementation).
        split_scale_threshold: Activated max-scale above which a Gaussian is
            split (2 children) instead of cloned, in normalized scene units.
        clone_noise: Positional jitter applied to cloned Gaussians.
        prune_min_opacity: Gaussians below this activated opacity are pruned.
        prune_max_scale: Gaussians whose max activated scale exceeds this are pruned.
        normalize_scene: Rescale init points and cameras into a unit box so
            scale-dependent thresholds (split/prune/noise) stay meaningful.
        max_displacement: Clamp for per-frame d_pos output of the deformation
            MLP, in normalized scene units (guards against fp16 blowups).
        checkpoint_interval_minutes: Interval in minutes between checkpoints.
        checkpoint_iterations: Also checkpoint every N iterations (0 disables).
        keep_last_k_checkpoints: Delete older checkpoints beyond the newest K
            (0 disables pruning of old checkpoints).
        resume_from: Path to checkpoint to resume from, or "latest".
        eval_interval: Iterations between validation renders + PSNR (0 disables).
        mixed_precision: Whether to use mixed precision training.
        is_4d: Enable the 4D deformation pipeline.
        reclassify_interval: Interval between static/dynamic reclassifications.
        reclassify_threshold: Accumulated-deformation threshold for reclassification.
        warmup_iterations: 3DGS-only warmup iterations before deformation starts.
    """

    # Scene
    scene_dir: str = "data/lego"
    output_dir: str = "outputs/lego_exp1"

    # Training
    max_iterations: int = 30_000
    gradient_accumulation_steps: int = 1
    debug_single_batch: bool = False
    seed: int = 42
    lr_position: float = 1.6e-4
    lr_sh: float = 2.5e-3
    lr_opacity: float = 5e-2
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_deformation: float = 1e-4
    lr_final_factor: float = 0.01

    # SH
    sh_degree: int = 3

    # Losses
    lambda_depth: float = 0.1
    lambda_deform_reg: float = 0.01
    lambda_temporal_smoothness: float = 0.1
    temporal_sample_size: int = 4096
    ssim_weight: float = 0.2         # in combined L1+SSIM loss

    # Densification & pruning
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    prune_interval: int = 500
    max_gaussians: int = 350_000     # hard cap for VRAM safety
    densify_grad_threshold: float = 2e-4
    split_scale_threshold: float = 0.05
    clone_noise: float = 1e-3
    prune_min_opacity: float = 0.005
    prune_max_scale: float = 0.5

    # Scene normalization
    normalize_scene: bool = True

    # Deformation safety
    max_displacement: float = 1.0

    # Checkpoint
    checkpoint_interval_minutes: int = 30
    checkpoint_iterations: int = 10_000
    keep_last_k_checkpoints: int = 3
    resume_from: Optional[str] = None   # path to .ckpt or "latest"

    # Validation
    eval_interval: int = 2000

    # Hardware
    mixed_precision: bool = True

    # 4D settings
    is_4d: bool = False
    reclassify_interval: int = 2000
    reclassify_threshold: float = 0.01
    warmup_iterations: int = 3000
