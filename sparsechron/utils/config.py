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
        lr_position: Learning rate for Gaussian positions.
        lr_sh: Learning rate for spherical harmonics.
        lr_opacity: Learning rate for opacity.
        lr_scale: Learning rate for scale.
        lr_rotation: Learning rate for rotation.
        lr_deformation: Learning rate for deformation network.
        lambda_depth: Weight for depth loss.
        lambda_deform_reg: Weight for deformation regularization loss.
        ssim_weight: Weight for SSIM in the combined L1+SSIM loss.
        densify_from_iter: Iteration to start densification.
        densify_until_iter: Iteration to stop densification.
        densify_interval: Interval between densification steps.
        prune_interval: Interval between pruning steps.
        max_gaussians: Maximum number of Gaussians to keep in memory.
        checkpoint_interval_minutes: Interval in minutes between checkpoints.
        resume_from: Path to checkpoint to resume from, or "latest".
        mixed_precision: Whether to use mixed precision training.
    """

    # Scene
    scene_dir: str = "data/lego"
    output_dir: str = "outputs/lego_exp1"

    # Training
    max_iterations: int = 30_000
    gradient_accumulation_steps: int = 1
    debug_single_batch: bool = False
    lr_position: float = 1.6e-4
    lr_sh: float = 2.5e-3
    lr_opacity: float = 5e-2
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_deformation: float = 1e-4

    # SH
    sh_degree: int = 3

    # Losses
    lambda_depth: float = 0.1
    lambda_deform_reg: float = 0.01
    ssim_weight: float = 0.2         # in combined L1+SSIM loss

    # Densification & pruning
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    prune_interval: int = 500
    max_gaussians: int = 500_000     # hard cap for VRAM safety

    # Checkpoint
    checkpoint_interval_minutes: int = 30
    checkpoint_iterations: int = 10_000
    resume_from: Optional[str] = None   # path to .ckpt or "latest"

    # Hardware
    mixed_precision: bool = True

    # 4D settings
    is_4d: bool = False
    reclassify_interval: int = 2000
    reclassify_threshold: float = 0.01
    warmup_iterations: int = 3000

