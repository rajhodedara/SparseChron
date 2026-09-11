"""Trainer for Gaussian Splatting."""

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from sparsechron.utils.config import TrainConfig
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.data.dataset import SceneDataset
from sparsechron.training.scheduler import DensificationScheduler
from sparsechron.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    prune_checkpoints,
)
from sparsechron.utils.timer import Timer
from sparsechron.losses.photometric import photometric_loss
from sparsechron.losses.depth import depth_loss
from sparsechron.losses.regularization import (
    texture_regularization_loss,
    temporal_smoothness_loss,
)
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.classifier import StaticDynamicClassifier


class Trainer:
    """Trainer for the SparseChron model."""

    def __init__(
        self,
        config: TrainConfig,
        model: GaussianModel,
        optimizer: torch.optim.Optimizer,
        dataset: SceneDataset,
        renderer: GaussianRenderer,
        scheduler: DensificationScheduler,
        val_dataset: Optional[SceneDataset] = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            config: Training configuration.
            model: The Gaussian model.
            optimizer: The optimizer.
            dataset: The scene dataset.
            renderer: The Gaussian renderer.
            scheduler: The densification scheduler.
            val_dataset: Optional held-out dataset for periodic validation.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.renderer = renderer
        self.scheduler = scheduler
        self.val_dataset = val_dataset
        self.timer = Timer()

        # Mixed precision: torch >= 2.3 exposes torch.amp.GradScaler with a
        # device argument; fall back for older builds.
        self.scaler: Optional[object] = None
        if self.config.mixed_precision and torch.cuda.is_available():
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except (TypeError, AttributeError):
                self.scaler = torch.cuda.amp.GradScaler()

        # Exponential LR decay (3DGS-style): every LR decays to
        # lr * lr_final_factor by max_iterations.
        self.lr_scheduler = None
        if self.config.lr_final_factor < 1.0:
            gamma = self.config.lr_final_factor ** (
                1.0 / max(1, self.config.max_iterations)
            )
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )

        self.deformation_mlp = None
        self.classifier = None
        if self.config.is_4d:
            self.deformation_mlp = DeformationMLP(
                max_displacement=self.config.max_displacement
            ).to(self.model.positions.device)
            self.classifier = StaticDynamicClassifier(num_gaussians=self.model.positions.shape[0])
            self.classifier.cum_deformation = self.classifier.cum_deformation.to(self.model.positions.device)

            # Add deformation MLP to optimizer
            self.optimizer.add_param_group({
                "params": self.deformation_mlp.parameters(),
                "lr": self.config.lr_deformation,
                "name": "deformation_mlp"
            })

    def train(self) -> None:
        """Runs the training loop."""
        self.timer.reset()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_iter = 1
        if self.config.resume_from:
            ckpt_path = find_latest_checkpoint(self.config.output_dir) if self.config.resume_from == "latest" else self.config.resume_from
            if ckpt_path and Path(ckpt_path).exists():
                start_iter, _ = load_checkpoint(
                    ckpt_path,
                    self.model,
                    self.optimizer,
                    deformation_mlp=self.deformation_mlp,
                    classifier=self.classifier,
                    scaler=self.scaler,
                )
                start_iter += 1 # start at next iteration

                # Fast-forward the LR decay to match the resumed iteration.
                if self.lr_scheduler is not None:
                    steps_done = (start_iter - 1) // max(1, self.config.gradient_accumulation_steps)
                    for _ in range(steps_done):
                        self.lr_scheduler.step()

        for iteration in range(start_iter, self.config.max_iterations + 1):
            if self.config.debug_single_batch:
                idx = 0
            else:
                idx = random.randint(0, len(self.dataset) - 1)

            item = self.dataset[idx]
            camera = item["camera"]

            gt_image = item["image"].to(self.model.positions.device)
            gt_depth = item["depth"].to(self.model.positions.device) if item["depth"] is not None else None

            deformed_params = None
            timestep = float(item.get("timestep", 0.5))
            if self.config.is_4d and iteration > self.config.warmup_iterations:
                # Use the dataset-provided timestep: it maps the split index back
                # to the original frame timeline, which stays consistent between
                # train and held-out splits.
                deformed_params = self.model.get_deformed(self.deformation_mlp, timestep)

                # Update classifier with dynamic d_pos
                d_pos_full = deformed_params["d_pos"]
                self.classifier.update(d_pos_full, mask=self.model.is_dynamic)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config.mixed_precision):
                render_dict = self.renderer.render(self.model, camera, deformed_params=deformed_params)
                pred_image = render_dict["rgb"].permute(2, 0, 1)

                loss = photometric_loss(pred_image, gt_image, self.config.ssim_weight)

                if gt_depth is not None and self.config.lambda_depth > 0:
                    loss_d = depth_loss(render_dict["depth"], gt_depth)
                    loss = loss + self.config.lambda_depth * loss_d

                if self.config.is_4d and deformed_params is not None:
                    # Calculate texture regularization loss
                    projected_points = render_dict["means2d"]
                    d_pos_dyn = deformed_params["d_pos"][self.model.is_dynamic]
                    proj_pts_dyn = projected_points[self.model.is_dynamic].detach()

                    reg_loss = texture_regularization_loss(d_pos_dyn, proj_pts_dyn, gt_image)
                    loss = loss + self.config.lambda_deform_reg * reg_loss

                    # NOTE: static_regularization_loss was removed: d_pos for
                    # static Gaussians is always exactly zero (the deformation MLP
                    # only runs on dynamic points), so that loss was identically 0
                    # and did nothing. Static behavior is enforced by the
                    # classifier mask + temporal smoothness on dynamic points.

                    # Temporal Smoothness (Random subsample of dynamic points)
                    dynamic_mask = self.model.is_dynamic
                    if dynamic_mask.any():
                        num_dynamic = dynamic_mask.sum().item()
                        sample_size = min(getattr(self.config, 'temporal_sample_size', 4096), num_dynamic)

                        dynamic_indices = torch.nonzero(dynamic_mask).squeeze(1)
                        perm = torch.randperm(num_dynamic, device=self.model.positions.device)[:sample_size]
                        subset_indices = dynamic_indices[perm]

                        d_pos_t1 = deformed_params["d_pos"][subset_indices]

                        delta = 0.05
                        timestep_t2 = min(1.0, timestep + delta)
                        subset_pos = self.model.positions[subset_indices]
                        times_t2 = torch.full((sample_size, 1), timestep_t2, device=subset_pos.device, dtype=subset_pos.dtype)

                        d_pos_t2, _, _ = self.deformation_mlp(subset_pos, times_t2)

                        loss_temporal = temporal_smoothness_loss(d_pos_t1, d_pos_t2)
                        loss = loss + getattr(self.config, 'lambda_temporal_smoothness', 0.1) * loss_temporal

            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at iteration {iteration}. Skipping step (LR unchanged).")
                self.optimizer.zero_grad()
                # NOTE: Do NOT permanently decay learning rates here. Repeated NaN
                # events used to drive LR -> 0 and silently flatline training.
                # GradScaler handles inf/NaN gradients itself by skipping steps
                # and reducing its internal scale factor.
                continue

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights only every gradient_accumulation_steps
            if iteration % self.config.gradient_accumulation_steps == 0:
                # Unscale gradients BEFORE optimizer.step() and BEFORE the
                # densification scheduler reads ._positions.grad. Otherwise the
                # raw (scaled) gradients inflate the accumulated-grad statistics
                # and densification clones indiscriminately until the hard cap.
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if self.config.densify_from_iter <= iteration <= self.config.densify_until_iter:
                    self.scheduler.step(iteration)
                    if self.config.is_4d and self.classifier.cum_deformation.shape[0] != self.model.positions.shape[0]:
                        self.classifier.resize(self.model.positions.shape[0])

                if self.config.is_4d and iteration > self.config.warmup_iterations and iteration % self.config.reclassify_interval == 0:
                    self.classifier.reclassify(self.model, threshold=self.config.reclassify_threshold)

                self.optimizer.zero_grad()

            if (
                self.val_dataset is not None
                and self.config.eval_interval > 0
                and iteration % self.config.eval_interval == 0
            ):
                self._validate(iteration, output_dir)

            if iteration % 100 == 0:
                max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(
                    f"Iteration {iteration}: Loss {loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                    f"Active Gaussians {self.model.positions.shape[0]}, "
                    f"Max VRAM: {max_vram_gb:.2f} GB"
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

            time_save = self.timer.elapsed_minutes() >= self.config.checkpoint_interval_minutes
            iter_save = self.config.checkpoint_iterations > 0 and iteration % self.config.checkpoint_iterations == 0
            if time_save or iter_save or iteration == self.config.max_iterations:
                checkpoint_path = output_dir / f"checkpoint_{iteration}.ckpt"
                save_checkpoint(
                    checkpoint_path,
                    iteration,
                    self.model,
                    self.optimizer,
                    deformation_mlp=self.deformation_mlp,
                    classifier=self.classifier,
                    scaler=self.scaler,
                )
                # Keep only the newest K checkpoints so /kaggle/working does not
                # fill up during long sessions.
                prune_checkpoints(output_dir, self.config.keep_last_k_checkpoints)
                # Cache cleanup only on save boundaries: empty_cache() forces a
                # device sync and measurably slows the training loop.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.timer.reset()
                if iteration == self.config.max_iterations:
                    print(f"Training complete! Final checkpoint saved to {checkpoint_path}")

    @torch.no_grad()
    def _validate(self, iteration: int, output_dir: Path) -> float:
        """Renders the first held-out view, logs PSNR, and saves a PNG preview.

        On Kaggle there is no live monitoring, so this is the primary way to
        spot black renders / floater artifacts mid-session.

        Args:
            iteration: Current training iteration.
            output_dir: Training output directory.

        Returns:
            The validation PSNR in dB.
        """
        item = self.val_dataset[0]
        camera = item["camera"]
        gt_image = item["image"].to(self.model.positions.device)

        deformed_params = None
        if self.config.is_4d and iteration > self.config.warmup_iterations:
            timestep = float(item.get("timestep", 0.5))
            deformed_params = self.model.get_deformed(self.deformation_mlp, timestep)

        render_dict = self.renderer.render(self.model, camera, deformed_params=deformed_params)
        pred_image = render_dict["rgb"].permute(2, 0, 1).clamp(0.0, 1.0)

        mse = torch.mean((pred_image - gt_image) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        print(f"[val] Iteration {iteration}: PSNR {psnr:.2f} dB")

        # Save a side-by-side (pred | GT) preview
        canvas = torch.cat([pred_image, gt_image.clamp(0.0, 1.0)], dim=2)  # concat along width
        arr = (
            canvas.permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        val_dir = output_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(val_dir / f"iter_{iteration:06d}.png")
        return psnr
