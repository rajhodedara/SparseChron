"""Trainer for Gaussian Splatting."""

import random
import torch
from pathlib import Path

from sparsechron.utils.config import TrainConfig
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.data.dataset import SceneDataset
from sparsechron.training.scheduler import DensificationScheduler
from sparsechron.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from sparsechron.utils.timer import Timer
from sparsechron.losses.photometric import photometric_loss
from sparsechron.losses.depth import depth_loss
from sparsechron.losses.regularization import texture_regularization_loss
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
    ) -> None:
        """Initializes the trainer.

        Args:
            config: Training configuration.
            model: The Gaussian model.
            optimizer: The optimizer.
            dataset: The scene dataset.
            renderer: The Gaussian renderer.
            scheduler: The densification scheduler.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.renderer = renderer
        self.scheduler = scheduler
        self.timer = Timer()

        self.deformation_mlp = None
        self.classifier = None
        if self.config.is_4d:
            self.deformation_mlp = DeformationMLP().to(self.model.positions.device)
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
                    classifier=self.classifier
                )
                start_iter += 1 # start at next iteration
        
        for iteration in range(start_iter, self.config.max_iterations + 1):
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            camera = item["camera"]
            
            gt_image = item["image"].to(self.model.positions.device)
            gt_depth = item["depth"].to(self.model.positions.device) if item["depth"] is not None else None
            
            deformed_params = None
            if self.config.is_4d:
                # Sample a random timestep in [0, 1]
                timestep = random.random()
                deformed_params = self.model.get_deformed(self.deformation_mlp, timestep)
                
                # Update classifier with dynamic d_pos
                d_pos_full = deformed_params["d_pos"]
                self.classifier.update(d_pos_full, mask=self.model.is_dynamic)

            render_dict = self.renderer.render(self.model, camera, deformed_params=deformed_params)
            pred_image = render_dict["rgb"].permute(2, 0, 1)
            
            loss = photometric_loss(pred_image, gt_image, self.config.ssim_weight)
            
            if gt_depth is not None and self.config.lambda_depth > 0:
                loss_d = depth_loss(render_dict["depth"], gt_depth)
                loss = loss + self.config.lambda_depth * loss_d

            if self.config.is_4d:
                # Calculate texture regularization loss
                projected_points = render_dict["means2d"]
                d_pos_dyn = deformed_params["d_pos"][self.model.is_dynamic]
                proj_pts_dyn = projected_points[self.model.is_dynamic]
                
                reg_loss = texture_regularization_loss(d_pos_dyn, proj_pts_dyn, gt_image)
                loss = loss + self.config.lambda_deform_reg * reg_loss
                
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at iteration {iteration}. Halving learning rates and skipping step.")
                self.optimizer.zero_grad()
                for group in self.optimizer.param_groups:
                    group["lr"] *= 0.5
                continue
                
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss {loss.item():.4f}, Active Gaussians {self.model.positions.shape[0]}")
            
            if self.config.densify_from_iter <= iteration <= self.config.densify_until_iter:
                self.scheduler.step(iteration)
                if self.config.is_4d and self.classifier.cum_deformation.shape[0] != self.model.positions.shape[0]:
                    self.classifier.resize(self.model.positions.shape[0])

            if self.config.is_4d and iteration % self.config.reclassify_interval == 0:
                self.classifier.reclassify(self.model, threshold=0.01)  # Using 0.01 as threshold for now

            if self.timer.elapsed_minutes() >= self.config.checkpoint_interval_minutes:
                checkpoint_path = output_dir / f"checkpoint_{iteration}.ckpt"
                save_checkpoint(
                    checkpoint_path, 
                    iteration, 
                    self.model, 
                    self.optimizer,
                    deformation_mlp=self.deformation_mlp,
                    classifier=self.classifier
                )
                self.timer.reset()
