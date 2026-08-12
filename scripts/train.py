"""Training entry point for SparseChron."""

import tyro
import torch

from sparsechron.utils.config import TrainConfig
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.data.dataset import SceneDataset
from sparsechron.training.scheduler import DensificationScheduler
from sparsechron.training.trainer import Trainer
from sparsechron.preprocessing.init_gaussians import initialize_gaussians


def main() -> None:
    """Parses config from CLI and runs training."""
    config = tyro.cli(TrainConfig)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SceneDataset(config.scene_dir)
    
    num_points = 10_000
    points = (torch.rand(num_points, 3, device=device) - 0.5) * 5.0
    colors = torch.rand(num_points, 3, device=device)
    
    init_dict = initialize_gaussians(points, colors)
    model = GaussianModel(init_dict).to(device)
    
    optimizer = torch.optim.Adam(
        [
            {"params": [model._positions], "lr": config.lr_position, "name": "positions"},
            {"params": [model._scales], "lr": config.lr_scale, "name": "scales"},
            {"params": [model._rotations], "lr": config.lr_rotation, "name": "rotations"},
            {"params": [model._opacities], "lr": config.lr_opacity, "name": "opacities"},
            {"params": [model._sh_coeffs], "lr": config.lr_sh, "name": "sh_coeffs"},
        ],
        lr=0.0,
        eps=1e-15,
    )
    
    renderer = GaussianRenderer()
    
    scheduler = DensificationScheduler(
        model=model,
        optimizer=optimizer,
        max_gaussians=config.max_gaussians,
        densify_interval=config.densify_interval,
        prune_interval=config.prune_interval,
    )
    
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        renderer=renderer,
        scheduler=scheduler,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
