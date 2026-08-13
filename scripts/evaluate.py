"""Evaluate a trained SparseChron model."""

import pathlib
import tyro
import torch

from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.data.dataset import SceneDataset
from sparsechron.evaluation.render_novel_views import render_novel_views
from sparsechron.evaluation.benchmark import (
    compute_metrics,
    save_metrics,
)


def main(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str = "evaluation_output",
    is_4d: bool = True,
) -> None:
    """Evaluates a trained SparseChron model on a dataset.

    Args:
        checkpoint_path: Path to the model checkpoint.
        dataset_path: Path to the dataset directory.
        output_dir: Directory to save outputs.
        is_4d: Whether the model is 4D (deformation).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Load checkpoint ---
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path, map_location=device
    )

    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )
    num_gaussians = state_dict["_positions"].shape[0]
    sh_degree = state_dict["_sh_coeffs"].shape[1]

    dummy_values = {
        "positions": torch.zeros(
            (num_gaussians, 3), device=device
        ),
        "scales": torch.zeros(
            (num_gaussians, 3), device=device
        ),
        "rotations": torch.zeros(
            (num_gaussians, 4), device=device
        ),
        "opacities": torch.zeros(
            (num_gaussians, 1), device=device
        ),
        "sh_coeffs": torch.zeros(
            (num_gaussians, sh_degree, 3), device=device
        ),
    }

    model = GaussianModel(dummy_values)
    # Use strict=False to handle is_dynamic buffer size mismatch
    model.load_state_dict(state_dict, strict=False)
    # Overwrite the is_dynamic buffer from checkpoint
    if "is_dynamic" in state_dict:
        model.is_dynamic = state_dict["is_dynamic"].to(device)
    model.to(device)
    model.eval()

    # --- Load deformation MLP ---
    deformation_mlp = None
    if is_4d:
        deformation_mlp = DeformationMLP()
        if "deformation_mlp_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(
                checkpoint["deformation_mlp_state_dict"]
            )
        elif "deformation_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(
                checkpoint["deformation_state_dict"]
            )
        deformation_mlp.to(device)
        deformation_mlp.eval()

    # --- Load dataset ---
    print(f"Loading dataset from {dataset_path}...")
    dataset = SceneDataset(dataset_path)
    print(f"  {len(dataset)} images loaded.")

    # --- Render novel views ---
    renderer = GaussianRenderer()

    print("Rendering novel views...")
    render_out_dir = pathlib.Path(output_dir) / "renders"
    rendered_results = render_novel_views(
        model=model,
        renderer=renderer,
        dataset=dataset,
        deformation_mlp=deformation_mlp,
        output_dir=str(render_out_dir),
    )

    # --- Compute metrics ---
    # SceneDataset[i] returns dict with "image" (C,H,W)
    pred_images = []
    gt_images = []

    for i in range(len(dataset)):
        batch = dataset[i]
        gt_img = batch["image"]  # (C, H, W) tensor

        if i < len(rendered_results):
            pred_img = rendered_results[i]["image"]
            # Ensure same shape
            if pred_img.shape == gt_img.shape:
                pred_images.append(pred_img.cpu())
                gt_images.append(gt_img.cpu())

    if len(pred_images) > 0:
        print(f"Computing metrics over {len(pred_images)} images...")
        metrics = compute_metrics(pred_images, gt_images)

        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")

        metrics_path = pathlib.Path(output_dir) / "metrics.json"
        save_metrics(metrics, metrics_path)
        print(f"Metrics saved to {metrics_path}")
    else:
        print(
            "WARNING: Could not compute metrics "
            "(shape mismatch or no data)."
        )

    print(f"Renders saved to {render_out_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    tyro.cli(main)
