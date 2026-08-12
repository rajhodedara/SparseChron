"""Script to launch the interactive viewer."""

import argparse
import time
from pathlib import Path

import torch
import tyro

from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.classifier import StaticDynamicClassifier
from sparsechron.training.checkpoint import load_checkpoint
from sparsechron.viewer.app import ViewerApp


def main(checkpoint: str, port: int = 8080) -> None:
    """Launches the interactive viewer.

    Args:
        checkpoint (str): Path to the checkpoint file.
        port (int): Port to run the viewer on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_sd = ckpt["model_state_dict"]

    initial_values = {
        "positions": model_sd["_positions"],
        "scales": model_sd["_scales"],
        "rotations": model_sd["_rotations"],
        "opacities": model_sd["_opacities"],
        "sh_coeffs": model_sd["_sh_coeffs"],
    }

    model = GaussianModel(initial_values).to(device)
    deformation_mlp = DeformationMLP().to(device)
    classifier = StaticDynamicClassifier(num_gaussians=model_sd["_positions"].shape[0])

    load_checkpoint(
        ckpt_path,
        model,
        deformation_mlp=deformation_mlp,
        classifier=classifier,
    )

    app = ViewerApp(model, deformation_mlp, classifier, port=port)

    print(f"Viewer running on port {port}. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    tyro.cli(main)
