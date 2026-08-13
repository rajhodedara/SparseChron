import os
import torch
import torchvision.utils as vutils
from typing import Dict, List, Optional, Any
import pathlib


def render_novel_views(
    model: Any,
    renderer: Any,
    dataset: Any,
    deformation_mlp: Optional[Any] = None,
    output_dir: Optional[str] = None
) -> List[Dict[str, torch.Tensor]]:
    """Renders novel views for a given dataset.

    Args:
        model: The GaussianModel.
        renderer: The GaussianRenderer instance.
        dataset: SceneDataset. Each item is a dict with
            keys "camera" (Camera), "image" (C,H,W), "depth".
        deformation_mlp: Optional DeformationMLP for 4D.
        output_dir: Directory to save rendered PNG images.

    Returns:
        List of dicts, each containing "image" (C,H,W tensor).
    """
    if output_dir is not None:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    rendered_results = []

    for i in range(len(dataset)):
        batch = dataset[i]

        # SceneDataset returns a dict with "camera", "image", "depth"
        if isinstance(batch, dict):
            camera = batch["camera"]
            timestep = batch.get("timestep", 0.0)
        elif isinstance(batch, (tuple, list)):
            camera = batch[0]
            timestep = batch[1] if len(batch) > 1 else 0.0
        else:
            camera = batch
            timestep = 0.0

        with torch.no_grad():
            if deformation_mlp is not None:
                deformed_params = model.get_deformed(
                    deformation_mlp, timestep
                )
                render_output = renderer.render(
                    model, camera, deformed_params
                )
            else:
                render_output = renderer.render(model, camera)

        # renderer returns {"rgb": (H,W,3), "depth": (H,W,1)}
        rgb = render_output["rgb"]

        # Convert (H,W,3) -> (C,H,W) for torchvision
        image = rgb.permute(2, 0, 1).clamp(0.0, 1.0)

        if output_dir is not None:
            image_path = os.path.join(
                output_dir, f"render_{i:05d}.png"
            )
            vutils.save_image(image, image_path)

        # Move to CPU immediately to free GPU memory
        rendered_results.append({
            "image": image.cpu(),
        })

        # Clear GPU cache periodically to avoid OOM
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == len(dataset) - 1:
            print(
                f"  Rendered {i + 1}/{len(dataset)} views"
            )

    return rendered_results
