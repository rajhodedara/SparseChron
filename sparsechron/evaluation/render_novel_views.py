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
    """Renders novel views for a given dataset using the model and renderer.
    
    Args:
        model (Any): The 3D Gaussian model.
        renderer (Any): The renderer instance.
        dataset (Any): The dataset to iterate over. Should yield (camera, timestep) or camera if not 4D.
        deformation_mlp (Optional[Any], optional): The deformation MLP for 4D. Defaults to None.
        output_dir (Optional[str], optional): Directory to save rendered images. Defaults to None.
        
    Returns:
        List[Dict[str, torch.Tensor]]: A list of dictionaries containing rendered images.
    """
    if output_dir is not None:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    rendered_results = []
    
    for i, batch in enumerate(dataset):
        if isinstance(batch, tuple) or isinstance(batch, list):
            camera = batch[0]
            timestep = batch[1] if len(batch) > 1 else 0.0
        elif hasattr(batch, 'camera') and hasattr(batch, 'timestep'):
            camera = batch.camera
            timestep = batch.timestep
        else:
            camera = batch
            timestep = 0.0
            
        if deformation_mlp is not None:
            deformed_params = model.get_deformed(deformation_mlp, timestep)
            render_output = renderer.render(model, camera, deformed_params)
        else:
            render_output = renderer.render(model, camera)
            
        if isinstance(render_output, dict):
            image = render_output.get("rgb", render_output.get("image"))
        else:
            image = render_output
            render_output = {"image": image}
            
        # image is (H, W, 3). vutils.save_image expects (C, H, W)
        if image is not None and image.ndim == 3:
            image = image.permute(2, 0, 1)
            
        render_output["image"] = image
        rendered_results.append(render_output)
        
        if output_dir is not None and image is not None:
            image_path = os.path.join(output_dir, f"render_{i:05d}.png")
            # Assumes image is (C, H, W) in [0, 1]
            vutils.save_image(image, image_path)
            
    return rendered_results
