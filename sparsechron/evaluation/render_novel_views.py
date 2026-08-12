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
            # Assuming get_deformed is a method of model or we apply it here.
            # Depending on project structure, but common approach is:
            deformed_model = model.get_deformed(deformation_mlp, timestep)
        else:
            deformed_model = model
            
        # Assuming renderer.render returns a dict with 'image' key or just the image tensor
        render_output = renderer.render(camera, deformed_model)
        
        if isinstance(render_output, dict):
            image = render_output.get("image", render_output.get("render"))
        else:
            image = render_output
            render_output = {"image": image}
            
        rendered_results.append(render_output)
        
        if output_dir is not None and image is not None:
            image_path = os.path.join(output_dir, f"render_{i:05d}.png")
            # Assumes image is (C, H, W) in [0, 1]
            vutils.save_image(image, image_path)
            
    return rendered_results
