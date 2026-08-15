"""Utility to visualize the first batch from the dataset."""

import tyro
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from sparsechron.utils.config import TrainConfig
from sparsechron.data.dataset import SceneDataset

def main() -> None:
    config = tyro.cli(TrainConfig)
    
    dataset = SceneDataset(config.scene_dir)
    print(f"Dataset loaded with {len(dataset)} items.")
    
    if len(dataset) == 0:
        print("No items in dataset!")
        return
        
    item = dataset[0]
    image = item["image"]
    depth = item["depth"]
    
    # Save image
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save("debug_batch_image.png")
    print("Saved debug_batch_image.png")
    
    # Save depth if exists
    if depth is not None:
        depth_np = depth.cpu().numpy()
        # Normalize for visualization
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_vis = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save("debug_batch_depth.png")
        print("Saved debug_batch_depth.png")
    else:
        print("No depth data found for this item.")

if __name__ == "__main__":
    main()
