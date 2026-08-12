"""DUSt3R camera loader."""

import json
from pathlib import Path
from typing import List

import torch

from sparsechron.utils.camera import Camera


def load_dust3r_cameras(scene_dir: Path | str) -> List[Camera]:
    """Loads DUSt3R cameras from cameras.json.
    
    The cameras in cameras.json are expected to be ordered the same as the
    images in the directory, or contain an 'image_name' key to sort by.
    
    Args:
        scene_dir (Path | str): Path to the scene directory.
        
    Returns:
        List[Camera]: List of parsed Camera objects.
    """
    scene_dir = Path(scene_dir)
    cam_file = scene_dir / "cameras.json"
    
    if not cam_file.exists():
        raise FileNotFoundError(f"Could not find {cam_file}")
        
    with open(cam_file, "r") as f:
        data = json.load(f)
        
    # If the json has image names, sort by image name to ensure alignment with image files
    if len(data) > 0 and "image_name" in data[0]:
        data = sorted(data, key=lambda x: x["image_name"])
        
    cameras = []
    for item in data:
        # R and T are stored as lists
        R = torch.tensor(item["R"], dtype=torch.float32)
        T = torch.tensor(item["T"], dtype=torch.float32)
        
        cam = Camera(
            fx=float(item["fx"]),
            fy=float(item["fy"]),
            cx=float(item["cx"]),
            cy=float(item["cy"]),
            width=int(item["width"]),
            height=int(item["height"]),
            R=R,
            T=T
        )
        cameras.append(cam)
        
    return cameras
