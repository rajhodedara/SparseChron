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
        
    # Support for the dictionary format output by pose_estimation.py
    if isinstance(data, dict):
        cameras = []
        sorted_keys = sorted(data.keys())
        for key in sorted_keys:
            item = data[key]
            intrinsics = item["intrinsics"]
            pose = torch.tensor(item["pose"], dtype=torch.float32)
            
            # Convert c2w to w2c
            R_c2w = pose[:3, :3]
            T_c2w = pose[:3, 3]
            
            R_w2c = R_c2w.T
            T_w2c = -R_w2c @ T_c2w
            
            width = item.get("width", int(intrinsics[2] * 2))
            height = item.get("height", int(intrinsics[3] * 2))

            cam = Camera(
                fx=float(intrinsics[0]),
                fy=float(intrinsics[1]),
                cx=float(intrinsics[2]),
                cy=float(intrinsics[3]),
                width=int(width),
                height=int(height),
                R=R_w2c,
                T=T_w2c
            )
            cameras.append(cam)
        return cameras
        
    # Fallback to the old list format
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
