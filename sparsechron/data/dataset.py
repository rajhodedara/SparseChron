"""Dataset loading and processing."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from sparsechron.utils.camera import Camera
from sparsechron.data.dust3r_loader import load_dust3r_cameras
from sparsechron.data.colmap_loader import load_colmap_cameras
import numpy as np


class SceneDataset(Dataset):
    """Dataset for scene images, cameras, and optional depth maps."""
    
    def __init__(
        self, scene_dir: Path | str, split: str = "train", downscale: int = 1
    ) -> None:
        """Initializes the SceneDataset.
        
        Args:
            scene_dir (Path | str): Path to the scene directory.
            split (str): Split to use ("train" or "test").
            downscale (int): Downscale factor for images and cameras. Defaults to 1.
        """
        self.scene_dir = Path(scene_dir)
        self.split = split
        if downscale <= 0:
            raise ValueError(f"downscale must be strictly positive, got {downscale}")
        self.downscale = downscale
        
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene directory does not exist: {self.scene_dir}")
            
        self.images_dir = self.scene_dir / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        self.image_paths = sorted([
            p for p in self.images_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in valid_exts
        ])
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")
            
        # Determine loader
        if (self.scene_dir / "cameras.json").exists():
            self.cameras = load_dust3r_cameras(self.scene_dir)
        elif (self.scene_dir / "sparse" / "0").exists() or \
             (self.scene_dir / "cameras.txt").exists() or \
             (self.scene_dir / "cameras.bin").exists():
            self.cameras = load_colmap_cameras(self.scene_dir)
        else:
            raise RuntimeError(f"No camera data found in {self.scene_dir}")
            
        if len(self.cameras) != len(self.image_paths):
            raise RuntimeError(
                f"Mismatch between number of images ({len(self.image_paths)}) "
                f"and cameras ({len(self.cameras)})"
            )
            
        # Adjust cameras for downscale
        if self.downscale > 1:
            for i in range(len(self.cameras)):
                cam = self.cameras[i]
                self.cameras[i] = Camera(
                    fx=cam.fx / self.downscale,
                    fy=cam.fy / self.downscale,
                    cx=cam.cx / self.downscale,
                    cy=cam.cy / self.downscale,
                    width=cam.width // self.downscale,
                    height=cam.height // self.downscale,
                    R=cam.R.clone(),
                    T=cam.T.clone()
                )
                
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets dataset item by index.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            Dict[str, Any]: A dictionary containing the camera, image tensor,
                and optionally depth.
        """
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        orig_w, orig_h = image.width, image.height
        new_w = max(1, orig_w // self.downscale) if self.downscale > 1 else orig_w
        new_h = max(1, orig_h // self.downscale) if self.downscale > 1 else orig_h
        
        if self.downscale > 1:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        img_tensor = TF.to_tensor(image)  # [0, 1] tensor of shape (C, H, W)
        
        # Check for depth map
        depth_tensor: Optional[torch.Tensor] = None
        depth_dir = self.scene_dir / "depth"
        
        if depth_dir.exists():
            npy_path = depth_dir / f"{img_path.stem}.npy"
            if npy_path.exists():
                depth_arr = np.load(npy_path)
                depth_tensor = torch.tensor(depth_arr, dtype=torch.float32)
            else:
                png_path = depth_dir / f"{img_path.stem}.png"
                if png_path.exists():
                    with Image.open(png_path) as depth_img:
                        depth_arr = np.array(depth_img, dtype=np.float32)
                        depth_tensor = torch.tensor(depth_arr, dtype=torch.float32)

            if depth_tensor is not None and self.downscale > 1:
                # Resize depth using nearest neighbor
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor, size=(new_h, new_w), mode='nearest'
                ).squeeze(0).squeeze(0)  # (H, W)
            elif depth_tensor is not None:
                # Ensure it's (H, W)
                if depth_tensor.dim() > 2:
                    depth_tensor = depth_tensor.squeeze()

        return {
            "camera": self.cameras[idx],
            "image": img_tensor,
            "depth": depth_tensor
        }
