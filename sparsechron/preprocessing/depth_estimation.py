"""Depth estimation from images."""

import logging
import random
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def estimate_depths(
    image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    device: str = "cuda"
) -> None:
    """
    Estimate relative depth maps for a list of images using Depth Anything V2.

    If the `depth_anything_v2` module is not available, falls back to a mock
    implementation that generates random depth maps for local development.

    Args:
        image_paths: List of paths to input images.
        output_dir: Directory where the output `.npy` depth maps will be saved.
        device: Device to run the model on (e.g., "cuda" or "cpu").
    """
    if not image_paths:
        raise ValueError("image_paths list cannot be empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise RuntimeError(
            "depth_anything_v2 is required but not found. "
            "Please install it before running depth estimation."
        )

    logger.info("Loaded depth_anything_v2 successfully. Running real implementation.")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = 'vitl'
        model = DepthAnythingV2(**model_configs[encoder])
        model = model.to(device).eval()

        for img_path in image_paths:
            img_path = Path(img_path)
            raw_image = cv2.imread(str(img_path))
            if raw_image is None:
                logger.warning(f"Failed to read image {img_path}")
                continue
                
            depth = model.infer_image(raw_image) # HxW numpy array
            
            out_path = output_dir / f"{img_path.stem}_depth.npy"
            np.save(str(out_path), depth)
            logger.info(f"Saved depth map for {img_path.name} to {out_path}")

