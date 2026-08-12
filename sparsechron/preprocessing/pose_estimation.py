"""Camera pose estimation."""

import json
import logging
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


def _write_ply_ascii(ply_path: Union[str, Path], points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY file from points and colors."""
    # Ensure no extra trailing newline before the first data row
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def estimate_poses(
    image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    device: str = "cuda"
) -> None:
    """
    Estimate camera poses and initial point cloud using DUSt3R.

    If the `dust3r` module is not available, falls back to a mock
    implementation that creates dummy camera parameters and a random point cloud.

    Args:
        image_paths: List of paths to input images.
        output_dir: Directory where the output `cameras.json` and 
                    `init_gaussians.ply` will be saved.
        device: Device to run the model on (e.g., "cuda" or "cpu").
    """
    if not image_paths:
        raise ValueError("image_paths list cannot be empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_path = output_dir / "cameras.json"
    ply_path = output_dir / "init_gaussians.ply"

    try:
        import torch
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        HAS_DUST3R = True
    except ImportError:
        HAS_DUST3R = False

    if HAS_DUST3R:
        logger.info("Loaded dust3r successfully. Running real implementation.")

        # Real implementation using DUSt3R
        model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(device)

        str_paths = [str(p) for p in image_paths]
        imgs = load_images(str_paths, size=512)

        pairs = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(pairs, model, device, batch_size=2)

        # Global alignment
        scene = global_aligner(
            output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        loss = scene.compute_global_alignment(
            init="mst", niter=300, schedule="linear", lr=0.01
        )

        # Extract intrinsics and extrinsics
        cameras = {}
        for i, img_path in enumerate(image_paths):
            img_path = Path(img_path)
            pose = scene.get_im_poses()[i].detach().cpu().numpy()
            focal = scene.get_focals()[i].detach().cpu().item()
            
            cameras[img_path.name] = {
                "intrinsics": [focal, focal, 256.0, 256.0], 
                "pose": pose.tolist()
            }
            
        with open(cameras_path, "w") as f:
            json.dump(cameras, f, indent=4)
        
        # Save point cloud
        pts3d = scene.get_pts3d().detach().cpu().numpy()
        valid_mask = scene.get_valid_masks().detach().cpu().numpy()
        colors = scene.get_colors().detach().cpu().numpy()
        
        pts = pts3d[valid_mask]
        cols = colors[valid_mask]
        cols = (cols * 255).astype(np.uint8)
        
        _write_ply_ascii(ply_path, pts, cols)
                
        logger.info(f"Saved real cameras and point cloud to {output_dir}")

    else:
        logger.warning(
            "dust3r not found. Using mock pose estimation for local development."
        )
        
        cameras = {}
        for img_path in image_paths:
            img_path = Path(img_path)
            cameras[img_path.name] = {
                "intrinsics": [800.0, 800.0, 320.0, 240.0],
                "pose": np.eye(4).tolist()
            }
            
        with open(cameras_path, "w") as f:
            json.dump(cameras, f, indent=4)
            
        num_points = 1000
        mock_pts = np.random.randn(num_points, 3).astype(np.float32)
        mock_colors = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)
        
        _write_ply_ascii(ply_path, mock_pts, mock_colors)

        logger.info(f"Saved mock cameras.json and init_gaussians.ply to {output_dir}")
