"""CLI for preprocessing scene data (depths and poses)."""

import logging
import shutil
from pathlib import Path
from typing import List

import tyro

from sparsechron.data.dataset import SceneDataset
from sparsechron.preprocessing.depth_estimation import estimate_depths
from sparsechron.preprocessing.pose_estimation import estimate_poses


def setup_logger() -> logging.Logger:
    """Sets up the logger for the script."""
    logger = logging.getLogger("sparsechron.preprocess")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(scene_dir: Path, output_dir: Path, device: str = "cuda") -> None:
    """Runs the preprocessing pipeline.

    Args:
        scene_dir: Path to the scene directory containing images, or an images directory.
        output_dir: Path to the output directory to save results.
        device: Device to run models on (e.g., "cuda" or "cpu").
    """
    logger = setup_logger()
    logger.info("Starting preprocessing pipeline.")

    # 1. Check input directory and find images
    images_in_dir = scene_dir / "images"
    if not images_in_dir.exists():
        images_in_dir = scene_dir

    if not images_in_dir.exists() or not images_in_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {images_in_dir}")
        raise FileNotFoundError(f"Directory not found: {images_in_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_paths: List[Path] = []
    for p in images_in_dir.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_paths.append(p)

    image_paths.sort()
    if not image_paths:
        logger.error(f"No valid images found in {images_in_dir}")
        raise ValueError(f"No valid images found in {images_in_dir}")

    logger.info(f"Found {len(image_paths)} images.")

    # 2. Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory if they aren't already there
    logger.info("Copying images to output directory...")
    out_image_paths: List[Path] = []
    for img_path in image_paths:
        out_path = images_out_dir / img_path.name
        if not out_path.exists():
            shutil.copy2(img_path, out_path)
        out_image_paths.append(out_path)

    # 3. Estimate Depths
    logger.info("Running depth estimation...")
    estimate_depths(out_image_paths, output_dir=output_dir, device=device)

    # 4. Estimate Poses
    logger.info("Running pose estimation...")
    estimate_poses(out_image_paths, output_dir=output_dir, device=device)

    # 5. Verify Dataset
    logger.info("Verifying SceneDataset loading...")
    try:
        dataset = SceneDataset(scene_dir=output_dir)
        logger.info(f"SceneDataset loaded successfully with {len(dataset)} images.")
    except Exception as e:
        logger.error(f"Failed to load SceneDataset: {e}")
        raise RuntimeError(f"SceneDataset loading failed: {e}") from e

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    tyro.cli(main)
