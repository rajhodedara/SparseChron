"""Converts HyperNeRF dataset cameras to SparseChron format."""

import json
import argparse
import numpy as np
import shutil
from pathlib import Path

def convert_cameras(scene_dir: str, output_dir: str):
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    
    camera_dir = scene_dir / "camera"
    if not camera_dir.exists():
        print(f"Error: Could not find {camera_dir}")
        return
        
    # We will put cameras.json directly in the output folder, 
    # but we ALSO need to copy the images there so the dataset loader finds everything together.
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(exist_ok=True)
    
    rgb_dir = scene_dir / "rgb" / "2x" # Prefer 2x
    if not rgb_dir.exists():
        rgb_dir = scene_dir / "rgb" / "1x"
        
    print(f"Copying images from {rgb_dir} to {images_out_dir}...")
    for img in rgb_dir.glob("*.png"):
        shutil.copy2(img, images_out_dir / img.name)
    
    cameras_list = []
    for cam_file in sorted(camera_dir.glob("*.json")):
        with open(cam_file, 'r') as f:
            cam_data = json.load(f)
            
        # HyperNeRF cameras are Camera-to-World (C2W)
        R_c2w = np.array(cam_data["orientation"], dtype=np.float32)
        T_c2w = np.array(cam_data["position"], dtype=np.float32)
        
        # Convert to World-to-Camera (W2C)
        R_w2c = R_c2w.T
        T_w2c = -R_w2c @ T_c2w
        
        fx = float(cam_data["focal_length"])
        # HyperNeRF images were downscaled (e.g. 2x), but we must adjust intrinsics!
        # The camera.json contains intrinsics for the 1x resolution!
        # If we use 2x images, they are half the size, so we must divide intrinsics by 2.
        scale_factor = 1.0
        if "2x" in str(rgb_dir):
            scale_factor = 2.0
        
        fx = fx / scale_factor
        fy = fx
        if "pixel_aspect_ratio" in cam_data:
            fy = (fx * scale_factor / float(cam_data["pixel_aspect_ratio"])) / scale_factor
            
        cx = float(cam_data["principal_point"][0]) / scale_factor
        cy = float(cam_data["principal_point"][1]) / scale_factor
        width = int(int(cam_data["image_size"][0]) / scale_factor)
        height = int(int(cam_data["image_size"][1]) / scale_factor)
        
        image_name = cam_file.stem + ".png"
        
        cameras_list.append({
            "image_name": image_name,
            "R": R_w2c.tolist(),
            "T": T_w2c.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height
        })
        
    out_file = output_dir / "cameras.json"
    with open(out_file, 'w') as f:
        json.dump(cameras_list, f, indent=2)
        
    print(f"Successfully converted {len(cameras_list)} cameras to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=str, required=True, help="Path to original cut-lemon1 dataset root (contains camera/ and rgb/ folders)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated dataset for training")
    args = parser.parse_args()
    
    convert_cameras(args.scene_dir, args.output_dir)
