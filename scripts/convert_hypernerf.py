import json
import argparse
import numpy as np
import shutil
from pathlib import Path

def _write_ply_ascii(ply_path, points, colors):
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
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def convert_cameras(scene_dir: str, output_dir: str):
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    
    camera_dir = scene_dir / "camera"
    if not camera_dir.exists():
        print(f"Error: Could not find {camera_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(exist_ok=True)
    
    rgb_dir = scene_dir / "rgb" / "2x"
    if not rgb_dir.exists():
        rgb_dir = scene_dir / "rgb" / "1x"
    if not rgb_dir.exists():
        rgb_dir = scene_dir / "rgb"
        
    print(f"Copying images from {rgb_dir} to {images_out_dir}...")
    for img in rgb_dir.glob("*.png"):
        shutil.copy2(img, images_out_dir / img.name)
    
    cameras_list = []
    for cam_file in sorted(camera_dir.glob("*.json")):
        with open(cam_file, 'r') as f:
            cam_data = json.load(f)
            
        R_c2w = np.array(cam_data["orientation"], dtype=np.float32)
        T_c2w = np.array(cam_data["position"], dtype=np.float32)
        
        R_w2c = R_c2w.T
        T_w2c = -R_w2c @ T_c2w
        
        fx = float(cam_data["focal_length"])
        scale_factor = 1.0
        if "2x" in str(rgb_dir):
            scale_factor = 2.0
        elif "4x" in str(rgb_dir):
            scale_factor = 4.0
        
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

    points_path = scene_dir / "points.npy"
    if points_path.exists():
        print(f"Found points.npy, converting to init_gaussians.ply...")
        points = np.load(points_path)
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
        ply_path = output_dir / "init_gaussians.ply"
        _write_ply_ascii(ply_path, points, colors)
        print(f"Created {ply_path} with {len(points)} points.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=str, required=True, help="Path to original dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated dataset for training")
    args = parser.parse_args()
    
    convert_cameras(args.scene_dir, args.output_dir)
