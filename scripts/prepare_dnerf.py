import os
import json
import numpy as np
import zipfile
import shutil
from pathlib import Path
import argparse

def download_file(url, filepath):
    print(f"Downloading D-NeRF dataset from {url}...")
    os.system(f"wget -O {filepath} {url}")

def process_dnerf(scene_name="mutant", base_dir="data"):
    url = "https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1"
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    zip_path = base_path / "data.zip"
    
    if not zip_path.exists():
        download_file(url, str(zip_path))
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)
        
    scene_path = base_path / scene_name
    if not scene_path.exists():
        nested_path = base_path / "data" / scene_name
        if nested_path.exists():
            scene_path = nested_path
        else:
            raise FileNotFoundError(f"Scene {scene_name} not found! Check extracted contents.")
        
    # We will format this into our standard structure
    out_dir = base_path / f"dnerf_{scene_name}"
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    with open(scene_path / "transforms_train.json", 'r') as f:
        meta = json.load(f)
        
    camera_angle_x = float(meta['camera_angle_x'])
    
    # We will read the first image to get width and height
    first_frame = meta['frames'][0]['file_path'] + ".png"
    import cv2
    img = cv2.imread(str(scene_path / first_frame))
    h, w = img.shape[:2]
    
    focal = .5 * w / np.tan(.5 * camera_angle_x)
    
    cameras_dict = {}
    
    print("Converting cameras and copying images...")
    for i, frame in enumerate(meta['frames']):
        file_path = frame['file_path'] + ".png"
        src_img_path = scene_path / file_path
        
        # New image name
        img_name = f"image_{i:04d}.png"
        dst_img_path = out_images_dir / img_name
        
        shutil.copy(src_img_path, dst_img_path)
        
        c2w = np.array(frame['transform_matrix'])
        
        # Convert c2w (OpenGL/NeRF) to w2c (OpenCV)
        c2w[0:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        cameras_dict[img_name] = {
            "R": R.tolist(),
            "T": T.tolist(),
            "focal_length": [focal, focal],
            "principal_point": [w / 2, h / 2],
            "width": w,
            "height": h
        }
        
    with open(out_dir / "cameras.json", 'w') as f:
        json.dump(cameras_dict, f, indent=4)
        
    print(f"\nDone! Dataset prepared at: {out_dir}")
    print(f"To train, run:")
    print(f"!PYTHONPATH=/kaggle/working/SparseChron python scripts/train.py --scene-dir {out_dir} --output-dir /kaggle/working/outputs_mutant --is-4d --mixed-precision")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="mutant", help="Scene name (e.g. mutant, lego, trex)")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    process_dnerf(args.scene, args.out)
