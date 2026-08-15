import json
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-camera-dir", type=str, required=True, help="Path to original camera/ directory (e.g. cut-lemon1/camera)")
    parser.add_argument("--output-json", type=str, required=True, help="Path to output cameras.json")
    args = parser.parse_args()

    cameras = {}
    for cam_file in sorted(os.listdir(args.source_camera_dir)):
        if not cam_file.endswith('.json'):
            continue
            
        base_name = os.path.splitext(cam_file)[0]
        img_name = base_name + ".png"  # Assume png
        
        full_cam_path = os.path.join(args.source_camera_dir, cam_file)
        with open(full_cam_path, "r") as f:
            cam_data = json.load(f)

        R = np.array(cam_data["orientation"])
        t = np.array(cam_data["position"])
        fx = float(cam_data["focal_length"])
        fy = float(cam_data.get("focal_length_y", fx))
        
        if "pixel_aspect_ratio" in cam_data:
            fy = fx * cam_data["pixel_aspect_ratio"]
            
        cx, cy = cam_data["principal_point"]

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t

        cameras[img_name] = {
            "intrinsics": [fx, fy, cx, cy],
            "pose": pose.tolist()
        }

    with open(args.output_json, "w") as f:
        json.dump(cameras, f, indent=4)

    print(f"Successfully converted {len(cameras)} cameras to {args.output_json}")

if __name__ == "__main__":
    main()
