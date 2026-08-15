import os
import json
import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory containing cameras.json and depth maps")
    args = parser.parse_args()

    data_dir = args.data_dir
    cameras_file = os.path.join(data_dir, "cameras.json")

    with open(cameras_file, "r") as f:
        cameras = json.load(f)

    # Sort images to get chronological order
    images = sorted(cameras.keys())
    points_3d = []
    colors_3d = []

    # FIX 4: Calculate actual scene center by intersecting camera optical axes
    # Some datasets (like HyperNeRF) do not have the object at the origin [0,0,0]
    rays = []
    for img_name in images:
        cam = cameras[img_name]
        pose = np.array(cam["pose"])
        t = pose[:3, 3]
        d = pose[:3, :3] @ np.array([0, 0, 1])
        rays.append((t, d))
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for t, d in rays:
        I = np.eye(3)
        d = d / np.linalg.norm(d)
        proj = I - np.outer(d, d)
        A += proj
        b += proj @ t
    
    scene_center = np.linalg.solve(A, b)
    print(f"Calculated scene center: {scene_center}")

    # FIX 3: Sample 10 frames evenly across the entire video trajectory!
    step = max(1, len(images) // 10)
    sampled_images = images[::step][:10] 

    for img_name in sampled_images:
        cam = cameras[img_name]
        pose = np.array(cam["pose"])  # 4x4 matrix
        fx, fy, cx, cy = cam["intrinsics"]
        
        img_path = os.path.join(data_dir, "images", img_name)
        depth_name = img_name.replace(".png", "_depth.npy").replace(".jpg", "_depth.npy")
        depth_path = os.path.join(data_dir, depth_name)
        
        if not os.path.exists(depth_path): 
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        
        # FIX 1: Disparity Inversion (Depth Anything outputs inverse depth)
        # High values = close, Low values = far
        # We scale depth to match the actual distance from camera to scene center
        dist_to_center = np.linalg.norm(pose[:3, 3] - scene_center)
        depth_min_val = max(0.5, dist_to_center - 1.5)
        depth_max_val = dist_to_center + 1.5
        
        depth_min_arr = depth.min()
        depth_max_arr = depth.max()
        # If depth == depth_max_arr (closest), depth_norm == depth_min_val
        depth_norm = depth_max_val - (depth_max_val - depth_min_val) * (depth - depth_min_arr) / (depth_max_arr - depth_min_arr + 1e-8)
        
        H, W = depth.shape
        img = cv2.resize(img, (W, H))
        
        # FIX 2: Correct Meshgrid generation (Standard X, Y image coordinates)
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        
        z = depth_norm
        x3 = (x - cx) * z / fx
        y3 = (y - cy) * z / fy
        
        pts_cam = np.stack([x3, y3, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
        pts_world = (pose @ pts_cam.T).T[:, :3]
        
        col = img.reshape(-1, 3)
        
        # Sample 10,000 points per image
        idx = np.random.choice(len(pts_world), 10000, replace=False)
        points_3d.append(pts_world[idx])
        colors_3d.append(col[idx])

    if len(points_3d) == 0:
        print("Error: No depth maps found. Cannot generate point cloud.")
        return

    points_3d = np.concatenate(points_3d, axis=0)
    colors_3d = np.concatenate(colors_3d, axis=0)

    ply_path = os.path.join(data_dir, "init_gaussians.ply")
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(ply_path, "w") as f:
        f.write(header)
        for p, c in zip(points_3d, colors_3d):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    print(f"SUCCESS: Generated {ply_path} with inverted scale & full trajectory! Total points: {len(points_3d)}")

if __name__ == "__main__":
    main()
