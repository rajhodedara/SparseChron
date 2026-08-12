"""COLMAP camera loader."""

import struct
from pathlib import Path
from typing import Any, Dict, List

import torch

from sparsechron.utils.camera import Camera


def qvec2rotmat(qvec: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a rotation matrix.
    
    Args:
        qvec (torch.Tensor): Quaternion (w, x, y, z) as a 1D tensor of shape (4,).
        
    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    qvec = qvec / torch.norm(qvec)
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ], dtype=torch.float32)


def _read_cameras_txt(path: Path) -> Dict[int, Dict[str, Any]]:
    """Reads COLMAP cameras.txt."""
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            elif model == "PINHOLE":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            else:
                # Fallback approximation for other models
                fx = params[0]
                fy = params[1] if len(params) > 1 else params[0]
                cx = params[2] if len(params) > 2 else width / 2.0
                cy = params[3] if len(params) > 3 else height / 2.0
                
            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
    return cameras


def _read_images_txt(path: Path) -> List[Dict[str, Any]]:
    """Reads COLMAP images.txt."""
    images = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    read_next_as_points = False
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
            
        if read_next_as_points:
            read_next_as_points = False
            continue
            
        parts = line.split()
        if len(parts) >= 10:
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
                
                images.append({
                    "camera_id": camera_id,
                    "name": name,
                    "qvec": torch.tensor([qw, qx, qy, qz], dtype=torch.float32),
                    "tvec": torch.tensor([tx, ty, tz], dtype=torch.float32)
                })
                read_next_as_points = True
            except ValueError:
                pass
    return images


def _read_cameras_bin(path: Path) -> Dict[int, Dict[str, Any]]:
    """Reads COLMAP cameras.bin."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            
            if model_id == 0:  # SIMPLE_PINHOLE
                params = struct.unpack("<3d", f.read(24))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                params = struct.unpack("<4d", f.read(32))
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model_id == 2:  # SIMPLE_RADIAL
                params = struct.unpack("<4d", f.read(32))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise NotImplementedError(
                    f"Unsupported camera model ID {model_id} in binary format"
                )
                
            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
    return cameras


def _read_images_bin(path: Path) -> List[Dict[str, Any]]:
    """Reads COLMAP images.bin."""
    images = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            
            name = b""
            char = f.read(1)
            while char != b"\x00":
                name += char
                char = f.read(1)
            name_str = name.decode("utf-8")
            
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Each point is 2 doubles (x,y) and 1 uint64 (point3D_id) = 24 bytes
            f.seek(num_points2d * 24, 1)
            
            images.append({
                "camera_id": camera_id,
                "name": name_str,
                "qvec": torch.tensor(qvec, dtype=torch.float32),
                "tvec": torch.tensor(tvec, dtype=torch.float32)
            })
    return images


def load_colmap_cameras(scene_dir: Path | str) -> List[Camera]:
    """Loads COLMAP cameras from cameras and images files.
    
    Args:
        scene_dir (Path | str): Path to the scene directory.
        
    Returns:
        List[Camera]: List of parsed Camera objects sorted by image name.
    """
    scene_dir = Path(scene_dir)
    
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_dir
        
    # Read cameras
    cameras_dict = {}
    if (sparse_dir / "cameras.bin").exists():
        cameras_dict = _read_cameras_bin(sparse_dir / "cameras.bin")
    elif (sparse_dir / "cameras.txt").exists():
        cameras_dict = _read_cameras_txt(sparse_dir / "cameras.txt")
    else:
        raise FileNotFoundError(
            f"Could not find cameras.bin or cameras.txt in {sparse_dir}"
        )
        
    # Read images
    images_list = []
    if (sparse_dir / "images.bin").exists():
        images_list = _read_images_bin(sparse_dir / "images.bin")
    elif (sparse_dir / "images.txt").exists():
        images_list = _read_images_txt(sparse_dir / "images.txt")
    else:
        raise FileNotFoundError(
            f"Could not find images.bin or images.txt in {sparse_dir}"
        )
        
    # Sort images by name to ensure alignment with image files
    images_list = sorted(images_list, key=lambda x: x["name"])
    
    output_cameras = []
    for img in images_list:
        cam_data = cameras_dict[img["camera_id"]]
        R = qvec2rotmat(img["qvec"])
        T = img["tvec"]
        
        cam = Camera(
            fx=float(cam_data["fx"]),
            fy=float(cam_data["fy"]),
            cx=float(cam_data["cx"]),
            cy=float(cam_data["cy"]),
            width=int(cam_data["width"]),
            height=int(cam_data["height"]),
            R=R,
            T=T
        )
        output_cameras.append(cam)
        
    return output_cameras
