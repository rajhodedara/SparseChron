"""Render a 4D orbit video from a trained SparseChron checkpoint.

Designed for GPU-less users: run this on Kaggle (T4), download the MP4, and
watch it anywhere. No local GPU is needed for viewing.

The virtual camera orbits the scene center at the capture radius while the
deformation timestep sweeps through [0, 1], producing a turntable video of
the reconstructed dynamic scene.
"""

import argparse
import math
import pathlib
import subprocess

import numpy as np
import torch
from PIL import Image

from sparsechron.data.dataset import SceneDataset
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.utils.camera import Camera


def load_model(checkpoint_path: str, device: torch.device, is_4d: bool):
    """Loads model (+ optional deformation MLP) from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    num_gaussians = state_dict["_positions"].shape[0]
    sh_degree = state_dict["_sh_coeffs"].shape[1]
    dummy_values = {
        "positions": torch.zeros((num_gaussians, 3), device=device),
        "scales": torch.zeros((num_gaussians, 3), device=device),
        "rotations": torch.zeros((num_gaussians, 4), device=device),
        "opacities": torch.zeros((num_gaussians, 1), device=device),
        "sh_coeffs": torch.zeros((num_gaussians, sh_degree, 3), device=device),
    }
    model = GaussianModel(dummy_values)
    model.load_state_dict(state_dict, strict=False)
    if "is_dynamic" in state_dict:
        model.is_dynamic = state_dict["is_dynamic"].to(device)
    model.to(device)
    model.eval()

    deformation_mlp = None
    if is_4d:
        deformation_mlp = DeformationMLP()
        if "deformation_mlp_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(checkpoint["deformation_mlp_state_dict"])
        elif "deformation_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(checkpoint["deformation_state_dict"])
        else:
            deformation_mlp = None
            print("Warning: no deformation MLP in checkpoint; rendering static.")
        if deformation_mlp is not None:
            deformation_mlp.to(device)
            deformation_mlp.eval()
    return model, deformation_mlp


def look_at_w2c(eye: np.ndarray, center: np.ndarray, up: np.ndarray):
    """Builds a w2c pose (R, T) for an OpenCV camera (x right, y down, z forward)."""
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        # Looking (almost) straight along the up axis: nudge the side vector.
        up = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u2 = np.cross(s, f)  # true up
    y_c = -u2            # OpenCV y points down
    R_c2w = np.stack([s, y_c, f], axis=-1)          # columns: x, y, z
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ eye
    return R_w2c, T_w2c


def orbit_camera(
    center: np.ndarray,
    radius: float,
    azimuth: float,
    elevation: float,
    up_axis: str,
    ref_cam: Camera,
    downscale: int,
):
    """Returns a Camera on a spherical orbit around the scene center."""
    if up_axis == "y":
        up = np.array([0.0, 1.0, 0.0])
        eye = center + radius * np.array(
            [math.cos(elevation) * math.cos(azimuth), math.sin(elevation),
             math.cos(elevation) * math.sin(azimuth)]
        )
    else:  # z-up (HyperNeRF)
        up = np.array([0.0, 0.0, 1.0])
        eye = center + radius * np.array(
            [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
             math.sin(elevation)]
        )
    R, T = look_at_w2c(eye, center, up)

    width = max(1, ref_cam.width // downscale)
    height = max(1, ref_cam.height // downscale)
    return Camera(
        fx=ref_cam.fx / downscale,
        fy=ref_cam.fy / downscale,
        cx=ref_cam.cx / downscale,
        cy=ref_cam.cy / downscale,
        width=width,
        height=height,
        R=torch.tensor(R, dtype=torch.float32),
        T=torch.tensor(T, dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Scene dir converted for training (provides intrinsics + normalization).")
    parser.add_argument("--output-dir", type=str, default="orbit_video")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--orbit-frac", type=float, default=1.0,
                        help="Fraction of a full turn to orbit (1.0 = full 360).")
    parser.add_argument("--elevation-deg", type=float, default=20.0)
    parser.add_argument("--up-axis", type=str, default="z", choices=["z", "y"])
    parser.add_argument("--downscale", type=int, default=2,
                        help="Render resolution scale relative to training images (2 = half res).")
    parser.add_argument("--radius-mult", type=float, default=1.1,
                        help="Orbit radius as a multiple of the mean capture-camera distance.")
    parser.add_argument("--static", action="store_true",
                        help="Freeze time (canonical shape) and only orbit the camera.")
    parser.add_argument("--is-4d", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: rendering on CPU will be very slow; run on Kaggle's GPU.")

    model, deformation_mlp = load_model(args.checkpoint_path, device, args.is_4d)
    renderer = GaussianRenderer()

    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = SceneDataset(args.dataset_path, split="train", test_every_n=0)

    # Apply the training-time scene normalization so cameras match the model.
    if getattr(dataset, "scene_transform", None) is None:
        candidate = pathlib.Path(args.checkpoint_path).parent / "scene_transform.json"
        if candidate.exists():
            import json

            with open(candidate, "r") as f:
                tdata = json.load(f)
            dataset.apply_scene_transform(
                (np.asarray(tdata["center"], dtype=np.float32), float(tdata["scale"]))
            )
            print(f"  Applied scene transform from {candidate}")

    ref_cam = dataset.cameras[0]
    center = np.asarray(dataset.scene_center, dtype=np.float64)

    dists = []
    for cam in dataset.cameras:
        eye = -(cam.R.numpy().T @ cam.T.numpy().reshape(3))
        dists.append(np.linalg.norm(eye - center))
    radius = float(np.mean(dists)) * args.radius_mult
    print(f"Orbit: center={center.round(3).tolist()}, radius={radius:.3f}, "
          f"up={args.up_axis}, {args.num_frames} frames @ {args.fps} fps")

    out_dir = pathlib.Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_frames):
        frac = i / max(1, args.num_frames - 1)
        azimuth = 2.0 * math.pi * args.orbit_frac * frac
        elevation = math.radians(args.elevation_deg)
        cam = orbit_camera(center, radius, azimuth, elevation, args.up_axis, ref_cam, args.downscale)

        timestep = 0.5 if (args.static or deformation_mlp is None) else frac
        deformed = None
        if deformation_mlp is not None and not args.static:
            deformed = model.get_deformed(deformation_mlp, float(timestep))

        with torch.no_grad():
            out = renderer.render(model, cam, deformed_params=deformed)
        rgb = out["rgb"].clamp(0.0, 1.0).cpu().numpy()
        Image.fromarray((rgb * 255.0).round().astype(np.uint8)).save(
            frames_dir / f"frame_{i:05d}.png"
        )
        if (i + 1) % 25 == 0 or i == args.num_frames - 1:
            print(f"  Rendered {i + 1}/{args.num_frames} frames")
        if (i + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stitch frames into an MP4 (ffmpeg is installed by the Kaggle setup cell).
    mp4_path = out_dir / "orbit_4d.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(args.fps),
                "-i", str(frames_dir / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(mp4_path),
            ],
            check=True,
        )
        print(f"Video saved to {mp4_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"ffmpeg unavailable or failed; PNG frames remain in {frames_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
