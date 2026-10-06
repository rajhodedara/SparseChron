"""Export a trained SparseChron model to a standard 3DGS PLY file.

The exported PLY opens in browser-based Gaussian-splat viewers such as
SuperSplat (https://superspl.at/editor) or any 3DGS-compatible tool — no
dedicated GPU or CUDA required locally, since those viewers render via WebGL.

Note: a PLY stores a single canonical pose; the 4D motion itself lives in the
deformation MLP and is best shared as an orbit video (see render_video.py).
"""

import argparse
import pathlib

import numpy as np
import torch

from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.gaussians import GaussianModel


def load_model(checkpoint_path: str, device: torch.device, is_4d: bool):
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

    deformation_mlp = None
    if is_4d:
        deformation_mlp = DeformationMLP()
        if "deformation_mlp_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(checkpoint["deformation_mlp_state_dict"])
        elif "deformation_state_dict" in checkpoint:
            deformation_mlp.load_state_dict(checkpoint["deformation_state_dict"])
        else:
            deformation_mlp = None
        if deformation_mlp is not None:
            deformation_mlp.to(device)
    return model, deformation_mlp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="splats.ply")
    parser.add_argument("--is-4d", action="store_true", default=True)
    parser.add_argument("--time", type=float, default=0.0,
                        help="Canonical timestep to bake into the exported positions.")
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Truncate SH to this degree (0 = view-independent colors, smallest file).")
    parser.add_argument("--max-scale", type=float, default=0.0,
                        help="Drop Gaussians whose max activated scale exceeds this "
                             "(0 = keep all). Useful for removing huge background-filler "
                             "splats that fog up external viewers.")
    parser.add_argument("--min-opacity", type=float, default=0.0,
                        help="Drop Gaussians whose activated opacity is below this (0 = keep all).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, deformation_mlp = load_model(args.checkpoint_path, device, args.is_4d)

    if args.max_scale > 0 or args.min_opacity > 0:
        with torch.no_grad():
            keep = torch.ones(model._positions.shape[0], dtype=torch.bool,
                              device=model._positions.device)
            if args.max_scale > 0:
                keep &= torch.exp(model._scales).max(dim=-1).values <= args.max_scale
            if args.min_opacity > 0:
                keep &= torch.sigmoid(model._opacities).squeeze(-1) > args.min_opacity
        dropped = int((~keep).sum())
        print(f"Cleanup: dropping {dropped:,} filler Gaussians "
              f"({100 * dropped / keep.shape[0]:.1f}%), keeping {int(keep.sum()):,}.")
        for name in ["_positions", "_scales", "_rotations", "_opacities", "_sh_coeffs"]:
            param = getattr(model, name)
            setattr(model, name, torch.nn.Parameter(param.data[keep], requires_grad=False))
        if model.is_dynamic.shape[0] == keep.shape[0]:
            model.is_dynamic = model.is_dynamic[keep]

    with torch.no_grad():
        if deformation_mlp is not None:
            deformed = model.get_deformed(deformation_mlp, float(args.time))
            positions = deformed["positions"].cpu().numpy()
        else:
            positions = model.positions.cpu().numpy()

        # Raw (unactivated) values match the 3DGS PLY convention:
        # scales stored as log(), opacities as logit().
        scales_raw = model._scales.data.cpu().numpy()
        rotations = torch.nn.functional.normalize(model._rotations.data, dim=-1).cpu().numpy()
        opacities_raw = model._opacities.data.cpu().numpy()
        sh = model._sh_coeffs.data.cpu().numpy()  # (N, D, 3)

    f_dc = sh[:, 0, :]
    keep_coeffs = (args.sh_degree + 1) ** 2 - 1
    # Channel-major flattening, matching the reference 3DGS PLY layout.
    f_rest = sh[:, 1 : 1 + keep_coeffs, :].transpose(0, 2, 1).reshape(positions.shape[0], -1)

    n = positions.shape[0]
    header_lines = [
        "ply", "format binary_little_endian 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
    ]
    for i in range(f_rest.shape[1]):
        header_lines.append(f"property float f_rest_{i}")
    header_lines += [
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
    ]

    dtype = np.dtype(
        [(name, "<f4") for name in
         ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
         + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]]
    )
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[1]):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacities_raw[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scales_raw[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = rotations[:, i]

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        f.write(arr.tobytes())

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Exported {n} Gaussians to {out} ({size_mb:.1f} MB, SH degree {args.sh_degree}).")
    print("Open it at https://superspl.at/editor (browser, no GPU needed).")


if __name__ == "__main__":
    main()
