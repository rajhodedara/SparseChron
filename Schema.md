# SparseChron — Schema & Data Models

SparseChron has no traditional database. All state lives in **files on disk**.
This document defines every data structure, file format, and storage schema used
across the pipeline.

---

## 1. On-Disk Directory Schema

```
data/<scene_name>/
├── images/                     # Raw input
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── depths/                     # Stage 0 output
│   ├── 000.npy                 # float32 (H, W)
│   ├── 001.npy
│   └── ...
├── timestamps.json             # Optional — user-provided or auto-generated
├── cameras.json                # Stage 1 output — all camera parameters
├── pointmaps/                  # Stage 1 output — DUSt3R dense point maps
│   ├── 000.npy                 # float32 (H, W, 3)
│   └── ...
├── init_gaussians.ply          # Stage 1 output — initial Gaussian positions
└── metadata.json               # Scene-level metadata

outputs/<experiment_name>/
├── config.yaml                 # Frozen training config for this run
├── checkpoints/
│   ├── ckpt_00000.ckpt         # iteration 0 (initial)
│   ├── ckpt_05000.ckpt
│   └── ckpt_latest.ckpt        # symlink to most recent
├── logs/
│   └── events.out.tfevents.*   # TensorBoard logs
├── renders/
│   ├── train/                  # Periodic training-view renders
│   │   ├── iter_05000_view_00.png
│   │   └── ...
│   └── test/                   # Final test-view renders
│       ├── view_00.png
│       └── ...
├── final/
│   ├── gaussians.ply           # Exported static Gaussian params
│   ├── deformation.pt          # Deformation MLP weights
│   └── model_info.json         # Export metadata
└── metrics.json                # Evaluation results
```

---

## 2. Core Data Structures

### 2.1 Gaussian Parameters

Each Gaussian is defined by these parameters, stored as contiguous tensors:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `positions` | `(N, 3)` | float32 | XYZ world-space centers |
| `scales` | `(N, 3)` | float32 | Log-space axis-aligned scales |
| `rotations` | `(N, 4)` | float32 | Unit quaternions `(w, x, y, z)` |
| `opacities` | `(N, 1)` | float32 | Pre-sigmoid opacity values |
| `sh_coeffs` | `(N, K, 3)` | float32 | Spherical harmonics; K = (degree+1)², default degree=3 → K=16 |
| `is_dynamic` | `(N,)` | bool | Static/dynamic classification mask |

**Total per Gaussian** (degree 3): 3 + 3 + 4 + 1 + 48 = **59 floats = 236 bytes**

### 2.2 Deformation MLP State

```python
DeformationMLP:
    layers: [
        Linear(4, 64),      # input: (x, y, z, t)
        Linear(64, 64),
        Linear(64, 64),
        Linear(64, 64),
        Linear(64, 10),     # output: (dx, dy, dz, dw, dqx, dqy, dqz, dsx, dsy, dsz)
    ]
    activation: ReLU (between hidden layers)
    output_activation: None (raw offsets)
```

| Layer | Parameters | Size |
|---|---|---|
| Input → Hidden 1 | 4×64 + 64 | 320 |
| Hidden 1 → 2 | 64×64 + 64 | 4,160 |
| Hidden 2 → 3 | 64×64 + 64 | 4,160 |
| Hidden 3 → 4 | 64×64 + 64 | 4,160 |
| Hidden 4 → Output | 64×10 + 10 | 650 |
| **Total** | | **13,450 params (~54 KB)** |

---

## 3. File Format Specifications

### 3.1 `timestamps.json`

```json
{
    "format": "normalized",
    "unit": "float [0, 1]",
    "timestamps": {
        "000.png": 0.0,
        "001.png": 0.083,
        "002.png": 0.167,
        "...": "..."
    }
}
```

If not provided, timestamps are auto-assigned as uniformly spaced values in
`[0, 1]` based on filename sort order.

### 3.2 `cameras.json`

```json
{
    "source": "dust3r",
    "coordinate_system": "opencv",
    "frames": [
        {
            "image_name": "000.png",
            "width": 800,
            "height": 600,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 400.0,
            "cy": 300.0,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "world_to_camera": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "timestamp": 0.0
        }
    ],
    "reprojection_error_px": 1.23,
    "estimation_method": "dust3r_global_alignment"
}
```

| Field | Type | Notes |
|---|---|---|
| `world_to_camera` | 4×4 float matrix | Extrinsics (OpenCV convention: +X right, +Y down, +Z forward) |
| `fx, fy, cx, cy` | float | Intrinsics (pixels) |
| `distortion` | float[5] | Radial + tangential (k1, k2, p1, p2, k3); zeros if undistorted |
| `timestamp` | float [0, 1] | Normalized time for this frame |

### 3.3 `metadata.json`

```json
{
    "scene_name": "lego",
    "num_frames": 12,
    "image_resolution": [800, 600],
    "depth_resolution": [800, 600],
    "num_initial_gaussians": 50000,
    "source_dataset": "nerf_synthetic",
    "created_at": "2026-09-15T10:30:00Z",
    "preprocessing": {
        "depth_model": "depth_anything_v2_vitl",
        "pose_model": "dust3r_512",
        "depth_alignment_method": "least_squares_scale_shift"
    }
}
```

### 3.4 `init_gaussians.ply`

Standard PLY format with custom properties:

```
ply
format binary_little_endian 1.0
element vertex 50000
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float sh_0
property float sh_1
property float sh_2
... (up to sh_47 for degree 3)
end_header
<binary data>
```

### 3.5 Checkpoint (`.ckpt`)

Serialized via `torch.save()` as a Python dict:

```python
{
    "iteration": 15000,
    "gaussians": {
        "positions":  Tensor(N, 3),
        "scales":     Tensor(N, 3),
        "rotations":  Tensor(N, 4),
        "opacities":  Tensor(N, 1),
        "sh_coeffs":  Tensor(N, 16, 3),
        "is_dynamic": Tensor(N,),
    },
    "deformation_mlp": OrderedDict,    # nn.Module.state_dict()
    "optimizer": dict,                  # optimizer.state_dict()
    "scheduler_state": {
        "densify_count": 45,
        "prune_count": 12,
        "last_densify_iter": 14900,
        "last_prune_iter": 14500,
    },
    "rng_states": {
        "python":  tuple,               # random.getstate()
        "numpy":   dict,                # np.random.get_state()
        "torch":   Tensor,              # torch.random.get_rng_state()
        "cuda":    Tensor,              # torch.cuda.get_rng_state()
    },
    "config": dict,                     # Frozen TrainConfig as dict
    "metrics": {
        "train_psnr": 28.5,
        "train_loss": 0.012,
    },
    "wall_clock_seconds": 5400.0,
    "timestamp": "2026-09-15T12:00:00Z",
    "version": "0.1.0",
}
```

**Checkpoint size estimate**: 200K Gaussians → ~50 MB per checkpoint.

### 3.6 `metrics.json`

```json
{
    "experiment": "lego_exp1",
    "scene": "lego",
    "num_train_views": 12,
    "num_test_views": 25,
    "iterations": 30000,
    "wall_clock_hours": 4.2,
    "peak_vram_gb": 11.3,
    "num_gaussians_final": 185432,
    "static_ratio": 0.72,
    "per_view": [
        {
            "view_id": 0,
            "psnr": 30.12,
            "ssim": 0.952,
            "lpips": 0.041
        }
    ],
    "mean": {
        "psnr": 29.45,
        "ssim": 0.941,
        "lpips": 0.053
    },
    "baseline_comparison": {
        "method": "vanilla_4dgs",
        "mean_psnr": 25.10,
        "mean_ssim": 0.890,
        "mean_lpips": 0.098
    }
}
```

### 3.7 `model_info.json` (Export Metadata)

```json
{
    "version": "0.1.0",
    "num_gaussians": 185432,
    "num_static": 133511,
    "num_dynamic": 51921,
    "sh_degree": 3,
    "deformation_mlp": {
        "hidden_dim": 64,
        "num_layers": 4,
        "input_dim": 4,
        "output_dim": 10
    },
    "num_timesteps": 12,
    "exported_from_checkpoint": "ckpt_30000.ckpt",
    "exported_at": "2026-09-15T14:00:00Z"
}
```

---

## 4. Depth Map Storage

| Property | Value |
|---|---|
| Format | NumPy `.npy` (float32) |
| Shape | `(H, W)` — same resolution as input image |
| Units | **Relative** (from Depth Anything V2) until alignment; **metric** (meters) after alignment |
| Invalid pixels | Marked as `0.0` or `NaN`; filtered out during depth loss |

### Aligned Depth Map Convention

After scale-shift alignment in Stage 1b:

$$D_{\text{aligned}} = s \cdot D_{\text{relative}} + t$$

where $s$ (scale) and $t$ (shift) are fit per-frame via least-squares against
DUSt3R metric depths. The aligned depths are **overwritten** into the same
`depths/` directory (originals are not preserved).

---

## 5. Tensor Shape Reference

Quick lookup for debugging shape mismatches:

| Tensor | Shape | Notes |
|---|---|---|
| Input image | `(3, H, W)` | CHW, float32, [0, 1] |
| Depth map | `(1, H, W)` | CHW, float32, meters |
| Camera intrinsics | `(3, 3)` | K matrix |
| Camera extrinsics | `(4, 4)` | World-to-camera, OpenCV |
| Gaussian positions | `(N, 3)` | World space |
| Gaussian scales | `(N, 3)` | Log-space |
| Gaussian rotations | `(N, 4)` | Quaternion (w, x, y, z) |
| Gaussian opacities | `(N, 1)` | Pre-sigmoid |
| SH coefficients | `(N, 16, 3)` | Degree 3, RGB |
| Deformation input | `(N_dyn, 4)` | (x, y, z, t) |
| Deformation output | `(N_dyn, 10)` | (Δpos, Δrot, Δscale) |
| Rendered image | `(3, H, W)` | CHW, float32, [0, 1] |
| Rendered depth | `(1, H, W)` | CHW, float32, meters |
| Rendered alpha | `(1, H, W)` | CHW, float32, [0, 1] |

---

*Derived from [TechSpec.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/TechSpec.md).
Every tensor shape and file format referenced in code must match this document.*
