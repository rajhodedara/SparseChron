# SparseChron — Technical Specification

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SparseChron Pipeline                     │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐  │
│  │  Stage 0  │──▶│  Stage 1  │──▶│  Stage 2   │──▶│  Stage 3   │ │
│  │  Ingest   │   │  Poses +  │   │  4DGS      │   │  Export +  │ │
│  │  & Depth  │   │  Init     │   │  Training   │   │  Viewer   │ │
│  └──────────┘   └──────────┘   └───────────┘   └────────────┘  │
│                                                                 │
│       ▲               ▲               ▲               ▲         │
│       │               │               │               │         │
│  raw images     DUSt3R poses    gsplat + losses    Viser app    │
│  + depth maps   + aligned depths  + checkpoints   + .ply export │
└─────────────────────────────────────────────────────────────────┘
```

The system is a **four-stage offline pipeline**. Each stage is independently
runnable and persists its outputs to disk, so any stage can be re-run without
repeating earlier work.

---

## 2. Stage Breakdown

### Stage 0 — Ingest & Monocular Depth

| Item | Detail |
|---|---|
| **Input** | 10–15 images (JPEG/PNG) in a directory, optional `timestamps.json` |
| **Process** | Run Depth Anything V2 on each frame → relative depth maps (`.npy`) |
| **Output** | `data/<scene>/depths/` directory with per-frame `.npy` depth arrays |
| **Runs on** | Kaggle (GPU) — Depth Anything V2 needs CUDA |

### Stage 1 — Pose Estimation & Initialization

| Item | Detail |
|---|---|
| **Input** | Raw images + depth maps from Stage 0 |
| **Process — DUSt3R** | Pairwise dense matching → global alignment → camera intrinsics + extrinsics + metric point maps |
| **Process — Depth alignment** | Align Depth Anything relative depths to DUSt3R metric depths via per-frame least-squares scale+shift fitting |
| **Process — Initialization** | Sample initial Gaussian positions from the fused point cloud; initialize scales from local point density; set SH coefficients from image colors |
| **Output** | `data/<scene>/cameras.json`, `data/<scene>/init_gaussians.ply` |
| **Runs on** | Kaggle (GPU) |
| **Fallback** | If DUSt3R fails (reprojection error > threshold), fall back to COLMAP |

### Stage 2 — 4DGS Training

| Item | Detail |
|---|---|
| **Input** | Images, cameras, depth maps, initial Gaussians |
| **Process** | Differentiable rendering loop with gsplat; optimize Gaussians + deformation MLPs |
| **Output** | Checkpoints (`.ckpt`) every 30 min; final model (`gaussians.ply` + `deformation.pt`) |
| **Runs on** | Kaggle (T4 GPU) |
| **Budget** | ~20,000–30,000 iterations; ~3–6 hours per scene depending on Gaussian count |

### Stage 3 — Export & Viewer

| Item | Detail |
|---|---|
| **Input** | Final trained checkpoint |
| **Process** | Export static Gaussians as `.ply`; bundle deformation weights as `.pt`; launch Viser server |
| **Output** | Local web viewer at `localhost:8080` |
| **Runs on** | Local Windows PC (CPU-only rendering via software fallback, or Kaggle for GPU-accelerated preview) |

---

## 3. Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | Kaggle's default runtime |
| PyTorch | 2.1+ | Tensor computation, autograd, CUDA |
| gsplat | 1.0+ | CUDA-accelerated Gaussian rasterization & backprop |
| viser | 0.2+ | Interactive 3D web viewer |
| numpy | 1.24+ | Array ops, depth map storage |
| Pillow | 10+ | Image I/O |
| torchmetrics | 1.0+ | PSNR, SSIM computation |
| lpips | 0.1.4 | Learned perceptual metric |

### Pose Estimation

| Package | Version | Purpose |
|---|---|---|
| dust3r | latest (from GitHub) | Sparse-view camera pose estimation + dense point maps |
| pycolmap | 0.6+ | COLMAP fallback (Python bindings) |

### Depth Estimation

| Package | Version | Purpose |
|---|---|---|
| depth-anything-v2 | latest (from GitHub) | Monocular relative depth prediction |

### Dev & Utility

| Package | Version | Purpose |
|---|---|---|
| tyro | 0.7+ | CLI argument parsing via dataclasses |
| tqdm | 4.65+ | Progress bars |
| tensorboard | 2.14+ | Training loss/metric logging |
| matplotlib | 3.7+ | Plotting evaluation charts |
| plyfile | 1.0+ | Reading/writing `.ply` point clouds |
| safetensors | 0.4+ | Fast, safe checkpoint serialization (optional alternative to `torch.save`) |

---

## 4. Project Directory Structure

```
SparseChron/
├── README.md
├── PRD.md                          # (planning doc)
├── TechSpec.md                     # (this file)
├── ... (other planning docs)
│
├── pyproject.toml                  # Package metadata + dependency pins
├── requirements.txt                # Flat pip requirements for Kaggle
│
├── configs/
│   ├── default.yaml                # Default hyperparameters
│   ├── nerf_synthetic.yaml         # Dataset-specific overrides
│   └── hypernerf.yaml
│
├── sparsechron/                    # Main Python package
│   ├── __init__.py
│   │
│   ├── data/                       # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py              # Scene dataset class (images, depths, cameras)
│   │   ├── colmap_loader.py        # COLMAP data parser
│   │   └── dust3r_loader.py        # DUSt3R data parser
│   │
│   ├── models/                     # Core model components
│   │   ├── __init__.py
│   │   ├── gaussians.py            # GaussianModel: parameters, densification, pruning
│   │   ├── deformation.py          # DeformationMLP: temporal offsets
│   │   ├── classifier.py           # Static/dynamic Gaussian classifier
│   │   └── renderer.py             # gsplat-based differentiable renderer
│   │
│   ├── losses/                     # Loss functions
│   │   ├── __init__.py
│   │   ├── photometric.py          # L1 + SSIM image loss
│   │   ├── depth.py                # Depth supervision loss
│   │   └── regularization.py       # Texture-aware deformation regularization
│   │
│   ├── training/                   # Training loop & checkpoint logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── checkpoint.py           # Save / load / resume logic
│   │   └── scheduler.py            # Densification & pruning schedule
│   │
│   ├── preprocessing/              # Stage 0 & 1 scripts
│   │   ├── __init__.py
│   │   ├── depth_estimation.py     # Depth Anything V2 wrapper
│   │   ├── pose_estimation.py      # DUSt3R wrapper + COLMAP fallback
│   │   ├── depth_alignment.py      # Scale-shift alignment
│   │   └── init_gaussians.py       # Point cloud → initial Gaussians
│   │
│   ├── evaluation/                 # Metrics & benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py              # PSNR, SSIM, LPIPS wrappers
│   │   ├── render_novel_views.py   # Render test views from trained model
│   │   └── benchmark.py            # Run full evaluation suite
│   │
│   ├── viewer/                     # Viser interactive viewer
│   │   ├── __init__.py
│   │   └── app.py                  # Viser server: load model, render, serve UI
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── camera.py               # Camera model (intrinsics, extrinsics, projection)
│       ├── transforms.py           # Rotation/quaternion helpers
│       ├── timer.py                # Wall-clock timer for checkpoint scheduling
│       └── config.py               # Dataclass-based config (parsed by tyro)
│
├── scripts/                        # Entry-point scripts
│   ├── preprocess.py               # Run Stage 0 + Stage 1
│   ├── train.py                    # Run Stage 2
│   ├── evaluate.py                 # Run Stage 3 (eval only)
│   ├── viewer.py                   # Run Stage 3 (viewer only)
│   └── export.py                   # Export final .ply + .pt
│
├── notebooks/                      # Kaggle notebooks
│   ├── sparsechron_train.ipynb     # Main training notebook for Kaggle
│   └── sparsechron_eval.ipynb      # Evaluation notebook
│
├── data/                           # Data directory (git-ignored)
│   └── <scene_name>/
│       ├── images/                 # Raw input images
│       ├── depths/                 # Depth maps (.npy)
│       ├── cameras.json            # Camera parameters
│       └── init_gaussians.ply      # Initial point cloud
│
├── outputs/                        # Training outputs (git-ignored)
│   └── <experiment_name>/
│       ├── checkpoints/            # .ckpt files (every 30 min)
│       ├── logs/                   # TensorBoard logs
│       ├── renders/                # Rendered images during training
│       └── final/                  # Exported model (.ply + .pt)
│
└── tests/                          # Unit & integration tests
    ├── test_gaussians.py
    ├── test_deformation.py
    ├── test_renderer.py
    ├── test_checkpoint.py
    └── test_losses.py
```

---

## 5. Key Interfaces

### 5.1 GaussianModel

```python
class GaussianModel:
    """Manages all Gaussian parameters and densification/pruning."""

    positions: Tensor        # (N, 3) — xyz centers
    scales: Tensor           # (N, 3) — axis-aligned scales (log-space)
    rotations: Tensor        # (N, 4) — quaternions
    opacities: Tensor        # (N, 1) — sigmoid-space
    sh_coeffs: Tensor        # (N, K, 3) — spherical harmonic coefficients
    is_dynamic: Tensor       # (N,) — boolean mask

    def densify_and_prune(self, grads, iteration): ...
    def get_deformed(self, deformation_mlp, timestep): ...
    def export_ply(self, path): ...
```

### 5.2 DeformationMLP

```python
class DeformationMLP(nn.Module):
    """Predicts per-Gaussian temporal offsets.

    Input:  (N_dynamic, 3+1)  — Gaussian position + normalized timestep
    Output: (N_dynamic, 3+4+3) — offsets to position, rotation, scale
    """

    def __init__(self, hidden_dim=64, num_layers=4): ...
    def forward(self, positions, timestep): ...
```

### 5.3 Checkpoint Format

```python
checkpoint = {
    "iteration": int,
    "gaussians": {
        "positions": Tensor,
        "scales": Tensor,
        "rotations": Tensor,
        "opacities": Tensor,
        "sh_coeffs": Tensor,
        "is_dynamic": Tensor,
    },
    "deformation_mlp": state_dict,
    "optimizer": optimizer.state_dict(),
    "scheduler_state": dict,       # densification/pruning schedule state
    "rng_states": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    },
    "config": dataclass_as_dict,
    "metrics": {"psnr": float, "ssim": float, "lpips": float},
    "wall_clock_seconds": float,
    "timestamp": str,              # ISO 8601
}
```

### 5.4 Config Dataclass (parsed by tyro)

```python
@dataclass
class TrainConfig:
    # Scene
    scene_dir: str = "data/lego"
    output_dir: str = "outputs/lego_exp1"

    # Training
    max_iterations: int = 30_000
    lr_position: float = 1.6e-4
    lr_sh: float = 2.5e-3
    lr_opacity: float = 5e-2
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_deformation: float = 1e-4

    # Losses
    lambda_depth: float = 0.1
    lambda_deform_reg: float = 0.01
    ssim_weight: float = 0.2         # in combined L1+SSIM loss

    # Densification & pruning
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    prune_interval: int = 500
    max_gaussians: int = 500_000     # hard cap for VRAM safety

    # Checkpoint
    checkpoint_interval_minutes: int = 30
    resume_from: str | None = None   # path to .ckpt or "latest"

    # Hardware
    mixed_precision: bool = True
```

---

## 6. Loss Function Design

The total training loss at each iteration is:

$$\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_d \mathcal{L}_{\text{depth}} + \lambda_r \mathcal{L}_{\text{deform}}$$

### Photometric Loss

$$\mathcal{L}_{\text{photo}} = (1 - \alpha) \cdot \|I_{\text{pred}} - I_{\text{gt}}\|_1 + \alpha \cdot (1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}}))$$

where $\alpha = 0.2$ (configurable via `ssim_weight`).

### Depth Loss

$$\mathcal{L}_{\text{depth}} = \| D_{\text{rendered}} - D_{\text{prior}} \|_1$$

Only applied where depth priors are confident (validity mask from DUSt3R).

### Texture-Aware Deformation Regularization (from Sparse4DGS)

$$\mathcal{L}_{\text{deform}} = \sum_{i \in \text{dynamic}} w_i \cdot \|\Delta \mathbf{x}_i\|^2$$

where $w_i = \exp(-\beta \cdot \|\nabla I_i\|)$ — higher weight (more
regularization) in textureless regions, lower weight in texture-rich regions
where photometric gradients already guide deformation.

---

## 7. Memory Budget (T4 16 GB)

| Component | Estimated VRAM |
|---|---|
| Gaussians (200K, FP32 params) | ~800 MB |
| Deformation MLP (64-dim, 4 layers) | ~50 MB |
| gsplat rasterization buffers | ~2–4 GB (resolution dependent) |
| Optimizer states (Adam, 2× params) | ~1.6 GB |
| Images + depth maps batch | ~500 MB |
| PyTorch overhead + fragmentation | ~2 GB |
| **Total estimated** | **~7–9 GB** |
| **Headroom** | **~7–9 GB** |

Headroom is sufficient. If Gaussian count reaches 500K, total rises to ~14 GB — 
hence the `max_gaussians` hard cap.

---

## 8. Kaggle Integration Strategy

### Session Workflow

```
Session start
  └─▶ Check for existing checkpoint in /kaggle/input/<dataset>/
        ├─ Found → resume training from checkpoint
        └─ Not found → start from scratch (or from Stage 1 output)
  └─▶ Train with 30-min checkpoint timer
  └─▶ On session end (or manual stop):
        └─ Save final checkpoint → commit to Kaggle Dataset output
```

### Data Persistence

| What | Where | Why |
|---|---|---|
| Raw images + preprocessed data | Kaggle Dataset (input) | Persistent across sessions |
| Checkpoints | `/kaggle/working/` → committed to Kaggle Dataset (output) | Survive session timeouts |
| Final model + renders | Kaggle Dataset (output) | Download for local viewer |
| TensorBoard logs | `/kaggle/working/logs/` | Monitor training curves |

### Notebook Cell Structure

The Kaggle notebook (`sparsechron_train.ipynb`) will have these cells:

1. **Setup**: Install deps (`pip install gsplat viser tyro`)
2. **Mount data**: Symlink Kaggle Dataset to `data/`
3. **Resume check**: Find latest checkpoint
4. **Train**: Call `scripts/train.py` with config
5. **Save outputs**: Copy checkpoint + logs to output directory
6. **Evaluate** (optional): Run metrics on test views

---

## 9. Testing Strategy

| Level | What | How |
|---|---|---|
| **Unit** | Gaussian parameter shapes, deformation MLP forward pass, loss computations | `pytest` with small synthetic tensors (CPU) |
| **Integration** | Full pipeline on a tiny 3-frame synthetic scene | Script that runs all 4 stages in sequence |
| **Smoke test** | Kaggle notebook runs for 100 iterations without crash | Manual run on Kaggle before long training |
| **Regression** | Metrics on NeRF-Synthetic `lego` don't drop below threshold | Automated after model changes |

Tests in `tests/` run locally on CPU (no GPU required) using small tensor
fixtures.

---

## 10. External API Boundaries

SparseChron has **no network APIs** — it is a local pipeline.  The only
"interfaces" are:

1. **CLI** (`scripts/*.py` via `tyro`) — the primary way to run each stage.
2. **Kaggle notebook** — wraps the CLI scripts in cells.
3. **Viser viewer** — serves a local HTTP page at `localhost:8080` with
   WebSocket-based 3D rendering.

---

*Derived from [PRD.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/PRD.md).
All architecture decisions serve the constraints defined there.*
