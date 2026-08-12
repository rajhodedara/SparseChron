# SparseChron — Implementation Plan

A phased roadmap ordered by dependency. Each phase produces a testable
milestone.

---

## Phase 1 — Project Scaffolding

> **Where**: Local Windows PC (no GPU needed)
> **Milestone**: `python -m sparsechron --help` runs; tests pass on CPU

### Tasks

- [ ] 1.1 Initialize Git repo, add `.gitignore` (ignore `data/`, `outputs/`,
      `__pycache__/`, `*.ckpt`, `*.npy`)
- [ ] 1.2 Create directory structure matching TechSpec §4
- [ ] 1.3 Write `pyproject.toml` with all pinned dependencies
- [ ] 1.4 Generate `requirements.txt` for Kaggle (`pip freeze` subset)
- [ ] 1.5 Implement `sparsechron/utils/config.py` — `TrainConfig` dataclass
      with all hyperparameters from TechSpec §5.4
- [ ] 1.6 Implement `sparsechron/utils/camera.py` — `Camera` dataclass with
      `project()`, `unproject()`, `world_to_camera()` methods
- [ ] 1.7 Implement `sparsechron/utils/transforms.py` — quaternion ↔ rotation
      matrix, quaternion multiplication, normalization
- [ ] 1.8 Write unit tests for camera projection round-trip and quaternion ops
- [ ] 1.9 Verify `pytest tests/` passes on CPU

### Deliverable

Repo skeleton with working config system and tested math utilities.

---

## Phase 2 — Data Pipeline (Stages 0 & 1)

> **Where**: Kaggle (GPU needed for Depth Anything V2 and DUSt3R)
> **Milestone**: Given 12 images → produces `cameras.json`, `depths/*.npy`,
> `init_gaussians.ply`

### Tasks

- [ ] 2.1 Implement `sparsechron/preprocessing/depth_estimation.py`
    - Wrap Depth Anything V2 (ViT-L variant)
    - Input: directory of images → Output: `depths/*.npy`
    - Handle resolution: resize to max 518px for depth model, save at original
      resolution (bilinear upsample)
- [ ] 2.2 Implement `sparsechron/preprocessing/pose_estimation.py`
    - Wrap DUSt3R global alignment pipeline
    - Input: directory of images → Output: `cameras.json` + `pointmaps/*.npy`
    - Validate poses: compute mean reprojection error, warn if > 5px
    - COLMAP fallback path (call `pycolmap` if DUSt3R error exceeds threshold)
- [ ] 2.3 Implement `sparsechron/preprocessing/depth_alignment.py`
    - Per-frame least-squares scale+shift fitting: align Depth Anything relative
      depths to DUSt3R metric depths
    - Use DUSt3R confidence maps to weight the fitting
    - Overwrite `depths/*.npy` with aligned metric depths
- [ ] 2.4 Implement `sparsechron/preprocessing/init_gaussians.py`
    - Fuse DUSt3R point maps into a single point cloud
    - Downsample to target count (e.g. 50K) via voxel grid filtering
    - Initialize Gaussian scales from local point density (k-NN radius)
    - Initialize SH coefficients from nearest image pixel colors
    - Save as `init_gaussians.ply`
- [ ] 2.5 Implement `sparsechron/data/dataset.py`
    - `SceneDataset` class: loads images, depths, cameras from a scene directory
    - Returns `(image, depth, camera, timestamp)` tuples
    - Handles train/test splits (random or specified)
- [ ] 2.6 Implement `sparsechron/data/dust3r_loader.py` and
      `sparsechron/data/colmap_loader.py` — parsers for each format into the
      unified `cameras.json` schema
- [ ] 2.7 Write `scripts/preprocess.py` — CLI entry point that runs 2.1–2.4
      in sequence
- [ ] 2.8 Test on NeRF-Synthetic `lego` scene (download, subsample 12 frames,
      run preprocessing, visually inspect point cloud)

### Deliverable

`python scripts/preprocess.py --scene_dir data/lego` produces all Stage 0+1
outputs. Visually verify that the initial point cloud roughly matches the scene.

---

## Phase 3 — Static 3DGS Baseline

> **Where**: Kaggle (GPU) for training; local for code editing
> **Milestone**: Static 3DGS renders novel views of a single timestep on
> NeRF-Synthetic `lego` with PSNR > 25 dB

### Tasks

- [ ] 3.1 Implement `sparsechron/models/gaussians.py`
    - `GaussianModel` class: holds all parameter tensors from Schema §2.1
    - `from_ply(path)` — load initial Gaussians
    - `get_covariance()` — compute 3D covariance from scales + rotations
    - `create_optimizer(config)` — per-parameter learning rates
- [ ] 3.2 Implement `sparsechron/models/renderer.py`
    - Wrap `gsplat.rasterization()` for forward rendering
    - Input: GaussianModel + Camera → Output: rendered RGB (3, H, W) + depth
      (1, H, W) + alpha (1, H, W)
    - Handle SH evaluation for view-dependent color
- [ ] 3.3 Implement `sparsechron/losses/photometric.py`
    - Combined L1 + SSIM loss as specified in TechSpec §6
- [ ] 3.4 Implement `sparsechron/losses/depth.py`
    - L1 depth loss with validity masking (ignore NaN / zero pixels)
- [ ] 3.5 Implement `sparsechron/training/scheduler.py`
    - Densification: clone/split Gaussians with high position gradients
    - Pruning: remove Gaussians with low opacity or huge scale
    - Schedule: densify every 100 iters (500–15000), prune every 500 iters
    - `max_gaussians` hard cap enforcement
- [ ] 3.6 Implement `sparsechron/training/checkpoint.py`
    - `save_checkpoint(path, iteration, model, optimizer, ...)` — full state
      as specified in Schema §3.5
    - `load_checkpoint(path)` — restore all state including RNG
    - `find_latest_checkpoint(directory)` — scan for most recent valid `.ckpt`
    - Corruption guard: save to temp file, then atomic rename
- [ ] 3.7 Implement `sparsechron/training/trainer.py` (static-only version)
    - Training loop: forward render → compute loss → backward → step → densify/prune
    - 30-minute checkpoint timer (via `sparsechron/utils/timer.py`)
    - Mixed-precision via `torch.cuda.amp.GradScaler`
    - TensorBoard logging (loss, PSNR, Gaussian count, learning rate)
    - NaN guard: skip iteration + halve LR on NaN loss
- [ ] 3.8 Write `scripts/train.py` — CLI entry point with `tyro`
- [ ] 3.9 Write unit tests:
    - `test_gaussians.py` — parameter shapes, from_ply loading, covariance
    - `test_renderer.py` — output shapes, gradient flow (tiny synthetic scene)
    - `test_checkpoint.py` — save/load round-trip, resume iteration count
    - `test_losses.py` — loss computation shapes, gradient exists
- [ ] 3.10 Train on NeRF-Synthetic `lego` (12 sparse views, 30K iterations)
      — verify PSNR > 25 dB

### Deliverable

Working static 3DGS that trains on Kaggle with checkpointing. This is the
foundation that Phase 4 extends.

---

## Phase 4 — Temporal Deformation (3DGS → 4DGS)

> **Where**: Kaggle (GPU) for training; local for code editing
> **Milestone**: 4DGS renders temporally varying views on HyperNeRF with
> visible motion

### Tasks

- [ ] 4.1 Implement `sparsechron/models/deformation.py`
    - `DeformationMLP` as specified in Schema §2.2
    - Input: `(N_dynamic, 4)` — (x, y, z, t)
    - Output: `(N_dynamic, 10)` — (Δpos, Δrot, Δscale)
    - Apply offsets: new_pos = pos + Δpos, new_rot = quat_mul(rot, Δrot),
      new_scale = scale + Δscale
- [ ] 4.2 Extend `GaussianModel.get_deformed(deformation_mlp, timestep)`
    - Apply deformation only to dynamic Gaussians (`is_dynamic == True`)
    - Return new parameter set (deformed positions, rotations, scales)
    - Static Gaussians pass through unchanged
- [ ] 4.3 Implement `sparsechron/models/classifier.py`
    - `StaticDynamicClassifier`: track accumulated deformation magnitude per
      Gaussian across training iterations
    - Reclassify every N iterations: if cumulative ‖Δx‖ < threshold →
      mark static; else → mark dynamic
    - Initially all Gaussians start as dynamic (first M iterations)
- [ ] 4.4 Implement `sparsechron/losses/regularization.py`
    - Texture-aware deformation regularization (TechSpec §6)
    - Compute image gradient magnitude ‖∇I‖ at each Gaussian's projected
      2D position
    - Weight: w_i = exp(-β · ‖∇I_i‖)
    - Loss: Σ w_i · ‖Δx_i‖²
- [ ] 4.5 Extend `trainer.py` for 4DGS
    - Sample random timestep per iteration
    - Apply deformation before rendering
    - Add deformation MLP parameters to optimizer
    - Add regularization loss to total loss
    - Run classifier periodically
    - Update checkpoint to include deformation MLP + classifier state
- [ ] 4.6 Write unit tests:
    - `test_deformation.py` — MLP forward shapes, offset application, gradient flow
    - `test_regularization.py` — texture weight computation, loss shape
- [ ] 4.7 Train on HyperNeRF `interp` scene (12 sparse views, 30K iterations)
      — verify temporal motion is reconstructed

### Deliverable

Full 4DGS training with static/dynamic split and deformation regularization.
Rendered sequences should show smooth motion.

---

## Phase 5 — Kaggle Notebook Integration

> **Where**: Kaggle
> **Milestone**: End-to-end training completes across multiple Kaggle sessions
> via checkpoint resume

### Tasks

- [ ] 5.1 Create `notebooks/sparsechron_train.ipynb`
    - Cell 1: Install dependencies (pin versions matching `requirements.txt`)
    - Cell 2: Clone repo or upload as Kaggle Dataset
    - Cell 3: Mount input dataset (images + preprocessed data)
    - Cell 4: Resume check — find latest checkpoint
    - Cell 5: Run training (calls `scripts/train.py`)
    - Cell 6: Copy outputs to Kaggle output directory
- [ ] 5.2 Test the full resume cycle:
    - Session 1: Train for 10K iterations, session times out
    - Session 2: Resume from checkpoint, train to 20K
    - Session 3: Resume from checkpoint, train to 30K (complete)
    - Verify metrics are consistent (no degradation from resume)
- [ ] 5.3 Create `notebooks/sparsechron_eval.ipynb`
    - Loads trained model from Kaggle Dataset
    - Runs evaluation on test views
    - Saves metrics + rendered images
- [ ] 5.4 Document the Kaggle workflow in README (step-by-step with
      screenshots)

### Deliverable

Battle-tested notebook that survives Kaggle session interruptions.

---

## Phase 6 — Evaluation & Benchmarking

> **Where**: Kaggle (GPU) for rendering; local for analysis
> **Milestone**: Complete metrics table for ≥2 NeRF-Synthetic + ≥1 HyperNeRF scene

### Tasks

- [ ] 6.1 Implement `sparsechron/evaluation/metrics.py`
    - PSNR via `torchmetrics.PeakSignalNoiseRatio`
    - SSIM via `torchmetrics.StructuralSimilarityIndexMeasure`
    - LPIPS via `lpips.LPIPS(net='vgg')`
    - All metrics operate on `(B, 3, H, W)` float32 tensors
- [ ] 6.2 Implement `sparsechron/evaluation/render_novel_views.py`
    - Load trained model (`.ply` + `.pt`)
    - Render at each test camera + timestep
    - Save rendered images to `outputs/.../renders/test/`
- [ ] 6.3 Implement `sparsechron/evaluation/benchmark.py`
    - Orchestrate: render all test views → compute metrics → aggregate →
      save `metrics.json`
    - Support baseline comparison mode: load two `metrics.json` and produce
      a delta table
- [ ] 6.4 Write `scripts/evaluate.py` — CLI entry point
- [ ] 6.5 Train + evaluate vanilla 4DGS baseline (same sparse inputs, no depth
      prior, no deformation regularization, no static/dynamic split) for
      comparison
- [ ] 6.6 Run full benchmark suite:
    - NeRF-Synthetic: `lego`, `drums` (minimum), more scenes if time permits
    - HyperNeRF: `interp/cut-lemon` (minimum)
    - 12 sparse input views for each
- [ ] 6.7 Generate results table (markdown) and per-scene bar charts
      (matplotlib)

### Deliverable

Quantitative results demonstrating SparseChron's advantage over vanilla 4DGS
on sparse inputs.

---

## Phase 7 — Viser Interactive Viewer

> **Where**: Local Windows PC (CPU software rendering) or Kaggle for
> GPU-accelerated preview
> **Milestone**: Viewer serves animated 3D scene at `localhost:8080`

### Tasks

- [ ] 7.1 Implement `sparsechron/viewer/app.py`
    - Load exported model (`gaussians.ply` + `deformation.pt` + `model_info.json`)
    - Set up Viser scene with point cloud rendering
    - Implement orbit camera controls (Viser provides this natively)
- [ ] 7.2 Add time slider widget
    - Viser GUI slider: [0.0, 1.0] mapped to animation timestep
    - On change: apply deformation MLP at new timestep, update Gaussian
      positions, re-render
- [ ] 7.3 Add play/pause button
    - Auto-increment timestep at configurable FPS
    - Loop animation
- [ ] 7.4 Add UI controls:
    - Background toggle (show/hide static Gaussians)
    - Stats panel (Gaussian count, static/dynamic ratio, FPS)
    - Snapshot button (save current view as PNG)
    - Reset camera button
- [ ] 7.5 Write `scripts/viewer.py` — CLI entry point
- [ ] 7.6 Test viewer with exported model — verify all controls work

### Deliverable

Interactive viewer ready for demo day. Loads model, scrubs through time,
orbits camera.

---

## Phase 8 — Polish & Deliverables

> **Where**: Local + Kaggle
> **Milestone**: Complete GitHub repo ready for submission

### Tasks

- [ ] 8.1 Write `README.md`:
    - Project overview with teaser image/GIF
    - Installation instructions (local + Kaggle)
    - Quick start (preprocess → train → view)
    - Results table with comparison
    - Citation info for the two papers
    - Architecture diagram
- [ ] 8.2 Write demo video script (2–3 minutes):
    - 0:00–0:20 — Problem statement (sparse photos → 4D reconstruction)
    - 0:20–0:50 — Architecture walkthrough (pipeline diagram)
    - 0:50–1:30 — Live demo (viewer with orbit + time scrub)
    - 1:30–2:10 — Results (metrics table, comparison renders)
    - 2:10–2:30 — Key technical contributions
    - 2:30–2:50 — Future work
- [ ] 8.3 Prepare viva defense materials:
    - Key concepts cheat sheet (see Appendix A below)
    - Anticipated questions and answers
    - Ablation results (if completed)
- [ ] 8.4 Final code cleanup:
    - Docstrings on all public functions
    - Type hints throughout
    - Remove dead code, debug prints
    - Run `ruff` linter, fix all issues
- [ ] 8.5 Final benchmark run with clean code (verify metrics unchanged)
- [ ] 8.6 Tag release `v1.0.0`, push to GitHub

### Deliverable

Submission-ready repository, demo video script, defense prep materials.

---

## Phase Summary

| Phase | Focus | Dependencies |
|---|---|---|
| 1 | Project scaffolding | None |
| 2 | Data pipeline | Phase 1 |
| 3 | Static 3DGS | Phase 2 |
| 4 | 4DGS temporal | Phase 3 |
| 5 | Kaggle notebooks | Phase 4 |
| 6 | Evaluation | Phase 4 |
| 7 | Viewer | Phase 4 |
| 8 | Polish | Phases 5–7 |

Phases 5, 6, and 7 can run **in parallel** after Phase 4 completes — they are
independent of each other.

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──┬──▶ Phase 5 ──┐
                                                ├──▶ Phase 6 ──┼──▶ Phase 8
                                                └──▶ Phase 7 ──┘
```

---

## Appendix A — Viva Defense Concepts

Key topics you must be able to explain fluently:

### Gaussian Splatting Fundamentals
- **What is a 3D Gaussian?** Parameterized by mean (position), covariance
  (scale + rotation), opacity, and color (SH coefficients). Each Gaussian
  represents a small "blob" of radiance in 3D space.
- **Splatting vs ray marching**: Splatting projects Gaussians to 2D and
  alpha-composites them (forward rendering), while NeRF-style methods march
  rays through a volume (backward). Splatting is orders of magnitude faster.
- **Spherical harmonics (SH)**: Basis functions on the sphere. SH coefficients
  encode view-dependent color — when you look at a Gaussian from different
  angles, the SH evaluation produces different colors (specular highlights).
- **Differentiable rasterization**: gsplat implements CUDA kernels that compute
  gradients of the rendered image w.r.t. all Gaussian parameters, enabling
  optimization via backpropagation.

### Sparse-View Challenges
- **Why sparse views are hard**: With few input views, the photometric loss
  has too few constraints → Gaussians can float to incorrect positions,
  collapse to needles, or produce blurry reconstructions.
- **Depth priors as regularization**: Monocular depth maps provide geometric
  guidance that is unavailable from sparse photometric signal alone. They
  anchor Gaussians to plausible surfaces.
- **Deformation regularization**: Without it, dynamic Gaussians can overfit
  to individual frames — producing correct renders at training times but
  garbage at intermediate times.

### Your Two Key Contributions (from the papers)
- **Texture-aware regularization** (Sparse4DGS): In textured regions, the
  photometric loss already provides strong gradients to guide deformation —
  so regularization weight is low. In textureless regions, the photometric
  signal is weak — so regularization weight is high, preventing wild
  deformations. The weight formula is w = exp(-β · ‖∇I‖).
- **Adaptive static/dynamic allocation** (Hybrid 3D-4DGS): Most Gaussians in
  a dynamic scene are actually static (e.g. background). Attaching a
  deformation MLP to static Gaussians wastes compute and VRAM. The classifier
  detects low-deformation Gaussians and marks them static, removing them from
  the deformation pipeline.

### Metrics
- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy.
  Higher is better. 30+ dB is considered good for novel view synthesis.
- **SSIM** (Structural Similarity Index): Measures structural similarity
  (luminance, contrast, structure). Range [0, 1], higher is better.
- **LPIPS** (Learned Perceptual Image Patch Similarity): Uses a pre-trained
  VGG network to measure perceptual similarity. Lower is better. More
  aligned with human judgment than PSNR/SSIM.

---

*Derived from [PRD.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/PRD.md),
[TechSpec.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/TechSpec.md),
[AppFlow.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/AppFlow.md), and
[Schema.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/Schema.md).*
