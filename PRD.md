# SparseChron — Product Requirements Document

## 1. Project Summary

**SparseChron** is a 4D scene reconstruction system that takes 10–15 sparse
burst photos as input and produces an interactive, animated 3D scene viewable in
a browser.  It is a third-year undergraduate capstone project targeting faculty
evaluation and potential academic conference submission.

The system demonstrates that a small handful of burst photos — far fewer than the
dense multi-view sequences required by existing methods — are sufficient to
reconstruct temporally coherent dynamic scenes.  It does so by combining ideas
from two recent papers:

| Paper | Key Idea Borrowed |
|---|---|
| **Sparse4DGS** (arXiv:2511.07122, AAAI 2026) | Texture-aware deformation regularization for sparse-frame dynamic reconstruction |
| **Hybrid 3D-4DGS** (arXiv:2505.13215) | Adaptive static/dynamic Gaussian allocation for memory & compute efficiency |

---

## 2. Target Audience

| Audience | Needs |
|---|---|
| **Capstone committee / faculty panel** | Clear demonstration of technical depth, novelty, and engineering rigor |
| **Academic reviewers** (potential conference) | Reproducible benchmarks, ablation studies, and comparison with baselines |
| **The student (you)** | A codebase you fully understand and can defend in a viva |

---

## 3. Goals & Success Criteria

### Must-Have Goals

1. **End-to-end pipeline**: Raw burst photos → camera poses → depth priors →
   4D Gaussian Splat → animated 3D viewer.
2. **Sparse-input advantage**: Demonstrate competitive or superior PSNR / SSIM /
   LPIPS compared to vanilla 4DGS when both are given only 10–15 input frames.
3. **Kaggle-safe training**: Checkpoint every 30 minutes, resume cleanly,
   complete training within Kaggle's T4 session limits.
4. **Interactive viewer**: Local Viser-based web viewer for demo day
   presentation.
5. **Benchmark results**: Quantitative table on NeRF-Synthetic and HyperNeRF
   datasets.
6. **Clean GitHub repo**: Well-documented, reproducible, with README.

### Nice-to-Have Goals

- Ablation study (with/without depth prior, with/without deformation
  regularization, with/without static-dynamic split).
- Novel view synthesis quality at varying sparsity levels (5, 10, 15 views).
- Training efficiency metrics (wall-clock time, peak VRAM, convergence speed).
- Real-world casual capture demo (phone burst shots of a moving object).
- 2–3 minute demo video.

---

## 4. Core Features

### F1 — Camera Pose Estimation (from sparse photos)

- **Primary method: DUSt3R** (recommended for 10–15 sparse frames).
  - Reason: COLMAP's feature matching degrades significantly below ~20 views.
    DUSt3R uses a transformer-based dense stereo approach that produces reliable
    relative poses and dense point maps even from very few, wide-baseline images.
  - DUSt3R also outputs dense depth maps as a byproduct, which can bootstrap
    depth initialization (partially overlapping with F2).
- **Fallback: COLMAP** for controlled synthetic datasets where camera parameters
  are already known or views are close-baseline.

### F2 — Monocular Depth Initialization

- Use **Depth Anything V2** to generate per-frame monocular depth maps.
- These depth priors regularize the initial Gaussian positions and scales,
  preventing the degenerate geometries that plague sparse-input 3DGS.
- Align DUSt3R metric depths with Depth Anything relative depths via scale-shift
  fitting (least-squares).

### F3 — 4D Gaussian Splatting Core

- **Static Gaussians**: Standard 3D Gaussian Splatting (position, scale,
  rotation, opacity, SH coefficients) for rigid/background regions.
- **Dynamic Gaussians**: Each Gaussian additionally carries a lightweight
  deformation MLP (or temporal basis functions) that predicts per-timestep
  offsets to position, rotation, and scale.
- **Adaptive allocation**: Classify each Gaussian as static or dynamic during
  training based on its accumulated deformation magnitude (Hybrid 3D-4DGS
  strategy).  This saves VRAM and compute by not attaching deformation networks
  to static regions.
- **Texture-aware deformation regularization** (Sparse4DGS): Penalize large
  deformations in texture-rich regions where the photometric loss already
  provides strong gradients; allow larger deformations in textureless regions
  where regularization must fill the gap.

### F4 — Training Pipeline

- End-to-end differentiable: rasterize via **gsplat**, backprop through
  photometric loss (L1 + SSIM) + depth loss + deformation regularization.
- Mixed-precision training (FP16 via `torch.cuda.amp`) to fit within 16 GB
  VRAM.
- Densification and pruning schedule adapted for sparse inputs (more aggressive
  early densification, gentler pruning).

### F5 — Checkpoint & Resume System

- Save full training state (Gaussians, optimizer, deformation nets, scheduler,
  iteration counter, RNG state) every 30 minutes.
- On resume: detect the latest valid checkpoint and continue training
  seamlessly.
- Persist checkpoints and final outputs to Kaggle Datasets for durability.

### F6 — Interactive Viewer

- **Viser**-based local web viewer.
- Features: orbit camera, time slider (scrub through the animation), play/pause,
  background toggle, Gaussian count display.
- Loads the final trained model (exported `.ply` + deformation weights).

### F7 — Evaluation & Benchmarking

- Datasets: **NeRF-Synthetic** (8 synthetic scenes) and **HyperNeRF** (real
  dynamic scenes).
- Metrics: **PSNR**, **SSIM**, **LPIPS** (via `torchmetrics` or `lpips`
  library).
- Baseline comparison: Vanilla 4DGS trained on the same sparse subset.
- Results presented in a clean table + per-scene bar charts.

---

## 5. Input / Output Specification

| | Description |
|---|---|
| **Input** | 10–15 JPEG/PNG images (burst capture or subsampled from a video), optionally with timestamps |
| **Intermediate** | Camera poses (DUSt3R / COLMAP), depth maps (Depth Anything V2), initial point cloud |
| **Output — Training** | Trained 4DGS checkpoint (`.ckpt`), exported Gaussians (`.ply`), deformation weights (`.pt`) |
| **Output — Viewer** | Local Viser web app serving the animated 3D scene |
| **Output — Eval** | Metrics JSON/CSV, rendered novel-view images, comparison tables |

---

## 6. Constraints

| Constraint | Detail |
|---|---|
| **GPU** | Kaggle T4 (16 GB VRAM), 30 hrs/week, 9-hour session limit |
| **Local dev** | Windows PC, no GPU — used for code editing, testing (CPU-only), and viewer development |
| **Checkpointing** | Must save every 30 minutes to survive Kaggle session timeouts |
| **Dependencies** | Must work with PyTorch 2.x, gsplat (CUDA), and Kaggle's pre-installed environment |
| **Reproducibility** | Fixed random seeds, deterministic data loading, documented hyperparameters |

---

## 7. Out of Scope

- Real-time training (training is offline/batch).
- Mobile or cloud deployment of the viewer.
- Video input (only still images; video→frames extraction is a pre-processing
  step the user does manually).
- Multi-GPU or distributed training.
- Novel architecture research beyond combining the two cited papers.

---

## 8. Risks (High-Level)

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle session timeout mid-training | Lost progress | 30-min checkpoint cycle, resume logic |
| 16 GB VRAM insufficient for dense scenes | OOM crashes | Mixed precision, aggressive pruning, Gaussian budget cap |
| DUSt3R produces poor poses on certain scenes | Bad reconstruction | Fall back to COLMAP; validate poses with reprojection error |
| Sparse inputs → blurry/degenerate output | Weak results for defense | Depth priors + deformation regularization are specifically designed to counter this |
| gsplat version mismatch on Kaggle | Build failures | Pin versions, pre-build wheels, test in Kaggle env early |

---

*This document is the single source of truth for what SparseChron does and
doesn't do.  All subsequent planning files derive from it.*
