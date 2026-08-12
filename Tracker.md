# SparseChron — Tracker

Progress log. Append-only — never rewrite history. Check items off as they are
completed and add notes where useful.

---

## Phase 1 — Project Scaffolding

- [x] 1.1 Initialize Git repo + `.gitignore`
- [x] 1.2 Create directory structure
- [x] 1.3 Write `pyproject.toml`
- [x] 1.4 Generate `requirements.txt`
- [x] 1.5 Implement `utils/config.py`
- [x] 1.6 Implement `utils/camera.py`
- [x] 1.7 Implement `utils/transforms.py`
- [x] 1.8 Write unit tests (camera + quaternion)
- [x] 1.9 Verify `pytest` passes on CPU

## Phase 2 — Data Pipeline

- [x] 2.1 Depth estimation wrapper (Depth Anything V2)
- [x] 2.2 Pose estimation wrapper (DUSt3R + COLMAP fallback)
- [x] 2.3 Depth alignment (scale-shift fitting)
- [x] 2.4 Gaussian initialization from point cloud
- [x] 2.5 `SceneDataset` class
- [x] 2.6 DUSt3R + COLMAP loaders
- [x] 2.7 `scripts/preprocess.py` CLI
- [x] 2.8 Test on NeRF-Synthetic `lego`

## Phase 3 — Static 3DGS Baseline

- [x] 3.1 `GaussianModel` class
- [x] 3.2 gsplat renderer wrapper
- [x] 3.3 Photometric loss (L1 + SSIM)
- [x] 3.4 Depth loss
- [x] 3.5 Densification & pruning scheduler
- [x] 3.6 Checkpoint save/load/resume
- [x] 3.7 Training loop (static-only)
- [x] 3.8 `scripts/train.py` CLI
- [x] 3.9 Unit tests (gaussians, renderer, checkpoint, losses)
- [ ] 3.10 Train `lego` — verify PSNR > 25 dB

## Phase 4 — Temporal Deformation (4DGS)

- [x] 4.1 Implement `sparsechron/models/deformation.py`
- [x] 4.2 Extend `GaussianModel.get_deformed(deformation_mlp, timestep)`
- [x] 4.3 Implement `sparsechron/models/classifier.py`
- [x] 4.4 Implement `sparsechron/losses/regularization.py`
- [x] 4.5 Extend `trainer.py` for 4DGS
- [x] 4.6 Write unit tests:
    - `test_deformation.py` — MLP forward shapes, offset application, gradient flow
    - `test_regularization.py` — texture weight computation, loss shape
- [ ] 4.7 Train on HyperNeRF `interp` scene (12 sparse views, 30K iterations)

## Phase 5 — Kaggle Notebook Integration

- [x] 5.1 Create `notebooks/sparsechron_train.ipynb`
- [x] 5.2 Validate memory usage (<15GB VRAM)
- [x] 5.3 Test 9-hour limit checkpointing
- [x] 5.4 Document Kaggle workflow in README

## Phase 6 — Evaluation & Benchmarking

- [x] 6.1 Implement `sparsechron/evaluation/metrics.py`
- [x] 6.2 Implement `sparsechron/evaluation/render_novel_views.py`
- [x] 6.3 Implement `sparsechron/evaluation/benchmark.py`
- [x] 6.4 Write `scripts/evaluate.py`
- [ ] 6.5 Run benchmark suite on Kaggle
- [ ] 6.6 Generate results table + bar charts

## Phase 7 — Viser Interactive Viewer

- [x] 7.1 Implement `sparsechron/viewer/app.py`
- [x] 7.2 Add time slider widget
- [x] 7.3 Add play/pause button
- [x] 7.4 Add UI controls (stats, snapshots, background toggle)
- [x] 7.5 Write `scripts/viewer.py`
- [ ] 7.6 Test viewer with exported model

## Phase 8 — Polish & Deliverables

- [ ] 8.1 Write `README.md`
- [ ] 8.2 Demo video script
- [ ] 8.3 Viva defense materials
- [ ] 8.4 Final code cleanup (docstrings, type hints, ruff)
- [ ] 8.5 Final benchmark run
- [ ] 8.6 Tag `v1.0.0`, push to GitHub

---

## Log

| Date | Note |
|---|---|
| — | Project planning complete (PRD, TechSpec, AppFlow, Schema, ImplementationPlan, Rules, Tracker) |

---

*Check items off as work is completed. Add dated notes in the Log table for
significant milestones, blockers, or decisions.*
