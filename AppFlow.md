# SparseChron — Application Flow

Since SparseChron is a research pipeline (not a traditional web/mobile app),
"application flow" means the **developer/researcher workflows** through the
system.  There are four primary flows.

---

## Flow 1 — Full Pipeline (New Scene)

The end-to-end path from raw photos to interactive viewer.

```mermaid
flowchart TD
    A["📷 Capture 10–15 burst photos"] --> B["Place images in data/scene/images/"]
    B --> C{"Timestamps available?"}
    C -- Yes --> D["Create timestamps.json"]
    C -- No --> E["Auto-assign uniform timestamps"]
    D --> F["Run: python scripts/preprocess.py --scene data/scene"]
    E --> F

    F --> G["Stage 0: Depth Anything V2\n→ data/scene/depths/*.npy"]
    G --> H["Stage 1a: DUSt3R pose estimation\n→ camera poses + dense point maps"]
    H --> I{"Reprojection error\n< threshold?"}
    I -- Yes --> J["Stage 1b: Depth alignment\n(scale-shift fit)"]
    I -- No --> K["Fallback: COLMAP\npycolmap pipeline"]
    K --> J
    J --> L["Stage 1c: Initialize Gaussians\n→ data/scene/init_gaussians.ply"]

    L --> M["Run: python scripts/train.py --scene_dir data/scene"]
    M --> N["Stage 2: 4DGS Training Loop\n(gsplat + deformation MLP)"]
    N --> O["Checkpoint saved every 30 min"]
    O --> P{"Training complete?\n(max_iterations reached)"}
    P -- No --> N
    P -- Yes --> Q["Run: python scripts/export.py --checkpoint outputs/.../final.ckpt"]
    Q --> R["Export: gaussians.ply + deformation.pt"]
    R --> S["Run: python scripts/viewer.py --model_dir outputs/.../final/"]
    S --> T["🖥️ Viser viewer at localhost:8080"]
```

---

## Flow 2 — Kaggle Training Session

The workflow inside a single Kaggle session, designed for interruption safety.

```mermaid
flowchart TD
    A["🟢 Kaggle session starts"] --> B["Cell 1: pip install deps"]
    B --> C["Cell 2: Mount Kaggle Dataset\n(symlink to data/)"]
    C --> D{"Cell 3: Checkpoint\nexists in input dataset?"}

    D -- Yes --> E["Load latest .ckpt\nRestore: model + optimizer +\nRNG + iteration counter"]
    D -- No --> F{"init_gaussians.ply\nexists?"}

    F -- Yes --> G["Start training from iteration 0"]
    F -- No --> H["Run preprocessing first\n(Stage 0 + Stage 1)"]
    H --> G

    E --> I["Resume training from\niteration N"]
    G --> I

    I --> J["Training loop"]
    J --> K{"30 min since\nlast save?"}
    K -- Yes --> L["Save checkpoint to\n/kaggle/working/checkpoints/"]
    L --> J
    K -- No --> M{"Training\ncomplete?"}
    M -- No --> J
    M -- Yes --> N["Save final checkpoint"]

    N --> O["Cell 5: Copy outputs to\nKaggle Dataset (output)"]
    O --> P["🔴 Session ends\n(or commit output dataset)"]

    style E fill:#2d5a27,stroke:#4a8,color:#fff
    style L fill:#5a4a00,stroke:#aa8,color:#fff
    style P fill:#5a1a1a,stroke:#a44,color:#fff
```

### Session Recovery

When a session times out unexpectedly:

1. Start a new Kaggle session
2. Attach the output dataset from the previous session as input
3. The resume logic auto-detects the latest valid checkpoint
4. Training continues from exactly where it left off (same iteration, same RNG state)

---

## Flow 3 — Evaluation & Benchmarking

```mermaid
flowchart TD
    A["Trained model\n(gaussians.ply + deformation.pt)"] --> B["Run: python scripts/evaluate.py\n--model_dir outputs/.../final\n--test_dir data/scene/test"]

    B --> C["For each test view:"]
    C --> D["Render novel view\nat test camera + timestep"]
    D --> E["Compute PSNR, SSIM, LPIPS\nvs ground truth"]
    E --> F["Save rendered image\nto outputs/.../renders/"]

    F --> G{"More test views?"}
    G -- Yes --> C
    G -- No --> H["Aggregate metrics\n→ outputs/.../metrics.json"]

    H --> I["Generate comparison table\n(per-scene + mean)"]
    I --> J["Generate bar charts\n(matplotlib)"]
    J --> K["📊 Results ready for report"]
```

### Evaluation Variants

| Variant | Command Flag | Purpose |
|---|---|---|
| Standard | `--eval_mode standard` | Full test set, all metrics |
| Quick | `--eval_mode quick` | 5 random test views, PSNR only |
| Baseline comparison | `--baseline_dir outputs/baseline/` | Side-by-side with vanilla 4DGS |
| Ablation | `--ablation no_depth` | Disable specific components |

---

## Flow 4 — Viewer Interaction

The Viser viewer is the demo-day deliverable. Here's what the user can do:

```mermaid
flowchart TD
    A["Launch: python scripts/viewer.py"] --> B["Viser server starts\nat localhost:8080"]
    B --> C["Browser opens viewer"]

    C --> D["Orbit Camera\n(click + drag to rotate,\nscroll to zoom,\nright-drag to pan)"]

    C --> E["Time Slider\n(drag to scrub through\nanimation timesteps)"]

    C --> F["Play / Pause\n(auto-animate through\ntimesteps in loop)"]

    C --> G["Background Toggle\n(show/hide background\nGaussians)"]

    C --> H["Stats Panel\n(Gaussian count,\nstatic vs dynamic split,\nFPS)"]

    C --> I["Render Snapshot\n(save current view\nas PNG)"]
```

### Viewer Controls Summary

| Control | Input | Action |
|---|---|---|
| Rotate | Left-click + drag | Orbit camera around scene center |
| Zoom | Scroll wheel | Move camera closer/further |
| Pan | Right-click + drag | Translate camera laterally |
| Time scrub | Slider widget | Set current animation timestep |
| Play/Pause | Button | Auto-cycle through timesteps |
| Background | Toggle button | Show/hide static Gaussians |
| Snapshot | Button | Save current rendered view as PNG |
| Reset | Button | Return camera to default position |

---

## Flow 5 — Error & Recovery Paths

| Scenario | Detection | Recovery |
|---|---|---|
| Kaggle session timeout | Training loop doesn't finish | Resume from last 30-min checkpoint (Flow 2) |
| OOM during training | `RuntimeError: CUDA out of memory` | Reduce `max_gaussians`, enable more aggressive pruning, restart |
| DUSt3R pose failure | Reprojection error > 5px threshold | Automatic fallback to COLMAP (Flow 1) |
| Corrupted checkpoint | `torch.load()` raises exception | Skip corrupted file, load previous checkpoint; warn user |
| gsplat build failure on Kaggle | Import error | Use pre-built wheel pinned to Kaggle's CUDA version |
| NaN in loss | `torch.isnan(loss)` check | Log warning, skip iteration, reduce learning rate by 0.5× |
| Depth Anything V2 OOM | CUDA OOM on high-res images | Resize images to max 512px for depth estimation, then upsample |

---

## Data Flow Summary

```
raw_images/
    │
    ▼
┌──────────────────┐     ┌──────────────────┐
│  Depth Anything  │     │     DUSt3R       │
│  V2              │     │  (pose + points) │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
    depths/*.npy          cameras.json
         │               pointmaps/*.npy
         │                        │
         └──────────┬─────────────┘
                    │
              depth_alignment
              init_gaussians
                    │
                    ▼
           init_gaussians.ply
                    │
                    ▼
         ┌──────────────────┐
         │   4DGS Training  │
         │   (gsplat loop)  │
         └────────┬─────────┘
                  │
         checkpoints/*.ckpt
                  │
                  ▼
         gaussians.ply + deformation.pt
                  │
          ┌───────┴────────┐
          ▼                ▼
    Viser Viewer      Evaluation
    (localhost)       (metrics.json)
```

---

*Derived from [PRD.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/PRD.md) and [TechSpec.md](file:///c:/Users/odeda/Desktop/Projects/IVP%20Project/TechSpec.md).*
