<h1 align="center">
  SparseChron 🌌
</h1>

<p align="center">
  <i>A highly optimized 4D Gaussian Splatting engine designed to synthesize dynamic scenes from sparse, uncalibrated datasets on heavily constrained hardware.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Compute-CUDA_11.8%2B-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Hardware-T4_GPU_Optimized-black?style=flat-square" alt="T4 GPU" />
</p>

---

## 📖 Abstract

Traditional 4D Gaussian Splatting requires massive amounts of VRAM, dense multi-view video arrays, and pre-computed COLMAP camera poses. 

**SparseChron** breaks these constraints. It is a custom 4DGS pipeline that synthesizes highly accurate 4D structures from just **10-15 uncalibrated sparse images** while operating strictly within a **16GB VRAM limit** (e.g., free Kaggle T4 GPUs). By fusing **Depth Anything V2** with **DUSt3R** for pose extraction and implementing dynamic compute-saving classifiers, SparseChron brings state-of-the-art scene synthesis to accessible hardware.

---

## ✨ Key Contributions

- **🚫 Zero COLMAP Required:** Replaces traditional Structure-from-Motion (SfM) pipelines. Poses and metrics are dynamically extracted using DUSt3R and depth alignment.
- **📉 Strict Memory Optimization:** Implements a hard cap of 500,000 Gaussians to prevent Out-Of-Memory (OOM) crashes on 16GB VRAM limits.
- **🧊 Static-Dynamic Freezing:** Introduces a `StaticDynamicClassifier` that identifies Gaussians that have stopped moving over time and freezes them, significantly reducing rasterization compute load.
- **🎭 Texture-Aware Regularization:** Eliminates temporal "floating" artifacts in texture-less environments using a custom image gradient loss function.

---

## 🏗️ Pipeline Architecture

```mermaid
graph TD
    A[Uncalibrated Sparse Images] --> B[DUSt3R Engine]
    A --> C[Depth Anything V2]
    
    B -->|Camera Poses & Point Clouds| D{Alignment & Scaling Module}
    C -->|Monocular Depth Maps| D
    
    D -->|Initialized 3D Gaussians| E[4D Gaussian Splatting Rasterizer]
    
    E --> F[Temporal Optimization Loop]
    F -->|Detect non-moving points| G[StaticDynamicClassifier]
    G -->|Freeze to save compute| F
    
    F -->|Final Output| H((Rendered 4D Scene))
```

---

## 🚀 Quick Start & Installation

```bash
git clone https://github.com/rajhodedara/SparseChron.git
cd SparseChron

# We highly recommend using a conda environment
conda create -n sparsechron python=3.10
conda activate sparsechron

# Install PyTorch and dependencies
pip install -r requirements.txt
```

---

## ☁️ The Kaggle Training Workflow

SparseChron was specifically engineered to be trained across disjoint, 9-hour free Kaggle sessions without losing progress.

1. **Upload Notebook**: Import `notebooks/sparsechron_train.ipynb` into a new Kaggle session.
2. **Mount Data**: Attach your image dataset to the Kaggle notebook environment.
3. **Run All**: The notebook will autonomously:
   - Install all required dependencies and submodules.
   - Look for the latest `.ckpt` file in your output directory to resume training instantly.
   - Run the training loop while printing `max_vram_allocated()` logs every 100 iterations.
4. **Seamless Resumption**: When Kaggle inevitably terminates your session at the 9-hour limit, simply restart the machine. The custom `find_latest_checkpoint` utility will load your progress effortlessly!

---

## 🛡️ License
This project is licensed under the MIT License.
