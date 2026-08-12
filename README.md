# SparseChron

SparseChron is a 4D Gaussian Splatting engine optimized for sparse temporal datasets and constrained computing environments (specifically Kaggle T4 GPUs). By fusing Depth Anything V2 and DUSt3R, SparseChron builds highly accurate 4D structures from just 10-15 uncalibrated sparse images.

## Features
- **Uncalibrated Sparse 4DGS**: No COLMAP required. Poses and metrics are extracted dynamically using DUSt3R and depth alignment.
- **T4 GPU Optimized**: Implements a hard cap of 500k Gaussians to prevent Out-Of-Memory (OOM) crashes on 16GB VRAM limits.
- **Dynamic Compute Savings**: Uses a `StaticDynamicClassifier` to freeze Gaussians that stop moving over time, saving rasterization compute.
- **Texture-Aware Regularization**: Penalizes temporal floating artifacts in texture-less environments using a custom image gradient loss.

## Installation

```bash
git clone https://github.com/rajhodedara/SparseChron.git
cd SparseChron
pip install -r requirements.txt
```

## Kaggle Workflow

SparseChron is designed to be trained across multiple disjoint 9-hour Kaggle sessions seamlessly.

1. **Upload Notebook**: Import `notebooks/sparsechron_train.ipynb` into a new Kaggle session.
2. **Mount Data**: Attach your image dataset to the Kaggle notebook.
3. **Run All**: The notebook will:
   - Install dependencies.
   - Clone this repository.
   - Look for the latest `.ckpt` file in your output directory and resume automatically if found.
   - Run training while printing `max_vram_allocated()` limits every 100 iterations.
4. **Resuming**: When Kaggle terminates your session at the 9-hour limit, simply restart the session. The `find_latest_checkpoint` utility will load your progress effortlessly!

## License
MIT
