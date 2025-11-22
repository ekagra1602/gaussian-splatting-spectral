# Gaussian Splatting for Campus Reconstruction

Reconstruction of the lantern at Hayden Lawn (ASU) using **3D Gaussian Splatting** and **VGGT** for pose estimation. This project introduces a custom **Spectral Anisotropy Regularization** to reduce needle-like artifacts in reconstructed Gaussians.

## 🚀 Quick Start

### 1. Prepare Data
Extract and filter frames from video:
```bash
python src/dataset_prep.py --video <video.mp4> --out data/campus_scene
```

### 2. Train (Spectral Regularization)
Train with custom spectral loss to improve Gaussian shapes:
```bash
python src/gaussian_spectral_training.py \
    --data_dir data/campus_scene \
    --result_dir Results/Spectral \
    --use_spectral_loss \
    --spectral_lambda 0.01
```

### 3. Train (Baseline)
Train standard model for comparison:
```bash
python src/gaussian_spectral_training.py \
    --data_dir data/campus_scene \
    --result_dir Results/Baseline \
    --spectral_lambda 0.0
```

## 🧪 Custom Feature: Spectral Regularization
Implemented in `src/spectral_loss.py`, this loss penalizes anisotropic (needle-like) Gaussians by maximizing the **Spectral Entropy** of their scales.
- **Goal**: Encourage spherical shapes, improve geometry, and reduce overfitting.
- **Loss**: $L_{spectral} = 1 - H_{norm}$ (where $H_{norm}$ is normalized spectral entropy).

## 📂 Structure
- `src/`: Custom training and prep scripts.
- `gsplat/` & `vggt/`: Core libraries.
- `Results/`: Checkpoints (`.pt`) and meshes (`.ply`).

## 📚 References
- **VGGT** (CVPR 2025), **gsplat** (JMLR 2025), **3DGS** (SIGGRAPH 2023)
