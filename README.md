# ViT-SE + Modified Firefly Algorithm (MFA) for EuroSAT Land Cover Classification

A PyTorch implementation of a **Squeeze-and-Excitation Vision Transformer (ViT-SE)** optimized via a **Modified Firefly Algorithm (MFA)** for Sentinel-2 satellite imagery classification on the **EuroSAT RGB** dataset, featuring a multi-channel image legibility enhancement pipeline.

This repository includes:

1. **Standard ViT-SE + MFA (`train_vitse.py`):** Implementation of the ViT-SE architecture and Modified Firefly Algorithm for 10-class satellite land-cover classification.
2. **Legibility-Enhanced 5-Channel ViT-SE (`train_vitse_legibility.py`):** An enhanced multi-channel pipeline that fuses standard RGB channels with **Contrast-Limited Adaptive Histogram Equalization (CLAHE)** and **Median-Filtered Canny Edge** maps into a 5-channel Vision Transformer input.

## Table of Contents

* [Overview & Architecture](#overview--architecture)
  * [1. Vision Transformer with Squeeze-and-Excitation (ViT-SE)](#1-vision-transformer-with-squeeze-and-excitation-vit-se)
  * [2. Modified Firefly Algorithm (MFA) for Hyperparameter Tuning](#2-modified-firefly-algorithm-mfa-for-hyperparameter-tuning)
  * [3. 5-Channel Legibility Preprocessing Pipeline](#3-5-channel-legibility-preprocessing-pipeline)
* [Repository Structure](#repository-structure)
* [Dataset Setup (EuroSAT RGB)](#dataset-setup-eurosat-rgb)
* [Installation & Requirements](#installation--requirements)
* [Usage & Command-Line Arguments](#usage--command-line-arguments)
* [Model Configurations](#model-configurations)
* [Evaluation Metrics](#evaluation-metrics)
* [References](#references)

## Overview & Architecture

### 1. Vision Transformer with Squeeze-and-Excitation (ViT-SE)

Standard Vision Transformers (ViTs) capture global spatial dependencies across image patches using Multi-Head Self-Attention (MHSA), but they treat all feature channels equally. In satellite land-cover classification, rural classes such as *Annual Crop*, *Permanent Crop*, *Pasture*, and *Herbaceous Vegetation* share similar spatial patterns and overlap in feature space.

To address this, **ViT-SE** integrates a **Squeeze-and-Excitation (SE)** block immediately after every Transformer Encoder block (`ViTSEBlock`):

1. **Patch Embedding & Sinusoidal Positional Encoding:** Input images are partitioned into $16 \times 16$ non-overlapping patches via a 2D convolution (`PatchEmbedding`), prepended with a learnable `[CLS]` token, and summed with fixed sinusoidal positional encodings:
   
   $$
   P_{(pos, 2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right), \quad P_{(pos, 2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)
   $$

2. **Spatial Attention (Transformer Encoder):** Applies Multi-Head Self-Attention, residual connections, LayerNorm, and a GELU Feed-Forward Network.

3. **Channel Recalibration (`SEBlock`):**
   * **Squeeze:** Global Average Pooling across all patch tokens compresses the $(B, N, D)$ sequence into a channel descriptor $s \in \mathbb{R}^{B \times D}$.
   * **Excitation:** A bottleneck MLP with reduction ratio $r = 16$ (`Linear(D, D/16) -> ReLU -> Linear(D/16, D) -> Sigmoid`) computes channel-wise attention weights $w \in [0, 1]^{B \times 1 \times D}$.
   * **Scale:** Recalibrates the token embeddings via element-wise multiplication ($x \odot w$), emphasizing discriminative spectral features and suppressing noisy channels.

### 2. Modified Firefly Algorithm (MFA) for Hyperparameter Tuning

Instead of relying on manual grid search for the learning rate $\xi \in [10^{-4}, 10^{-2}]$, the framework includes an optional **Modified Firefly Algorithm (`--mfa`)** where each firefly's position represents a candidate learning rate and its brightness corresponds to validation accuracy.

The position update rule incorporates two mathematical enhancements over the standard Firefly Algorithm:

* **Linearly Decreasing Inertia Weight ($\varpi$):** Balances global exploration and local exploitation across iterations:
  
  $$
  \varpi = \varpi_{\max} - \frac{(\varpi_{\max} - \varpi_{\min}) \times \text{iter}}{1 + \max_{\text{iter}}}, \quad \varpi_{\max}=0.9, \; \varpi_{\min}=0.4
  $$

* **Tan-Sigmoid Geometric Diversity Factor ($\psi$):** Prevents premature convergence by modulating the stochastic walk:
  
  $$
  \psi = \frac{1 - e^{-z}}{1 + e^{-z}}, \quad z = \text{rand}() \times \frac{\text{iter} + 1}{\max_{\text{iter}}}
  $$

* **Updated Position Equation:**
  
  $$
  r_i(t+1) = \varpi r_i(t) + \beta_0 e^{-\gamma d_{ij}^2}(r_j - r_i) + \alpha \psi \left(\text{rand}() - \frac{1}{2}\right)
  $$

### 3. 5-Channel Legibility Preprocessing Pipeline

Inspired by *Jayanthi & Maheswari (2024)* on image enhancement and structural edge preservation, `train_vitse_legibility.py` introduces `LegibilityTransform`, which expands the 3-channel RGB satellite image into a **5-channel composite tensor** $(5 \times H \times W)$:

* **Channels 0–2:** Standard normalized RGB channels.
* **Channel 3 (CLAHE):** Grayscale conversion followed by **Contrast-Limited Adaptive Histogram Equalization** (`clipLimit=2.0`, `tileGridSize=(8,8)`) to enhance local contrast across sub-regions while clipping noise amplification.
* **Channel 4 (Median Blur + Canny Edge):** A $3 \times 3$ nonlinear **Median Filter** to remove impulse/speckle noise, followed by **Canny Edge Detection** (`low=50`, `high=150`) to capture structural boundaries (e.g., roads, rivers, field borders).
* The `PatchEmbedding` convolutional projection layer is dynamically expanded from `in_channels=3` to `in_channels=5`.

## Repository Structure

```
├── train_vitse.py               # 3-channel RGB ViT-SE + MFA training script
├── train_vitse_legibility.py    # 5-channel Legibility-Enhanced ViT-SE + MFA training script
└── README.md                    # Project documentation
```

## Dataset Setup (EuroSAT RGB)

The project uses the **EuroSAT RGB** dataset derived from Sentinel-2 satellite imagery, consisting of **27,000 images** ($64 \times 64$ native resolution) across **10 balanced classes**:
`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.

1. Download the [EuroSAT RGB dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) (or via `torchvision.datasets.EuroSAT`).
2. Organize the dataset in standard `ImageFolder` format:

   ```
   working/
   ├── AnnualCrop/
   ├── Forest/
   ├── HerbaceousVegetation/
   ├── Highway/
   ├── Industrial/
   ├── Pasture/
   ├── PermanentCrop/
   ├── Residential/
   ├── River/
   └── SeaLake/
   ```

3. The scripts automatically perform a **seeded 80/20 stratified train-test split** (`stratified_split`) to preserve exact class ratios between training and evaluation.

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision numpy Pillow opencv-python
```

## Usage & Command-Line Arguments

### Running 3-Channel ViT-SE (`train_vitse.py`)

**1. Standard Training (`small` config):**

```bash
python train_vitse.py --data ./working --config small --epochs 20 --batch-size 32 --lr 0.0001
```

**2. Training with Modified Firefly Algorithm (MFA) Learning Rate Search:**

```bash
python train_vitse.py --data ./working --config small --mfa --mfa-fireflies 5 --mfa-iter 3 --mfa-eval-epochs 1
```

**3. Running the `exact` Configuration:**

```bash
python train_vitse.py --data ./working --config exact
```

### Running 5-Channel Legibility ViT-SE (`train_vitse_legibility.py`)

**1. 5-Channel Training (RGB + CLAHE + Canny) with `small` config:**

```bash
python train_vitse_legibility.py --data ./working --output ./outputs --config small --epochs 20
```

**2. Overnight ViT-Base Scale Run (`overnight` config):**

```bash
python train_vitse_legibility.py --data ./working --output ./outputs --config overnight --epochs 100
```

### CLI Options Reference

| Argument | Default | Description |
| ----- | ----- | ----- |
| `--data` | `./working` | Path to the EuroSAT `ImageFolder` root directory |
| `--output` | `./outputs` | Directory to save `best_vitse.pth` (in `train_vitse_legibility.py`) |
| `--config` | `small` | Model scale preset: `small`, `overnight`, or `exact` |
| `--epochs` | `None` | Override number of training epochs from config |
| `--batch-size` | `None` | Override batch size from config |
| `--lr` | `None` | Override learning rate (ignored if `--mfa` is enabled) |
| `--device` | `auto` | Compute device (`cuda`, `cpu`, or `auto`) |
| `--mfa` | `False` | Enable Modified Firefly Algorithm learning-rate optimization |
| `--mfa-fireflies` | `5` | Number of fireflies in MFA swarm |
| `--mfa-iter` | `3` | Number of MFA swarm iterations |
| `--mfa-train-samples` | `512` | Subset size of training images for fast MFA fitness evaluation |

## Model Configurations

We provide three model scale presets to support both rapid experimentation on single GPUs and large-scale training runs:

| Parameter | `small` (Default) | `overnight` | `exact` |
| ----- | ----- | ----- | ----- |
| **Input Resolution** | $224 \times 224$ | $224 \times 224$ | $384 \times 384$ |
| **Patch Size** | $16 \times 16$ | $16 \times 16$ | $16 \times 16$ |
| **Embedding Dim ($D$)** | $256$ | $768$ | $1024$ |
| **Transformer Depth ($L$)** | $6$ | $12$ | $24$ |
| **Attention Heads** | $4$ | $12$ | $16$ |
| **MLP Feedforward Dim** | $512$ | $3072$ | $4096$ |
| **Dropout** | $0.2$ | $0.2$ | $0.2$ |
| **Default Batch Size** | $32$ | $16$ | $4$ |
| **Default Epochs** | $20$ | $100$ | $100$ |

## Evaluation Metrics

All models are evaluated at the end of every epoch using a $10 \times 10$ confusion matrix (`metrics_from_confusion`) to compute:

* **Overall Accuracy**
* **Macro-Precision**
* **Macro-Recall**
* **Macro-F1 Score**
* **Cohen's Kappa ($\Xi$)**

The model checkpoint achieving the highest test accuracy is automatically saved as `best_vitse.pth`.

## References

1. **Wassan, S., Bilal, A., Alzahrani, A., Almohammadi, K., Alrashidi, M., & Mousavirad, S. J. (2025).** *A modified vision transformer framework for image-based land cover segmentation in rural architectural design and planning.* Scientific Reports, 15, 32658. https://doi.org/10.1038/s41598-025-19234-w
2. **Jayanthi, J., & Maheswari, P. U. (2024).** *Comparative study: enhancing legibility of ancient Indian script images from diverse stone background structures using 34 different pre-processing methods.* Heritage Science, 12, 63. https://doi.org/10.1186/s40494-024-01169-6
