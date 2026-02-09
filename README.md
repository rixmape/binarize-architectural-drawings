# Binarize: Shallow Convolutional Autoencoders for Historical Architectural Drawings

A Python replication of the end-to-end machine learning pipeline described in:

> Narag, M. J. G., Lico, G. R., & Soriano, M. (2025). Binarizing historical architectural drawings with shallow convolutional autoencoders. Engineering Applications of Artificial Intelligence, 148, 110400. [doi:10.1016/j.engappai.2025.110400](https://doi.org/10.1016/j.engappai.2025.110400)

Hyperparameters not specified in the paper (loss function, batch size, patch stride, reconstruction logic) were clarified directly by the corresponding author.

## What It Does

Historical architectural drawings accumulate stains that can be as dark as the drawn lines themselves, causing traditional thresholding techniques to fail. This script trains shallow convolutional autoencoders on manually cleaned patches from a *single* drawing (less than 1% of a collection), then generalizes to binarize every other drawing automatically — separating ink lines from stains, linen texture, pencil marks, and tape residue.

## Pipeline Overview

```plaintext
Dirty Image → Grayscale → Patch Extraction → Autoencoder → Binarization → Stitched Output
```

1. **Patch extraction** — High-resolution scans are sliced into small overlapping patches (training) or non-overlapping grids (inference).
2. **Shallow autoencoder** — A single-hidden-layer convolutional autoencoder maps dirty patches to clean binary patches, trained with MSE loss via Adam.
3. **Reconstruction** — Predicted patches are tiled back into the full-resolution binarized drawing.

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.12 with TensorFlow 2.x. Runs on CPU or GPU. Models 1–4 fit within Google Colab's free tier; Models 5–8 require more RAM.

## Usage

**Train** a shallow model on a dirty/clean image pair:

```bash
python binarize.py train \
    --dirty data/drawing1_q3_dirty.tif \
    --clean data/drawing1_q3_clean.tif \
    --model-id 6 \
    --output-dir runs/model6
```

**Binarize** an unseen drawing:

```bash
python binarize.py infer \
    --image data/drawing2.tif \
    --weights runs/model6/model_6_best.weights.h5 \
    --model-id 6 \
    --output results/drawing2_binarized.png
```

**Evaluate** against a ground truth:

```bash
python binarize.py evaluate \
    --predicted results/drawing2_binarized.png \
    --ground-truth data/drawing2_gt.png
```

**Fine-tune** with additional patches to handle edge cases (faint lines, tape marks):

```bash
python binarize.py finetune \
    --dirty data/extra_dirty/ \
    --clean data/extra_clean/ \
    --weights runs/model6/model_6_best.weights.h5 \
    --model-id 6 \
    --output-dir runs/model6_finetuned
```

**Compare** against traditional baselines (Simple, Otsu, Adaptive Gaussian, Adaptive Mean):

```bash
python binarize.py compare \
    --image data/drawing2.tif \
    --ground-truth data/drawing2_gt.png
```

## Available Models

| Model | Type | Input Size | Params |
|-------|------|-----------|--------|
| 1 | Deep | 32×32 | 148,865 |
| 2 | Shallow | 32×32 | 1,217 |
| 3 | Deep | 64×64 | 739,073 |
| 4 | Shallow | 64×64 | 1,217 |
| 5 | Deep | 128×128 | 739,073 |
| 6 | Shallow | 128×128 | 1,217 |
| 7 | Deep | 256×256 | 3,099,137 |
| 8 | Shallow | 256×256 | 1,217 |

The paper found that shallow models consistently outperform their deep counterparts (F1 up to 0.977 vs. 0.968), while requiring orders of magnitude fewer parameters.

## Key Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss | MSE | Author |
| Optimizer | Adam (lr=0.001) | Paper §3.2 |
| Batch size | 32 | Author |
| Epochs | 100 | Paper §3.2 |
| Train/val split | 70/30 | Paper §3.2 |
| Binarization threshold | 0.5 | Standard default |
