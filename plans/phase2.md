# Phase 2 — Evaluate on Test Set and Run Baselines

## Objective

Infer on the held-out test image (quadrant 2 of Drawing #1), compute F1/IoU/PSNR
for each shallow model, run classical binarization baselines for comparison,
and assess how close our results are to the paper's Table 1.

---

## Prerequisites

Phase 1 must be complete:

- Trained weights exist at `runs/model_{2,4,6,8}/model_{id}_best.weights.h5`.
- Training metadata has been recorded.
- Loss curves show healthy convergence.

---

## Context: Test Data

The test set comes from the authors as `gt_input.png` / `gt_output.png`:

| File            | Dimensions (H x W) | Content                                  |
| --------------- | ------------------ | ---------------------------------------- |
| `gt_input.png`  | 7348 x 9913        | Quadrant 2 of Drawing #1 (degraded scan) |
| `gt_output.png` | 7348 x 9913        | Manually binarized ground truth          |

**Important**: The test image was **never seen during training** — training
used quadrant 3, while this is quadrant 2. This is the paper's exact
evaluation protocol.

**Height mismatch note**: gt images are 7348 px tall vs. 7349 for the training
images. The script handles this via `pad_to_multiple()` for inference and
crops back to original dimensions. The `_cli_infer` function at line 864-866
also uses `min_h, min_w` cropping when comparing against ground truth.

---

## Task 2.1: Infer With All Four Shallow Models

### Commands

Run inference for each model on the test image, saving binarized outputs
and computing metrics against ground truth:

```bash
mkdir -p results

for mid in 2 4 6 8; do
    python binarize.py infer \
        --image data/gt_input.png \
        --weights runs/model_${mid}/model_${mid}_best.weights.h5 \
        --model-id $mid \
        --output results/model_${mid}_binarized.png \
        --ground-truth data/gt_output.png
done
```

### Expected behavior per model

**Model 2 (32x32)**:

- Patches: (7360/32) x (9920/32) = 230 x 310 = 71,300 non-overlapping patches
  (image padded from 7348x9913 to nearest multiples of 32).
- Inference batch_size=64 by default — fast.
- May show more "blocky" artifacts at patch boundaries due to small receptive field.

**Model 4 (64x64)**:

- Patches: ~17,600 non-overlapping patches.
- Better context per patch; fewer boundary artifacts expected.

**Model 6 (128x128)**:

- Patches: ~4,500 non-overlapping patches.
- Paper's best model — expecting highest F1 here.

**Model 8 (256x256)**:

- Patches: ~1,170 non-overlapping patches.
- Largest receptive field but fewest patches.
- Padding more significant (7348 -> 7424, 9913 -> 10112).

### Output files

```
results/
  model_2_binarized.png     # Binarized test image from Model 2
  model_4_binarized.png     # Binarized test image from Model 4
  model_6_binarized.png     # Binarized test image from Model 6
  model_8_binarized.png     # Binarized test image from Model 8
```

Metrics (F1, IoU, PSNR) will be printed to stdout/logs during each run.

### Pass criteria

- All four inferences complete without error.
- Output images have correct dimensions (7348 x 9913).
- All F1 scores are > 0.90 (if below, something is fundamentally wrong).

---

## Task 2.2: Run Traditional Baselines

### Why

The paper's Table 2 (Section 4) reports baseline binarization performance
using four classical methods. Running these on the same test image lets us:

1. Confirm the autoencoder's advantage over traditional techniques.
2. Validate our metric computation against the paper's reported baselines.

### Command

```bash
python binarize.py compare \
    --image data/gt_input.png \
    --ground-truth data/gt_output.png \
    --output-dir results/baselines/
```

### Expected behavior

This command runs four methods:

1. **Simple thresholding** — sweeps all 256 threshold values, picks best F1.
   This is computationally intensive on a 7348x9913 image (256 full-image
   comparisons). May take several minutes.
2. **Otsu's method** — single automatic threshold. Fast.
3. **Adaptive Gaussian** — sweeps kernel sizes [31, 63, 127, 255] x
   C values [0..100]. That's 4 x 101 = 404 iterations on a ~73M pixel
   image. **This will be slow** (potentially 30-60 minutes).
4. **Adaptive Mean** — same parameter sweep. Equally slow.

### Performance warning

The adaptive thresholding sweeps (Tasks 3 & 4 in `traditional_binarization()`,
lines 753-798) iterate over 404 parameter combinations each, computing
`cv2.adaptiveThreshold` + `compute_metrics` on the full 73M-pixel image
each time. Total: ~808 iterations x ~73M pixels.

**Estimated time**: 30-90 minutes depending on CPU. This is a known
bottleneck in the script but is faithful to the paper's methodology
(finding the best hyperparameters for each baseline).

**Mitigation**: If too slow, the simple and Otsu baselines alone are
sufficient for a meaningful comparison. The adaptive methods can be run
overnight.

### Output files

```
results/baselines/
  simple_binarized.png
  otsu_binarized.png
  adaptive_gaussian_binarized.png
  adaptive_mean_binarized.png
```

### Pass criteria

- All four baselines produce output images.
- Otsu F1 should be notably lower than autoencoder F1 (the paper's
  central claim is that autoencoders beat traditional methods on
  degraded drawings).

---

## Task 2.3: Compare Results Against Table 1

### Paper's reported metrics (shallow models, Drawing #1 quadrant 2)

| Model | Patch | F1    | IoU   | PSNR  |
| ----- | ----- | ----- | ----- | ----- |
| 2     | 32    | 0.960 | 0.923 | 18.51 |
| 4     | 64    | 0.974 | 0.949 | 20.17 |
| 6     | 128   | 0.977 | 0.955 | 20.71 |
| 8     | 256   | 0.975 | 0.952 | 20.35 |

### Steps

1. Compile our metrics from Task 2.1 logs into a comparison table:

```
| Model | Paper F1 | Our F1 | Delta  | Paper IoU | Our IoU | Paper PSNR | Our PSNR |
| ----- | -------- | ------ | ------ | --------- | ------- | ---------- | -------- |
| 2     | 0.960    | ?      | ?      | 0.923     | ?       | 18.51      | ?        |
| 4     | 0.974    | ?      | ?      | 0.949     | ?       | 20.17      | ?        |
| 6     | 0.977    | ?      | ?      | 0.955     | ?       | 20.71      | ?        |
| 8     | 0.975    | ?      | ?      | 0.952     | ?       | 20.35      | ?        |
```

2. Analyze discrepancies:

   **If F1 is within +/- 0.02**: Successful replication. Differences are
   attributable to different train/val split boundaries, hardware-level
   floating-point differences, and/or TF version differences.

   **If F1 is within +/- 0.05**: Partial replication. The model is learning
   the right mapping but some factor differs. Investigate:
   - Was the stitching order correct? (Task 0.3 should have caught this.)
   - Are stride fractions correct? (Models 6 and 8 use inferred values.)
   - Is the training data genuinely equivalent to the authors' quadrant 3?

   **If F1 is < 0.90**: Something is fundamentally wrong. Debug:
   - Check if `gt_output.png` is truly binary or has intermediate values.
   - Verify the stitched Q3 matches the original TIFF region.
   - Try training on the pre-split data directly (skip recombination).

3. Save the comparison table to `results/replication_report.md`.

### Pass criteria

- At least 3 of 4 models achieve F1 within +/- 0.02 of the paper.
- Model 6 (128x128) achieves the highest F1 among our models (matching
  the paper's finding).

---

## Task 2.4: Visual Comparison of Binarized Outputs

### Why

Metrics alone don't tell the full story. Visual inspection reveals
qualitative differences: patch boundary artifacts, missed fine lines,
false positive stain remnants, etc.

### Steps

1. For each model's output, zoom into three types of regions:
   - **Dense line work** — verify thin lines are preserved, not broken.
   - **Stained areas** — verify stains are removed (white in output).
   - **Patch boundaries** — look for grid-like artifacts, especially
     in Model 2 (smallest 32x32 patches).

2. Compare Model 2 vs. Model 6 side-by-side on the same region to see
   how patch size affects visual quality.

3. Compare the best autoencoder output against Otsu baseline on a stained
   region — this should dramatically illustrate the paper's claim.

### Implementation

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load outputs
m6 = cv2.imread("results/model_6_binarized.png", cv2.IMREAD_GRAYSCALE)
otsu = cv2.imread("results/baselines/otsu_binarized.png", cv2.IMREAD_GRAYSCALE)
gt = cv2.imread("data/gt_output.png", cv2.IMREAD_GRAYSCALE)

# Pick a region (adjust coordinates to a stained area)
r = slice(2000, 2512), slice(3000, 3512)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt[r], cmap="gray"); axes[0].set_title("Ground Truth")
axes[1].imshow(m6[r], cmap="gray"); axes[1].set_title("Model 6 (Ours)")
axes[2].imshow(otsu[r], cmap="gray"); axes[2].set_title("Otsu Baseline")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig("results/visual_comparison.png", dpi=200)
```

### Pass criteria

- Autoencoder outputs visually remove stains while preserving lines.
- No severe patch boundary artifacts in any model.

---

## Task 2.5: Compile Final Replication Report

### Steps

Create `results/replication_report.md` containing:

1. **Setup**: Hardware, TF version, Python version, commit hash.
2. **Data preparation**: Stitching method, verification results.
3. **Training summary**: Table from Task 1.5.
4. **Evaluation results**: Comparison table from Task 2.3.
5. **Baseline comparison**: Classical method metrics from Task 2.2.
6. **Visual comparison**: Reference to `visual_comparison.png`.
7. **Discrepancies and analysis**: Why metrics differ from the paper.
8. **Limitations**: What was not replicated (fine-tuning, 25 other
   drawings, Arellano collection).

This document serves as the primary deliverable of the replication effort.

### Pass criteria

- Report is complete with all sections filled in.
- Conclusions are supported by the data.

---

## Checklist

- [ ] Task 2.1: Run inference with Models 2, 4, 6, 8 on `gt_input.png` with metrics
- [ ] Task 2.2: Run traditional baselines via `compare` command
- [ ] Task 2.3: Build comparison table against the paper's Table 1
- [ ] Task 2.4: Visual comparison of binarized outputs (autoencoder vs. baselines)
- [ ] Task 2.5: Compile final replication report in `results/replication_report.md`
