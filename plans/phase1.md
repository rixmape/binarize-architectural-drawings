# Phase 1 — Train Shallow Models (2, 4, 6, 8)

## Objective

Train all four shallow autoencoder architectures on the reconstructed
quadrant 3 of Drawing #1 and produce saved weights ready for Phase 2
evaluation. This is the core replication step for the paper's Table 1
(shallow model rows).

---

## Prerequisites

All Phase 0 tasks must be complete:

- `data/q3_dirty.png` (7349 x 9913) and `data/q3_clean.png` exist and
  have been visually verified.
- Smoke test passed for all four models.
- GPU availability confirmed (or CPU-only accepted).

---

## Context: What the Paper Reports for Shallow Models

From Table 1 (Section 5, tested on quadrant 2 of Drawing #1):

| Model | Patch | F1    | IoU   | PSNR (dB) |
| ----- | ----- | ----- | ----- | --------- |
| 2     | 32    | 0.960 | 0.923 | 18.51     |
| 4     | 64    | 0.974 | 0.949 | 20.17     |
| 6     | 128   | 0.977 | 0.955 | 20.71     |
| 8     | 256   | 0.975 | 0.952 | 20.35     |

These are our target metrics. Exact reproduction is unlikely due to:

- Different random seed effects on the 70/30 split (our split boundaries
  will differ from the authors' pre-split).
- Potential floating-point differences across hardware/TF versions.
- Inferred stride fractions for 128 and 256 (not author-confirmed).

A reasonable success criterion is F1 within +/- 0.02 of the paper's values.

---

## Hyperparameters (Fixed)

All values are from the paper (Section 3.2) and author email (Feb 7 2026):

| Parameter      | Value | Source                               |
| -------------- | ----- | ------------------------------------ |
| Loss           | MSE   | Author email (confirmed, not BCE)    |
| Optimizer      | Adam  | Paper Section 3.2                    |
| Learning rate  | 0.001 | Paper Section 3.2 (Adam default)     |
| Batch size     | 32    | Author email                         |
| Epochs         | 100   | Paper Section 3.2                    |
| Early stopping | 15    | Script default (not stated in paper) |
| Val split      | 0.30  | Paper Section 3.2                    |
| Threshold      | 0.5   | Standard sigmoid decision boundary   |
| Seed           | 42    | Script default (reproducibility)     |

### Stride fractions (training only)

| Patch size | Stride fraction | Overlap | Source                                |
| ---------- | --------------- | ------- | ------------------------------------- |
| 32         | 1.00            | 0%      | Author: "does not need overlap"       |
| 64         | 0.75            | 25%     | Author: "stride = 0.75 of input size" |
| 128        | 0.50            | 50%     | Inferred from author's pattern        |
| 256        | 0.50            | 50%     | Inferred from author's pattern        |

---

## Task 1.1: Train Model 2 (Shallow, 32x32)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 2 \
    --output-dir runs/model_2 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Stride = 32 (1.00 x 32), no overlap.
- Patch extraction from 7349 x 9913 image should yield ~70,000+ patches.
- 70/30 split: ~49,000 train / ~21,000 validation patches.
- Model has 1,217 parameters.
- Architecture: Input(32,32,1) -> Conv2D(64,k=3,s=2) -> ConvT(1,k=3,s=2,sigmoid)
- Training should complete in minutes on GPU.
- Early stopping may trigger before epoch 100.

### Output files

```
runs/model_2/
  model_2_best.weights.h5     # Best weights (min val_loss)
  model_2_final.weights.h5    # Final epoch weights
  model_2_loss_curve.png      # Train/val loss plot
```

### Monitoring

- Watch for val_loss convergence. If val_loss plateaus early (< 20 epochs),
  the model has learned the mapping quickly — expected for shallow models.
- If val_loss diverges from train_loss, this could indicate the split created
  a distribution mismatch. Note the gap for comparison with other models.

---

## Task 1.2: Train Model 4 (Shallow, 64x64)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 4 \
    --output-dir runs/model_4 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Stride = 48 (0.75 x 64), 25% overlap — **author-confirmed**.
- Fewer but overlapping patches (~20,000-30,000 estimated).
- 70/30 split applied to extracted patches.
- Same 1,217 parameters as Model 2.
- This is the model where stride is most confidently correct.

---

## Task 1.3: Train Model 6 (Shallow, 128x128)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 6 \
    --output-dir runs/model_6 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Stride = 64 (0.50 x 128), 50% overlap — **inferred, not confirmed**.
- Fewer patches than Model 4 due to larger patch size.
- This is the paper's best-performing model (F1 = 0.977).
- If our F1 deviates significantly, the stride fraction is the most
  likely cause and should be revisited.

---

## Task 1.4: Train Model 8 (Shallow, 256x256)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 8 \
    --output-dir runs/model_8 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Stride = 128 (0.50 x 256), 50% overlap — **inferred, not confirmed**.
- Fewest patches of all configurations.
- Largest receptive field per patch — may capture more global context
  but risks overfitting with fewer training samples.
- Patch extraction and training will be fastest due to small dataset.

---

## Task 1.5: Record Training Metadata

### Why

For reproducibility and comparison with the paper, log key facts from
each training run into a summary table.

### Steps

After all four models are trained, create `runs/training_summary.md` with:

| Model | Patches | Train | Val | Best epoch | Final train loss | Final val loss | Early stopped? |
| ----- | ------- | ----- | --- | ---------- | ---------------- | -------------- | -------------- |
| 2     | ?       | ?     | ?   | ?          | ?                | ?              | ?              |
| 4     | ?       | ?     | ?   | ?          | ?                | ?              | ?              |
| 6     | ?       | ?     | ?   | ?          | ?                | ?              | ?              |
| 8     | ?       | ?     | ?   | ?          | ?                | ?              | ?              |

Fill in from the training logs. This table will be compared against Phase 2
evaluation metrics to identify any anomalies.

### Pass criteria

- All four models produced best weights with val_loss < 0.01 (expected
  for a near-binary reconstruction task).
- No model diverged (val_loss increasing monotonically).

---

## Task 1.6: Inspect Loss Curves

### Why

The loss curve plots reveal training dynamics — convergence speed,
overfitting tendencies, and whether early stopping fired appropriately.

### Steps

1. Open each `model_*_loss_curve.png` and look for:
   - **Convergence** — both train and val loss should decrease and plateau.
   - **No divergence** — val_loss should not increase while train_loss
     decreases (would indicate overfitting).
   - **Early stopping** — if triggered, verify it fired at a reasonable
     epoch (not too early, not after clear overfitting).

2. Compare across models:
   - Larger patch sizes may converge slower (fewer patches, more complex
     spatial patterns to learn).
   - Model 2 (32x32) may converge fastest due to most training patches
     and simplest spatial structure.

### Pass criteria

- All four loss curves show healthy convergence.
- No model shows severe overfitting (val_loss >> train_loss).

---

## Risk Factors and Mitigations

### Risk 1: Memory pressure from large patch counts

Model 2 (32x32, stride=32) may produce 70,000+ patches, each 32x32x1
float32 = 4 KB. Total: ~280 MB. The `train_test_split` call creates
copies, temporarily doubling memory. If RAM is tight:

- **Mitigation**: Train Model 2 last, or reduce to `--stride-fraction 0.5`
  (acknowledged deviation from paper).

### Risk 2: Image loading time

The stitched `q3_dirty.png` is ~120 MB. `cv2.imread` on large PNGs can be
slow (10-30 seconds). This is a one-time cost per training run.

- **Mitigation**: None needed — just wait.

### Risk 3: TensorFlow/Keras API changes

The script uses `tensorflow.keras` (TF's bundled Keras). TF 2.16+ ships
Keras 3 by default, which changed some APIs (e.g., `.weights.h5` saving).

- **Mitigation**: Check `pip show tensorflow keras` versions before training.
  If Keras 3 issues arise, pin `tf-keras` or adjust save format.

---

## Checklist

- [ ] Task 1.1: Train Model 2 (shallow, 32x32, stride=1.00)
- [ ] Task 1.2: Train Model 4 (shallow, 64x64, stride=0.75)
- [ ] Task 1.3: Train Model 6 (shallow, 128x128, stride=0.50)
- [ ] Task 1.4: Train Model 8 (shallow, 256x256, stride=0.50)
- [ ] Task 1.5: Record training metadata in `runs/training_summary.md`
- [ ] Task 1.6: Inspect all four loss curves for healthy convergence
