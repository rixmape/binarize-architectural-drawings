# Phase 3 — Train Deep Models (1, 3, 5, 7)

## Objective

Train all four deep autoencoder architectures to complete the full Table 1
comparison. This phase validates the paper's central finding: shallow models
(1,217 params) outperform deep models (148K–3.1M params) on this task.

---

## Prerequisites

Phase 2 must be complete:

- Shallow model metrics are compiled and compared against the paper.
- No fundamental issues found (F1 > 0.90 for shallow models).
- `data/q3_dirty.png` and `data/q3_clean.png` remain available.

---

## Context: What the Paper Reports for Deep Models

From Table 1 (Section 5, tested on quadrant 2 of Drawing #1):

| Model | Patch | Params    | F1    | IoU   | PSNR  |
| ----- | ----- | --------- | ----- | ----- | ----- |
| 1     | 32    | 148,865   | 0.957 | 0.918 | 18.14 |
| 3     | 64    | 739,073   | 0.968 | 0.938 | 19.38 |
| 5     | 128   | 739,073   | 0.968 | 0.939 | 19.41 |
| 7     | 256   | 3,099,137 | 0.966 | 0.935 | 19.07 |

Key observation: **every deep model underperforms its shallow counterpart**
on all three metrics. This is the paper's main contribution — demonstrating
that task-specific shallow architectures can beat deep ones when the
problem structure is simple enough.

---

## Deep Model Architectures (from binarize.py lines 349-401)

### Model 1 (32x32 deep, 148,865 params)

```
Input(32,32,1) -> Conv(64,s=2) -> Conv(128,s=2) ->
Conv(128,s=2) -> ConvT(64,s=2) -> ConvT(1,s=2,sigmoid)
Bottleneck: 8x8x128
```

### Model 3 (64x64 deep, 739,073 params)

```
Input(64,64,1) -> Conv(64,s=2) -> Conv(128,s=2) -> Conv(256,s=2) ->
ConvT(128,s=2) -> ConvT(64,s=2) -> ConvT(1,s=2,sigmoid)
Bottleneck: 8x8x256
```

### Model 5 (128x128 deep, 739,073 params)

```
Input(128,128,1) -> Conv(64,s=2) -> Conv(128,s=2) -> Conv(256,s=2) ->
ConvT(128,s=2) -> ConvT(64,s=2) -> ConvT(1,s=2,sigmoid)
Bottleneck: 16x16x256
```

### Model 7 (256x256 deep, 3,099,137 params)

```
Input(256,256,1) -> Conv(64,s=2) -> Conv(128,s=2) -> Conv(256,s=2) -> Conv(512,s=2) ->
ConvT(256,s=2) -> ConvT(128,s=2) -> ConvT(64,s=2) -> ConvT(1,s=2,sigmoid)
Bottleneck: 16x16x512
```

---

## Task 3.1: Train Model 1 (Deep, 32x32)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 1 \
    --output-dir runs/model_1 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Same patch count as Model 2 (~70,000+ patches at stride=32).
- 148,865 params — 122x more than shallow counterpart.
- **Longer training time**: More params means slower per-epoch compute.
  On GPU, expect 5-15 minutes total (vs. ~2 min for Model 2).
- May overfit more easily — watch val_loss divergence from train_loss.

### Output files

```
runs/model_1/
  model_1_best.weights.h5
  model_1_final.weights.h5
  model_1_loss_curve.png
```

---

## Task 3.2: Train Model 3 (Deep, 64x64)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 3 \
    --output-dir runs/model_3 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Same patch count as Model 4 (stride=48, ~20,000-30,000 patches).
- 739,073 params — 607x more than shallow counterpart.
- Three encoder + three decoder layers vs. one each in Model 4.
- Training time: 10-30 minutes on GPU.

---

## Task 3.3: Train Model 5 (Deep, 128x128)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 5 \
    --output-dir runs/model_5 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Same patch count as Model 6 (stride=64, 50% overlap).
- 739,073 params — same as Model 3 despite larger input (deeper bottleneck
  at 16x16x256 instead of 8x8x256, but same layer structure).
- Training time: 15-30 minutes on GPU.

---

## Task 3.4: Train Model 7 (Deep, 256x256)

### Command

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 7 \
    --output-dir runs/model_7 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Expected behavior

- Same patch count as Model 8 (stride=128, 50% overlap).
- **3,099,137 params** — the largest model, 2,546x more than shallow.
- Four encoder + four decoder layers.
- **Longest training time**: 30-60+ minutes on GPU.
- **Highest memory usage**: May need to reduce batch_size to 16 if OOM.
  Each 256x256 patch is 256 KB float32 — a batch of 32 is ~8 MB, but
  internal activations at 128x128x64 etc. multiply memory.

### Memory mitigation

If TensorFlow raises an OOM error:

```bash
python binarize.py train \
    --dirty data/q3_dirty.png \
    --clean data/q3_clean.png \
    --model-id 7 \
    --output-dir runs/model_7 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001
```

Note: changing batch size from 32 to 16 is a known deviation from the
paper's protocol. Document this if it occurs.

---

## Task 3.5: Evaluate Deep Models on Test Set

### Commands

```bash
for mid in 1 3 5 7; do
    python binarize.py infer \
        --image data/gt_input.png \
        --weights runs/model_${mid}/model_${mid}_best.weights.h5 \
        --model-id $mid \
        --output results/model_${mid}_binarized.png \
        --ground-truth data/gt_output.png
done
```

### Expected results

Deep models should score **lower** than their shallow counterparts:

| Model | Type | Expected F1 range |
| ----- | ---- | ----------------- |
| 1     | Deep | 0.94 - 0.97       |
| 3     | Deep | 0.95 - 0.98       |
| 5     | Deep | 0.95 - 0.98       |
| 7     | Deep | 0.95 - 0.98       |

If any deep model outperforms its shallow counterpart, this would be a
notable deviation from the paper's findings worth investigating.

---

## Task 3.6: Complete the Full Table 1 Comparison

### Steps

Combine Phase 2 (shallow) and Phase 3 (deep) metrics into the full table:

```
| Model | Type    | Patch | Params    | Paper F1 | Our F1 | Paper IoU | Our IoU | Paper PSNR | Our PSNR |
| ----- | ------- | ----- | --------- | -------- | ------ | --------- | ------- | ---------- | -------- |
| 1     | Deep    | 32    | 148,865   | 0.957    | ?      | 0.918     | ?       | 18.14      | ?        |
| 2     | Shallow | 32    | 1,217     | 0.960    | ?      | 0.923     | ?       | 18.51      | ?        |
| 3     | Deep    | 64    | 739,073   | 0.968    | ?      | 0.938     | ?       | 19.38      | ?        |
| 4     | Shallow | 64    | 1,217     | 0.974    | ?      | 0.949     | ?       | 20.17      | ?        |
| 5     | Deep    | 128   | 739,073   | 0.968    | ?      | 0.939     | ?       | 19.41      | ?        |
| 6     | Shallow | 128   | 1,217     | 0.977    | ?      | 0.955     | ?       | 20.71      | ?        |
| 7     | Deep    | 256   | 3,099,137 | 0.966    | ?      | 0.935     | ?       | 19.07      | ?        |
| 8     | Shallow | 256   | 1,217     | 0.975    | ?      | 0.952     | ?       | 20.35      | ?        |
```

### Analysis points

1. **Shallow vs. Deep delta**: For each patch size, compute
   `shallow_F1 - deep_F1`. Paper shows +0.003 to +0.009 advantage for
   shallow. Do we reproduce this ranking?

2. **Best model**: Does Model 6 (128x128 shallow) still come out on top?

3. **Parameter efficiency**: Shallow models achieve higher F1 with
   1,217 params vs. 148K–3.1M. This is the paper's headline result.

4. **Patch size trend**: F1 should generally increase from 32 to 128,
   then slightly decrease at 256, for both shallow and deep families.

### Pass criteria

- All 8 rows are filled in with valid metrics.
- For at least 3 of 4 patch sizes, shallow F1 > deep F1 (matching paper).
- The overall best model is either Model 6 or Model 8 (128 or 256 shallow).

---

## Task 3.7: Update Replication Report

### Steps

Update `results/replication_report.md` (from Phase 2, Task 2.5) to include:

1. Full 8-model Table 1 comparison.
2. Deep model training metadata (epochs, loss, early stopping).
3. Deep model loss curves analysis.
4. Final conclusion: do we confirm the paper's finding that shallow > deep?
5. Updated limitations section noting any batch size changes or anomalies.

### Pass criteria

- Report contains complete data for all 8 models.
- Conclusion directly addresses the paper's central claim.

---

## Risk Factors and Mitigations

### Risk 1: GPU OOM on Model 7 (3.1M params, 256x256 patches)

Model 7 has 4 encoder and 4 decoder layers with up to 512 filters. At
batch_size=32 with 256x256 patches, intermediate activations can consume
several GB of VRAM.

- **Mitigation**: Reduce batch_size to 16 or 8. Document the change.

### Risk 2: Deep models take much longer to train

With 100 epochs, patience=15 early stopping, and 600-3000x more params,
expect training times 10-60x longer than shallow models.

- **Mitigation**: Train models in parallel if multiple GPUs available.
  Otherwise, run overnight.

### Risk 3: Deep models may behave differently with our train/val split

Since we recombined and re-split the data (rather than using the authors'
exact split), the deep models might be more sensitive to the specific
split than shallow models due to higher capacity and overfitting risk.

- **Mitigation**: Compare val_loss dynamics between deep and shallow models
  of the same patch size. Large divergence in training behavior suggests
  split sensitivity.

---

## Checklist

- [ ] Task 3.1: Train Model 1 (deep, 32x32, 148K params)
- [ ] Task 3.2: Train Model 3 (deep, 64x64, 739K params)
- [ ] Task 3.3: Train Model 5 (deep, 128x128, 739K params)
- [ ] Task 3.4: Train Model 7 (deep, 256x256, 3.1M params)
- [ ] Task 3.5: Evaluate deep Models 1, 3, 5, 7 on `gt_input.png` with metrics
- [ ] Task 3.6: Complete the full 8-model Table 1 comparison
- [ ] Task 3.7: Update replication report with deep model results and final conclusion
