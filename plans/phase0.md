# Phase 0 — Pre-flight Checks

## Objective

Verify that the provided data is usable by the existing script and prepare
the combined training image that the script's `train` command expects.
Phase 0 produces no trained models — only validated, pipeline-ready inputs.

---

## Context: Why This Phase Exists

The authors (Dr. Mark Jeremy Narag, email2.txt, Feb 20 2026) shared data that
is **pre-split** into three pairs:

| File pair                     | Dimensions (H x W) | Represents                           |
| ----------------------------- | ------------------ | ------------------------------------ |
| `train_input/output.png`      | 7349 x 7000        | ~70% of Drawing #1, quadrant 3       |
| `validation_input/output.png` | 7349 x 2913        | ~30% of Drawing #1, quadrant 3       |
| `gt_input/output.png`         | 7348 x 9913        | Drawing #1, quadrant 2 (unseen test) |

But `binarize.py` line 808-838 (`_cli_train`) expects a **single** dirty/clean
image pair. It extracts patches itself and performs its own 70/30 split at line
476-481 via `train_test_split()`. Feeding the pre-split `train_input.png`
directly would cause a **double split** — 70% of the already-70% portion =
~49% of quadrant 3 used for actual training, violating the paper's protocol.

The agreed solution: **horizontally concatenate** train + validation back into
the full quadrant 3, then let the script split from scratch.

Evidence the concat axis is horizontal:

- Train width (7000) + validation width (2913) = 9913 = gt_input width.
- Both have identical height (7349).
- gt_input is quadrant 2 with width 9913 — matching the reconstructed Q3.

---

## Task 0.1: Verify cv2.IMREAD_GRAYSCALE Handles RGBA Inputs

### Why

The input images (`train_input.png`, `validation_input.png`, `gt_input.png`)
are 8-bit RGBA (4-channel), while the output images are grayscale. The
script's `load_grayscale()` at line 108 uses `cv2.IMREAD_GRAYSCALE`:

```python
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

OpenCV's documentation states this flag converts any image to single-channel
grayscale on load, including RGBA. But we need to confirm there are no
edge-case issues (e.g., alpha channel influencing the conversion).

### Steps

1. Run a quick Python snippet that loads each RGBA input with
   `cv2.IMREAD_GRAYSCALE` and verifies:
   - The returned array is 2-D (H, W), not 3-D (H, W, C).
   - dtype is uint8.
   - Values are in a reasonable range (not all zeros, not all 255).
   - No warnings or errors printed.

2. Cross-check by loading the same image with `cv2.IMREAD_UNCHANGED`, manually
   discarding the alpha channel, converting BGR to gray, and comparing pixel
   values to the `IMREAD_GRAYSCALE` result. They should match within rounding
   tolerance (max absolute difference <= 1).

### Pass criteria

- All three RGBA inputs produce valid 2-D uint8 arrays.
- The grayscale conversion matches manual BGR-to-gray conversion.

---

## Task 0.2: Stitch Train + Validation Into Full Quadrant 3

### Why

The script needs a single dirty image and a single clean image representing
the full quadrant 3 of Drawing #1. We must reconstruct these from the
pre-split pieces.

### Steps

1. Load `train_input.png` and `validation_input.png` as grayscale float32.
2. Horizontally concatenate: `np.hstack([train, validation])`.
3. Verify the resulting shape is (7349, 9913) — matching gt_input's width.
4. Repeat for `train_output.png` + `validation_output.png`.
5. Save the stitched results as:
   - `data/q3_dirty.png` (stitched input)
   - `data/q3_clean.png` (stitched output)

### Stitching order ambiguity

We do not know whether train is the LEFT or RIGHT portion. The default
assumption is **train = left, validation = right** (natural reading order,
and 70% is the larger portion on the left). We verify this visually in
Task 0.3. If the seam looks wrong, swap the order.

### Implementation

```python
import cv2
import numpy as np

train_in = cv2.imread("data/train_input.png", cv2.IMREAD_GRAYSCALE)
val_in   = cv2.imread("data/validation_input.png", cv2.IMREAD_GRAYSCALE)
q3_dirty = np.hstack([train_in, val_in])
assert q3_dirty.shape == (7349, 9913), f"Unexpected shape: {q3_dirty.shape}"
cv2.imwrite("data/q3_dirty.png", q3_dirty)

train_out = cv2.imread("data/train_output.png", cv2.IMREAD_GRAYSCALE)
val_out   = cv2.imread("data/validation_output.png", cv2.IMREAD_GRAYSCALE)
q3_clean  = np.hstack([train_out, val_out])
assert q3_clean.shape == (7349, 9913), f"Unexpected shape: {q3_clean.shape}"
cv2.imwrite("data/q3_clean.png", q3_clean)
```

### Pass criteria

- Both stitched images have shape (7349, 9913).
- File sizes are reasonable (dirty should be ~120 MB, clean ~2 MB since
  binary images compress well).

---

## Task 0.3: Visual Inspection of the Stitch

### Why

A bad stitch (wrong order, misaligned seam) would silently corrupt training.
We need to confirm the boundary at column 7000 is seamless.

### Steps

1. Open `data/q3_dirty.png` in an image viewer or generate a matplotlib
   figure zoomed into the seam region (columns 6990-7010).
2. Look for:
   - **Continuity** — architectural lines should flow across the boundary.
   - **No abrupt intensity shift** — the left and right halves should have
     similar background tone at the join.
   - **No duplication or gap** — no repeated or missing column of pixels.
3. If the seam is wrong, swap the concatenation order to
   `np.hstack([val, train])` and re-check.
4. Repeat for `data/q3_clean.png`.

### Implementation (automated seam check)

```python
import numpy as np
import cv2

img = cv2.imread("data/q3_dirty.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
col_left  = img[:, 6999]   # Last column of train portion
col_right = img[:, 7000]   # First column of validation portion
diff = np.abs(col_left - col_right)
print(f"Seam diff — mean: {diff.mean():.2f}, max: {diff.max():.2f}, "
      f"std: {diff.std():.2f}")
# Compare to interior column differences as baseline
interior_diff = np.abs(img[:, 3499].astype(float) - img[:, 3500].astype(float))
print(f"Interior diff — mean: {interior_diff.mean():.2f}, "
      f"max: {interior_diff.max():.2f}")
```

If the seam diff is dramatically higher than interior diff, the order is
likely wrong.

### Pass criteria

- Seam column-pair difference is within the same order of magnitude as an
  arbitrary interior column-pair difference.
- Visual inspection confirms continuity of architectural features.

---

## Task 0.4: Verify Dimension Compatibility With All 4 Patch Sizes

### Why

The stitched quadrant 3 (7349 x 9913) and the test image (7348 x 9913) are
**not** multiples of any patch size (32, 64, 128, 256). The script handles
this via `pad_to_multiple()` at line 224-262 using reflect padding. But
during training, `extract_patches()` at line 169 discards incomplete patches
at the edges (no padding). We need to confirm how many patches each
configuration yields and whether any important data is lost.

### Steps

For the stitched Q3 (7349 x 9913), compute for each patch size:

| Patch | Stride frac | Stride (px) | Rows             | Cols             | Total patches |
| ----- | ----------- | ----------- | ---------------- | ---------------- | ------------- |
| 32    | 1.00        | 32          | (7349-32)/32+1   | (9913-32)/32+1   | ?             |
| 64    | 0.75        | 48          | (7349-64)/48+1   | (9913-64)/48+1   | ?             |
| 128   | 0.50        | 64          | (7349-128)/64+1  | (9913-128)/64+1  | ?             |
| 256   | 0.50        | 128         | (7349-256)/128+1 | (9913-256)/128+1 | ?             |

Verify that the total counts are reasonable (thousands for 32x32, hundreds
for 256x256) and that edge-pixel loss is minimal.

### Pass criteria

- All four configurations produce > 0 patches.
- Edge pixel loss (pixels never covered by any patch) is < 1% of total area.

---

## Task 0.5: Dry-Run the Script's Train Command (Smoke Test)

### Why

Before committing to a full training run, verify the script can load the
stitched data, extract patches, build each shallow model, and reach the
`model.fit()` call without errors. This catches import issues, shape
mismatches, or TensorFlow/Keras version incompatibilities.

### Steps

1. Run each shallow model with `--epochs 1` on the stitched data:

```bash
for mid in 2 4 6 8; do
    python binarize.py train \
        --dirty data/q3_dirty.png \
        --clean data/q3_clean.png \
        --model-id $mid \
        --output-dir runs/smoke_test/model_${mid} \
        --epochs 1
done
```

2. Verify for each model:
   - No Python errors or TensorFlow exceptions.
   - Weights files are created (`model_*_best.weights.h5`).
   - Loss curve PNG is created.
   - Logged patch count matches Task 0.4 expectations.
   - The 70/30 split is logged (check "Split: X training / Y validation").

3. Clean up smoke test outputs: `rm -rf runs/smoke_test/`

### Pass criteria

- All four shallow models complete 1 epoch without error.
- Logged split ratios are approximately 70/30.

---

## Task 0.6: Confirm GPU Availability

### Why

Training 4 shallow models (1,217 params) on GPU will take minutes; on CPU it
will still be feasible but slower. We should confirm TensorFlow detects the
GPU before committing to Phase 1.

### Steps

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output: a non-empty list containing at least one GPU device.

If no GPU is detected:

- Check `nvidia-smi` output.
- Verify `tensorflow[and target]` version matches CUDA version.
- If GPU setup would take significant time, note it and proceed with CPU
  (shallow models are small enough).

### Pass criteria

- TensorFlow reports at least one GPU, **or** CPU-only is accepted and noted.

---

## Checklist

- [ ] Task 0.1: Verify `cv2.IMREAD_GRAYSCALE` handles RGBA inputs correctly
- [ ] Task 0.2: Stitch train + validation into `data/q3_dirty.png` and `data/q3_clean.png`
- [ ] Task 0.3: Visually inspect the stitch seam for continuity
- [ ] Task 0.4: Compute patch counts for all 4 patch sizes on stitched Q3
- [ ] Task 0.5: Dry-run `train` with `--epochs 1` for Models 2, 4, 6, 8
- [ ] Task 0.6: Confirm GPU availability for TensorFlow
