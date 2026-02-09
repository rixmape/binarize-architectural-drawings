"""Binarize historical architectural drawings with shallow convolutional autoencoders."""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Section 1 — Configuration
# ===================================================================

# Default stride fractions per patch size (training only).
# Derived from author email: stride = 0.75 of patch size for 64×64.
# Extrapolated for other sizes following the stated pattern.
DEFAULT_STRIDE_FRACTIONS: dict[int, float] = {
    32: 1.00,  # No overlap — sufficient patches without it
    64: 0.75,  # 25% overlap — confirmed by author
    128: 0.50,  # 50% overlap — inferred for larger inputs
    256: 0.50,  # 50% overlap — inferred for largest inputs
}

# Model ID → (patch_size, is_deep)
MODEL_REGISTRY: dict[int, tuple[int, bool]] = {
    1: (32, True),
    2: (32, False),
    3: (64, True),
    4: (64, False),
    5: (128, True),
    6: (128, False),
    7: (256, True),
    8: (256, False),
}


@dataclass
class TrainConfig:
    """Training hyperparameters matching the paper and author clarifications."""

    loss: str = "mse"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split_ratio: float = 0.30
    binarization_threshold: float = 0.5


# ===================================================================
# Section 2 — Image I/O and Preprocessing
# ===================================================================


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as single-channel grayscale with float32 in [0, 1].

    Parameters
    ----------
    path : str or Path
        Filesystem path to any raster image (TIFF, PNG, JPEG, …).

    Returns
    -------
    np.ndarray
        2-D array of shape (H, W) with dtype float32 in [0.0, 1.0].

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV failed to decode: {path}")
    return img.astype(np.float32) / 255.0


def save_binary_image(
    image: np.ndarray,
    path: str | Path,
) -> None:
    """Save a binary float image ([0, 1]) as an 8-bit PNG or TIFF.

    Parameters
    ----------
    image : np.ndarray
        2-D array with values in {0.0, 1.0}.
    path : str or Path
        Destination file path.  Format inferred from extension.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = (image * 255).astype(np.uint8)
    success = cv2.imwrite(path, out)
    if not success:
        raise IOError(f"Failed to write image: {path}")
    logger.info("Saved binary image → %s", path)


# ===================================================================
# Section 3 — Patch Extraction
# ===================================================================


def extract_patches(
    image: np.ndarray,
    patch_size: int,
    stride: int | None = None,
) -> np.ndarray:
    """Extract square patches from a 2-D grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D array of shape (H, W), dtype float32.
    patch_size : int
        Side length of each square patch.
    stride : int or None
        Step size between consecutive patches.  Defaults to *patch_size*
        (non-overlapping).

    Returns
    -------
    np.ndarray
        4-D array of shape (N, patch_size, patch_size, 1).
    """
    if stride is None:
        stride = patch_size

    h, w = image.shape[:2]
    patches: list[np.ndarray] = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    arr = np.array(patches, dtype=np.float32)
    # Add channel dimension → (N, H, W, 1)
    return arr[..., np.newaxis]


def extract_training_patches(
    dirty: np.ndarray,
    clean: np.ndarray,
    patch_size: int,
    stride_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired dirty/clean patches for supervised training.

    Parameters
    ----------
    dirty : np.ndarray
        Degraded image, shape (H, W), float32 in [0, 1].
    clean : np.ndarray
        Manually binarized counterpart, same shape as *dirty*.
    patch_size : int
        Side length of each square patch (32, 64, 128, or 256).
    stride_fraction : float or None
        Fraction of *patch_size* used as stride.  If ``None``, uses the
        default from ``DEFAULT_STRIDE_FRACTIONS``.

    Returns
    -------
    x_patches, y_patches : tuple of np.ndarray
        Both of shape (N, patch_size, patch_size, 1).
    """
    if dirty.shape != clean.shape:
        raise ValueError(f"Shape mismatch: dirty {dirty.shape} vs. clean {clean.shape}")
    if stride_fraction is None:
        stride_fraction = DEFAULT_STRIDE_FRACTIONS.get(patch_size, 0.75)

    stride = max(1, int(patch_size * stride_fraction))
    logger.info(
        "Extracting training patches: size=%d, stride=%d (%.0f%% overlap)",
        patch_size,
        stride,
        (1 - stride / patch_size) * 100,
    )

    x_patches = extract_patches(dirty, patch_size, stride)
    y_patches = extract_patches(clean, patch_size, stride)

    logger.info("Extracted %d paired patches.", len(x_patches))
    return x_patches, y_patches


def pad_to_multiple(
    image: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Pad an image so both dimensions are exact multiples of *patch_size*.

    Uses reflect padding to minimise edge artifacts.

    Parameters
    ----------
    image : np.ndarray
        2-D array of shape (H, W).
    patch_size : int
        The divisor to align to.

    Returns
    -------
    np.ndarray
        Padded image with shape (H', W') where H' and W' are multiples
        of *patch_size*.
    """
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return image
    padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    logger.info(
        "Padded image from (%d, %d) to (%d, %d).",
        h,
        w,
        padded.shape[0],
        padded.shape[1],
    )
    return padded


# ===================================================================
# Section 4 — Model Construction
# ===================================================================


def _conv_block(
    x: tf.Tensor,
    filters: int,
) -> tf.Tensor:
    """Single encoder block: Conv2D(stride=2) + ReLU."""
    return layers.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu",
    )(x)


def _deconv_block(
    x: tf.Tensor,
    filters: int,
    activation: str = "relu",
) -> tf.Tensor:
    """Single decoder block: Conv2DTranspose(stride=2) + activation."""
    return layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=activation,
    )(x)


def build_model(model_id: int) -> keras.Model:
    """Construct one of the eight autoencoder architectures from Figure 3.

    Parameters
    ----------
    model_id : int
        Integer 1–8 identifying the architecture.
        Odd = deep, even = shallow.

    Returns
    -------
    keras.Model
        Uncompiled Keras model.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"model_id must be 1–8, got {model_id}. " f"Valid IDs: {sorted(MODEL_REGISTRY)}")

    patch_size, is_deep = MODEL_REGISTRY[model_id]
    inp = layers.Input(shape=(patch_size, patch_size, 1))

    if is_deep:
        x = _build_deep_encoder_decoder(inp, model_id)
    else:
        x = _build_shallow_encoder_decoder(inp)

    model = keras.Model(inputs=inp, outputs=x, name=f"model_{model_id}")
    logger.info(
        "Built Model %d (%s, %dx%d): %s params",
        model_id,
        "deep" if is_deep else "shallow",
        patch_size,
        patch_size,
        f"{model.count_params():,}",
    )
    return model


def _build_shallow_encoder_decoder(inp: tf.Tensor) -> tf.Tensor:
    """Shallow autoencoder: one hidden (bottleneck) layer.

    Architecture (all sizes):
        Input → Conv(64, s=2) → ConvT(1, s=2, sigmoid) → Output
    """
    # Encoder — single convolution halves spatial dims
    x = _conv_block(inp, filters=64)
    # Decoder — single transposed convolution restores dims
    x = _deconv_block(x, filters=1, activation="sigmoid")
    return x


def _build_deep_encoder_decoder(
    inp: tf.Tensor,
    model_id: int,
) -> tf.Tensor:
    """Deep autoencoder: multiple hidden layers per Figure 3.

    The filter counts and depth vary by model_id.
    """
    # ----- Model 1: 32×32 deep -----
    # 32→16×64→8×128 | 8×128→16×64→32×1
    if model_id == 1:
        x = _conv_block(inp, 64)  # 16×16×64
        x = _conv_block(x, 128)  # 8×8×128
        x = _deconv_block(x, 64)  # 16×16×64
        x = _deconv_block(x, 1, "sigmoid")  # 32×32×1

    # ----- Model 3: 64×64 deep -----
    # 64→32×64→16×128→8×256 | 8→16×128→32×64→64×1
    elif model_id == 3:
        x = _conv_block(inp, 64)  # 32×32×64
        x = _conv_block(x, 128)  # 16×16×128
        x = _conv_block(x, 256)  # 8×8×256
        x = _deconv_block(x, 128)  # 16×16×128
        x = _deconv_block(x, 64)  # 32×32×64
        x = _deconv_block(x, 1, "sigmoid")  # 64×64×1

    # ----- Model 5: 128×128 deep -----
    # 128→64×64→32×128→16×256 | 16→32×128→64×64→128×1
    elif model_id == 5:
        x = _conv_block(inp, 64)  # 64×64×64
        x = _conv_block(x, 128)  # 32×32×128
        x = _conv_block(x, 256)  # 16×16×256
        x = _deconv_block(x, 128)  # 32×32×128
        x = _deconv_block(x, 64)  # 64×64×64
        x = _deconv_block(x, 1, "sigmoid")  # 128×128×1

    # ----- Model 7: 256×256 deep -----
    # 256→128×64→64×128→32×256→16×512
    # | 16→32×256→64×128→128×64→256×1
    elif model_id == 7:
        x = _conv_block(inp, 64)  # 128×128×64
        x = _conv_block(x, 128)  # 64×64×128
        x = _conv_block(x, 256)  # 32×32×256
        x = _conv_block(x, 512)  # 16×16×512
        x = _deconv_block(x, 256)  # 32×32×256
        x = _deconv_block(x, 128)  # 64×64×128
        x = _deconv_block(x, 64)  # 128×128×64
        x = _deconv_block(x, 1, "sigmoid")  # 256×256×1

    else:
        raise ValueError(f"No deep architecture for model_id={model_id}")

    return x


# ===================================================================
# Section 5 — Training
# ===================================================================


def compile_model(
    model: keras.Model,
    config: TrainConfig | None = None,
) -> keras.Model:
    """Compile a model with the paper's specified optimizer and loss.

    Parameters
    ----------
    model : keras.Model
        An uncompiled autoencoder from ``build_model``.
    config : TrainConfig or None
        Hyperparameters.  Uses paper defaults when ``None``.

    Returns
    -------
    keras.Model
        The same model, now compiled.
    """
    if config is None:
        config = TrainConfig()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=config.loss,
    )
    logger.info(
        "Compiled with optimizer=%s (lr=%s), loss=%s",
        config.optimizer,
        config.learning_rate,
        config.loss,
    )
    return model


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig | None = None,
    output_dir: str | Path = "runs",
) -> keras.callbacks.History:
    """Train the autoencoder and save weights + history plot.

    Parameters
    ----------
    model : keras.Model
        A compiled autoencoder.
    x_train : np.ndarray
        Dirty patches, shape (N, H, W, 1).
    y_train : np.ndarray
        Clean patches, same shape.
    config : TrainConfig or None
        Hyperparameters.
    output_dir : str or Path
        Directory where weights and plots are saved.

    Returns
    -------
    keras.callbacks.History
        Training history object.
    """
    if config is None:
        config = TrainConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 70/30 train-validation split
    x_trn, x_val, y_trn, y_val = train_test_split(
        x_train,
        y_train,
        test_size=config.validation_split_ratio,
        random_state=SEED,
    )
    logger.info(
        "Split: %d training / %d validation patches.",
        len(x_trn),
        len(x_val),
    )

    # Callbacks
    best_weights_path = output_dir / f"{model.name}_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_weights_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_trn,
        y_trn,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final weights
    final_weights_path = output_dir / f"{model.name}_final.weights.h5"
    model.save_weights(str(final_weights_path))

    # Plot training curves
    _plot_training_history(history, output_dir, model.name)

    logger.info("Training complete. Weights saved to %s", output_dir)
    return history


def _plot_training_history(
    history: keras.callbacks.History,
    output_dir: Path,
    model_name: str,
) -> None:
    """Save a loss-curve plot to *output_dir*."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training History — {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_loss_curve.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    logger.info("Loss curve saved → %s", plot_path)


# ===================================================================
# Section 6 — Inference and Reconstruction
# ===================================================================


def binarize_image(
    model: keras.Model,
    image: np.ndarray,
    patch_size: int,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> np.ndarray:
    """Binarize a full-resolution image using patch-wise inference.

    The image is padded to be divisible by *patch_size*, split into a
    non-overlapping grid, predicted patch-by-patch, and tiled back
    together.  A hard threshold produces the final binary output.

    Parameters
    ----------
    model : keras.Model
        Trained autoencoder.
    image : np.ndarray
        Grayscale image, shape (H, W), float32 in [0, 1].
    patch_size : int
        Must match the model's expected input size.
    threshold : float
        Binarization cutoff applied to sigmoid output.
    batch_size : int
        Inference batch size (does not affect results).

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W) with values in {0.0, 1.0},
        cropped back to the original dimensions.
    """
    orig_h, orig_w = image.shape[:2]

    # Pad to exact multiples of patch_size
    padded = pad_to_multiple(image, patch_size)
    pad_h, pad_w = padded.shape[:2]

    # Extract non-overlapping patches (stride = patch_size)
    patches = extract_patches(padded, patch_size, stride=patch_size)
    logger.info("Inference: %d patches of %dx%d.", len(patches), patch_size, patch_size)

    # Predict in batches
    predicted = model.predict(patches, batch_size=batch_size, verbose=0)

    # Binarize
    predicted = (predicted >= threshold).astype(np.float32)

    # Reconstruct full image by tiling
    cols = pad_w // patch_size
    rows = pad_h // patch_size

    result = np.zeros((pad_h, pad_w), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * patch_size
            x0 = c * patch_size
            result[y0 : y0 + patch_size, x0 : x0 + patch_size] = predicted[idx, :, :, 0]
            idx += 1

    # Crop back to original dimensions
    result = result[:orig_h, :orig_w]
    return result


# ===================================================================
# Section 7 — Evaluation Metrics
# ===================================================================


def compute_metrics(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute F1 score, IoU, and PSNR between two binary images.

    Both images must contain values in {0, 1} (or {0.0, 1.0}).
    Black (0) pixels are treated as the **positive** class (foreground).

    Parameters
    ----------
    ground_truth : np.ndarray
        Reference binary image, shape (H, W).
    predicted : np.ndarray
        Model output binary image, same shape.

    Returns
    -------
    dict with keys ``"f1"``, ``"iou"``, ``"psnr"``.
    """
    if ground_truth.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: GT {ground_truth.shape} vs. " f"Pred {predicted.shape}")

    # Work with integer masks: foreground (black) = 1, background = 0
    gt = (ground_truth < 0.5).astype(np.int32)
    pr = (predicted < 0.5).astype(np.int32)

    tp = int(np.sum(gt & pr))
    fp = int(np.sum((~gt.astype(bool)) & pr.astype(bool)))
    fn = int(np.sum(gt.astype(bool) & (~pr.astype(bool))))

    # F1 Score (Equations 4–6)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Intersection over Union (Equation 7)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0

    # PSNR (Equations 8–9) — computed on the original [0, 255] scale
    gt_255 = ground_truth * 255.0
    pr_255 = predicted * 255.0
    mse = np.mean((gt_255 - pr_255) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(255.0**2 / mse)

    return {"f1": f1, "iou": iou, "psnr": psnr}


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    logger.info(
        "F1 = %.4f  |  IoU = %.4f  |  PSNR = %.2f dB",
        metrics["f1"],
        metrics["iou"],
        metrics["psnr"],
    )


# ===================================================================
# Section 8 — Comparison with Traditional Techniques
# ===================================================================


def traditional_binarization(
    image: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> dict[str, dict]:
    """Apply traditional thresholding techniques for baseline comparison.

    Implements the four methods from Section 4 of the paper:
        1. Simple thresholding (best threshold from 0–255 sweep)
        2. Otsu's automatic thresholding
        3. Adaptive Gaussian thresholding
        4. Adaptive Mean thresholding

    Parameters
    ----------
    image : np.ndarray
        Grayscale image in [0, 1].
    ground_truth : np.ndarray or None
        If provided, metrics are computed for each technique.

    Returns
    -------
    dict
        Keys are technique names; values contain ``"image"`` (the
        binarized result) and optionally ``"metrics"``.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    results: dict[str, dict] = {}

    # --- 1. Simple thresholding (sweep for best if GT available) ---
    if ground_truth is not None:
        best_f1 = -1.0
        best_thresh = 128
        for t in range(0, 256):
            _, binary = cv2.threshold(img_uint8, t, 255, cv2.THRESH_BINARY)
            binary_float = binary.astype(np.float32) / 255.0
            m = compute_metrics(ground_truth, binary_float)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thresh = t
        _, simple_bin = cv2.threshold(img_uint8, best_thresh, 255, cv2.THRESH_BINARY)
        simple_float = simple_bin.astype(np.float32) / 255.0
        results["simple"] = {
            "image": simple_float,
            "threshold": best_thresh,
            "metrics": compute_metrics(ground_truth, simple_float),
        }
    else:
        _, simple_bin = cv2.threshold(img_uint8, 128, 255, cv2.THRESH_BINARY)
        results["simple"] = {
            "image": simple_bin.astype(np.float32) / 255.0,
            "threshold": 128,
        }

    # --- 2. Otsu's thresholding ---
    otsu_thresh, otsu_bin = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_float = otsu_bin.astype(np.float32) / 255.0
    results["otsu"] = {"image": otsu_float, "threshold": otsu_thresh}
    if ground_truth is not None:
        results["otsu"]["metrics"] = compute_metrics(ground_truth, otsu_float)

    # --- 3 & 4. Adaptive thresholding (Gaussian and Mean) ---
    kernel_sizes = [31, 63, 127, 255]
    c_values = range(0, 101)

    for method_name, method_flag in [
        ("adaptive_gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C),
        ("adaptive_mean", cv2.ADAPTIVE_THRESH_MEAN_C),
    ]:
        if ground_truth is not None:
            best_f1 = -1.0
            best_k = 31
            best_c = 0
            for k in kernel_sizes:
                for c in c_values:
                    try:
                        binary = cv2.adaptiveThreshold(
                            img_uint8,
                            255,
                            method_flag,
                            cv2.THRESH_BINARY,
                            k,
                            c,
                        )
                        binary_float = binary.astype(np.float32) / 255.0
                        m = compute_metrics(ground_truth, binary_float)
                        if m["f1"] > best_f1:
                            best_f1 = m["f1"]
                            best_k = k
                            best_c = c
                    except cv2.error:
                        continue
            binary = cv2.adaptiveThreshold(img_uint8, 255, method_flag, cv2.THRESH_BINARY, best_k, best_c)
            binary_float = binary.astype(np.float32) / 255.0
            results[method_name] = {
                "image": binary_float,
                "kernel_size": best_k,
                "C": best_c,
                "metrics": compute_metrics(ground_truth, binary_float),
            }
        else:
            binary = cv2.adaptiveThreshold(img_uint8, 255, method_flag, cv2.THRESH_BINARY, 255, 19)
            results[method_name] = {
                "image": binary.astype(np.float32) / 255.0,
                "kernel_size": 255,
                "C": 19,
            }

    return results


# ===================================================================
# Section 9 — CLI Interface
# ===================================================================


def _cli_train(args: argparse.Namespace) -> None:
    """Handle the ``train`` sub-command."""
    model_id = args.model_id
    patch_size, _ = MODEL_REGISTRY[model_id]
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("=== TRAIN — Model %d (patch %dx%d) ===", model_id, patch_size, patch_size)

    # Load dirty/clean pair
    dirty = load_grayscale(args.dirty)
    clean = load_grayscale(args.clean)

    # Extract training patches
    x_patches, y_patches = extract_training_patches(
        dirty,
        clean,
        patch_size,
        stride_fraction=args.stride_fraction,
    )

    # Build and compile
    model = build_model(model_id)
    compile_model(model, config)
    model.summary(print_fn=logger.info)

    # Train
    train_model(model, x_patches, y_patches, config, args.output_dir)


def _cli_infer(args: argparse.Namespace) -> None:
    """Handle the ``infer`` sub-command."""
    model_id = args.model_id
    patch_size, _ = MODEL_REGISTRY[model_id]
    threshold = args.threshold

    logger.info("=== INFER — Model %d ===", model_id)

    # Build model and load weights
    model = build_model(model_id)
    compile_model(model)
    model.load_weights(args.weights)
    logger.info("Loaded weights from %s", args.weights)

    # Load and binarize
    image = load_grayscale(args.image)
    result = binarize_image(model, image, patch_size, threshold=threshold)
    save_binary_image(result, args.output)

    # Optional evaluation
    if args.ground_truth:
        gt = load_grayscale(args.ground_truth)
        # Ensure dimensions match
        min_h = min(gt.shape[0], result.shape[0])
        min_w = min(gt.shape[1], result.shape[1])
        metrics = compute_metrics(gt[:min_h, :min_w], result[:min_h, :min_w])
        print_metrics(metrics)


def _cli_evaluate(args: argparse.Namespace) -> None:
    """Handle the ``evaluate`` sub-command."""
    logger.info("=== EVALUATE ===")

    predicted = load_grayscale(args.predicted)
    gt = load_grayscale(args.ground_truth)

    # Binarize loaded images (in case they are not perfectly 0/1)
    predicted = (predicted >= 0.5).astype(np.float32)
    gt = (gt >= 0.5).astype(np.float32)

    min_h = min(gt.shape[0], predicted.shape[0])
    min_w = min(gt.shape[1], predicted.shape[1])
    metrics = compute_metrics(
        gt[:min_h, :min_w],
        predicted[:min_h, :min_w],
    )
    print_metrics(metrics)


def _cli_finetune(args: argparse.Namespace) -> None:
    """Handle the ``finetune`` sub-command."""
    model_id = args.model_id
    patch_size, _ = MODEL_REGISTRY[model_id]
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("=== FINE-TUNE — Model %d ===", model_id)

    # Collect dirty/clean pairs from directories
    dirty_dir = Path(args.dirty)
    clean_dir = Path(args.clean)

    dirty_paths = sorted(glob.glob(str(dirty_dir / "*")))
    clean_paths = sorted(glob.glob(str(clean_dir / "*")))

    if len(dirty_paths) != len(clean_paths):
        raise ValueError(f"Mismatch: {len(dirty_paths)} dirty vs. " f"{len(clean_paths)} clean images.")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for dp, cp in zip(dirty_paths, clean_paths):
        d = load_grayscale(dp)
        c = load_grayscale(cp)
        xp, yp = extract_training_patches(d, c, patch_size)
        all_x.append(xp)
        all_y.append(yp)

    x_patches = np.concatenate(all_x, axis=0)
    y_patches = np.concatenate(all_y, axis=0)
    logger.info("Fine-tune dataset: %d total patches.", len(x_patches))

    # Build model and load existing weights
    model = build_model(model_id)
    compile_model(model, config)
    model.load_weights(args.weights)
    logger.info("Loaded pre-trained weights from %s", args.weights)

    # Continue training
    train_model(model, x_patches, y_patches, config, args.output_dir)


def _cli_compare(args: argparse.Namespace) -> None:
    """Handle the ``compare`` sub-command — traditional baselines."""
    logger.info("=== COMPARE TRADITIONAL TECHNIQUES ===")

    image = load_grayscale(args.image)
    gt = None
    if args.ground_truth:
        gt = load_grayscale(args.ground_truth)
        gt = (gt >= 0.5).astype(np.float32)

    results = traditional_binarization(image, gt)

    for name, info in results.items():
        msg = f"  {name:25s}"
        if "threshold" in info:
            msg += f"  thresh={info['threshold']}"
        if "kernel_size" in info:
            msg += f"  k={info['kernel_size']}  C={info['C']}"
        if "metrics" in info:
            m = info["metrics"]
            msg += f"  F1={m['f1']:.4f}  IoU={m['iou']:.4f}  " f"PSNR={m['psnr']:.2f}"
        logger.info(msg)

    # Optionally save outputs
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, info in results.items():
            save_binary_image(info["image"], out / f"{name}_binarized.png")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with sub-commands."""
    parser = argparse.ArgumentParser(
        prog="binarize",
        description="Binarize historical architectural drawings with shallow convolutional autoencoders (Narag et al., 2025).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train an autoencoder model.")
    p_train.add_argument("--dirty", required=True, help="Path to dirty training image.")
    p_train.add_argument("--clean", required=True, help="Path to clean (ground truth) training image.")
    p_train.add_argument("--model-id", type=int, required=True, choices=range(1, 9), help="Model ID (1-8).")
    p_train.add_argument("--output-dir", default="runs", help="Directory for weights and plots.")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--learning-rate", type=float, default=0.001)
    p_train.add_argument("--stride-fraction", type=float, default=None, help="Override default stride fraction.")

    # ---- infer ----
    p_infer = sub.add_parser("infer", help="Binarize an image using a trained model.")
    p_infer.add_argument("--image", required=True, help="Path to image to binarize.")
    p_infer.add_argument("--weights", required=True, help="Path to saved model weights.")
    p_infer.add_argument("--model-id", type=int, required=True, choices=range(1, 9))
    p_infer.add_argument("--output", required=True, help="Path for the binarized output image.")
    p_infer.add_argument("--ground-truth", default=None, help="Optional GT for evaluation.")
    p_infer.add_argument("--threshold", type=float, default=0.5)

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Compute F1, IoU, PSNR between two images.")
    p_eval.add_argument("--predicted", required=True)
    p_eval.add_argument("--ground-truth", required=True)

    # ---- finetune ----
    p_ft = sub.add_parser("finetune", help="Fine-tune a trained model with additional data.")
    p_ft.add_argument("--dirty", required=True, help="Directory of additional dirty images.")
    p_ft.add_argument("--clean", required=True, help="Directory of corresponding clean images.")
    p_ft.add_argument("--weights", required=True, help="Pre-trained model weights to load.")
    p_ft.add_argument("--model-id", type=int, required=True, choices=range(1, 9))
    p_ft.add_argument("--output-dir", default="runs_finetuned")
    p_ft.add_argument("--epochs", type=int, default=100)
    p_ft.add_argument("--batch-size", type=int, default=32)
    p_ft.add_argument("--learning-rate", type=float, default=0.001)

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="Run traditional binarization baselines.")
    p_cmp.add_argument("--image", required=True, help="Path to grayscale image.")
    p_cmp.add_argument("--ground-truth", default=None, help="Optional GT for metric computation.")
    p_cmp.add_argument("--output-dir", default=None, help="Optional directory to save binarized outputs.")

    return parser


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Entry point — dispatch to the appropriate sub-command handler."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": _cli_train,
        "infer": _cli_infer,
        "evaluate": _cli_evaluate,
        "finetune": _cli_finetune,
        "compare": _cli_compare,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
