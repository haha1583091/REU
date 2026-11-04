# -*- coding: utf-8 -*-
"""
NC-REU real-rock downstream demonstration:
REU-guided spectral-prototype classification
===============================================================================

Purpose
-------
This script provides ONE deliberately simple downstream application of the
already-established rock REU values.

It does NOT re-estimate REU and does NOT use classification accuracy to tune
REU.  Instead, it asks a practical question:

    If a labelled training region is used to build one representative spectral
    prototype for each rock class, how does classification performance change
    when the prototype-support size is chosen relative to each class REU?

The downstream task is intentionally simple:
    - one spectral prototype per rock class;
    - prototype = mean raw spectrum inside one labelled square training ROI;
    - classifier = nearest prototype by spectral angle (SAM);
    - test samples = fixed, spatially separated single pixels;
    - no deep network, no hyperparameter search, no feature selection.

Why this avoids class-dependent inference leakage
-------------------------------------------------
Class-specific REU is used ONLY when constructing labelled TRAINING prototypes.
At inference, every test pixel is compared with all five prototypes using the
same SAM rule.  The true test class is never used to choose an inference window.

Important limitation / intended role
------------------------------------
The frozen REUs were obtained in the preceding REU study and are treated here
as prior scale descriptors.  Therefore this is a downstream proof-of-concept,
NOT an independent benchmark intended to establish classification state of the
art or re-prove the validity of REU.

Frozen final rock REUs
----------------------
    class 1 = 128 px
    class 2 = 160 px
    class 3 =  80 px
    class 4 =  96 px
    class 5 =  56 px

REU-normalized prototype supports
---------------------------------
    L / L_REU = 0.25, 0.50, 1.00, 1.25

For each condition, the script repeats prototype construction at multiple
geometry-only training locations.  The spatial test set is fixed across all
conditions and repeats.

A full-training-region mean spectrum is also evaluated as an upper-reference
prototype.  It is NOT used to select the best REU condition.

Default input
-------------
    D:\\LS_REU\\data\\5rock_mat\\5rock.mat
    D:\\LS_REU\\data\\5rock_mat\\5rock_label.mat

Outputs
-------
D:\\LS_REU\\results\\nc_reu_rock_downstream_prototype_classification\\

    downstream_config.json
    spatial_split_audit.csv
    prototype_window_audit.csv
    test_pixel_audit.csv
    classification_repeat_metrics.csv
    classification_summary.csv
    per_class_f1_repeat.csv
    per_class_f1_summary.csv
    full_train_reference_metrics.csv
    confusion_matrix_REU_mean.csv
    confusion_matrix_FULL_TRAIN.csv

    Fig_1_classification_vs_REU_normalized_support.jpg
    Fig_2_accuracy_efficiency_tradeoff.jpg

Run
---
    D:\\Anaconda\\envs\\DL\\python.exe
        reu_nc_rock_downstream_spectral_prototype_classification.py
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import label as ndi_label


# =============================================================================
# 0. USER SETTINGS
# =============================================================================

DATA_PATH = Path(r"D:\LS_REU\data\5rock_mat\5rock.mat")
LABEL_PATH = Path(r"D:\LS_REU\data\5rock_mat\5rock_label.mat")
DATA_KEY = "data"
LABEL_KEY = "data"

OUTPUT_ROOT = Path(
    r"D:\LS_REU\results\nc_reu_rock_downstream_prototype_classification"
)

# Frozen values from the final rock REU analysis.  DO NOT tune these using the
# classification output.
ROCK_REU_PX: Mapping[int, int] = {
    1: 128,
    2: 160,
    3: 80,
    4: 96,
    5: 56,
}

CLASS_NAMES: Mapping[int, str] = {
    1: "Rock class 1",
    2: "Rock class 2",
    3: "Rock class 3",
    4: "Rock class 4",
    5: "Rock class 5",
}

# Four deliberately simple supports: clearly sub-REU, intermediate, REU, and
# slightly super-REU.  No classification-guided search is performed.
LAMBDA_VALUES: Tuple[float, ...] = (0.25, 0.50, 1.00, 1.25)

# Spatial split of each largest connected rock component along its long axis.
# One side is training, a central gap is excluded, and the other side is test.
TRAIN_FRACTION = 0.60
BUFFER_FRACTION = 0.15
# Test fraction is the remainder (= 0.25).

# Candidate prototype windows must lie almost entirely inside the labelled
# training part of the rock.
MIN_TRAIN_COVERAGE = 0.98
CANDIDATE_STRIDE_PX = 4

# Repeated geometry-only prototype locations.
N_REPEATS = 20
BASE_SEED = 3407

# Fixed balanced test set.  Same pixels are used for every condition/repeat.
MAX_TEST_PIXELS_PER_CLASS = 4000

# Optional physical pixel size for reporting only.
PIXEL_SIZE_MM: Optional[float] = 0.1

FIG_DPI = 600


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


def stable_seed(*tokens: Any) -> int:
    x = 2166136261
    for token in tokens:
        for byte in str(token).encode("utf-8"):
            x ^= int(byte)
            x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def setup_plot_fonts() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "savefig.dpi": FIG_DPI,
    })


def load_mat_array(path: Path, key: Optional[str], ndim: int) -> np.ndarray:
    """Robustly load MATLAB arrays; HxWx1 labels are squeezed to HxW."""
    if not path.exists():
        raise FileNotFoundError(path)

    data = loadmat(path)

    if key is not None and key in data:
        arr = np.asarray(data[key])
        selected_name = key
    else:
        candidates: List[Tuple[str, np.ndarray]] = []
        for name, value in data.items():
            if name.startswith("__"):
                continue
            a = np.asarray(value)
            if np.squeeze(a).ndim == ndim:
                candidates.append((name, a))
        if not candidates:
            shapes = {
                name: np.asarray(value).shape
                for name, value in data.items()
                if not name.startswith("__")
            }
            raise KeyError(
                f"No array reducible to {ndim}D found in {path}. "
                f"Available variables/shapes: {shapes}"
            )
        candidates.sort(key=lambda item: item[1].size, reverse=True)
        selected_name, arr = candidates[0]

    original_shape = tuple(arr.shape)
    arr = np.squeeze(arr)

    if arr.ndim != ndim:
        raise ValueError(
            f"{path}: variable '{selected_name}' expected {ndim}D after squeeze; "
            f"original={original_shape}, squeezed={arr.shape}"
        )

    print(
        f"[MAT loader] {path.name}: '{selected_name}', "
        f"original={original_shape}, loaded={arr.shape}, dtype={arr.dtype}"
    )
    return np.asarray(arr)


def largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labelled, n = ndi_label(mask.astype(bool), structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0
    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    cid = int(np.argmax(counts))
    return labelled == cid, int(n)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(yy) == 0:
        raise ValueError("Empty mask.")
    return int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1


def integral_image_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    ii = np.cumsum(np.cumsum(a, axis=0), axis=1)
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = ii
    return out


def rect_sum(
    ii: np.ndarray,
    y0: np.ndarray,
    x0: np.ndarray,
    y1: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


# =============================================================================
# 2. SPATIAL TRAIN / BUFFER / TEST SPLIT
# =============================================================================

@dataclass(frozen=True)
class CandidateWindow:
    y0: int
    x0: int
    y1: int
    x1: int
    size_px: int
    valid_train_pixels: int
    coverage: float


@dataclass
class RockPartition:
    class_id: int
    class_name: str
    reu_px: int

    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int

    component_crop: np.ndarray
    train_mask: np.ndarray
    buffer_mask: np.ndarray
    test_mask: np.ndarray

    split_axis: str
    train_side: str
    component_count: int


def build_partition_masks(
    component_crop: np.ndarray,
    axis: str,
    train_side: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = component_crop.shape

    if axis == "y":
        coord = np.arange(h, dtype=np.float64) / max(h - 1, 1)
        field = coord[:, None]
    else:
        coord = np.arange(w, dtype=np.float64) / max(w - 1, 1)
        field = coord[None, :]

    train_end = TRAIN_FRACTION
    test_start = TRAIN_FRACTION + BUFFER_FRACTION

    if train_side == "low":
        train_zone = field < train_end
        buffer_zone = (field >= train_end) & (field < test_start)
        test_zone = field >= test_start
    else:
        train_zone = field > (1.0 - train_end)
        buffer_zone = (
            (field <= (1.0 - train_end))
            & (field > (1.0 - test_start))
        )
        test_zone = field <= (1.0 - test_start)

    train = component_crop & train_zone
    buffer = component_crop & buffer_zone
    test = component_crop & test_zone
    return train, buffer, test


def generate_candidate_windows(
    train_mask: np.ndarray,
    size_px: int,
    stride: int = CANDIDATE_STRIDE_PX,
) -> List[CandidateWindow]:
    h, w = train_mask.shape
    size = int(size_px)
    if size <= 0 or size > h or size > w:
        return []

    max_y = h - size
    max_x = w - size

    ys = list(range(0, max_y + 1, max(1, stride)))
    xs = list(range(0, max_x + 1, max(1, stride)))
    if ys[-1] != max_y:
        ys.append(max_y)
    if xs[-1] != max_x:
        xs.append(max_x)

    yg, xg = np.meshgrid(
        np.asarray(ys, dtype=np.int32),
        np.asarray(xs, dtype=np.int32),
        indexing="ij",
    )
    y0 = yg.ravel()
    x0 = xg.ravel()
    y1 = y0 + size
    x1 = x0 + size

    ii = integral_image_2d(train_mask.astype(np.float64))
    counts = rect_sum(ii, y0, x0, y1, x1)
    coverage = counts / float(size * size)

    valid = np.where(coverage >= MIN_TRAIN_COVERAGE)[0]
    out: List[CandidateWindow] = []
    for idx in valid:
        out.append(CandidateWindow(
            y0=int(y0[idx]),
            x0=int(x0[idx]),
            y1=int(y1[idx]),
            x1=int(x1[idx]),
            size_px=size,
            valid_train_pixels=int(round(float(counts[idx]))),
            coverage=float(coverage[idx]),
        ))
    return out


def choose_partition_for_class(
    component_full: np.ndarray,
    class_id: int,
    component_count: int,
) -> RockPartition:
    y0, x0, y1, x1 = bbox_from_mask(component_full)
    crop = component_full[y0:y1, x0:x1]
    h, w = crop.shape
    axis = "y" if h >= w else "x"

    reu_px = int(ROCK_REU_PX[class_id])
    largest_support = int(round(max(LAMBDA_VALUES) * reu_px))

    choices = []
    for side in ("low", "high"):
        train, buffer, test = build_partition_masks(crop, axis, side)
        n_candidates = len(
            generate_candidate_windows(
                train, largest_support, stride=CANDIDATE_STRIDE_PX
            )
        )
        choices.append((
            n_candidates,
            int(train.sum()),
            int(test.sum()),
            side,
            train,
            buffer,
            test,
        ))

    # Geometry-only decision.  No spectra or classifier performance are used.
    choices.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
    (
        _,
        _,
        _,
        best_side,
        train_mask,
        buffer_mask,
        test_mask,
    ) = choices[0]

    return RockPartition(
        class_id=class_id,
        class_name=CLASS_NAMES.get(class_id, f"Rock class {class_id}"),
        reu_px=reu_px,
        bbox_y0=y0,
        bbox_x0=x0,
        bbox_y1=y1,
        bbox_x1=x1,
        component_crop=crop,
        train_mask=train_mask,
        buffer_mask=buffer_mask,
        test_mask=test_mask,
        split_axis=axis,
        train_side=best_side,
        component_count=component_count,
    )


# =============================================================================
# 3. SELECT TRAINING PROTOTYPE WINDOWS
# =============================================================================

def select_windows_for_repeats(
    candidates: Sequence[CandidateWindow],
    class_id: int,
    lambda_value: float,
) -> List[CandidateWindow]:
    if len(candidates) == 0:
        return []

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, "prototype", class_id, lambda_value)
    )
    replace = len(candidates) < N_REPEATS
    idx = rng.choice(
        len(candidates),
        size=N_REPEATS,
        replace=replace,
    )
    return [candidates[int(i)] for i in idx]


def compute_selected_window_means(
    cube_crop: np.ndarray,
    train_mask: np.ndarray,
    selected: Sequence[Tuple[float, int, CandidateWindow]],
) -> Tuple[Dict[Tuple[float, int], np.ndarray], np.ndarray]:
    """
    Efficiently compute raw mean spectra for all selected windows in one class.

    selected entries:
        (lambda_value, repeat_id, CandidateWindow)

    Returns
    -------
    prototype_map[(lambda, repeat)] -> mean spectrum
    full_train_mean -> mean spectrum over all training pixels
    """
    if len(selected) == 0:
        raise ValueError("No selected prototype windows.")

    b = cube_crop.shape[2]

    y0 = np.asarray([x[2].y0 for x in selected], dtype=np.int32)
    x0 = np.asarray([x[2].x0 for x in selected], dtype=np.int32)
    y1 = np.asarray([x[2].y1 for x in selected], dtype=np.int32)
    x1 = np.asarray([x[2].x1 for x in selected], dtype=np.int32)
    counts = np.asarray(
        [x[2].valid_train_pixels for x in selected],
        dtype=np.float64,
    )

    spectra = np.empty((len(selected), b), dtype=np.float64)
    full_train = np.empty(b, dtype=np.float64)

    n_train = int(train_mask.sum())
    if n_train == 0:
        raise ValueError("Empty training mask.")

    for band in range(b):
        z = np.asarray(cube_crop[:, :, band], dtype=np.float64)
        masked = np.where(train_mask, z, 0.0)

        full_train[band] = float(masked.sum(dtype=np.float64)) / float(n_train)

        ii = integral_image_2d(masked)
        sums = rect_sum(ii, y0, x0, y1, x1)
        spectra[:, band] = sums / np.maximum(counts, 1.0)

    mapping: Dict[Tuple[float, int], np.ndarray] = {}
    for i, (lam, repeat_id, _) in enumerate(selected):
        mapping[(float(lam), int(repeat_id))] = spectra[i].copy()

    return mapping, full_train


# =============================================================================
# 4. FIXED TEST SET
# =============================================================================

def collect_fixed_test_pixels(
    cube: np.ndarray,
    partitions: Mapping[int, RockPartition],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    spectra: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    audit_rows: List[Dict[str, Any]] = []

    for class_id in sorted(partitions):
        p = partitions[class_id]
        crop_cube = cube[p.bbox_y0:p.bbox_y1, p.bbox_x0:p.bbox_x1, :]

        yy, xx = np.where(p.test_mask)
        if len(yy) == 0:
            raise RuntimeError(f"Class {class_id}: empty spatial test partition.")

        # Use only pixels finite in all bands.
        candidate_spectra = crop_cube[yy, xx, :]
        finite = np.all(np.isfinite(candidate_spectra), axis=1)
        yy = yy[finite]
        xx = xx[finite]
        candidate_spectra = candidate_spectra[finite]

        if len(yy) == 0:
            raise RuntimeError(f"Class {class_id}: no finite test pixels.")

        rng = np.random.default_rng(
            stable_seed(BASE_SEED, "fixed_test_pixels", class_id)
        )
        n_take = min(MAX_TEST_PIXELS_PER_CLASS, len(yy))
        chosen = np.sort(
            rng.choice(len(yy), size=n_take, replace=False)
        )

        X = np.asarray(candidate_spectra[chosen], dtype=np.float64)
        y = np.full(n_take, class_id, dtype=np.int32)

        spectra.append(X)
        labels.append(y)

        for local_i, cidx in enumerate(chosen):
            audit_rows.append({
                "class_id": class_id,
                "class_name": p.class_name,
                "test_sample_id_within_class": local_i,
                "y_global": int(yy[cidx] + p.bbox_y0),
                "x_global": int(xx[cidx] + p.bbox_x0),
            })

    return (
        np.vstack(spectra),
        np.concatenate(labels),
        pd.DataFrame(audit_rows),
    )


# =============================================================================
# 5. SAM PROTOTYPE CLASSIFIER AND METRICS
# =============================================================================

def l2_normalize_rows(X: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def predict_by_sam(
    X_test: np.ndarray,
    prototypes: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    """
    Nearest prototype by spectral angle.
    Minimizing SAM is equivalent to maximizing cosine similarity.
    """
    Xn = l2_normalize_rows(X_test)
    Pn = l2_normalize_rows(prototypes)
    similarity = Xn @ Pn.T
    idx = np.argmax(similarity, axis=1)
    class_ids_arr = np.asarray(class_ids, dtype=np.int32)
    return class_ids_arr[idx]


def confusion_matrix_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    class_ids = list(class_ids)
    index = {c: i for i, c in enumerate(class_ids)}
    cm = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if int(t) in index and int(p) in index:
            cm[index[int(t)], index[int(p)]] += 1
    return cm


def classification_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    cm = np.asarray(cm, dtype=np.float64)
    total = float(cm.sum())
    tp = np.diag(cm)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    recall = np.divide(
        tp, support,
        out=np.zeros_like(tp),
        where=support > 0,
    )
    precision = np.divide(
        tp, predicted,
        out=np.zeros_like(tp),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    return {
        "oa": float(tp.sum() / total) if total > 0 else np.nan,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
    }


# =============================================================================
# 6. MAIN ANALYSIS
# =============================================================================

def main() -> None:
    t0 = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("NC-REU downstream demonstration: REU-guided spectral-prototype classification")
    print("No REU re-estimation; no classifier hyperparameter search.")
    print(f"Data : {DATA_PATH}")
    print(f"Label: {LABEL_PATH}")
    print(f"Frozen REUs: {dict(ROCK_REU_PX)}")
    print(f"Supports L/REU: {LAMBDA_VALUES}")
    print(f"Repeats: {N_REPEATS}")
    print("=" * 100)

    cube = load_mat_array(DATA_PATH, DATA_KEY, ndim=3).astype(np.float32, copy=False)
    gt = load_mat_array(LABEL_PATH, LABEL_KEY, ndim=2)
    gt = np.asarray(np.rint(gt), dtype=np.int32)

    if cube.shape[:2] != gt.shape:
        raise ValueError(
            f"Spatial mismatch: cube={cube.shape[:2]}, labels={gt.shape}"
        )

    finite_pixel_mask = np.all(np.isfinite(cube), axis=2)

    # -------------------------------------------------------------------------
    # 6.1 Build spatially separated partitions
    # -------------------------------------------------------------------------
    partitions: Dict[int, RockPartition] = {}
    split_rows: List[Dict[str, Any]] = []

    for class_id in sorted(ROCK_REU_PX):
        raw = (gt == class_id) & finite_pixel_mask
        component, component_count = largest_connected_component(raw)
        if component.sum() == 0:
            raise RuntimeError(f"Class {class_id}: no valid connected component.")

        p = choose_partition_for_class(
            component_full=component,
            class_id=class_id,
            component_count=component_count,
        )
        partitions[class_id] = p

        split_rows.append({
            "class_id": class_id,
            "class_name": p.class_name,
            "reu_px": p.reu_px,
            "component_count": p.component_count,
            "component_pixels": int(p.component_crop.sum()),
            "bbox_y0": p.bbox_y0,
            "bbox_x0": p.bbox_x0,
            "bbox_y1": p.bbox_y1,
            "bbox_x1": p.bbox_x1,
            "bbox_height": int(p.bbox_y1 - p.bbox_y0),
            "bbox_width": int(p.bbox_x1 - p.bbox_x0),
            "split_axis": p.split_axis,
            "train_side": p.train_side,
            "train_pixels": int(p.train_mask.sum()),
            "buffer_pixels": int(p.buffer_mask.sum()),
            "test_pixels": int(p.test_mask.sum()),
        })

        print(
            f"Class {class_id}: REU={p.reu_px:>3d}px | "
            f"bbox={p.component_crop.shape} | axis={p.split_axis} | "
            f"train_side={p.train_side} | train={int(p.train_mask.sum())} | "
            f"buffer={int(p.buffer_mask.sum())} | test={int(p.test_mask.sum())}"
        )

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(
        OUTPUT_ROOT / "spatial_split_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.2 Fixed balanced spatial test pixels
    # -------------------------------------------------------------------------
    X_test, y_test, test_audit = collect_fixed_test_pixels(cube, partitions)
    test_audit.to_csv(
        OUTPUT_ROOT / "test_pixel_audit.csv",
        index=False, encoding="utf-8-sig"
    )
    print(
        f"\nFixed test set: n={len(y_test)}, "
        f"per-class max={MAX_TEST_PIXELS_PER_CLASS}, bands={X_test.shape[1]}"
    )

    class_ids = sorted(ROCK_REU_PX)

    # -------------------------------------------------------------------------
    # 6.3 Candidate and selected training prototype windows
    # -------------------------------------------------------------------------
    prototype_maps: Dict[int, Dict[Tuple[float, int], np.ndarray]] = {}
    full_train_prototypes: Dict[int, np.ndarray] = {}
    window_audit_rows: List[Dict[str, Any]] = []

    for class_id in class_ids:
        p = partitions[class_id]
        crop_cube = cube[p.bbox_y0:p.bbox_y1, p.bbox_x0:p.bbox_x1, :]

        selected_all: List[Tuple[float, int, CandidateWindow]] = []

        for lam in LAMBDA_VALUES:
            size = int(round(lam * p.reu_px))
            candidates = generate_candidate_windows(
                p.train_mask,
                size_px=size,
                stride=CANDIDATE_STRIDE_PX,
            )

            if len(candidates) == 0:
                raise RuntimeError(
                    f"Class {class_id}, lambda={lam}: no valid training "
                    f"prototype ROI of size {size}px. "
                    f"Consider checking the spatial split geometry; do NOT "
                    f"tune REU or classification metrics."
                )

            selected = select_windows_for_repeats(
                candidates=candidates,
                class_id=class_id,
                lambda_value=lam,
            )

            print(
                f"  class={class_id} | lambda={lam:.2f} | "
                f"L={size:>3d}px | candidate ROIs={len(candidates):>4d}"
            )

            for repeat_id, win in enumerate(selected):
                selected_all.append((lam, repeat_id, win))
                window_audit_rows.append({
                    "class_id": class_id,
                    "class_name": p.class_name,
                    "reu_px": p.reu_px,
                    "lambda_value": lam,
                    "repeat_id": repeat_id,
                    "support_px": win.size_px,
                    "support_physical_mm": (
                        win.size_px * PIXEL_SIZE_MM
                        if PIXEL_SIZE_MM is not None else np.nan
                    ),
                    "n_candidates_available": len(candidates),
                    "sampled_with_replacement": len(candidates) < N_REPEATS,
                    "y0_global": win.y0 + p.bbox_y0,
                    "x0_global": win.x0 + p.bbox_x0,
                    "y1_global": win.y1 + p.bbox_y0,
                    "x1_global": win.x1 + p.bbox_x0,
                    "valid_train_pixels": win.valid_train_pixels,
                    "coverage": win.coverage,
                })

        prototype_map, full_train_mean = compute_selected_window_means(
            crop_cube,
            p.train_mask,
            selected_all,
        )
        prototype_maps[class_id] = prototype_map
        full_train_prototypes[class_id] = full_train_mean

    window_audit_df = pd.DataFrame(window_audit_rows)
    window_audit_df.to_csv(
        OUTPUT_ROOT / "prototype_window_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.4 Classification for each REU-normalized support and repeat
    # -------------------------------------------------------------------------
    repeat_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    cm_by_lambda: Dict[float, List[np.ndarray]] = {lam: [] for lam in LAMBDA_VALUES}

    for lam in LAMBDA_VALUES:
        for repeat_id in range(N_REPEATS):
            prototypes = np.vstack([
                prototype_maps[c][(float(lam), int(repeat_id))]
                for c in class_ids
            ])

            y_pred = predict_by_sam(X_test, prototypes, class_ids)
            cm = confusion_matrix_fixed(y_test, y_pred, class_ids)
            metrics = classification_metrics_from_cm(cm)
            cm_by_lambda[lam].append(cm)

            # Exact number of labelled pixels used to form the 5 prototypes.
            subaudit = window_audit_df[
                (np.isclose(window_audit_df["lambda_value"], lam))
                & (window_audit_df["repeat_id"] == repeat_id)
            ]
            total_training_pixels = int(
                subaudit["valid_train_pixels"].sum()
            )

            repeat_rows.append({
                "condition": f"{lam:.2f}_REU",
                "lambda_value": lam,
                "repeat_id": repeat_id,
                "oa": metrics["oa"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "prototype_training_pixels_total": total_training_pixels,
                "test_pixels_total": len(y_test),
            })

            for j, class_id in enumerate(class_ids):
                per_class_rows.append({
                    "condition": f"{lam:.2f}_REU",
                    "lambda_value": lam,
                    "repeat_id": repeat_id,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id, f"Rock class {class_id}"
                    ),
                    "precision": float(metrics["per_class_precision"][j]),
                    "recall": float(metrics["per_class_recall"][j]),
                    "f1": float(metrics["per_class_f1"][j]),
                })

    repeat_df = pd.DataFrame(repeat_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    repeat_df.to_csv(
        OUTPUT_ROOT / "classification_repeat_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    per_class_df.to_csv(
        OUTPUT_ROOT / "per_class_f1_repeat.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.5 Full-training-region prototype reference
    # -------------------------------------------------------------------------
    full_prototypes = np.vstack([
        full_train_prototypes[c] for c in class_ids
    ])
    y_pred_full = predict_by_sam(X_test, full_prototypes, class_ids)
    cm_full = confusion_matrix_fixed(y_test, y_pred_full, class_ids)
    metrics_full = classification_metrics_from_cm(cm_full)

    full_training_pixels = int(
        sum(int(partitions[c].train_mask.sum()) for c in class_ids)
    )

    full_reference_df = pd.DataFrame([{
        "condition": "FULL_TRAIN",
        "oa": metrics_full["oa"],
        "macro_precision": metrics_full["macro_precision"],
        "macro_recall": metrics_full["macro_recall"],
        "macro_f1": metrics_full["macro_f1"],
        "prototype_training_pixels_total": full_training_pixels,
        "test_pixels_total": len(y_test),
    }])
    full_reference_df.to_csv(
        OUTPUT_ROOT / "full_train_reference_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.6 Summaries
    # -------------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []

    for lam in LAMBDA_VALUES:
        sub = repeat_df[np.isclose(repeat_df["lambda_value"], lam)]
        summary_rows.append({
            "condition": f"{lam:.2f}_REU",
            "lambda_value": lam,
            "n_repeats": len(sub),
            "oa_mean": float(sub["oa"].mean()),
            "oa_std": float(sub["oa"].std(ddof=1)),
            "oa_q25": float(sub["oa"].quantile(0.25)),
            "oa_median": float(sub["oa"].median()),
            "oa_q75": float(sub["oa"].quantile(0.75)),
            "macro_f1_mean": float(sub["macro_f1"].mean()),
            "macro_f1_std": float(sub["macro_f1"].std(ddof=1)),
            "macro_f1_q25": float(sub["macro_f1"].quantile(0.25)),
            "macro_f1_median": float(sub["macro_f1"].median()),
            "macro_f1_q75": float(sub["macro_f1"].quantile(0.75)),
            "macro_recall_mean": float(sub["macro_recall"].mean()),
            "prototype_training_pixels_median": float(
                sub["prototype_training_pixels_total"].median()
            ),
            "prototype_training_pixels_min": int(
                sub["prototype_training_pixels_total"].min()
            ),
            "prototype_training_pixels_max": int(
                sub["prototype_training_pixels_total"].max()
            ),
            "training_fraction_vs_full_median": float(
                sub["prototype_training_pixels_total"].median()
                / full_training_pixels
            ),
            "macro_f1_fraction_of_full_reference": float(
                sub["macro_f1"].median()
                / max(metrics_full["macro_f1"], 1.0e-12)
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_ROOT / "classification_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    per_class_summary = (
        per_class_df
        .groupby(["lambda_value", "class_id", "class_name"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_median=("f1", "median"),
        )
    )
    per_class_summary.to_csv(
        OUTPUT_ROOT / "per_class_f1_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.7 Mean confusion matrix at REU and full-train reference
    # -------------------------------------------------------------------------
    cm_reu_mean = np.mean(
        np.stack(cm_by_lambda[1.0], axis=0).astype(np.float64),
        axis=0,
    )
    row_sum = cm_reu_mean.sum(axis=1, keepdims=True)
    cm_reu_norm = np.divide(
        cm_reu_mean,
        row_sum,
        out=np.zeros_like(cm_reu_mean),
        where=row_sum > 0,
    )

    cm_full_float = cm_full.astype(np.float64)
    row_sum_full = cm_full_float.sum(axis=1, keepdims=True)
    cm_full_norm = np.divide(
        cm_full_float,
        row_sum_full,
        out=np.zeros_like(cm_full_float),
        where=row_sum_full > 0,
    )

    class_labels = [CLASS_NAMES[c] for c in class_ids]
    pd.DataFrame(
        cm_reu_norm,
        index=class_labels,
        columns=class_labels,
    ).to_csv(
        OUTPUT_ROOT / "confusion_matrix_REU_mean.csv",
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        cm_full_norm,
        index=class_labels,
        columns=class_labels,
    ).to_csv(
        OUTPUT_ROOT / "confusion_matrix_FULL_TRAIN.csv",
        encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6.8 Figures
    # -------------------------------------------------------------------------
    setup_plot_fonts()

    # Fig 1: performance versus normalized training support.
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x = summary_df["lambda_value"].to_numpy(float)

    ax.errorbar(
        x,
        summary_df["oa_mean"].to_numpy(float),
        yerr=summary_df["oa_std"].to_numpy(float),
        marker="o",
        capsize=3,
        linewidth=1.3,
        label="OA",
    )
    ax.errorbar(
        x,
        summary_df["macro_f1_mean"].to_numpy(float),
        yerr=summary_df["macro_f1_std"].to_numpy(float),
        marker="s",
        capsize=3,
        linewidth=1.3,
        label="Macro-F1",
    )

    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(metrics_full["oa"], linestyle=":", linewidth=1.0)
    ax.axhline(metrics_full["macro_f1"], linestyle="-.", linewidth=1.0)

    ax.set_xlabel(r"Prototype support, $L/L_{\mathrm{REU}}$")
    ax.set_ylabel("Classification performance")
    ax.set_xlim(min(LAMBDA_VALUES) - 0.05, max(LAMBDA_VALUES) + 0.05)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_ROOT / "Fig_1_classification_vs_REU_normalized_support.jpg",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Fig 2: compact efficiency trade-off.
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    xp = summary_df["prototype_training_pixels_median"].to_numpy(float)
    yp = summary_df["macro_f1_median"].to_numpy(float)

    ax.plot(xp, yp, marker="o", linewidth=1.3)
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{row['lambda_value']:.2g} REU",
            (row["prototype_training_pixels_median"], row["macro_f1_median"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.scatter(
        [full_training_pixels],
        [metrics_full["macro_f1"]],
        marker="D",
        s=35,
        label="Full training region",
    )

    ax.set_xlabel("Labelled pixels used to construct five prototypes")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_ROOT / "Fig_2_accuracy_efficiency_tradeoff.jpg",
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6.9 Configuration / provenance
    # -------------------------------------------------------------------------
    save_json(
        OUTPUT_ROOT / "downstream_config.json",
        {
            "data_path": DATA_PATH,
            "label_path": LABEL_PATH,
            "cube_shape": cube.shape,
            "frozen_rock_reu_px": ROCK_REU_PX,
            "class_names": CLASS_NAMES,
            "lambda_values": LAMBDA_VALUES,
            "train_fraction": TRAIN_FRACTION,
            "buffer_fraction": BUFFER_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - BUFFER_FRACTION,
            "min_train_coverage": MIN_TRAIN_COVERAGE,
            "candidate_stride_px": CANDIDATE_STRIDE_PX,
            "n_repeats": N_REPEATS,
            "max_test_pixels_per_class": MAX_TEST_PIXELS_PER_CLASS,
            "classifier": "nearest raw spectral prototype by SAM",
            "test_input": "fixed single-pixel spectra, spatially separated",
            "full_train_reference": {
                "oa": metrics_full["oa"],
                "macro_f1": metrics_full["macro_f1"],
                "training_pixels": full_training_pixels,
            },
            "scientific_role": (
                "Downstream proof-of-concept only.  REU is frozen before "
                "classification and classification performance is not used "
                "to select or tune REU."
            ),
        },
    )

    # -------------------------------------------------------------------------
    # 6.10 Console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DOWNSTREAM CLASSIFICATION SUMMARY")
    print("=" * 100)
    print(
        summary_df[
            [
                "condition",
                "oa_mean",
                "oa_std",
                "macro_f1_mean",
                "macro_f1_std",
                "prototype_training_pixels_median",
                "training_fraction_vs_full_median",
                "macro_f1_fraction_of_full_reference",
            ]
        ].to_string(index=False)
    )

    print("\nFULL-TRAIN REFERENCE")
    print(full_reference_df.to_string(index=False))

    print("\nInterpretation target:")
    print(
        "Use this experiment only to show whether an REU-referenced labelled "
        "support can construct useful spectral prototypes with much less "
        "training area than the full training region.  Do not claim SOTA."
    )

    print("\n" + "=" * 100)
    print("FINISHED")
    print(f"Runtime: {time.time() - t0:.1f} s")
    print(f"Results: {OUTPUT_ROOT}")
    print("Primary files:")
    print("  classification_summary.csv")
    print("  classification_repeat_metrics.csv")
    print("  prototype_window_audit.csv")
    print("  per_class_f1_summary.csv")
    print("  full_train_reference_metrics.csv")
    print("  Fig_1_classification_vs_REU_normalized_support.jpg")
    print("  Fig_2_accuracy_efficiency_tradeoff.jpg")
    print("=" * 100)


if __name__ == "__main__":
    main()
