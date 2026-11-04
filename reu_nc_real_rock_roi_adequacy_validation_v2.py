# -*- coding: utf-8 -*-
"""
Real-rock ROI adequacy / spectral-statistical stability validation for NC-REU
=============================================================================

Scientific purpose
------------------
This script performs ONE deliberately simple real-data validation:

    Do real rock hyperspectral regions show the same reduction in observation-
    location dependence and spectral-statistical error as spatial support
    approaches the already-estimated REU?

This is NOT another REU estimator. REU values are frozen before this analysis.
No V2.7 threshold is retuned and no PCA / soft-state / REU metric is reused.

For each of the five rock regions, the script evaluates square observation
supports at

    lambda = L / L_REU = 0.25, 0.50, 0.75, 1.00, 1.25, 1.50

and compares every local ROI against the complete connected rock region using
independent RAW-SPECTRUM statistics:

1) Mean-spectrum spectral angle (SAM, degrees)
2) Mean-spectrum relative RMSE
3) Spectral standard-deviation-profile relative RMSE
4) Across-position dispersion of ROI mean spectra

The first three are evaluated for each ROI and summarized by q50/q95. The
fourth directly quantifies how strongly the estimated mean spectrum changes
with ROI location at each support.

Interpretation
--------------
The intended test is not that error becomes exactly zero at lambda=1. Instead:

    lambda < 1  : sub-representative support, stronger location dependence
    lambda ~ 1  : statistically representative regime is reached
    lambda > 1  : accuracy may continue to improve, but marginal gain may fall

The analysis also compares equal support increments:

    0.5 -> 1.0 REU    versus    1.0 -> 1.5 REU

to quantify diminishing marginal improvement without introducing a new
"plateau threshold".

Important design rules
----------------------
- Frozen REUs are taken from the final rock V2.7 results:
      class 1 = 128 px
      class 2 = 160 px
      class 3 =  80 px
      class 4 =  96 px
      class 5 =  56 px
- The largest 8-connected component of each rock label is used.
- Candidate square ROIs require >= 98% target-region coverage.
- ROI positions are selected deterministically and WITHOUT looking at spectra.
- The full connected rock region is the independent reference.
- No classification, clustering, PCA, soft-state model, anisotropy, bootstrap,
  knee, plateau detector or new REU threshold is used here.

Default data
------------
    D:\\LS_REU\\data\\5rock_mat\\5rock.mat
    D:\\LS_REU\\data\\5rock_mat\\5rock_label.mat

Run
---
    D:\\Anaconda\\envs\\DL\\python.exe reu_nc_real_rock_roi_adequacy_validation.py
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
from scipy.stats import wilcoxon


# =============================================================================
# 0. USER SETTINGS — freeze for reported experiment
# =============================================================================

DATA_PATH = Path(r"D:\LS_REU\data\5rock_mat\5rock.mat")
LABEL_PATH = Path(r"D:\LS_REU\data\5rock_mat\5rock_label.mat")
DATA_KEY = "data"
LABEL_KEY = "data"

OUTPUT_ROOT = Path(r"D:\LS_REU\results\nc_reu_real_rock_roi_adequacy")

# Frozen final rock V2.7 REU medians. DO NOT change after seeing validation.
ROCK_REU_PX: Mapping[int, int] = {
    1: 128,
    2: 160,
    3: 80,
    4: 96,
    5: 56,
}

# Replace names below with actual lithology names if desired.
CLASS_NAMES: Mapping[int, str] = {
    1: "Rock class 1",
    2: "Rock class 2",
    3: "Rock class 3",
    4: "Rock class 4",
    5: "Rock class 5",
}

# Prior NC-REU outputs used 0.1 mm per pixel. Set None for pixel-only reporting.
PIXEL_SIZE_MM: Optional[float] = 0.1

LAMBDA_VALUES: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

MIN_REGION_COVERAGE = 0.98
MAX_ROIS_PER_SCALE = 120
MIN_ROIS_FOR_Q95 = 12
MAX_POSITION_STRIDE_PX = 8
RANDOM_SEED = 3407

WAVELENGTH_CANDIDATE_KEYS = (
    "wavelength", "wavelengths", "wl", "lambda", "bands", "band_centers"
)

FIG_DPI = 600


# =============================================================================
# 1. UTILITIES
# =============================================================================

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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
    """
    Robust MAT-array loader.

    Important:
    MATLAB labels are often saved as HxWx1 or 1xHxW.  Those singleton axes do
    not carry spatial information, so they are safely removed before checking
    dimensionality.  This fixes files such as:

        5rock_label.mat -> shape (2279, 316, 1)

    which should be interpreted as a 2-D label image of shape (2279, 316).
    """
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
            a_squeezed = np.squeeze(a)

            # Accept arrays that become the requested dimensionality after
            # removal of singleton MATLAB dimensions.
            if a_squeezed.ndim == ndim:
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

    # Remove only singleton dimensions.  HxWx1 -> HxW; 1xHxW -> HxW.
    arr = np.squeeze(arr)

    if arr.ndim != ndim:
        raise ValueError(
            f"{path}: variable '{selected_name}' expected {ndim}D after "
            f"removing singleton dimensions, but original shape was "
            f"{original_shape} and squeezed shape is {arr.shape}."
        )

    print(
        f"[MAT loader] {path.name}: selected variable '{selected_name}', "
        f"original shape={original_shape}, loaded shape={arr.shape}, "
        f"dtype={arr.dtype}"
    )
    return np.asarray(arr)


def try_load_wavelengths(path: Path, n_bands: int) -> Optional[np.ndarray]:
    try:
        data = loadmat(path)
    except Exception:
        return None
    lower_map = {str(k).lower(): k for k in data.keys() if not str(k).startswith("__")}
    for candidate in WAVELENGTH_CANDIDATE_KEYS:
        key = lower_map.get(candidate.lower())
        if key is None:
            continue
        arr = np.asarray(data[key]).squeeze()
        if arr.ndim == 1 and len(arr) == n_bands and np.all(np.isfinite(arr)):
            return arr.astype(np.float64)
    return None


def largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labelled, n = ndi_label(mask.astype(bool), structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0
    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    cid = int(np.argmax(counts))
    return labelled == cid, cid, int(n)


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


def spectral_angle_deg(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = a @ b
    den = np.linalg.norm(a, axis=1) * max(float(np.linalg.norm(b)), eps)
    cosv = np.clip(num / np.maximum(den, eps), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def relative_rmse_rows(a: np.ndarray, ref: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    numerator = np.sqrt(np.mean((a - ref[None, :]) ** 2, axis=1))
    denominator = max(float(np.sqrt(np.mean(ref ** 2))), eps)
    return numerator / denominator


def quantile_safe(values: Sequence[float], probability: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, probability))


# =============================================================================
# 2. GEOMETRY-ONLY ROI SAMPLING
# =============================================================================

@dataclass
class ROIWindow:
    class_id: int
    class_name: str
    reu_px: int
    lambda_target: float
    support_px: int
    lambda_actual: float
    y0: int
    x0: int
    y1: int
    x1: int
    valid_pixel_count: int
    coverage: float


def generate_valid_rois(
    component_mask: np.ndarray,
    class_id: int,
    class_name: str,
    reu_px: int,
    lambda_target: float,
) -> List[ROIWindow]:
    h, w = component_mask.shape
    size = max(2, int(round(float(reu_px) * float(lambda_target))))
    if size > h or size > w:
        return []

    max_y = h - size
    max_x = w - size
    stride = max(1, min(MAX_POSITION_STRIDE_PX, max(1, size // 4)))

    ys = list(range(0, max_y + 1, stride))
    xs = list(range(0, max_x + 1, stride))
    if ys[-1] != max_y:
        ys.append(max_y)
    if xs[-1] != max_x:
        xs.append(max_x)

    y0_grid, x0_grid = np.meshgrid(
        np.asarray(ys, dtype=np.int32),
        np.asarray(xs, dtype=np.int32),
        indexing="ij",
    )
    y0 = y0_grid.ravel()
    x0 = x0_grid.ravel()
    y1 = y0 + size
    x1 = x0 + size

    ii = integral_image_2d(component_mask.astype(np.float64))
    counts = rect_sum(ii, y0, x0, y1, x1)
    coverage = counts / float(size * size)

    valid = np.where(coverage >= MIN_REGION_COVERAGE)[0]
    if len(valid) == 0:
        return []

    if len(valid) > MAX_ROIS_PER_SCALE:
        rng = np.random.default_rng(stable_seed(
            RANDOM_SEED, "roi_positions", class_id, reu_px, lambda_target, size
        ))
        chosen = np.sort(rng.choice(valid, size=MAX_ROIS_PER_SCALE, replace=False))
    else:
        chosen = valid

    out: List[ROIWindow] = []
    for idx in chosen:
        out.append(ROIWindow(
            class_id=int(class_id),
            class_name=str(class_name),
            reu_px=int(reu_px),
            lambda_target=float(lambda_target),
            support_px=int(size),
            lambda_actual=float(size / float(reu_px)),
            y0=int(y0[idx]),
            x0=int(x0[idx]),
            y1=int(y1[idx]),
            x1=int(x1[idx]),
            valid_pixel_count=int(round(float(counts[idx]))),
            coverage=float(coverage[idx]),
        ))
    return out


# =============================================================================
# 3. MEMORY-EFFICIENT RAW-SPECTRUM MOMENTS
# =============================================================================

def compute_reference_and_roi_moments(
    cube: np.ndarray,
    component_mask: np.ndarray,
    windows: Sequence[ROIWindow],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(windows) == 0:
        raise ValueError("No ROI windows supplied.")

    cube = np.asarray(cube)
    mask = np.asarray(component_mask, dtype=bool)
    _, _, b = cube.shape

    y0 = np.asarray([r.y0 for r in windows], dtype=np.int32)
    x0 = np.asarray([r.x0 for r in windows], dtype=np.int32)
    y1 = np.asarray([r.y1 for r in windows], dtype=np.int32)
    x1 = np.asarray([r.x1 for r in windows], dtype=np.int32)
    counts = np.asarray([r.valid_pixel_count for r in windows], dtype=np.float64)

    n = len(windows)
    roi_mean = np.empty((n, b), dtype=np.float64)
    roi_std = np.empty((n, b), dtype=np.float64)
    reference_mean = np.empty(b, dtype=np.float64)
    reference_std = np.empty(b, dtype=np.float64)

    n_ref = int(mask.sum())
    if n_ref < 2:
        raise ValueError("Reference component contains fewer than two valid pixels.")

    for band in range(b):
        z = np.asarray(cube[:, :, band], dtype=np.float64)
        masked = np.where(mask, z, 0.0)
        masked_sq = np.where(mask, z * z, 0.0)

        s_ref = float(masked.sum(dtype=np.float64))
        ss_ref = float(masked_sq.sum(dtype=np.float64))
        mu_ref = s_ref / float(n_ref)
        var_ref = max(ss_ref / float(n_ref) - mu_ref * mu_ref, 0.0)
        reference_mean[band] = mu_ref
        reference_std[band] = math.sqrt(var_ref)

        ii = integral_image_2d(masked)
        ii2 = integral_image_2d(masked_sq)
        sums = rect_sum(ii, y0, x0, y1, x1)
        sums2 = rect_sum(ii2, y0, x0, y1, x1)

        means = sums / np.maximum(counts, 1.0)
        variances = np.maximum(
            sums2 / np.maximum(counts, 1.0) - means * means, 0.0
        )
        roi_mean[:, band] = means
        roi_std[:, band] = np.sqrt(variances)

    return reference_mean, reference_std, roi_mean, roi_std


# =============================================================================
# 4. ONE ROCK CLASS
# =============================================================================

def analyze_one_class(
    cube_full: np.ndarray,
    gt_full: np.ndarray,
    class_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_name = CLASS_NAMES.get(class_id, f"Rock class {class_id}")
    reu_px = int(ROCK_REU_PX[class_id])

    raw_mask = gt_full == class_id
    component, component_id, component_count = largest_connected_component(raw_mask)
    if component.sum() == 0:
        raise RuntimeError(f"Class {class_id}: no labelled pixels.")

    finite_all = np.all(np.isfinite(cube_full), axis=2)
    component = component & finite_all
    if component.sum() == 0:
        raise RuntimeError(f"Class {class_id}: no finite hyperspectral pixels.")

    y0g, x0g, y1g, x1g = bbox_from_mask(component)
    cube = np.asarray(cube_full[y0g:y1g, x0g:x1g, :])
    mask = component[y0g:y1g, x0g:x1g]

    print("\n" + "-" * 100)
    print(
        f"Class {class_id}: {class_name} | frozen REU={reu_px} px | "
        f"component pixels={int(mask.sum())} | crop={mask.shape}"
    )

    all_windows: List[ROIWindow] = []
    for lam in LAMBDA_VALUES:
        rois = generate_valid_rois(
            component_mask=mask,
            class_id=class_id,
            class_name=class_name,
            reu_px=reu_px,
            lambda_target=lam,
        )
        all_windows.extend(rois)
        support_px = max(2, int(round(reu_px * lam)))
        print(
            f"  lambda={lam:>4.2f} | L={support_px:>3d} px | "
            f"valid ROIs={len(rois):>3d}"
        )

    if len(all_windows) == 0:
        raise RuntimeError(f"Class {class_id}: no valid ROI supports.")

    ref_mean, ref_std, roi_mean, roi_std = compute_reference_and_roi_moments(
        cube=cube,
        component_mask=mask,
        windows=all_windows,
    )

    mean_sam = spectral_angle_deg(roi_mean, ref_mean)
    mean_rel_rmse = relative_rmse_rows(roi_mean, ref_mean)
    std_rel_rmse = relative_rmse_rows(roi_std, ref_std)

    position_rows: List[Dict[str, Any]] = []
    for i, win in enumerate(all_windows):
        position_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "reu_px": reu_px,
            "reu_physical_mm": (
                reu_px * PIXEL_SIZE_MM if PIXEL_SIZE_MM is not None else np.nan
            ),
            "lambda_target": win.lambda_target,
            "lambda_actual": win.lambda_actual,
            "support_px": win.support_px,
            "support_physical_mm": (
                win.support_px * PIXEL_SIZE_MM if PIXEL_SIZE_MM is not None else np.nan
            ),
            "roi_id_within_class": i,
            "y0_global": win.y0 + y0g,
            "x0_global": win.x0 + x0g,
            "y1_global": win.y1 + y0g,
            "x1_global": win.x1 + x0g,
            "valid_pixel_count": win.valid_pixel_count,
            "coverage": win.coverage,
            "mean_spectrum_sam_deg": float(mean_sam[i]),
            "mean_spectrum_rel_rmse": float(mean_rel_rmse[i]),
            "std_profile_rel_rmse": float(std_rel_rmse[i]),
        })
    position_df = pd.DataFrame(position_rows)

    summary_rows: List[Dict[str, Any]] = []
    for lam in LAMBDA_VALUES:
        idx = np.asarray(
            [i for i, r in enumerate(all_windows) if abs(r.lambda_target - lam) < 1e-12],
            dtype=np.int64,
        )
        support_px = max(2, int(round(reu_px * lam)))

        if len(idx) == 0:
            summary_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "reu_px": reu_px,
                "lambda_target": lam,
                "lambda_actual": support_px / float(reu_px),
                "support_px": support_px,
                "n_valid_rois": 0,
                "q95_supported": False,
            })
            continue

        local_means = roi_mean[idx]
        position_std_spectrum = np.std(local_means, axis=0, ddof=0)
        position_dispersion_rel = (
            float(np.sqrt(np.mean(position_std_spectrum ** 2)))
            / max(float(np.sqrt(np.mean(ref_mean ** 2))), 1.0e-12)
        )

        sub_sam = mean_sam[idx]
        sub_mean_rmse = mean_rel_rmse[idx]
        sub_std_rmse = std_rel_rmse[idx]

        summary_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "reu_px": reu_px,
            "reu_physical_mm": (
                reu_px * PIXEL_SIZE_MM if PIXEL_SIZE_MM is not None else np.nan
            ),
            "lambda_target": lam,
            "lambda_actual": support_px / float(reu_px),
            "support_px": support_px,
            "support_physical_mm": (
                support_px * PIXEL_SIZE_MM if PIXEL_SIZE_MM is not None else np.nan
            ),
            "n_valid_rois": int(len(idx)),
            "q95_supported": bool(len(idx) >= MIN_ROIS_FOR_Q95),
            "mean_spectrum_sam_q50_deg": quantile_safe(sub_sam, 0.50),
            "mean_spectrum_sam_q95_deg": quantile_safe(sub_sam, 0.95),
            "mean_spectrum_rel_rmse_q50": quantile_safe(sub_mean_rmse, 0.50),
            "mean_spectrum_rel_rmse_q95": quantile_safe(sub_mean_rmse, 0.95),
            "std_profile_rel_rmse_q50": quantile_safe(sub_std_rmse, 0.50),
            "std_profile_rel_rmse_q95": quantile_safe(sub_std_rmse, 0.95),
            "position_mean_spectrum_dispersion_rel": position_dispersion_rel,
        })
    summary_df = pd.DataFrame(summary_rows)

    n_bands = cube.shape[2]
    wavelength = try_load_wavelengths(DATA_PATH, n_bands)
    reference_rows: List[Dict[str, Any]] = []
    for band in range(n_bands):
        reference_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "band_index_0based": band,
            "band_index_1based": band + 1,
            "wavelength_nm": (
                float(wavelength[band]) if wavelength is not None else np.nan
            ),
            "reference_mean_reflectance": float(ref_mean[band]),
            "reference_std_reflectance": float(ref_std[band]),
        })
    reference_df = pd.DataFrame(reference_rows)

    component_audit = pd.DataFrame([{
        "class_id": class_id,
        "class_name": class_name,
        "reu_px": reu_px,
        "reu_physical_mm": (
            reu_px * PIXEL_SIZE_MM if PIXEL_SIZE_MM is not None else np.nan
        ),
        "total_label_pixels": int(raw_mask.sum()),
        "connected_component_count": int(component_count),
        "selected_component_id": int(component_id),
        "selected_component_pixels_finite": int(mask.sum()),
        "bbox_y0": y0g,
        "bbox_x0": x0g,
        "bbox_y1": y1g,
        "bbox_x1": x1g,
        "bbox_height": int(y1g - y0g),
        "bbox_width": int(x1g - x0g),
        "bbox_fill_fraction": float(mask.mean()),
    }])

    return position_df, summary_df, component_audit, reference_df


# =============================================================================
# 5. DIMINISHING RETURNS
# =============================================================================

PRIMARY_METRICS: Mapping[str, str] = {
    "mean_spectrum_sam_q95_deg": "Mean-spectrum SAM q95",
    "mean_spectrum_rel_rmse_q95": "Mean-spectrum relRMSE q95",
    "std_profile_rel_rmse_q95": "Std-profile relRMSE q95",
    "position_mean_spectrum_dispersion_rel": "Position dispersion",
}


def get_scale_value(df: pd.DataFrame, class_id: int, lam: float, column: str) -> float:
    sub = df[
        (df["class_id"] == class_id)
        & np.isclose(df["lambda_target"].astype(float), float(lam))
    ]
    if sub.empty or column not in sub.columns:
        return float("nan")
    return float(pd.to_numeric(sub.iloc[0][column], errors="coerce"))


def diminishing_returns_tables(
    scale_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for class_id in sorted(ROCK_REU_PX):
        for metric_col, metric_label in PRIMARY_METRICS.items():
            e05 = get_scale_value(scale_df, class_id, 0.50, metric_col)
            e10 = get_scale_value(scale_df, class_id, 1.00, metric_col)
            e15 = get_scale_value(scale_df, class_id, 1.50, metric_col)

            pre = (
                (e05 - e10) / e05
                if np.isfinite(e05) and np.isfinite(e10) and e05 > 0 else np.nan
            )
            post = (
                (e10 - e15) / e10
                if np.isfinite(e10) and np.isfinite(e15) and e10 > 0 else np.nan
            )

            rows.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"Rock class {class_id}"),
                "metric": metric_col,
                "metric_label": metric_label,
                "error_at_0p5_REU": e05,
                "error_at_1p0_REU": e10,
                "error_at_1p5_REU": e15,
                "relative_improvement_0p5_to_1p0": pre,
                "relative_improvement_1p0_to_1p5": post,
                "diminishing_return": (
                    bool(pre > post) if np.isfinite(pre) and np.isfinite(post) else False
                ),
                "gain_difference_pre_minus_post": (
                    pre - post if np.isfinite(pre) and np.isfinite(post) else np.nan
                ),
            })

    by_class = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    tests: Dict[str, Any] = {
        "note": (
            "Wilcoxon tests use five rock classes as independent units. "
            "With n=5 these are supportive/descriptive, not primary evidence."
        )
    }

    for metric_col, metric_label in PRIMARY_METRICS.items():
        sub = by_class[by_class["metric"] == metric_col].copy()
        pre = pd.to_numeric(
            sub["relative_improvement_0p5_to_1p0"], errors="coerce"
        ).to_numpy(float)
        post = pd.to_numeric(
            sub["relative_improvement_1p0_to_1p5"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(pre) & np.isfinite(post)

        n = int(valid.sum())
        frac = float(np.mean(pre[valid] > post[valid])) if n else np.nan
        pre_med = float(np.median(pre[valid])) if n else np.nan
        post_med = float(np.median(post[valid])) if n else np.nan

        p_value = np.nan
        statistic = np.nan
        if n >= 3:
            try:
                test = wilcoxon(
                    pre[valid],
                    post[valid],
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except Exception:
                pass

        summary_rows.append({
            "metric": metric_col,
            "metric_label": metric_label,
            "n_classes_complete": n,
            "median_relative_improvement_0p5_to_1p0": pre_med,
            "median_relative_improvement_1p0_to_1p5": post_med,
            "fraction_classes_with_diminishing_return": frac,
            "wilcoxon_greater_statistic": statistic,
            "wilcoxon_greater_p_value": p_value,
        })
        tests[metric_col] = {
            "n": n,
            "alternative": "pre_REU_gain > post_REU_gain",
            "statistic": statistic,
            "p_value": p_value,
        }

    return by_class, pd.DataFrame(summary_rows), tests


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_metric_by_lambda(
    scale_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: Path,
) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    lambda_grid = np.asarray(LAMBDA_VALUES, dtype=float)
    class_matrix: List[np.ndarray] = []

    for class_id in sorted(ROCK_REU_PX):
        sub = scale_df[scale_df["class_id"] == class_id].sort_values("lambda_target")
        x = pd.to_numeric(sub["lambda_target"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)

        ax.plot(
            x, y, marker="o", markersize=3.8, linewidth=1.0, alpha=0.65,
            label=CLASS_NAMES.get(class_id, f"Rock class {class_id}")
        )

        vals = []
        for lam in lambda_grid:
            m = np.where(np.isclose(x, lam))[0]
            vals.append(y[m[0]] if len(m) else np.nan)
        class_matrix.append(np.asarray(vals, dtype=float))

    if class_matrix:
        matrix = np.vstack(class_matrix)
        median = np.nanmedian(matrix, axis=0)
        ax.plot(
            lambda_grid, median, marker="D", markersize=5.0, linewidth=2.0,
            label="Median across rocks"
        )

    ax.axvline(1.0, linestyle="--", linewidth=1.1)
    ax.text(
        1.0, 0.97, "REU", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9
    )
    ax.set_xlabel(r"Normalized observation support, $L/L_{\mathrm{REU}}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(LAMBDA_VALUES) - 0.04, max(LAMBDA_VALUES) + 0.04)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_summary(scale_df: pd.DataFrame, filename: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    lambda_grid = np.asarray(LAMBDA_VALUES, dtype=float)

    display_metrics = [
        ("mean_spectrum_sam_q95_deg", "Mean-spectrum SAM"),
        ("mean_spectrum_rel_rmse_q95", "Mean-spectrum relRMSE"),
        ("std_profile_rel_rmse_q95", "Std-profile relRMSE"),
        ("position_mean_spectrum_dispersion_rel", "Position dispersion"),
    ]

    for metric, label in display_metrics:
        normalized_curves: List[np.ndarray] = []
        for class_id in sorted(ROCK_REU_PX):
            sub = scale_df[
                scale_df["class_id"] == class_id
            ].sort_values("lambda_target")
            values = []
            for lam in lambda_grid:
                row = sub[np.isclose(sub["lambda_target"].astype(float), lam)]
                values.append(
                    float(pd.to_numeric(row.iloc[0][metric], errors="coerce"))
                    if not row.empty else np.nan
                )
            arr = np.asarray(values, dtype=float)
            baseline = arr[0]
            if np.isfinite(baseline) and baseline > 0:
                normalized_curves.append(arr / baseline)

        if normalized_curves:
            med = np.nanmedian(np.vstack(normalized_curves), axis=0)
            ax.plot(
                lambda_grid, med, marker="o", markersize=4.2,
                linewidth=1.5, label=label
            )

    ax.axvline(1.0, linestyle="--", linewidth=1.1)
    ax.text(
        1.0, 0.97, "REU", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9
    )
    ax.set_xlabel(r"Normalized observation support, $L/L_{\mathrm{REU}}$")
    ax.set_ylabel("Error normalized to 0.25 REU")
    ax.set_xlim(min(LAMBDA_VALUES) - 0.04, max(LAMBDA_VALUES) + 0.04)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("NC-REU real-rock ROI adequacy / spectral-statistical stability validation")
    print("Frozen REU -> independent raw-spectrum statistics -> REU-normalized support")
    print(f"Data : {DATA_PATH}")
    print(f"Label: {LABEL_PATH}")
    print(f"REU  : {dict(ROCK_REU_PX)}")
    print(f"lambda values: {LAMBDA_VALUES}")
    print(f"coverage >= {MIN_REGION_COVERAGE:.2f}; max ROIs/scale={MAX_ROIS_PER_SCALE}")
    print("=" * 100)

    cube = load_mat_array(DATA_PATH, DATA_KEY, ndim=3).astype(np.float32, copy=False)
    gt = load_mat_array(LABEL_PATH, LABEL_KEY, ndim=2)

    # Defensive check: labels must now be a genuine 2-D spatial image.
    if gt.ndim != 2:
        raise ValueError(f"Label array must be 2-D after loading, got {gt.shape}")

    if cube.shape[:2] != gt.shape:
        raise ValueError(f"Spatial mismatch: cube={cube.shape[:2]}, labels={gt.shape}")
    gt = np.asarray(np.rint(gt), dtype=np.int32)

    print(f"Cube shape={cube.shape}, dtype={cube.dtype}")
    print(f"Label values={np.unique(gt)}")

    expected = set(ROCK_REU_PX.keys())
    available = set(int(x) for x in np.unique(gt) if int(x) > 0)
    missing = expected - available
    if missing:
        raise ValueError(f"Required rock labels missing from GT: {sorted(missing)}")

    save_json(
        OUTPUT_ROOT / "validation_config.json",
        {
            "data_path": DATA_PATH,
            "label_path": LABEL_PATH,
            "data_key": DATA_KEY,
            "label_key": LABEL_KEY,
            "cube_shape": cube.shape,
            "frozen_rock_reu_px": ROCK_REU_PX,
            "class_names": CLASS_NAMES,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "lambda_values": LAMBDA_VALUES,
            "min_region_coverage": MIN_REGION_COVERAGE,
            "max_rois_per_scale": MAX_ROIS_PER_SCALE,
            "min_rois_for_q95": MIN_ROIS_FOR_Q95,
            "max_position_stride_px": MAX_POSITION_STRIDE_PX,
            "random_seed": RANDOM_SEED,
            "independent_metrics": list(PRIMARY_METRICS.keys()),
            "scientific_note": (
                "This script does not estimate REU. It validates already-frozen "
                "rock REUs using independent raw-spectrum statistics."
            ),
        },
    )

    position_tables: List[pd.DataFrame] = []
    scale_tables: List[pd.DataFrame] = []
    audit_tables: List[pd.DataFrame] = []
    reference_tables: List[pd.DataFrame] = []

    for class_id in sorted(ROCK_REU_PX):
        pos_df, scale_df, audit_df, ref_df = analyze_one_class(cube, gt, class_id)
        position_tables.append(pos_df)
        scale_tables.append(scale_df)
        audit_tables.append(audit_df)
        reference_tables.append(ref_df)

    position_all = pd.concat(position_tables, ignore_index=True)
    scale_all = pd.concat(scale_tables, ignore_index=True)
    audit_all = pd.concat(audit_tables, ignore_index=True)
    reference_all = pd.concat(reference_tables, ignore_index=True)

    position_all.to_csv(
        OUTPUT_ROOT / "rock_roi_position_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    scale_all.to_csv(
        OUTPUT_ROOT / "rock_scale_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    audit_all.to_csv(
        OUTPUT_ROOT / "rock_component_audit.csv",
        index=False, encoding="utf-8-sig"
    )
    reference_all.to_csv(
        OUTPUT_ROOT / "rock_reference_spectral_statistics.csv",
        index=False, encoding="utf-8-sig"
    )

    by_class, diminishing_summary, tests = diminishing_returns_tables(scale_all)
    by_class.to_csv(
        OUTPUT_ROOT / "diminishing_returns_by_class.csv",
        index=False, encoding="utf-8-sig"
    )
    diminishing_summary.to_csv(
        OUTPUT_ROOT / "diminishing_returns_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    save_json(OUTPUT_ROOT / "statistical_tests.json", tests)

    plot_metric_by_lambda(
        scale_all,
        "mean_spectrum_sam_q95_deg",
        "Mean-spectrum SAM q95 (deg)",
        OUTPUT_ROOT / "Fig_1_mean_spectrum_SAM_q95.jpg",
    )
    plot_metric_by_lambda(
        scale_all,
        "mean_spectrum_rel_rmse_q95",
        "Mean-spectrum relative RMSE q95",
        OUTPUT_ROOT / "Fig_2_mean_spectrum_relRMSE_q95.jpg",
    )
    plot_metric_by_lambda(
        scale_all,
        "std_profile_rel_rmse_q95",
        "Spectral std-profile relative RMSE q95",
        OUTPUT_ROOT / "Fig_3_std_profile_relRMSE_q95.jpg",
    )
    plot_metric_by_lambda(
        scale_all,
        "position_mean_spectrum_dispersion_rel",
        "Across-position mean-spectrum dispersion",
        OUTPUT_ROOT / "Fig_4_position_dispersion.jpg",
    )
    plot_normalized_summary(
        scale_all,
        OUTPUT_ROOT / "Fig_5_REU_normalized_spectral_stability.jpg",
    )

    print("\n" + "=" * 100)
    print("ROCK REAL-DATA ADEQUACY SUMMARY")
    print("=" * 100)
    cols = [
        "class_id", "class_name", "lambda_target", "support_px", "n_valid_rois",
        "mean_spectrum_sam_q95_deg",
        "mean_spectrum_rel_rmse_q95",
        "std_profile_rel_rmse_q95",
        "position_mean_spectrum_dispersion_rel",
    ]
    print(scale_all[cols].to_string(index=False))

    print("\nDiminishing returns summary")
    print(diminishing_summary.to_string(index=False))

    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print("FINISHED")
    print(f"Runtime: {elapsed:.1f} s")
    print(f"Results: {OUTPUT_ROOT}")
    print("Primary files:")
    print("  rock_scale_summary.csv")
    print("  diminishing_returns_summary.csv")
    print("  statistical_tests.json")
    print("  Fig_1_mean_spectrum_SAM_q95.jpg")
    print("  Fig_3_std_profile_relRMSE_q95.jpg")
    print("  Fig_4_position_dispersion.jpg")
    print("  Fig_5_REU_normalized_spectral_stability.jpg")
    print("=" * 100)


if __name__ == "__main__":
    main()
