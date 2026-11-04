# -*- coding: utf-8 -*-
"""
reu_downstream_stabilization_v2_rbf_svm.py

NC-REU | Independent downstream stabilization validation v2
============================================================

PURPOSE
-------
This script replaces the v1 nearest-prototype classifier with a fixed nonlinear
RBF-SVM pipeline and tests whether the information-defined REU has independent
consequences for downstream inference.

The downstream classifier NEVER enters the REU calculation.

The v2 experiment tests THREE independent outcomes:

    1. Predictive performance
       Macro-F1 and overall accuracy on one fixed, spatially separated test set.

    2. Prediction instability
       Repeat-to-repeat prediction disagreement / normalized prediction entropy
       on that same fixed test set.

    3. Agreement with a full-training-reservoir reference
       For every repeat, the support-limited model is compared with a paired
       full-training-reservoir model trained with EXACTLY THE SAME number of
       spectra per class.

NON-NEGOTIABLE FAIRNESS CONTROLS
--------------------------------
* Frozen REUs from the main V2.7 analysis:
      R1 = 128 px
      R2 = 160 px
      R3 =  80 px
      R4 =  96 px
      R5 =  56 px

* Same support factors as v1:
      0.25, 0.50, 0.75, 1.00, 1.25, 1.50 REU

* Exactly 128 training spectra per class at EVERY support.

* Exactly 128 training spectra per class for the full-reservoir reference.

* Exactly 800 FIXED test spectra per class.

* Test pixels are selected once from spatially separated held-out reservoirs and
  never reused for fitting the scaler, PCA or SVM.

* Within a repeat and class, all support windows share the SAME center and are
  nested.  The same random priority field is also used to sample the fixed
  number of training spectra at all support sizes and from the full reservoir.

* Classifier architecture and hyperparameters are fixed before inspecting v2:
      per-spectrum L2 normalization
      StandardScaler
      PCA(n_components=32, svd_solver="full")
      RBF-SVM(C=1.0, gamma="scale")

  There is NO grid search, cross-validation, support-dependent tuning, REU
  tuning, threshold tuning or selection of a best repeat.

SCIENTIFIC INTERPRETATION
-------------------------
The desired evidence is NOT that Macro-F1 must peak exactly at 1 REU.

The stronger and more appropriate downstream signature is that, as support
approaches the frozen REU:

    * predictive performance approaches the paired full-reservoir reference;
    * predictions become less sensitive to which local support is sampled;
    * support-limited predictions increasingly agree with the paired
      full-reservoir predictions.

A non-confirmed stabilization is retained as a valid result.  This script never
changes the REU or classifier after seeing the outcome.

EXPECTED INPUT
--------------
D:\\LS_REU\\data\\5rock_mat\\5rock.mat
D:\\LS_REU\\data\\5rock_mat\\5rock_label.mat

Expected cube shape:
    (2279, 316, 272)

Expected label shape:
    (2279, 316)

The MAT loader is robust to conventional scipy-readable MAT files and MATLAB
v7.3/HDF5 files.  The numeric variable is selected automatically, preferring
the variable named "data".

OUTPUT
------
D:\\LS_REU\\results\\nc_reu_downstream_stabilization_v2_rbf_svm\\

Main outputs:
    validation_design_predeclared_v2.json
    spatial_split_audit.csv
    fixed_test_set_audit.csv
    training_center_audit.csv
    training_window_audit.csv

    classification_repeat_metrics.csv
    classification_summary.csv
    full_reference_repeat_metrics.csv
    full_reference_summary.csv
    per_class_f1_repeat.csv
    per_class_f1_summary.csv

    full_field_agreement_repeat.csv
    full_field_agreement_summary.csv
    prediction_instability_summary.csv

    prediction_matrices.npz
    downstream_v2_assessment.json
    RESULTS_INTERPRETATION_v2.txt

    Downstream_REU_stabilization_v2.png
    Downstream_REU_stabilization_v2.pdf
    Downstream_REU_stabilization_v2.svg
    Downstream_REU_stabilization_v2.tiff

RUN
---
D:\\Anaconda\\envs\\DL\\python.exe ^
    D:\\LS_REU\\260909\\reu_downstream_stabilization_v2_rbf_svm.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Limit BLAS oversubscription before NumPy / sklearn imports.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.io import loadmat

try:
    import h5py
except Exception:
    h5py = None

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC


# =============================================================================
# 0. PREDECLARED EXPERIMENT SETTINGS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("NC_REU_ROOT", r"D:\LS_REU"))

DATA_PATH = PROJECT_ROOT / "data" / "5rock_mat" / "5rock.mat"
LABEL_PATH = PROJECT_ROOT / "data" / "5rock_mat" / "5rock_label.mat"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "nc_reu_downstream_stabilization_v2_rbf_svm"
)

# Frozen main-analysis REUs.  They MUST NOT be re-estimated by this script.
FROZEN_REU_PX: Dict[int, int] = {
    1: 128,
    2: 160,
    3: 80,
    4: 96,
    5: 56,
}

ROCK_NAMES: Dict[int, str] = {
    1: "Pyroxene pegmatite",
    2: "Breccia",
    3: "Coarse-grained biotite granite",
    4: "Banded migmatite",
    5: "Biotite monzogranite",
}

CLASS_IDS: Tuple[int, ...] = tuple(sorted(FROZEN_REU_PX))

# Retained exactly from v1 to avoid post-hoc support selection.
SUPPORT_FACTORS: Tuple[float, ...] = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
)

N_REPEATS = 20
N_TRAIN_PER_CLASS = 128
N_TEST_PER_CLASS = 800

# A nominal window must be overwhelmingly occupied by valid pixels of the
# target class and training reservoir.
MIN_WINDOW_CLASS_COVERAGE = 0.98

# Spatial split families retained from v1.
PRIMARY_SPLIT = {
    "name": "primary",
    "train_fraction": 0.72,
    "test_fraction": 0.18,
    "gap_fraction": 0.10,
}

FALLBACK_SPLIT = {
    "name": "fallback",
    "train_fraction": 0.80,
    "test_fraction": 0.15,
    "gap_fraction": 0.05,
}

# Classifier is FIXED.  No tuning and no CV are performed.
L2_NORMALIZE_EACH_SPECTRUM = True
PCA_N_COMPONENTS = 32
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_KERNEL = "rbf"
SVM_CLASS_WEIGHT = None

# Descriptive bootstrap only; not used to choose support or REU.
BOOTSTRAP_ITERATIONS = 5000

# Kept only as a descriptive continuity check with v1.
# It is NOT used to tune the classifier or REU.
F1_REFERENCE_MARGIN = 0.010

RANDOM_SEED = 3407

# Figure settings.
DPI = 600
SAVE_TIFF = True

# Save predictions for complete reproducibility.
SAVE_PREDICTIONS = True


# =============================================================================
# 1. UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2)


def stable_seed(*tokens: Any) -> int:
    """
    Deterministic FNV-1a-like seed generator.
    """
    x = 2166136261
    for token in tokens:
        for byte in str(token).encode("utf-8"):
            x ^= int(byte)
            x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def support_px(class_id: int, factor: float) -> int:
    """
    Convert lambda = L / L_REU to the nominal square side in pixels.
    """
    return max(
        1,
        int(round(float(FROZEN_REU_PX[class_id]) * float(factor))),
    )


def setup_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.85,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.dpi": DPI,
    })


def clean_axes(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=3.2,
        width=0.8,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(values),
        size=(int(n_boot), len(values)),
    )
    boot = values[idx].mean(axis=1)

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def bootstrap_median_ci(
    values: Sequence[float],
    n_boot: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(values),
        size=(int(n_boot), len(values)),
    )
    boot = np.median(values[idx], axis=1)

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def paired_bootstrap_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """
    Returns median(a-b), 95% bootstrap CI of that paired median difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]

    if len(d) == 0:
        return np.nan, np.nan, np.nan

    med = float(np.median(d))

    if len(d) == 1:
        return med, med, med

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    boot = np.median(d[idx], axis=1)

    return (
        med,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


# =============================================================================
# 2. ROBUST MAT LOADER
# =============================================================================

def _normalize_candidate_array(
    value: Any,
    desired_ndim: int,
) -> Optional[np.ndarray]:
    """
    Convert a MAT candidate to a numeric ndarray and remove singleton
    dimensions before checking dimensionality.

    This specifically fixes label MAT files stored as, for example:
        (1, H, W)
        (H, W, 1)
        [[numeric_array]]  # MATLAB cell/object wrapper
    """
    try:
        arr = np.asarray(value)
    except Exception:
        return None

    # Direct numeric ndarray.
    if np.issubdtype(arr.dtype, np.number):
        arr = np.squeeze(arr)

        if arr.ndim == desired_ndim and arr.size > 0:
            return arr

    return None


def _recursive_numeric_candidates(
    value: Any,
    desired_ndim: int,
    path: str,
    out: List[Tuple[str, np.ndarray]],
    depth: int = 0,
    max_depth: int = 6,
) -> None:
    """
    Recursively unwrap common MATLAB containers:
      - numeric ndarrays
      - singleton/object cell arrays
      - scipy mat_struct objects
      - structured arrays / np.void
      - nested dicts

    The recursion is deliberately shallow and only records arrays that become
    exactly `desired_ndim` after singleton dimensions are removed.
    """
    if depth > max_depth:
        return

    # 1) Direct numeric array.
    arr = _normalize_candidate_array(
        value,
        desired_ndim=desired_ndim,
    )

    if arr is not None:
        out.append((path, arr))
        return

    # 2) Dict-like structure.
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).startswith("__"):
                continue

            _recursive_numeric_candidates(
                child,
                desired_ndim=desired_ndim,
                path=f"{path}.{key}",
                out=out,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    # 3) scipy.io.matlab mat_struct-like object.
    fieldnames = getattr(
        value,
        "_fieldnames",
        None,
    )

    if fieldnames:
        for field in fieldnames:
            try:
                child = getattr(value, field)
            except Exception:
                continue

            _recursive_numeric_candidates(
                child,
                desired_ndim=desired_ndim,
                path=f"{path}.{field}",
                out=out,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    # 4) Structured scalar / structured ndarray.
    if isinstance(value, np.void) and value.dtype.names:
        for field in value.dtype.names:
            _recursive_numeric_candidates(
                value[field],
                desired_ndim=desired_ndim,
                path=f"{path}.{field}",
                out=out,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    if isinstance(value, np.ndarray):

        if value.dtype.names:
            for field in value.dtype.names:
                try:
                    child = value[field]
                except Exception:
                    continue

                _recursive_numeric_candidates(
                    child,
                    desired_ndim=desired_ndim,
                    path=f"{path}.{field}",
                    out=out,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            return

        # MATLAB cell arrays normally arrive as dtype=object.
        if value.dtype == object:
            # Do not recursively traverse a huge object array.
            # Label/cell wrappers are normally tiny; cap inspection.
            flat = value.ravel()

            if flat.size <= 256:
                for i, child in enumerate(flat):
                    _recursive_numeric_candidates(
                        child,
                        desired_ndim=desired_ndim,
                        path=f"{path}[{i}]",
                        out=out,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
            return


def _choose_numeric_array_from_dict(
    mapping: Mapping[str, Any],
    desired_ndim: int,
    preferred_name: str = "data",
) -> Tuple[str, np.ndarray]:
    """
    Robust scipy-MAT variable selector.

    Compared with the original v2 loader, this version:
      1) applies np.squeeze BEFORE dimensionality testing;
      2) unwraps MATLAB cell / struct wrappers recursively;
      3) strongly prefers the requested variable name ('data');
      4) emits the available top-level variable metadata if no candidate exists.
    """
    candidates: List[Tuple[str, np.ndarray]] = []

    # First inspect the preferred top-level variable so that a nested numeric
    # array inside `data` is preferred over unrelated MAT metadata.
    ordered_keys = list(mapping.keys())

    ordered_keys.sort(
        key=lambda k: (
            0 if str(k) == preferred_name else 1,
            str(k),
        )
    )

    for key in ordered_keys:
        if str(key).startswith("__"):
            continue

        _recursive_numeric_candidates(
            mapping[key],
            desired_ndim=desired_ndim,
            path=str(key),
            out=candidates,
        )

    if not candidates:
        diagnostics = []

        for key, value in mapping.items():
            if str(key).startswith("__"):
                continue

            try:
                arr = np.asarray(value)

                diagnostics.append({
                    "name": str(key),
                    "shape": tuple(arr.shape),
                    "dtype": str(arr.dtype),
                    "ndim": int(arr.ndim),
                })
            except Exception:
                diagnostics.append({
                    "name": str(key),
                    "shape": None,
                    "dtype": type(value).__name__,
                    "ndim": None,
                })

        raise ValueError(
            f"No numeric array reducible to {desired_ndim}D was found in the MAT file.\n"
            f"Top-level variables: {diagnostics}"
        )

    def candidate_score(
        item: Tuple[str, np.ndarray],
    ) -> Tuple[int, int, int, str]:
        path, arr = item

        root = path.split(".")[0].split("[")[0]

        preferred_root = (
            0 if root == preferred_name else 1
        )

        exact_preferred_path = (
            0 if path == preferred_name else 1
        )

        # Larger arrays are normally the actual image/label data.
        size_score = -int(arr.size)

        return (
            preferred_root,
            exact_preferred_path,
            size_score,
            path,
        )

    candidates.sort(
        key=candidate_score
    )

    path, arr = candidates[0]

    return (
        path,
        np.asarray(arr),
    )


def _load_hdf5_mat(
    path: Path,
    desired_ndim: int,
    preferred_name: str = "data",
) -> Tuple[str, np.ndarray]:
    """
    Robust MATLAB v7.3/HDF5 loader.

    Singleton dimensions are removed before dimensionality testing.
    For MATLAB v7.3 datasets, axes are reversed first (MATLAB HDF5 storage
    convention), then singleton axes are squeezed.
    """
    if h5py is None:
        raise RuntimeError(
            "This MAT file appears to require h5py, but h5py is unavailable."
        )

    datasets: List[Tuple[str, np.ndarray]] = []

    with h5py.File(path, "r") as f:

        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return

            try:
                raw = np.asarray(obj)
            except Exception:
                return

            if not np.issubdtype(raw.dtype, np.number):
                return

            # MATLAB v7.3 numeric datasets usually need axis reversal.
            if raw.ndim >= 2:
                arr = np.transpose(
                    raw,
                    axes=tuple(
                        reversed(
                            range(raw.ndim)
                        )
                    ),
                )
            else:
                arr = raw

            arr = np.squeeze(arr)

            if (
                arr.ndim == desired_ndim
                and arr.size > 0
            ):
                datasets.append(
                    (name, arr)
                )

        f.visititems(visitor)

    if not datasets:
        with h5py.File(path, "r") as f:
            diagnostics: List[Dict[str, Any]] = []

            def diagnostic_visitor(
                name: str,
                obj: Any,
            ) -> None:
                if isinstance(obj, h5py.Dataset):
                    diagnostics.append({
                        "name": name,
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                    })

            f.visititems(
                diagnostic_visitor
            )

        raise ValueError(
            f"No numeric HDF5 dataset reducible to {desired_ndim}D was found.\n"
            f"Datasets: {diagnostics}"
        )

    def score(
        item: Tuple[str, np.ndarray],
    ) -> Tuple[int, int, str]:
        name, arr = item

        leaf = name.split("/")[-1]

        return (
            0 if leaf == preferred_name else 1,
            -int(arr.size),
            name,
        )

    datasets.sort(
        key=score
    )

    return datasets[0]


def load_mat_numeric(
    path: Path,
    desired_ndim: int,
    preferred_name: str = "data",
) -> Tuple[str, np.ndarray]:
    """
    Load a numeric MATLAB array robustly.

    scipy-readable MAT is attempted first.  If it does not contain a suitable
    candidate, HDF5/v7.3 loading is attempted only when the file is actually
    HDF5-readable.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}"
        )

    scipy_error: Optional[Exception] = None

    try:
        # struct_as_record=False makes MATLAB structs easier to unwrap.
        mat = loadmat(
            path,
            struct_as_record=False,
            squeeze_me=False,
        )

        name, arr = (
            _choose_numeric_array_from_dict(
                mat,
                desired_ndim=desired_ndim,
                preferred_name=preferred_name,
            )
        )

        return (
            name,
            np.asarray(arr),
        )

    except (
        NotImplementedError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        scipy_error = exc

    # Only try HDF5 if h5py can really identify the file as HDF5.
    if h5py is not None:
        try:
            if h5py.is_hdf5(str(path)):
                return _load_hdf5_mat(
                    path,
                    desired_ndim=desired_ndim,
                    preferred_name=preferred_name,
                )
        except Exception:
            pass

    raise ValueError(
        f"Failed to load a {desired_ndim}D numeric array from:\n"
        f"  {path}\n\n"
        f"scipy loader error:\n"
        f"  {scipy_error}\n\n"
        "The loader now supports squeezed singleton dimensions and common "
        "MATLAB cell/struct wrappers. If this error remains, the printed "
        "variable diagnostics will show the actual storage layout."
    )


def align_cube_and_labels(
    cube: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    cube = np.asarray(cube)
    labels = np.asarray(labels).squeeze()

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D label array; got shape {labels.shape}."
        )

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D hyperspectral cube; got shape {cube.shape}."
        )

    if cube.shape[:2] == labels.shape:
        return cube, labels

    # Try common alternatives.
    candidates = [
        np.transpose(cube, (1, 0, 2)),
        np.transpose(cube, (2, 1, 0)),
        np.transpose(cube, (2, 0, 1)),
        np.transpose(cube, (0, 2, 1)),
        np.transpose(cube, (1, 2, 0)),
    ]

    for candidate in candidates:
        if candidate.shape[:2] == labels.shape:
            return candidate, labels

    # Try transposed labels.
    if cube.shape[:2] == labels.T.shape:
        return cube, labels.T

    for candidate in candidates:
        if candidate.shape[:2] == labels.T.shape:
            return candidate, labels.T

    raise ValueError(
        "Could not align cube and labels.\n"
        f"Cube shape: {cube.shape}\n"
        f"Label shape: {labels.shape}"
    )


def load_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    cube_var, cube = load_mat_numeric(
        DATA_PATH,
        desired_ndim=3,
        preferred_name="data",
    )

    label_var, labels = load_mat_numeric(
        LABEL_PATH,
        desired_ndim=2,
        preferred_name="data",
    )

    cube, labels = align_cube_and_labels(cube, labels)

    cube = np.asarray(cube, dtype=np.float32)
    labels = np.rint(np.asarray(labels)).astype(np.int16)

    # Pixel-level validity independent of class semantics.
    finite = np.all(np.isfinite(cube), axis=2)
    norm = np.linalg.norm(
        np.nan_to_num(cube, nan=0.0),
        axis=2,
    )
    nondegenerate = norm > 1.0e-10

    valid = finite & nondegenerate

    # Keep invalid spectral pixels labelled as background for all later masks.
    labels = labels.copy()
    labels[~valid] = 0

    info = {
        "cube_variable": cube_var,
        "label_variable": label_var,
        "cube_shape": cube.shape,
        "label_shape": labels.shape,
        "dtype": str(cube.dtype),
        "valid_spectral_pixels": int(valid.sum()),
        "class_pixel_counts_after_spectral_validity": {
            int(cid): int(np.sum(labels == cid))
            for cid in CLASS_IDS
        },
    }

    return cube, labels, info


# =============================================================================
# 3. CLASS COMPONENTS AND SPATIAL SPLIT
# =============================================================================

@dataclass
class ClassDomain:
    class_id: int
    rock: str

    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int

    cube_crop: np.ndarray
    component_mask_crop: np.ndarray

    train_mask_crop: np.ndarray
    test_mask_crop: np.ndarray

    split_name: str
    split_orientation: str
    split_axis: str

    train_fraction: float
    test_fraction: float
    gap_fraction: float

    nested_centers_yx: np.ndarray


def largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(mask, structure=structure)

    if n == 0:
        raise ValueError("No connected component exists for class mask.")

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    component_id = int(np.argmax(counts))
    component = labeled == component_id

    yy, xx = np.where(component)

    y0 = int(yy.min())
    y1 = int(yy.max()) + 1
    x0 = int(xx.min())
    x1 = int(xx.max()) + 1

    return component, (y0, x0, y1, x1)


def split_segment_masks(
    shape: Tuple[int, int],
    axis: str,
    train_fraction: float,
    test_fraction: float,
    gap_fraction: float,
    orientation: str,
) -> Tuple[np.ndarray, np.ndarray]:

    h, w = shape
    n = h if axis == "row" else w

    train_len = int(round(n * train_fraction))
    test_len = int(round(n * test_fraction))

    # Keep exact coverage of the axis; the remainder becomes the gap.
    train_len = min(train_len, n)
    test_len = min(test_len, n - train_len)
    gap_len = n - train_len - test_len

    if gap_len < 1:
        # Force at least a one-pixel separator.
        gap_len = 1
        if train_len >= test_len and train_len > 1:
            train_len -= 1
        elif test_len > 1:
            test_len -= 1

    train_region = np.zeros((h, w), dtype=bool)
    test_region = np.zeros((h, w), dtype=bool)

    if orientation == "low":
        train_start = 0
        train_end = train_len
        test_start = train_len + gap_len
        test_end = n

    elif orientation == "high":
        test_start = 0
        test_end = test_len
        train_start = test_len + gap_len
        train_end = n

    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    if axis == "row":
        train_region[train_start:train_end, :] = True
        test_region[test_start:test_end, :] = True

    elif axis == "col":
        train_region[:, train_start:train_end] = True
        test_region[:, test_start:test_end] = True

    else:
        raise ValueError(f"Unknown split axis: {axis}")

    return train_region, test_region


def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(
        mask.astype(np.int32),
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0).cumsum(axis=1)


def square_sum_at_centers(
    ii: np.ndarray,
    centers_yx: np.ndarray,
    size: int,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:

    centers_yx = np.asarray(centers_yx, dtype=int)

    if len(centers_yx) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool)

    h, w = shape

    cy = centers_yx[:, 0]
    cx = centers_yx[:, 1]

    y0 = cy - size // 2
    x0 = cx - size // 2
    y1 = y0 + size
    x1 = x0 + size

    valid = (
        (y0 >= 0)
        & (x0 >= 0)
        & (y1 <= h)
        & (x1 <= w)
    )

    sums = np.zeros(len(centers_yx), dtype=np.int32)

    idx = np.where(valid)[0]

    if len(idx):
        sums[idx] = (
            ii[y1[idx], x1[idx]]
            - ii[y0[idx], x1[idx]]
            - ii[y1[idx], x0[idx]]
            + ii[y0[idx], x0[idx]]
        )

    return sums, valid


def candidate_nested_centers(
    train_mask: np.ndarray,
    support_sizes: Sequence[int],
    min_coverage: float,
    min_train_spectra: int,
) -> np.ndarray:
    """
    Find centers at which EVERY support size is spatially valid.

    This makes all lambda windows nested and guarantees that no support gains an
    advantage merely because it was allowed to use a different center.
    """
    support_sizes = sorted(set(int(s) for s in support_sizes))
    max_size = max(support_sizes)

    h, w = train_mask.shape

    if max_size > h or max_size > w:
        return np.empty((0, 2), dtype=np.int32)

    ii = integral_image(train_mask)

    # Candidate centers from all top-left placements of the largest window.
    y0_grid, x0_grid = np.meshgrid(
        np.arange(0, h - max_size + 1, dtype=np.int32),
        np.arange(0, w - max_size + 1, dtype=np.int32),
        indexing="ij",
    )

    centers = np.column_stack([
        y0_grid.ravel() + max_size // 2,
        x0_grid.ravel() + max_size // 2,
    ])

    keep = np.ones(len(centers), dtype=bool)

    for size in support_sizes:
        sums, in_bounds = square_sum_at_centers(
            ii,
            centers,
            size=size,
            shape=train_mask.shape,
        )

        required = max(
            int(min_train_spectra),
            int(math.ceil(float(min_coverage) * size * size)),
        )

        keep &= in_bounds & (sums >= required)

        if not np.any(keep):
            return np.empty((0, 2), dtype=np.int32)

    return centers[keep].astype(np.int32)


def build_class_domains(
    cube: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[int, ClassDomain], pd.DataFrame]:

    domains: Dict[int, ClassDomain] = {}
    audit_rows: List[Dict[str, Any]] = []

    for class_id in CLASS_IDS:

        full_class_mask = labels == class_id
        component, bbox = largest_connected_component(full_class_mask)

        y0, x0, y1, x1 = bbox

        component_crop = component[y0:y1, x0:x1]
        cube_crop = cube[y0:y1, x0:x1, :]

        h, w = component_crop.shape

        # Separation is placed along the longer spatial dimension.
        axis = "row" if h >= w else "col"

        support_sizes = [
            support_px(class_id, factor)
            for factor in SUPPORT_FACTORS
        ]

        candidates: List[Dict[str, Any]] = []

        for split in (PRIMARY_SPLIT, FALLBACK_SPLIT):
            for orientation in ("low", "high"):

                train_region, test_region = split_segment_masks(
                    shape=component_crop.shape,
                    axis=axis,
                    train_fraction=split["train_fraction"],
                    test_fraction=split["test_fraction"],
                    gap_fraction=split["gap_fraction"],
                    orientation=orientation,
                )

                train_mask = component_crop & train_region
                test_mask = component_crop & test_region

                centers = candidate_nested_centers(
                    train_mask=train_mask,
                    support_sizes=support_sizes,
                    min_coverage=MIN_WINDOW_CLASS_COVERAGE,
                    min_train_spectra=N_TRAIN_PER_CLASS,
                )

                candidates.append({
                    "split": split,
                    "orientation": orientation,
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                    "centers": centers,
                    "center_count": int(len(centers)),
                    "train_pixels": int(train_mask.sum()),
                    "test_pixels": int(test_mask.sum()),
                })

        # Eligibility is purely geometric:
        # 1) enough valid nested training centers;
        # 2) enough held-out test pixels.
        eligible = [
            c
            for c in candidates
            if (
                c["center_count"] > 0
                and c["test_pixels"] >= N_TEST_PER_CLASS
            )
        ]

        if not eligible:
            details = [
                {
                    "split": c["split"]["name"],
                    "orientation": c["orientation"],
                    "center_count": c["center_count"],
                    "test_pixels": c["test_pixels"],
                }
                for c in candidates
            ]

            raise RuntimeError(
                f"No valid spatial split for class {class_id} "
                f"({ROCK_NAMES[class_id]}).\n"
                f"Candidates: {details}"
            )

        # Select the split with the largest number of admissible nested centers.
        # This selection is label-geometry based and does not use classifier
        # performance or downstream outcomes.
        eligible.sort(
            key=lambda c: (
                c["center_count"],
                c["test_pixels"],
                c["train_pixels"],
            ),
            reverse=True,
        )

        best = eligible[0]
        split = best["split"]

        domain = ClassDomain(
            class_id=class_id,
            rock=ROCK_NAMES[class_id],

            bbox_y0=y0,
            bbox_x0=x0,
            bbox_y1=y1,
            bbox_x1=x1,

            cube_crop=cube_crop,
            component_mask_crop=component_crop,

            train_mask_crop=best["train_mask"],
            test_mask_crop=best["test_mask"],

            split_name=split["name"],
            split_orientation=best["orientation"],
            split_axis=axis,

            train_fraction=float(split["train_fraction"]),
            test_fraction=float(split["test_fraction"]),
            gap_fraction=float(split["gap_fraction"]),

            nested_centers_yx=best["centers"],
        )

        domains[class_id] = domain

        candidate_count_map = {
            f"{c['split']['name']}_{c['orientation']}": c["center_count"]
            for c in candidates
        }

        audit_rows.append({
            "class_id": class_id,
            "rock": ROCK_NAMES[class_id],

            "component_pixels": int(component.sum()),
            "crop_height": int(h),
            "crop_width": int(w),

            "bbox_y0": y0,
            "bbox_x0": x0,
            "bbox_y1": y1,
            "bbox_x1": x1,

            "reu_px": int(FROZEN_REU_PX[class_id]),
            "max_support_px": int(max(support_sizes)),

            "selected_split": (
                f"{split['name']}_{best['orientation']}"
            ),
            "axis": axis,

            "train_pixels": int(best["train_pixels"]),
            "test_pixels": int(best["test_pixels"]),
            "valid_nested_centers": int(best["center_count"]),

            "train_fraction": float(split["train_fraction"]),
            "test_fraction": float(split["test_fraction"]),
            "gap_fraction": float(split["gap_fraction"]),

            "all_candidate_counts": json.dumps(
                candidate_count_map,
                ensure_ascii=False,
            ),
        })

    return domains, pd.DataFrame(audit_rows)


# =============================================================================
# 4. FIXED TEST SET
# =============================================================================

def make_fixed_test_set(
    domains: Mapping[int, ClassDomain],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    audit_rows: List[Dict[str, Any]] = []

    for class_id in CLASS_IDS:
        domain = domains[class_id]

        coords = np.argwhere(domain.test_mask_crop)

        if len(coords) < N_TEST_PER_CLASS:
            raise RuntimeError(
                f"Class {class_id} has only {len(coords)} held-out test pixels, "
                f"but {N_TEST_PER_CLASS} are required."
            )

        rng = np.random.default_rng(
            stable_seed("fixed_test", RANDOM_SEED, class_id)
        )

        selected_idx = rng.choice(
            len(coords),
            size=N_TEST_PER_CLASS,
            replace=False,
        )

        selected = coords[selected_idx]

        spectra = domain.cube_crop[
            selected[:, 0],
            selected[:, 1],
            :
        ]

        x_list.append(np.asarray(spectra, dtype=np.float32))
        y_list.append(
            np.full(
                N_TEST_PER_CLASS,
                class_id,
                dtype=np.int16,
            )
        )

        global_y = selected[:, 0] + domain.bbox_y0
        global_x = selected[:, 1] + domain.bbox_x0

        for i in range(N_TEST_PER_CLASS):
            audit_rows.append({
                "class_id": class_id,
                "rock": domain.rock,
                "test_index_within_class": i,
                "row_crop": int(selected[i, 0]),
                "col_crop": int(selected[i, 1]),
                "row_global": int(global_y[i]),
                "col_global": int(global_x[i]),
            })

    x_test = np.vstack(x_list).astype(np.float32)
    y_test = np.concatenate(y_list).astype(np.int16)

    return x_test, y_test, pd.DataFrame(audit_rows)


# =============================================================================
# 5. FIXED CLASSIFIER
# =============================================================================

def make_classifier() -> Pipeline:
    steps: List[Tuple[str, Any]] = []

    if L2_NORMALIZE_EACH_SPECTRUM:
        steps.append(
            ("l2_normalize", Normalizer(norm="l2"))
        )

    steps.extend([
        (
            "standardize",
            StandardScaler(
                with_mean=True,
                with_std=True,
            ),
        ),
        (
            "pca",
            PCA(
                n_components=PCA_N_COMPONENTS,
                svd_solver="full",
                whiten=False,
            ),
        ),
        (
            "rbf_svm",
            SVC(
                C=SVM_C,
                kernel=SVM_KERNEL,
                gamma=SVM_GAMMA,
                class_weight=SVM_CLASS_WEIGHT,
                probability=False,
                shrinking=True,
                cache_size=2048,
            ),
        ),
    ])

    return Pipeline(steps=steps)


# =============================================================================
# 6. TRAINING SAMPLE SELECTION
# =============================================================================

def square_bounds(
    center_yx: Sequence[int],
    size: int,
) -> Tuple[int, int, int, int]:

    cy = int(center_yx[0])
    cx = int(center_yx[1])

    y0 = cy - int(size) // 2
    x0 = cx - int(size) // 2

    y1 = y0 + int(size)
    x1 = x0 + int(size)

    return y0, x0, y1, x1


def make_priority_map(
    train_mask: np.ndarray,
    repeat: int,
    class_id: int,
) -> np.ndarray:
    """
    Uniform random priorities over the entire class-specific training reservoir.

    The SAME priority map is used for all lambda values and for the paired
    full-reservoir control within one repeat.  This strongly pairs the experiment
    and prevents unnecessary Monte Carlo noise from separate random samplers.
    """
    priority = np.full(
        train_mask.shape,
        np.inf,
        dtype=np.float64,
    )

    coords = np.argwhere(train_mask)

    rng = np.random.default_rng(
        stable_seed(
            "priority_map",
            RANDOM_SEED,
            repeat,
            class_id,
        )
    )

    priority_values = rng.random(len(coords))

    priority[
        coords[:, 0],
        coords[:, 1],
    ] = priority_values

    return priority


def sample_from_mask_by_priority(
    cube_crop: np.ndarray,
    eligible_mask: np.ndarray,
    priority_map: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:

    coords = np.argwhere(eligible_mask)

    if len(coords) < n_samples:
        raise RuntimeError(
            f"Only {len(coords)} eligible pixels, need {n_samples}."
        )

    priorities = priority_map[
        coords[:, 0],
        coords[:, 1],
    ]

    order = np.argpartition(
        priorities,
        kth=n_samples - 1,
    )[:n_samples]

    selected = coords[order]

    # Sort selected pixels by priority only for deterministic audit ordering.
    selected_priorities = priority_map[
        selected[:, 0],
        selected[:, 1],
    ]

    selected = selected[
        np.argsort(
            selected_priorities,
            kind="stable",
        )
    ]

    spectra = cube_crop[
        selected[:, 0],
        selected[:, 1],
        :
    ]

    return (
        np.asarray(spectra, dtype=np.float32),
        selected.astype(np.int32),
    )


def build_training_set_for_support(
    domains: Mapping[int, ClassDomain],
    centers: Mapping[int, np.ndarray],
    priority_maps: Mapping[int, np.ndarray],
    factor: float,
    repeat: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    audit_rows: List[Dict[str, Any]] = []

    for class_id in CLASS_IDS:
        domain = domains[class_id]

        size = support_px(class_id, factor)
        center = centers[class_id]

        y0, x0, y1, x1 = square_bounds(
            center,
            size,
        )

        h, w = domain.train_mask_crop.shape

        if (
            y0 < 0
            or x0 < 0
            or y1 > h
            or x1 > w
        ):
            raise RuntimeError(
                f"Support out of bounds: class={class_id}, "
                f"lambda={factor}, bounds={(y0, x0, y1, x1)}, "
                f"shape={(h, w)}"
            )

        support_mask = np.zeros_like(
            domain.train_mask_crop,
            dtype=bool,
        )

        support_mask[y0:y1, x0:x1] = True

        eligible = (
            support_mask
            & domain.train_mask_crop
        )

        available = int(eligible.sum())

        nominal_area = int(size * size)
        coverage = float(
            available / nominal_area
        )

        if (
            coverage + 1.0e-12
            < MIN_WINDOW_CLASS_COVERAGE
        ):
            raise RuntimeError(
                f"Coverage below predeclared limit: "
                f"class={class_id}, lambda={factor}, "
                f"coverage={coverage:.4f}"
            )

        spectra, coords = sample_from_mask_by_priority(
            cube_crop=domain.cube_crop,
            eligible_mask=eligible,
            priority_map=priority_maps[class_id],
            n_samples=N_TRAIN_PER_CLASS,
        )

        x_parts.append(spectra)

        y_parts.append(
            np.full(
                N_TRAIN_PER_CLASS,
                class_id,
                dtype=np.int16,
            )
        )

        audit_rows.append({
            "repeat": int(repeat),
            "lambda_value": float(factor),
            "class_id": int(class_id),
            "rock": domain.rock,

            "reu_px": int(FROZEN_REU_PX[class_id]),
            "support_px": int(size),

            "center_row_crop": int(center[0]),
            "center_col_crop": int(center[1]),

            "y0": int(y0),
            "x0": int(x0),
            "y1": int(y1),
            "x1": int(x1),

            "nominal_footprint_pixels": nominal_area,
            "available_valid_class_train_pixels": available,
            "window_class_coverage": coverage,
            "sampled_spectra": int(N_TRAIN_PER_CLASS),
        })

    return (
        np.vstack(x_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int16),
        audit_rows,
    )


def build_full_reference_training_set(
    domains: Mapping[int, ClassDomain],
    priority_maps: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for class_id in CLASS_IDS:
        domain = domains[class_id]

        spectra, _ = sample_from_mask_by_priority(
            cube_crop=domain.cube_crop,
            eligible_mask=domain.train_mask_crop,
            priority_map=priority_maps[class_id],
            n_samples=N_TRAIN_PER_CLASS,
        )

        x_parts.append(spectra)

        y_parts.append(
            np.full(
                N_TRAIN_PER_CLASS,
                class_id,
                dtype=np.int16,
            )
        )

    return (
        np.vstack(x_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int16),
    )


# =============================================================================
# 7. METRICS
# =============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
        "oa": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(CLASS_IDS),
                average="macro",
                zero_division=0,
            )
        ),
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, Dict[str, float]]:

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(CLASS_IDS),
            zero_division=0,
        )
    )

    out: Dict[int, Dict[str, float]] = {}

    for i, class_id in enumerate(CLASS_IDS):
        out[class_id] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return out


def js_divergence_discrete(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / max(p.sum(), 1.0e-15)
    q = q / max(q.sum(), 1.0e-15)

    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        ok = a > 0
        return float(
            np.sum(
                a[ok]
                * np.log(
                    a[ok] / np.maximum(b[ok], 1.0e-15)
                )
            )
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def full_prediction_agreement_metrics(
    support_pred: np.ndarray,
    full_pred: np.ndarray,
) -> Dict[str, float]:

    agreement = float(
        np.mean(support_pred == full_pred)
    )

    kappa = float(
        cohen_kappa_score(
            full_pred,
            support_pred,
            labels=list(CLASS_IDS),
        )
    )

    macro_f1_to_full = float(
        f1_score(
            full_pred,
            support_pred,
            labels=list(CLASS_IDS),
            average="macro",
            zero_division=0,
        )
    )

    support_counts = np.asarray([
        np.sum(support_pred == cid)
        for cid in CLASS_IDS
    ], dtype=float)

    full_counts = np.asarray([
        np.sum(full_pred == cid)
        for cid in CLASS_IDS
    ], dtype=float)

    class_distribution_js = js_divergence_discrete(
        support_counts,
        full_counts,
    )

    return {
        "prediction_agreement": agreement,
        "prediction_disagreement": 1.0 - agreement,
        "cohen_kappa_to_full": kappa,
        "macro_f1_to_full_predictions": macro_f1_to_full,
        "predicted_class_distribution_js_to_full": float(
            class_distribution_js
        ),
    }


def repeat_prediction_instability(
    pred_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    pred_matrix shape:
        [n_repeats, n_test]

    Returns label-independent stability descriptors.
    """
    pred_matrix = np.asarray(pred_matrix)

    n_repeats, n_test = pred_matrix.shape

    # Pairwise repeat-to-repeat disagreement.
    pairwise = []

    for i, j in combinations(range(n_repeats), 2):
        pairwise.append(
            float(
                np.mean(
                    pred_matrix[i] != pred_matrix[j]
                )
            )
        )

    pairwise = np.asarray(pairwise, dtype=float)

    # Per-test-sample predictive class distribution across repeats.
    counts = np.zeros(
        (n_test, len(CLASS_IDS)),
        dtype=np.float64,
    )

    for k, class_id in enumerate(CLASS_IDS):
        counts[:, k] = np.sum(
            pred_matrix == class_id,
            axis=0,
        )

    probabilities = counts / float(n_repeats)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        logp = np.where(
            probabilities > 0,
            np.log(probabilities),
            0.0,
        )

    entropy = -np.sum(
        probabilities * logp,
        axis=1,
    )

    entropy_normalized = entropy / math.log(
        len(CLASS_IDS)
    )

    consensus_idx = np.argmax(
        counts,
        axis=1,
    )

    consensus_pred = np.asarray(
        CLASS_IDS,
        dtype=np.int16,
    )[consensus_idx]

    consensus_fraction = (
        np.max(counts, axis=1)
        / float(n_repeats)
    )

    repeat_disagreement_to_consensus = np.asarray([
        np.mean(pred_matrix[r] != consensus_pred)
        for r in range(n_repeats)
    ], dtype=float)

    ci_low, ci_high = bootstrap_mean_ci(
        repeat_disagreement_to_consensus,
        seed=stable_seed(
            "instability_consensus_ci",
            RANDOM_SEED,
            n_repeats,
            n_test,
        ),
    )

    return {
        "pairwise_disagreement_mean": float(
            np.mean(pairwise)
        ),
        "pairwise_disagreement_median": float(
            np.median(pairwise)
        ),
        "pairwise_disagreement_std": float(
            np.std(pairwise, ddof=1)
            if len(pairwise) > 1
            else 0.0
        ),

        "normalized_prediction_entropy_mean": float(
            np.mean(entropy_normalized)
        ),
        "normalized_prediction_entropy_median": float(
            np.median(entropy_normalized)
        ),

        "consensus_fraction_mean": float(
            np.mean(consensus_fraction)
        ),
        "consensus_fraction_median": float(
            np.median(consensus_fraction)
        ),

        "repeat_to_consensus_disagreement_mean": float(
            np.mean(repeat_disagreement_to_consensus)
        ),
        "repeat_to_consensus_disagreement_std": float(
            np.std(
                repeat_disagreement_to_consensus,
                ddof=1,
            )
            if len(repeat_disagreement_to_consensus) > 1
            else 0.0
        ),
        "repeat_to_consensus_disagreement_ci95_low": ci_low,
        "repeat_to_consensus_disagreement_ci95_high": ci_high,
    }


# =============================================================================
# 8. MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    cube: np.ndarray,
    domains: Mapping[int, ClassDomain],
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:

    classifier_template = make_classifier()

    classification_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    agreement_rows: List[Dict[str, Any]] = []
    center_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []

    n_test = len(y_test)
    n_factors = len(SUPPORT_FACTORS)

    support_predictions = np.empty(
        (N_REPEATS, n_factors, n_test),
        dtype=np.uint8,
    )

    full_predictions = np.empty(
        (N_REPEATS, n_test),
        dtype=np.uint8,
    )

    total_models = N_REPEATS * (n_factors + 1)
    model_counter = 0

    for repeat in range(N_REPEATS):

        print()
        print("-" * 110)
        print(
            f"Repeat {repeat + 1:02d}/{N_REPEATS:02d}"
        )
        print("-" * 110)

        # Paired center selection and priority maps are frozen for this repeat.
        centers: Dict[int, np.ndarray] = {}
        priority_maps: Dict[int, np.ndarray] = {}

        for class_id in CLASS_IDS:
            domain = domains[class_id]

            rng = np.random.default_rng(
                stable_seed(
                    "center",
                    RANDOM_SEED,
                    repeat,
                    class_id,
                )
            )

            center_idx = int(
                rng.integers(
                    0,
                    len(domain.nested_centers_yx),
                )
            )

            center = domain.nested_centers_yx[
                center_idx
            ].astype(np.int32)

            centers[class_id] = center

            priority_maps[class_id] = make_priority_map(
                train_mask=domain.train_mask_crop,
                repeat=repeat,
                class_id=class_id,
            )

            center_rows.append({
                "class_id": class_id,
                "rock": domain.rock,
                "repeat": repeat,
                "center_row_crop": int(center[0]),
                "center_col_crop": int(center[1]),
                "valid_nested_center_count": int(
                    len(domain.nested_centers_yx)
                ),
            })

        # ---------------------------------------------------------------------
        # Full-training-reservoir paired reference
        # ---------------------------------------------------------------------
        x_full, y_full = build_full_reference_training_set(
            domains=domains,
            priority_maps=priority_maps,
        )

        model_counter += 1
        print(
            f"[{model_counter:03d}/{total_models:03d}] "
            "Fit paired full-reservoir RBF-SVM"
        )

        full_model = clone(classifier_template)
        full_model.fit(x_full, y_full)

        pred_full = full_model.predict(x_test).astype(np.uint8)
        full_predictions[repeat] = pred_full

        full_metric = classification_metrics(
            y_test,
            pred_full,
        )

        full_rows.append({
            "repeat": repeat,
            "oa": full_metric["oa"],
            "macro_f1": full_metric["macro_f1"],
            "train_spectra_per_class": N_TRAIN_PER_CLASS,
            "total_train_spectra": int(
                N_TRAIN_PER_CLASS * len(CLASS_IDS)
            ),
            "spatial_support": "full_training_reservoir",
        })

        full_pc = per_class_metrics(
            y_test,
            pred_full,
        )

        for class_id in CLASS_IDS:
            per_class_rows.append({
                "model_type": "full_reference",
                "repeat": repeat,
                "lambda_value": np.nan,
                "class_id": class_id,
                "rock": ROCK_NAMES[class_id],
                **full_pc[class_id],
            })

        del full_model
        del x_full
        del y_full
        gc.collect()

        # ---------------------------------------------------------------------
        # Nested support models
        # ---------------------------------------------------------------------
        for factor_index, factor in enumerate(SUPPORT_FACTORS):

            x_train, y_train, this_window_rows = (
                build_training_set_for_support(
                    domains=domains,
                    centers=centers,
                    priority_maps=priority_maps,
                    factor=factor,
                    repeat=repeat,
                )
            )

            window_rows.extend(this_window_rows)

            model_counter += 1

            print(
                f"[{model_counter:03d}/{total_models:03d}] "
                f"Fit lambda={factor:.2f} REU RBF-SVM"
            )

            model = clone(classifier_template)
            model.fit(x_train, y_train)

            pred = model.predict(x_test).astype(np.uint8)
            support_predictions[
                repeat,
                factor_index,
                :
            ] = pred

            metric = classification_metrics(
                y_test,
                pred,
            )

            # Spatial footprint is diagnostic only; train sample count is fixed.
            footprint_nominal = int(
                sum(
                    support_px(cid, factor) ** 2
                    for cid in CLASS_IDS
                )
            )

            valid_in_footprints = int(
                sum(
                    row["available_valid_class_train_pixels"]
                    for row in this_window_rows
                )
            )

            full_training_reservoir_pixels = int(
                sum(
                    domains[cid].train_mask_crop.sum()
                    for cid in CLASS_IDS
                )
            )

            classification_rows.append({
                "repeat": repeat,
                "lambda_value": float(factor),

                "oa": metric["oa"],
                "macro_f1": metric["macro_f1"],

                "train_spectra_per_class": N_TRAIN_PER_CLASS,
                "total_train_spectra": int(
                    N_TRAIN_PER_CLASS * len(CLASS_IDS)
                ),

                "total_nominal_footprint_pixels": footprint_nominal,
                "total_valid_class_pixels_in_footprints": (
                    valid_in_footprints
                ),
                "training_footprint_fraction_vs_full_reservoir": float(
                    valid_in_footprints
                    / max(full_training_reservoir_pixels, 1)
                ),

                "paired_full_reference_oa": full_metric["oa"],
                "paired_full_reference_macro_f1": full_metric["macro_f1"],
                "macro_f1_minus_full": float(
                    metric["macro_f1"]
                    - full_metric["macro_f1"]
                ),
                "absolute_macro_f1_gap_to_full": float(
                    abs(
                        metric["macro_f1"]
                        - full_metric["macro_f1"]
                    )
                ),
            })

            pc = per_class_metrics(
                y_test,
                pred,
            )

            for class_id in CLASS_IDS:
                per_class_rows.append({
                    "model_type": "support",
                    "repeat": repeat,
                    "lambda_value": float(factor),
                    "class_id": class_id,
                    "rock": ROCK_NAMES[class_id],
                    **pc[class_id],
                })

            agreement = full_prediction_agreement_metrics(
                support_pred=pred,
                full_pred=pred_full,
            )

            agreement_rows.append({
                "repeat": repeat,
                "lambda_value": float(factor),
                **agreement,
            })

            del model
            del x_train
            del y_train
            gc.collect()

    return {
        "classification_repeat": pd.DataFrame(
            classification_rows
        ),
        "full_repeat": pd.DataFrame(
            full_rows
        ),
        "per_class_repeat": pd.DataFrame(
            per_class_rows
        ),
        "agreement_repeat": pd.DataFrame(
            agreement_rows
        ),
        "center_audit": pd.DataFrame(
            center_rows
        ),
        "window_audit": pd.DataFrame(
            window_rows
        ),
        "support_predictions": support_predictions,
        "full_predictions": full_predictions,
    }


# =============================================================================
# 9. SUMMARIES
# =============================================================================

def summarize_classification(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    for factor, g in df.groupby(
        "lambda_value",
        sort=True,
    ):

        f1 = g["macro_f1"].to_numpy(dtype=float)
        oa = g["oa"].to_numpy(dtype=float)
        gap = g[
            "absolute_macro_f1_gap_to_full"
        ].to_numpy(dtype=float)

        f1_ci = bootstrap_mean_ci(
            f1,
            seed=stable_seed(
                "classification_f1_ci",
                RANDOM_SEED,
                factor,
            ),
        )

        oa_ci = bootstrap_mean_ci(
            oa,
            seed=stable_seed(
                "classification_oa_ci",
                RANDOM_SEED,
                factor,
            ),
        )

        gap_ci = bootstrap_mean_ci(
            gap,
            seed=stable_seed(
                "classification_gap_ci",
                RANDOM_SEED,
                factor,
            ),
        )

        rows.append({
            "lambda_value": float(factor),
            "n_repeats": int(len(g)),

            "macro_f1_mean": float(np.mean(f1)),
            "macro_f1_median": float(np.median(f1)),
            "macro_f1_std": float(np.std(f1, ddof=1)),
            "macro_f1_q25": float(np.quantile(f1, 0.25)),
            "macro_f1_q75": float(np.quantile(f1, 0.75)),
            "macro_f1_ci95_low": f1_ci[0],
            "macro_f1_ci95_high": f1_ci[1],

            "oa_mean": float(np.mean(oa)),
            "oa_median": float(np.median(oa)),
            "oa_std": float(np.std(oa, ddof=1)),
            "oa_ci95_low": oa_ci[0],
            "oa_ci95_high": oa_ci[1],

            "absolute_macro_f1_gap_to_full_mean": float(
                np.mean(gap)
            ),
            "absolute_macro_f1_gap_to_full_median": float(
                np.median(gap)
            ),
            "absolute_macro_f1_gap_to_full_ci95_low": gap_ci[0],
            "absolute_macro_f1_gap_to_full_ci95_high": gap_ci[1],

            "training_footprint_fraction_vs_full_median": float(
                g[
                    "training_footprint_fraction_vs_full_reservoir"
                ].median()
            ),
        })

    return pd.DataFrame(rows)


def summarize_full_reference(
    df: pd.DataFrame,
) -> pd.DataFrame:

    f1 = df["macro_f1"].to_numpy(dtype=float)
    oa = df["oa"].to_numpy(dtype=float)

    f1_ci = bootstrap_mean_ci(
        f1,
        seed=stable_seed(
            "full_reference_f1_ci",
            RANDOM_SEED,
        ),
    )

    oa_ci = bootstrap_mean_ci(
        oa,
        seed=stable_seed(
            "full_reference_oa_ci",
            RANDOM_SEED,
        ),
    )

    return pd.DataFrame([{
        "n_repeats": int(len(df)),

        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_median": float(np.median(f1)),
        "macro_f1_std": float(np.std(f1, ddof=1)),
        "macro_f1_q25": float(np.quantile(f1, 0.25)),
        "macro_f1_q75": float(np.quantile(f1, 0.75)),
        "macro_f1_ci95_low": f1_ci[0],
        "macro_f1_ci95_high": f1_ci[1],

        "oa_mean": float(np.mean(oa)),
        "oa_median": float(np.median(oa)),
        "oa_std": float(np.std(oa, ddof=1)),
        "oa_ci95_low": oa_ci[0],
        "oa_ci95_high": oa_ci[1],

        "train_spectra_per_class": N_TRAIN_PER_CLASS,
        "spatial_support": "full_training_reservoir",
    }])


def summarize_agreement(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    metric_names = [
        "prediction_agreement",
        "prediction_disagreement",
        "cohen_kappa_to_full",
        "macro_f1_to_full_predictions",
        "predicted_class_distribution_js_to_full",
    ]

    for factor, g in df.groupby(
        "lambda_value",
        sort=True,
    ):
        row: Dict[str, Any] = {
            "lambda_value": float(factor),
            "n_repeats": int(len(g)),
        }

        for metric in metric_names:
            values = g[metric].to_numpy(dtype=float)

            ci = bootstrap_mean_ci(
                values,
                seed=stable_seed(
                    "agreement_ci",
                    RANDOM_SEED,
                    factor,
                    metric,
                ),
            )

            row[f"{metric}_mean"] = float(
                np.mean(values)
            )
            row[f"{metric}_median"] = float(
                np.median(values)
            )
            row[f"{metric}_std"] = float(
                np.std(values, ddof=1)
            )
            row[f"{metric}_ci95_low"] = ci[0]
            row[f"{metric}_ci95_high"] = ci[1]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_per_class(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    support_df = df[
        df["model_type"] == "support"
    ].copy()

    for (factor, class_id), g in support_df.groupby(
        ["lambda_value", "class_id"],
        sort=True,
    ):
        values = g["f1"].to_numpy(dtype=float)

        ci = bootstrap_mean_ci(
            values,
            seed=stable_seed(
                "per_class_f1_ci",
                RANDOM_SEED,
                factor,
                class_id,
            ),
        )

        rows.append({
            "model_type": "support",
            "lambda_value": float(factor),
            "class_id": int(class_id),
            "rock": ROCK_NAMES[int(class_id)],

            "f1_mean": float(np.mean(values)),
            "f1_median": float(np.median(values)),
            "f1_std": float(np.std(values, ddof=1)),
            "f1_ci95_low": ci[0],
            "f1_ci95_high": ci[1],
        })

    full_df = df[
        df["model_type"] == "full_reference"
    ].copy()

    for class_id, g in full_df.groupby(
        "class_id",
        sort=True,
    ):
        values = g["f1"].to_numpy(dtype=float)

        ci = bootstrap_mean_ci(
            values,
            seed=stable_seed(
                "per_class_full_f1_ci",
                RANDOM_SEED,
                class_id,
            ),
        )

        rows.append({
            "model_type": "full_reference",
            "lambda_value": np.nan,
            "class_id": int(class_id),
            "rock": ROCK_NAMES[int(class_id)],

            "f1_mean": float(np.mean(values)),
            "f1_median": float(np.median(values)),
            "f1_std": float(np.std(values, ddof=1)),
            "f1_ci95_low": ci[0],
            "f1_ci95_high": ci[1],
        })

    return pd.DataFrame(rows)


def summarize_prediction_instability(
    support_predictions: np.ndarray,
    full_predictions: np.ndarray,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    for factor_index, factor in enumerate(SUPPORT_FACTORS):
        metrics = repeat_prediction_instability(
            support_predictions[:, factor_index, :]
        )

        rows.append({
            "model_type": "support",
            "lambda_value": float(factor),
            "n_repeats": N_REPEATS,
            **metrics,
        })

    full_metrics = repeat_prediction_instability(
        full_predictions
    )

    rows.append({
        "model_type": "full_reference",
        "lambda_value": np.nan,
        "n_repeats": N_REPEATS,
        **full_metrics,
    })

    return pd.DataFrame(rows)


# =============================================================================
# 10. DESCRIPTIVE ASSESSMENT
# =============================================================================

def build_assessment(
    classification_repeat: pd.DataFrame,
    classification_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    instability_summary: pd.DataFrame,
) -> Dict[str, Any]:

    def csum(factor: float) -> pd.Series:
        return classification_summary.loc[
            np.isclose(
                classification_summary["lambda_value"],
                factor,
            )
        ].iloc[0]

    def asum(factor: float) -> pd.Series:
        return agreement_summary.loc[
            np.isclose(
                agreement_summary["lambda_value"],
                factor,
            )
        ].iloc[0]

    def isum(factor: float) -> pd.Series:
        return instability_summary.loc[
            (
                instability_summary["model_type"] == "support"
            )
            & np.isclose(
                instability_summary["lambda_value"],
                factor,
            )
        ].iloc[0]

    c025 = csum(0.25)
    c100 = csum(1.00)
    c150 = csum(1.50)

    a025 = asum(0.25)
    a100 = asum(1.00)
    a150 = asum(1.50)

    i025 = isum(0.25)
    i100 = isum(1.00)
    i150 = isum(1.50)

    full_f1 = float(
        full_summary["macro_f1_mean"].iloc[0]
    )

    # Paired F1 comparison, repeat by repeat.
    p025 = classification_repeat.loc[
        np.isclose(
            classification_repeat["lambda_value"],
            0.25,
        )
    ].sort_values("repeat")

    p100 = classification_repeat.loc[
        np.isclose(
            classification_repeat["lambda_value"],
            1.00,
        )
    ].sort_values("repeat")

    p150 = classification_repeat.loc[
        np.isclose(
            classification_repeat["lambda_value"],
            1.50,
        )
    ].sort_values("repeat")

    gain_025_to_100 = paired_bootstrap_difference_ci(
        p100["macro_f1"].to_numpy(),
        p025["macro_f1"].to_numpy(),
        seed=stable_seed(
            "gain_025_100",
            RANDOM_SEED,
        ),
    )

    gain_100_to_150 = paired_bootstrap_difference_ci(
        p150["macro_f1"].to_numpy(),
        p100["macro_f1"].to_numpy(),
        seed=stable_seed(
            "gain_100_150",
            RANDOM_SEED,
        ),
    )

    instability_reduction = (
        1.0
        - float(
            i100["pairwise_disagreement_mean"]
        )
        / max(
            float(
                i025["pairwise_disagreement_mean"]
            ),
            1.0e-15,
        )
    )

    entropy_reduction = (
        1.0
        - float(
            i100[
                "normalized_prediction_entropy_mean"
            ]
        )
        / max(
            float(
                i025[
                    "normalized_prediction_entropy_mean"
                ]
            ),
            1.0e-15,
        )
    )

    return {
        "design_status": (
            "predeclared_fixed_classifier_no_support_or_reu_tuning"
        ),

        "frozen_reu_px": FROZEN_REU_PX,
        "support_factors": SUPPORT_FACTORS,

        "full_reference_macro_f1_mean": full_f1,

        "macro_f1_mean": {
            "0.25_REU": float(c025["macro_f1_mean"]),
            "1.00_REU": float(c100["macro_f1_mean"]),
            "1.50_REU": float(c150["macro_f1_mean"]),
        },

        "paired_median_macro_f1_gain_0p25_to_1p0": {
            "estimate": gain_025_to_100[0],
            "ci95_low": gain_025_to_100[1],
            "ci95_high": gain_025_to_100[2],
        },

        "paired_median_macro_f1_gain_1p0_to_1p5": {
            "estimate": gain_100_to_150[0],
            "ci95_low": gain_100_to_150[1],
            "ci95_high": gain_100_to_150[2],
        },

        "absolute_f1_gap_to_full_mean": {
            "0.25_REU": float(
                c025[
                    "absolute_macro_f1_gap_to_full_mean"
                ]
            ),
            "1.00_REU": float(
                c100[
                    "absolute_macro_f1_gap_to_full_mean"
                ]
            ),
            "1.50_REU": float(
                c150[
                    "absolute_macro_f1_gap_to_full_mean"
                ]
            ),
        },

        "full_field_prediction_agreement_mean": {
            "0.25_REU": float(
                a025["prediction_agreement_mean"]
            ),
            "1.00_REU": float(
                a100["prediction_agreement_mean"]
            ),
            "1.50_REU": float(
                a150["prediction_agreement_mean"]
            ),
        },

        "full_field_prediction_kappa_mean": {
            "0.25_REU": float(
                a025["cohen_kappa_to_full_mean"]
            ),
            "1.00_REU": float(
                a100["cohen_kappa_to_full_mean"]
            ),
            "1.50_REU": float(
                a150["cohen_kappa_to_full_mean"]
            ),
        },

        "pairwise_prediction_disagreement_mean": {
            "0.25_REU": float(
                i025["pairwise_disagreement_mean"]
            ),
            "1.00_REU": float(
                i100["pairwise_disagreement_mean"]
            ),
            "1.50_REU": float(
                i150["pairwise_disagreement_mean"]
            ),
        },

        "normalized_prediction_entropy_mean": {
            "0.25_REU": float(
                i025[
                    "normalized_prediction_entropy_mean"
                ]
            ),
            "1.00_REU": float(
                i100[
                    "normalized_prediction_entropy_mean"
                ]
            ),
            "1.50_REU": float(
                i150[
                    "normalized_prediction_entropy_mean"
                ]
            ),
        },

        "prediction_instability_reduction_0p25_to_1p0_percent": float(
            100.0 * instability_reduction
        ),

        "prediction_entropy_reduction_0p25_to_1p0_percent": float(
            100.0 * entropy_reduction
        ),

        "f1_reference_margin_retained_from_v1_for_description_only": (
            F1_REFERENCE_MARGIN
        ),

        "interpretation_rule": (
            "Strong downstream support requires a coherent pattern rather than "
            "a single favorable F1 point: performance should approach the "
            "paired full-reservoir reference, repeat-to-repeat prediction "
            "instability should decline, and agreement with the paired "
            "full-reservoir predictions should increase as support approaches "
            "the frozen REU. No REU or classifier parameter is altered based "
            "on these results."
        ),
    }


# =============================================================================
# 11. FIGURE
# =============================================================================

def plot_main_figure(
    classification_repeat: pd.DataFrame,
    classification_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    agreement_repeat: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    instability_summary: pd.DataFrame,
) -> None:

    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.4, 3.45),
    )

    ax1, ax2, ax3 = axes

    x = classification_summary[
        "lambda_value"
    ].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # a. Macro-F1
    # -------------------------------------------------------------------------
    for repeat, g in classification_repeat.groupby(
        "repeat",
        sort=True,
    ):
        g = g.sort_values("lambda_value")

        ax1.plot(
            g["lambda_value"],
            g["macro_f1"],
            linewidth=0.6,
            alpha=0.18,
            zorder=1,
        )

    y = classification_summary[
        "macro_f1_mean"
    ].to_numpy(dtype=float)

    yerr_low = (
        y
        - classification_summary[
            "macro_f1_ci95_low"
        ].to_numpy(dtype=float)
    )

    yerr_high = (
        classification_summary[
            "macro_f1_ci95_high"
        ].to_numpy(dtype=float)
        - y
    )

    ax1.errorbar(
        x,
        y,
        yerr=np.vstack([yerr_low, yerr_high]),
        marker="o",
        markersize=4.8,
        linewidth=1.4,
        capsize=2.2,
        zorder=3,
        label="Support-limited model",
    )

    full_mean = float(
        full_summary["macro_f1_mean"].iloc[0]
    )

    full_low = float(
        full_summary["macro_f1_ci95_low"].iloc[0]
    )

    full_high = float(
        full_summary["macro_f1_ci95_high"].iloc[0]
    )

    ax1.axhspan(
        full_low,
        full_high,
        alpha=0.10,
        zorder=0,
    )

    ax1.axhline(
        full_mean,
        linestyle="--",
        linewidth=1.0,
        label="Full-reservoir reference",
    )

    ax1.axvline(
        1.0,
        linestyle=":",
        linewidth=0.9,
    )

    ax1.set_xlabel(r"Observation support, $L/L_{\mathrm{REU}}$")
    ax1.set_ylabel("Macro-F1")
    ax1.set_xticks(SUPPORT_FACTORS)
    clean_axes(ax1)
    panel_label(ax1, "a")

    ax1.legend(
        frameon=False,
        loc="best",
    )

    # -------------------------------------------------------------------------
    # b. Prediction instability
    # -------------------------------------------------------------------------
    support_inst = instability_summary[
        instability_summary["model_type"] == "support"
    ].sort_values("lambda_value")

    ax2.plot(
        support_inst["lambda_value"],
        100.0
        * support_inst["pairwise_disagreement_mean"],
        marker="s",
        markersize=4.8,
        linewidth=1.4,
        label="Repeat-to-repeat disagreement",
    )

    full_inst = instability_summary[
        instability_summary["model_type"] == "full_reference"
    ].iloc[0]

    ax2.axhline(
        100.0
        * float(
            full_inst["pairwise_disagreement_mean"]
        ),
        linestyle="--",
        linewidth=1.0,
        label="Full-reservoir reference",
    )

    ax2.axvline(
        1.0,
        linestyle=":",
        linewidth=0.9,
    )

    ax2.set_xlabel(r"Observation support, $L/L_{\mathrm{REU}}$")
    ax2.set_ylabel("Prediction disagreement (%)")
    ax2.set_xticks(SUPPORT_FACTORS)
    clean_axes(ax2)
    panel_label(ax2, "b")

    ax2.legend(
        frameon=False,
        loc="best",
    )

    # -------------------------------------------------------------------------
    # c. Agreement with paired full-reservoir model
    # -------------------------------------------------------------------------
    for repeat, g in agreement_repeat.groupby(
        "repeat",
        sort=True,
    ):
        g = g.sort_values("lambda_value")

        ax3.plot(
            g["lambda_value"],
            100.0 * g["prediction_agreement"],
            linewidth=0.6,
            alpha=0.18,
            zorder=1,
        )

    y_agree = (
        100.0
        * agreement_summary[
            "prediction_agreement_mean"
        ].to_numpy(dtype=float)
    )

    y_agree_low = (
        y_agree
        - 100.0
        * agreement_summary[
            "prediction_agreement_ci95_low"
        ].to_numpy(dtype=float)
    )

    y_agree_high = (
        100.0
        * agreement_summary[
            "prediction_agreement_ci95_high"
        ].to_numpy(dtype=float)
        - y_agree
    )

    ax3.errorbar(
        agreement_summary["lambda_value"],
        y_agree,
        yerr=np.vstack([
            y_agree_low,
            y_agree_high,
        ]),
        marker="^",
        markersize=5.0,
        linewidth=1.4,
        capsize=2.2,
        zorder=3,
    )

    ax3.axvline(
        1.0,
        linestyle=":",
        linewidth=0.9,
    )

    ax3.set_xlabel(r"Observation support, $L/L_{\mathrm{REU}}$")
    ax3.set_ylabel("Agreement with full-field model (%)")
    ax3.set_xticks(SUPPORT_FACTORS)
    clean_axes(ax3)
    panel_label(ax3, "c")

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.96,
        wspace=0.34,
    )

    ensure_dir(OUTPUT_ROOT)

    stem = OUTPUT_ROOT / "Downstream_REU_stabilization_v2"

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    if SAVE_TIFF:
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=DPI,
            bbox_inches="tight",
        )

    plt.close(fig)


# =============================================================================
# 12. SAVE PREDECLARED DESIGN
# =============================================================================

def save_predeclared_design(
    data_info: Optional[Mapping[str, Any]] = None,
) -> None:

    design = {
        "script": Path(__file__).name,

        "scientific_question": (
            "Do predictions from an independently defined nonlinear "
            "classification task become more accurate, less sampling-sensitive "
            "and more consistent with a paired full-training-reservoir model "
            "as observation support approaches the frozen REU?"
        ),

        "reu_source": (
            PROJECT_ROOT
            / "results"
            / "nc_reu_pixel_v2_7_robust_scale_inference"
        ),

        "frozen_reu_px": FROZEN_REU_PX,

        "support_factors": SUPPORT_FACTORS,

        "n_repeats": N_REPEATS,
        "n_train_spectra_per_class_at_every_support": N_TRAIN_PER_CLASS,
        "n_train_spectra_per_class_full_reference": N_TRAIN_PER_CLASS,
        "n_fixed_test_spectra_per_class": N_TEST_PER_CLASS,

        "min_window_class_coverage": MIN_WINDOW_CLASS_COVERAGE,

        "spatial_split_primary": PRIMARY_SPLIT,
        "spatial_split_fallback": FALLBACK_SPLIT,

        "classifier": {
            "per_spectrum_l2_normalization": (
                L2_NORMALIZE_EACH_SPECTRUM
            ),
            "standardization": "StandardScaler fit on training data only",
            "pca_n_components": PCA_N_COMPONENTS,
            "pca_solver": "full",
            "svm_kernel": SVM_KERNEL,
            "svm_C": SVM_C,
            "svm_gamma": SVM_GAMMA,
            "svm_class_weight": SVM_CLASS_WEIGHT,
            "hyperparameter_search": False,
            "cross_validation_for_tuning": False,
        },

        "paired_design": (
            "same class-specific support center across all lambda values "
            "within each repeat; supports are nested; the same random priority "
            "map is used to select the fixed number of training spectra from "
            "each support and from the paired full reservoir"
        ),

        "fixed_test_design": (
            "one spatially separated held-out test sample is selected once and "
            "used unchanged for all support factors, all repeats and all "
            "full-reservoir reference models"
        ),

        "full_extent_control": (
            "same number of spectra per class as support-limited models, "
            "sampled from the complete geometry-defined training reservoir"
        ),

        "primary_downstream_outputs": {
            "predictive_performance": [
                "Macro-F1",
                "overall accuracy",
                "per-class F1",
                "absolute paired Macro-F1 gap to full reference",
            ],
            "prediction_instability": [
                "pairwise repeat-to-repeat prediction disagreement",
                "normalized repeat prediction entropy",
                "repeat-to-consensus disagreement",
            ],
            "full_field_agreement": [
                "prediction agreement fraction",
                "Cohen kappa to paired full prediction",
                "Macro-F1 to paired full prediction",
                "predicted-class-distribution JS divergence",
            ],
        },

        "f1_reference_margin_for_description_only": F1_REFERENCE_MARGIN,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "random_seed": RANDOM_SEED,

        "strict_no_tuning_statement": (
            "No REU, support factor, spatial split fraction, sample count, "
            "classifier architecture or classifier hyperparameter is altered "
            "after inspecting the v2 downstream outcomes."
        ),

        "data_info": data_info,
    }

    save_json(
        OUTPUT_ROOT / "validation_design_predeclared_v2.json",
        design,
    )


# =============================================================================
# 13. TEXT INTERPRETATION
# =============================================================================

def save_results_interpretation(
    assessment: Mapping[str, Any],
) -> None:

    f1 = assessment["macro_f1_mean"]
    agreement = assessment[
        "full_field_prediction_agreement_mean"
    ]
    disagreement = assessment[
        "pairwise_prediction_disagreement_mean"
    ]

    gain1 = assessment[
        "paired_median_macro_f1_gain_0p25_to_1p0"
    ]

    gain2 = assessment[
        "paired_median_macro_f1_gain_1p0_to_1p5"
    ]

    text = f"""
NC-REU independent downstream stabilization validation v2
================================================================================

Frozen REUs:
{FROZEN_REU_PX}

Classifier:
Per-spectrum L2 normalization -> StandardScaler -> PCA({PCA_N_COMPONENTS})
-> fixed RBF-SVM(C={SVM_C}, gamma={SVM_GAMMA})

Fairness controls:
- {N_TRAIN_PER_CLASS} spectra/class at every support
- {N_TRAIN_PER_CLASS} spectra/class in each paired full-reservoir reference
- fixed spatially separated test set: {N_TEST_PER_CLASS} spectra/class
- same center across all lambda values within each repeat
- nested supports
- no classifier or REU tuning from v2 outcomes

Predictive performance:
- Mean Macro-F1 at 0.25 REU : {f1['0.25_REU']:.6f}
- Mean Macro-F1 at 1.00 REU : {f1['1.00_REU']:.6f}
- Mean Macro-F1 at 1.50 REU : {f1['1.50_REU']:.6f}
- Mean full-reference Macro-F1: {assessment['full_reference_macro_f1_mean']:.6f}

Paired median Macro-F1 change:
- 0.25 -> 1.00 REU:
  {gain1['estimate']:.6f}
  95% bootstrap CI [{gain1['ci95_low']:.6f}, {gain1['ci95_high']:.6f}]

- 1.00 -> 1.50 REU:
  {gain2['estimate']:.6f}
  95% bootstrap CI [{gain2['ci95_low']:.6f}, {gain2['ci95_high']:.6f}]

Prediction instability:
- Pairwise disagreement at 0.25 REU: {100.0 * disagreement['0.25_REU']:.3f}%
- Pairwise disagreement at 1.00 REU: {100.0 * disagreement['1.00_REU']:.3f}%
- Pairwise disagreement at 1.50 REU: {100.0 * disagreement['1.50_REU']:.3f}%
- Reduction from 0.25 to 1.00 REU:
  {assessment['prediction_instability_reduction_0p25_to_1p0_percent']:.3f}%

Agreement with the paired full-reservoir model:
- 0.25 REU: {100.0 * agreement['0.25_REU']:.3f}%
- 1.00 REU: {100.0 * agreement['1.00_REU']:.3f}%
- 1.50 REU: {100.0 * agreement['1.50_REU']:.3f}%

INTERPRETATION RULE
-------------------
Do not judge the downstream validation from whether one F1 point is the maximum.

The strongest independent support for REU is a COHERENT three-part pattern:
1) Macro-F1 approaches the paired full-reservoir reference;
2) repeat-to-repeat prediction instability decreases with support;
3) support-limited predictions increasingly agree with the paired
   full-reservoir predictions.

A non-confirmed stabilization is a valid scientific outcome.  Do not alter
support factors, frozen REUs, split geometry, training sample count, test sample
count, PCA dimension, C or gamma after seeing this result.
""".strip()

    with (
        OUTPUT_ROOT / "RESULTS_INTERPRETATION_v2.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(text + "\n")


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> None:

    ensure_dir(OUTPUT_ROOT)

    print("=" * 114)
    print("NC-REU | Independent downstream stabilization validation v2")
    print(f"Data             : {DATA_PATH}")
    print(f"Labels           : {LABEL_PATH}")
    print(f"Output           : {OUTPUT_ROOT}")
    print(f"Frozen REUs      : {FROZEN_REU_PX}")
    print(f"Support factors  : {SUPPORT_FACTORS}")
    print(f"Repeats          : {N_REPEATS}")
    print(f"Train spectra    : {N_TRAIN_PER_CLASS} per class at EVERY support")
    print(f"Full reference   : {N_TRAIN_PER_CLASS} per class")
    print(f"Fixed test       : {N_TEST_PER_CLASS} per class")
    print(
        "Classifier        : "
        f"L2 -> StandardScaler -> PCA({PCA_N_COMPONENTS}) -> "
        f"RBF-SVM(C={SVM_C}, gamma={SVM_GAMMA})"
    )
    print("Hyperparameter CV : NONE")
    print("REU retuning      : NONE")
    print("=" * 114)

    t0 = time.time()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    cube, labels, data_info = load_data()

    print()
    print(
        f"[MAT loader] cube variable='{data_info['cube_variable']}', "
        f"loaded shape={tuple(data_info['cube_shape'])}, dtype={data_info['dtype']}"
    )

    print(
        f"[MAT loader] label variable='{data_info['label_variable']}', "
        f"loaded shape={tuple(data_info['label_shape'])}"
    )

    print(
        "[Data] class pixels after spectral validity:",
        data_info["class_pixel_counts_after_spectral_validity"],
    )

    save_predeclared_design(
        data_info=data_info
    )

    # -------------------------------------------------------------------------
    # Geometry-defined spatial split
    # -------------------------------------------------------------------------
    print()
    print("[1/5] Build geometry-defined spatial train/test reservoirs...")

    domains, split_audit = build_class_domains(
        cube=cube,
        labels=labels,
    )

    split_audit.to_csv(
        OUTPUT_ROOT / "spatial_split_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        split_audit[
            [
                "class_id",
                "rock",
                "selected_split",
                "axis",
                "train_pixels",
                "test_pixels",
                "valid_nested_centers",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Fixed spatially separated test set
    # -------------------------------------------------------------------------
    print()
    print("[2/5] Select ONE fixed spatially separated test set...")

    x_test, y_test, test_audit = make_fixed_test_set(
        domains=domains
    )

    test_audit.to_csv(
        OUTPUT_ROOT / "fixed_test_set_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[Test] X={x_test.shape}, y={y_test.shape}, "
        f"class counts="
        f"{dict(zip(*np.unique(y_test, return_counts=True)))}"
    )

    # -------------------------------------------------------------------------
    # Repeated paired support experiment
    # -------------------------------------------------------------------------
    print()
    print("[3/5] Run fixed RBF-SVM downstream experiment...")

    results = run_experiment(
        cube=cube,
        domains=domains,
        x_test=x_test,
        y_test=y_test,
    )

    classification_repeat = results[
        "classification_repeat"
    ]

    full_repeat = results[
        "full_repeat"
    ]

    per_class_repeat = results[
        "per_class_repeat"
    ]

    agreement_repeat = results[
        "agreement_repeat"
    ]

    center_audit = results[
        "center_audit"
    ]

    window_audit = results[
        "window_audit"
    ]

    support_predictions = results[
        "support_predictions"
    ]

    full_predictions = results[
        "full_predictions"
    ]

    # Save audits and repeat-level metrics immediately.
    classification_repeat.to_csv(
        OUTPUT_ROOT / "classification_repeat_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_repeat.to_csv(
        OUTPUT_ROOT / "full_reference_repeat_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_repeat.to_csv(
        OUTPUT_ROOT / "per_class_f1_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )

    agreement_repeat.to_csv(
        OUTPUT_ROOT / "full_field_agreement_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )

    center_audit.to_csv(
        OUTPUT_ROOT / "training_center_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_audit.to_csv(
        OUTPUT_ROOT / "training_window_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_PREDICTIONS:
        np.savez_compressed(
            OUTPUT_ROOT / "prediction_matrices.npz",

            support_predictions=support_predictions,
            full_predictions=full_predictions,

            y_test=y_test.astype(np.int16),

            support_factors=np.asarray(
                SUPPORT_FACTORS,
                dtype=np.float32,
            ),

            class_ids=np.asarray(
                CLASS_IDS,
                dtype=np.int16,
            ),
        )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    print()
    print("[4/5] Summarize performance, instability and full-field agreement...")

    classification_summary = summarize_classification(
        classification_repeat
    )

    full_summary = summarize_full_reference(
        full_repeat
    )

    agreement_summary = summarize_agreement(
        agreement_repeat
    )

    per_class_summary = summarize_per_class(
        per_class_repeat
    )

    instability_summary = summarize_prediction_instability(
        support_predictions=support_predictions,
        full_predictions=full_predictions,
    )

    classification_summary.to_csv(
        OUTPUT_ROOT / "classification_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_summary.to_csv(
        OUTPUT_ROOT / "full_reference_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    agreement_summary.to_csv(
        OUTPUT_ROOT / "full_field_agreement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_summary.to_csv(
        OUTPUT_ROOT / "per_class_f1_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    instability_summary.to_csv(
        OUTPUT_ROOT / "prediction_instability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assessment = build_assessment(
        classification_repeat=classification_repeat,
        classification_summary=classification_summary,
        full_summary=full_summary,
        agreement_summary=agreement_summary,
        instability_summary=instability_summary,
    )

    save_json(
        OUTPUT_ROOT / "downstream_v2_assessment.json",
        assessment,
    )

    save_results_interpretation(
        assessment
    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    print()
    print("[5/5] Draw NC-style downstream validation figure...")

    plot_main_figure(
        classification_repeat=classification_repeat,
        classification_summary=classification_summary,
        full_summary=full_summary,
        agreement_repeat=agreement_repeat,
        agreement_summary=agreement_summary,
        instability_summary=instability_summary,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 114)
    print("CLASSIFICATION SUMMARY")
    print("=" * 114)

    print(
        classification_summary[
            [
                "lambda_value",
                "macro_f1_mean",
                "macro_f1_std",
                "macro_f1_ci95_low",
                "macro_f1_ci95_high",
                "absolute_macro_f1_gap_to_full_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("FULL-RESERVOIR REFERENCE")
    print(
        full_summary[
            [
                "macro_f1_mean",
                "macro_f1_std",
                "macro_f1_ci95_low",
                "macro_f1_ci95_high",
                "oa_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("PREDICTION INSTABILITY")
    print(
        instability_summary[
            [
                "model_type",
                "lambda_value",
                "pairwise_disagreement_mean",
                "normalized_prediction_entropy_mean",
                "consensus_fraction_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("AGREEMENT WITH PAIRED FULL-RESERVOIR MODEL")
    print(
        agreement_summary[
            [
                "lambda_value",
                "prediction_agreement_mean",
                "prediction_agreement_ci95_low",
                "prediction_agreement_ci95_high",
                "cohen_kappa_to_full_mean",
                "predicted_class_distribution_js_to_full_mean",
            ]
        ].to_string(index=False)
    )

    runtime = time.time() - t0

    print()
    print("=" * 114)
    print("V2 downstream validation complete")
    print(f"Runtime : {runtime / 60.0:.2f} min")
    print(f"Output  : {OUTPUT_ROOT}")
    print()
    print("IMPORTANT INTERPRETATION")
    print(
        "Judge the validation from the joint pattern of Macro-F1, "
        "prediction instability and agreement with the paired full-reservoir "
        "model. Do not select a favorable support or retune the RBF-SVM after "
        "seeing the curves."
    )
    print("=" * 114)


if __name__ == "__main__":
    main()
