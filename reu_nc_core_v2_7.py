# -*- coding: utf-8 -*-
"""
NC-REU core algorithms V2.7 correlation-corrected
======================
A threshold- and uncertainty-controlled representative-scale framework for
hyperspectral spectral-spatial fields.

Core changes relative to the previous implementation
----------------------------------------------------
1. No weighted composite score and no argmax "optimal window".
2. A scale is accepted only when spectral-distribution, composition,
   configuration, uncertainty and stability constraints are satisfied.
3. Window position is treated as uncertainty rather than searched only for the
   single window closest to the global mean.
4. Pixel support size is the primary scale; physical conversion is optional.
5. The framework supports global REU, direction/shape-dependent REU, local
   failure diagnosis, and an optional representative-unit set.

The code is intended as a reproducible research framework. Thresholds must be
fixed before final test-set analysis and justified from measurement precision,
scientific tolerance or downstream task requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import jensenshannon
from scipy.signal import savgol_filter, fftconvolve
from scipy.optimize import curve_fit
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from skimage.measure import label as cc_label, regionprops

from reu_nc_geometry_v2_7 import ShapeSpec, ShapeGeometry, make_shape_geometry

CORE_VERSION = "2.7-robust-scale-inference"

EPS = 1.0e-12


@dataclass
class PreprocessConfig:
    """Spectral preprocessing used before state-space construction."""

    remove_nonfinite_pixels: bool = True
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    savgol_enabled: bool = False
    savgol_window: int = 9
    savgol_polyorder: int = 2
    l2_normalize_for_state_model: bool = True


@dataclass
class StateModelConfig:
    """Low-dimensional spectral-state model."""

    n_components: int = 6
    n_states: int = 8
    fit_sample_size: int = 50000
    soft_temperature_quantile: float = 0.5
    random_seed: int = 3407


@dataclass
class WindowConfig:
    """Multi-scale sampling configuration."""

    window_sizes_px: Sequence[int] = field(
        default_factory=lambda: [6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128]
    )
    # Equal-area support geometries. Angles are counter-clockwise from image x-axis.
    shape_specs: Sequence[ShapeSpec] = field(
        default_factory=lambda: [ShapeSpec("square", "rectangle", 1.0, 0.0)]
    )
    overlap: float = 0.75
    # Do not let the translation stride grow indefinitely with window size.
    # A percentage-only stride severely under-samples large windows and can
    # create an artificial "not enough windows" failure. The actual stride is
    # min(window_size * (1-overlap), max_position_stride_px).
    max_position_stride_px: Optional[int] = 8
    # High coverage avoids changing rectangular-window geometry at class boundaries.
    min_region_coverage: float = 0.98
    min_valid_pixels: int = 16
    min_windows_per_scale: int = 12
    max_windows_per_scale: int = 500
    max_spectra_per_window: int = 1200
    max_reference_spectra: int = 6000
    random_seed: int = 3407


@dataclass
class MetricConfig:
    """Distribution and spatial-configuration metrics."""

    projection_count: int = 24
    projection_quantiles: int = 33
    cooccurrence_distances_px: Sequence[int] = field(default_factory=lambda: [1, 2, 4])
    cooccurrence_directions: Sequence[str] = field(
        default_factory=lambda: ["horizontal", "vertical", "diag_down", "diag_up"]
    )
    rare_state_probability: float = 0.08


@dataclass
class CorrelationConfig:
    """Corrected multivariate spatial-correlation analysis.

    The estimator works in PCA latent space, compares directions by *Euclidean*
    lag rather than integer grid step, fits a normalized generalized-exponential
    semivariogram, and explicitly reports censoring instead of replacing an
    unobserved range by the largest tested lag.
    """

    max_lag_fraction_short_side: float = 0.45
    max_lag_px: int = 160
    radial_pair_count: int = 160000
    radial_bins: int = 24
    min_pairs_per_bin: int = 100
    directional_target_lags_px: Sequence[float] = field(
        default_factory=lambda: [1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 144, 160]
    )
    min_pairs_per_directional_lag: int = 100
    range_fraction: float = 0.95
    minimum_fit_r2: float = 0.60
    empirical_plateau_fraction: float = 0.80
    nugget_upper: float = 0.50
    power_lower: float = 0.50
    power_upper: float = 2.00
    bootstrap_iterations: int = 80
    bootstrap_max_pairs_per_lag: int = 3000
    adaptive_min_anisotropy_ratio: float = 1.15
    adaptive_min_principal_frequency: float = 0.60
    adaptive_min_reliable_directions: int = 4
    adaptive_max_aspect_ratio: float = 3.0

    # V2.7 robust scale inference.  A low-R2 parametric range can still be
    # retained when the fitted range and empirical first crossing agree closely.
    practical_range_low_r2_agreement_tolerance: float = 0.30
    practical_range_tail_min: float = 0.90

    # Dual-scale anisotropy gate: an adaptive ellipse is enabled only when
    # integral and practical ranges give concordant directional evidence.
    adaptive_integral_ratio_min: float = 1.25
    adaptive_integral_ci_lower_min: float = 1.15
    adaptive_practical_ratio_min: float = 1.15
    adaptive_min_practical_reliable_directions: int = 3
    adaptive_require_principal_concordance: bool = True

    random_seed: int = 3407


@dataclass
class ConvergenceConfig:
    """Pre-registered acceptance criteria for the representative scale."""

    spectral_error_q95_max: float = 0.10
    composition_js_q95_max: float = 0.10
    configuration_js_q95_max: float = 0.15
    spectral_funnel_width_max: float = 0.10
    composition_funnel_width_max: float = 0.10
    configuration_funnel_width_max: float = 0.15
    rare_state_coverage_q05_min: float = 0.80
    # Unsupervised spectral states are statistical clusters rather than verified
    # rare minerals/classes. Keep rare-state preservation as a diagnostic by
    # default; enforce it only when state labels have independent physical meaning.
    enforce_rare_state_constraint: bool = False
    # Stability is evaluated over a forward multi-scale block. The score is
    # the maximum metric range divided by that metric's acceptance threshold.
    relative_stability_max: float = 0.20
    stability_window_scales: int = 3
    consecutive_scales: int = 3
    attribute_error_q95_max: Optional[float] = None
    task_loss_q95_max: Optional[float] = None


@dataclass
class REUConfig:
    """Complete analysis configuration."""

    # Set pixel_size=None when no independently calibrated physical resolution exists.
    pixel_size: Optional[float] = None
    physical_unit: str = "pixel"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    state_model: StateModelConfig = field(default_factory=StateModelConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralStateModel:
    """Fitted latent spectral-state model and corrected spatial structure."""

    pca: PCA
    kmeans: MiniBatchKMeans
    temperature: float
    projection_vectors: np.ndarray
    projection_probabilities: np.ndarray
    reference_projection_quantiles: np.ndarray
    reference_projection_scale: np.ndarray
    global_composition: np.ndarray
    global_configuration: np.ndarray
    global_mean_spectrum: np.ndarray
    global_rare_states: np.ndarray
    state_map: np.ndarray
    soft_state_cube: np.ndarray
    latent_cube: np.ndarray
    valid_mask: np.ndarray

    # Corrected isotropic multivariate correlation range. ``None`` means the
    # 95%-sill range was not reliably observed inside the available field.
    correlation_length_px: Optional[float]
    correlation_length_fit_px: Optional[float]
    correlation_length_lower_bound_px: Optional[float]
    correlation_length_censored: bool
    correlation_fit_r2: Optional[float]
    correlation_length_ci95_px: Tuple[Optional[float], Optional[float]]

    composition_entropy: float
    configuration_entropy: float

    # Corrected directional structure diagnostics. The scalar anisotropy index
    # is range_ratio-1 only when all required directional ranges are reliable;
    # otherwise it is NaN rather than a fabricated directionality value.
    anisotropy_index: float
    directional_correlation_lengths_px: Dict[str, Optional[float]]
    directional_correlation_diagnostics: Dict[str, Dict[str, Any]]
    principal_direction_deg: float
    adaptive_aspect_ratio: float
    adaptive_geometry_status: str
    principal_direction_bootstrap_frequency: float
    anisotropy_ratio_ci95: Tuple[Optional[float], Optional[float]]
    correlation_diagnostics: Dict[str, Any]


# -----------------------------------------------------------------------------
# I/O and region extraction
# -----------------------------------------------------------------------------

def _normalise_loaded_array(value: Any) -> Optional[np.ndarray]:
    """Return a squeezed numeric ndarray, or None for metadata/object variables."""

    arr = np.asarray(value)
    if arr.dtype.fields is not None:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.squeeze(arr)


def _find_numeric_array(
    mat: Mapping[str, Any],
    ndim: int,
    preferred_key: Optional[str],
) -> Tuple[str, np.ndarray]:
    """Find a numeric MAT variable after singleton dimensions are removed.

    A preferred key is attempted first. If it is absent or has the wrong
    post-squeeze dimensionality, automatic detection is used instead of
    terminating. This supports label cubes stored as ``(H, W, 1)``.
    """

    if preferred_key is not None and preferred_key in mat:
        preferred = _normalise_loaded_array(mat[preferred_key])
        if preferred is not None and preferred.ndim == ndim:
            return preferred_key, preferred
        print(
            f"[MAT loader] preferred variable '{preferred_key}' is not a numeric "
            f"{ndim}-D array after squeeze; falling back to automatic detection."
        )

    candidates: List[Tuple[float, str, np.ndarray]] = []
    for key, value in mat.items():
        if key.startswith('__'):
            continue
        arr = _normalise_loaded_array(value)
        if arr is None or arr.ndim != ndim:
            continue

        score = float(arr.size)
        key_lower = key.lower()
        if ndim == 2:
            if any(token in key_lower for token in ('label', 'ground', 'truth', 'gt', 'mask', 'class')):
                score *= 10.0
            # Discrete label maps are usually low-cardinality arrays.
            if arr.size <= 2_000_000:
                sample = arr.ravel()[::max(1, arr.size // 10000)]
                finite = sample[np.isfinite(sample)]
                if finite.size:
                    unique_count = np.unique(finite).size
                    if unique_count <= 256:
                        score *= 2.0
        candidates.append((score, key, arr))

    if not candidates:
        available = []
        for key, value in mat.items():
            if key.startswith('__'):
                continue
            arr = np.asarray(value)
            available.append(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        raise ValueError(
            f"No numeric {ndim}-D variable found after squeeze. Available variables: "
            + '; '.join(available)
        )

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, key, arr = candidates[0]
    return key, arr


def load_array(
    path: str | Path,
    key: Optional[str] = None,
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load NPY/NPZ/MAT/TIFF arrays with singleton-dimension support.

    Parameters
    ----------
    expected_shape:
        Optional exact shape check after squeezing. For a label image this can
        be set to the first two dimensions of the hyperspectral cube.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    selected_key: Optional[str] = None

    if suffix == '.npy':
        arr = np.load(path)
    elif suffix == '.npz':
        data = np.load(path)
        if key is None:
            if len(data.files) != 1:
                raise KeyError(f"NPZ contains {data.files}; specify a key.")
            key = data.files[0]
        arr = data[key]
        selected_key = key
    elif suffix == '.mat':
        mat = loadmat(path)
        if expected_ndim is None:
            if key is None:
                raise ValueError('expected_ndim or key is required for MAT files.')
            arr = np.asarray(mat[key])
            selected_key = key
        else:
            selected_key, arr = _find_numeric_array(mat, expected_ndim, key)
    elif suffix in {'.tif', '.tiff'}:
        import tifffile
        arr = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.squeeze(np.asarray(arr))
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"Loaded array from {path} has shape {arr.shape}; expected "
            f"{expected_ndim} dimensions after squeeze."
        )
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"Loaded array from {path} has shape {arr.shape}; expected {expected_shape}."
        )

    if suffix == '.mat':
        print(
            f"[MAT loader] {path.name}: selected variable '{selected_key}', "
            f"shape={arr.shape}, dtype={arr.dtype}"
        )
    return arr


def load_wavelengths(
    path: Optional[str | Path],
    n_bands: int,
    key: Optional[str] = None,
) -> np.ndarray:
    """Load a wavelength vector, or return band indices when unavailable."""

    if path is None:
        return np.arange(n_bands, dtype=np.float64)
    arr = load_array(path, key=key, expected_ndim=1).astype(np.float64).ravel()
    if arr.size != n_bands:
        raise ValueError(f"Wavelength count {arr.size} does not match band count {n_bands}.")
    return arr


def extract_regions_from_ground_truth(
    ground_truth: np.ndarray,
    min_pixels: int = 100,
    split_connected_components: bool = True,
) -> List[Dict[str, Any]]:
    """Extract independent regions. Disconnected components are not merged by default."""

    gt = np.asarray(ground_truth)
    if gt.ndim != 2:
        raise ValueError("Ground truth must be a 2-D label image.")

    regions: List[Dict[str, Any]] = []
    for class_id in np.unique(gt):
        if class_id <= 0:
            continue
        class_mask = gt == class_id
        component_map = cc_label(class_mask, connectivity=2) if split_connected_components else class_mask.astype(int)
        for component_index, prop in enumerate(regionprops(component_map), start=1):
            if prop.area < min_pixels:
                continue
            mask = component_map == prop.label
            y1, x1, y2, x2 = prop.bbox
            region_id = f"class_{int(class_id)}_component_{component_index}"
            regions.append(
                {
                    "region_id": region_id,
                    "class_id": int(class_id),
                    "component_id": int(component_index),
                    "mask": mask,
                    "pixel_count": int(prop.area),
                    "bbox": (int(y1), int(x1), int(y2), int(x2)),
                    "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
                    "coverage_ratio": float(prop.area / max((y2 - y1) * (x2 - x1), 1)),
                }
            )
    return regions


# -----------------------------------------------------------------------------
# Preprocessing and state-space model
# -----------------------------------------------------------------------------

def preprocess_cube(cube: np.ndarray, valid_mask: np.ndarray, config: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray]:
    cube = np.asarray(cube, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=bool).copy()
    if cube.ndim != 3 or mask.shape != cube.shape[:2]:
        raise ValueError("Cube must be HxWxB and mask must be HxW.")

    finite = np.all(np.isfinite(cube), axis=2)
    if config.remove_nonfinite_pixels:
        mask &= finite
    cube[~np.isfinite(cube)] = 0.0

    if mask.sum() == 0:
        raise ValueError("No valid pixels remain after finite-value filtering.")

    low, high = config.clip_percentiles
    region_values = cube[mask]
    band_low = np.nanpercentile(region_values, low, axis=0)
    band_high = np.nanpercentile(region_values, high, axis=0)
    cube = np.clip(cube, band_low[None, None, :], band_high[None, None, :])

    if config.savgol_enabled:
        window = int(config.savgol_window)
        if window % 2 == 0:
            window += 1
        if window <= config.savgol_polyorder:
            window = config.savgol_polyorder + 3
            if window % 2 == 0:
                window += 1
        if window < cube.shape[2]:
            cube[mask] = savgol_filter(
                cube[mask], window_length=window, polyorder=config.savgol_polyorder, axis=1
            ).astype(np.float32)
    return cube, mask


def _soft_assign(distances: np.ndarray, temperature: float) -> np.ndarray:
    logits = -np.square(distances) / max(2.0 * temperature * temperature, EPS)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True) + EPS
    return probs


def _composition_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > EPS]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def _configuration_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > EPS]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def _direction_offset(name: str, distance: int) -> Tuple[int, int]:
    mapping = {
        "horizontal": (0, distance),
        "vertical": (distance, 0),
        "diag_down": (distance, distance),
        "diag_up": (distance, -distance),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported direction '{name}'.")
    return mapping[name]


def compute_soft_cooccurrence(
    soft_cube: np.ndarray,
    mask: np.ndarray,
    distances: Sequence[int],
    directions: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute normalized soft state co-occurrence over distances and directions."""

    h, w, k = soft_cube.shape
    vectors: List[np.ndarray] = []
    direction_strengths: Dict[str, float] = {}

    for direction in directions:
        direction_values: List[float] = []
        for distance in distances:
            dy, dx = _direction_offset(direction, int(distance))
            y0a, y1a = max(0, -dy), min(h, h - dy)
            x0a, x1a = max(0, -dx), min(w, w - dx)
            y0b, y1b = y0a + dy, y1a + dy
            x0b, x1b = x0a + dx, x1a + dx
            if y1a <= y0a or x1a <= x0a:
                vectors.append(np.zeros(k * k, dtype=np.float64))
                direction_values.append(0.0)
                continue
            pair_mask = mask[y0a:y1a, x0a:x1a] & mask[y0b:y1b, x0b:x1b]
            if pair_mask.sum() == 0:
                matrix = np.zeros((k, k), dtype=np.float64)
            else:
                a = soft_cube[y0a:y1a, x0a:x1a][pair_mask]
                b = soft_cube[y0b:y1b, x0b:x1b][pair_mask]
                matrix = a.T @ b
                matrix = 0.5 * (matrix + matrix.T)
                matrix /= matrix.sum() + EPS
            vectors.append(matrix.ravel())
            direction_values.append(float(np.trace(matrix)))
        direction_strengths[direction] = float(np.mean(direction_values))

    vector = np.concatenate(vectors) if vectors else np.zeros(k * k, dtype=np.float64)
    vector /= vector.sum() + EPS
    return vector, direction_strengths



def estimate_legacy_correlation_length(
    scalar_map: np.ndarray,
    mask: np.ndarray,
    random_seed: int,
    max_pairs: int = 60000,
    n_bins: int = 30,
) -> float:
    """Legacy V2.4 PC1 estimator retained only for audit/comparison."""
    coords = np.column_stack(np.where(mask))
    values = scalar_map[mask].astype(np.float64)
    if len(coords) < 20 or np.nanstd(values) < EPS:
        return 1.0
    rng = np.random.default_rng(random_seed)
    n_pairs = min(max_pairs, max(len(coords) * 10, 2000))
    ia = rng.integers(0, len(coords), size=n_pairs)
    ib = rng.integers(0, len(coords), size=n_pairs)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    distances = np.linalg.norm(coords[ia] - coords[ib], axis=1)
    semivariance = 0.5 * np.square(values[ia] - values[ib])
    max_distance = np.quantile(distances, 0.90)
    bins = np.linspace(0.0, max_distance, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    gamma = np.full(n_bins, np.nan)
    for idx in range(n_bins):
        sel = (distances >= bins[idx]) & (distances < bins[idx + 1])
        if sel.sum() >= 20:
            gamma[idx] = np.median(semivariance[sel])
    finite = np.isfinite(gamma)
    if finite.sum() < 5:
        return float(max(1.0, np.sqrt(mask.sum()) / 10.0))
    sill = np.nanmedian(gamma[finite][-max(3, finite.sum() // 4):])
    if sill <= EPS:
        return 1.0
    crossing = np.where(gamma >= 0.95 * sill)[0]
    if crossing.size == 0:
        return float(centers[np.nanargmax(gamma)])
    return float(max(centers[crossing[0]], 1.0))


def _generalized_exponential_variogram(
    h: np.ndarray | float,
    nugget: float,
    scale: float,
    power: float,
) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return nugget + (1.0 - nugget) * (1.0 - np.exp(-np.power(np.maximum(h, 0.0) / max(scale, EPS), power)))


def _fit_normalized_variogram(
    lag_px: Sequence[float],
    gamma_normalized: Sequence[float],
    pair_count: Sequence[int],
    cfg: CorrelationConfig,
) -> Dict[str, Any]:
    """Fit a normalized generalized-exponential semivariogram.

    Because the multivariate semivariance is normalized by its theoretical
    independent-pair sill, the model asymptote is fixed at 1. This avoids the
    V2.4 inconsistency of comparing a median semivariance with a variance sill.
    """
    h = np.asarray(lag_px, dtype=float)
    g = np.asarray(gamma_normalized, dtype=float)
    n = np.asarray(pair_count, dtype=float)
    finite = np.isfinite(h) & np.isfinite(g) & np.isfinite(n) & (h > 0) & (n > 0)
    h, g, n = h[finite], g[finite], n[finite]
    if len(h) < 5 or np.ptp(h) <= EPS:
        return {
            "status": "insufficient_lags", "range_px": None,
            "fit_range_px": None, "range_censored": False,
            "fit_r2": None, "largest_tested_lag_px": float(np.max(h)) if len(h) else None,
        }
    order = np.argsort(h)
    h, g, n = h[order], g[order], n[order]
    max_h = float(np.max(h))
    # Fit tolerates moderate overshoot above sill, but the physical asymptote is 1.
    gfit = np.clip(g, 0.0, 2.5)
    nug0 = float(np.clip(np.nanmin(gfit[:min(3, len(gfit))]), 0.0, min(0.20, cfg.nugget_upper)))
    target = nug0 + 0.63 * (1.0 - nug0)
    a0 = float(h[int(np.argmin(np.abs(gfit - target)))])
    a0 = float(np.clip(a0, 0.5, max_h * 2.0))
    sigma = 1.0 / np.sqrt(np.maximum(n, 1.0))
    try:
        popt, _ = curve_fit(
            _generalized_exponential_variogram,
            h, gfit,
            p0=(nug0, a0, 1.0),
            sigma=sigma,
            absolute_sigma=False,
            bounds=(
                (0.0, 0.25, cfg.power_lower),
                (cfg.nugget_upper, max(max_h * 8.0, 2.0), cfg.power_upper),
            ),
            maxfev=30000,
        )
        nugget, scale, power = map(float, popt)
        pred = _generalized_exponential_variogram(h, *popt)
        ss_res = float(np.sum((gfit - pred) ** 2))
        ss_tot = float(np.sum((gfit - np.mean(gfit)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, EPS))
        target_fraction = float(np.clip(cfg.range_fraction, 0.50, 0.999))
        remaining = (1.0 - target_fraction) / max(1.0 - nugget, EPS)
        if remaining <= 0 or remaining >= 1:
            range_fit = np.nan
        else:
            range_fit = float(scale * np.power(-np.log(remaining), 1.0 / power))
        tail = gfit[h >= 0.70 * max_h]
        empirical_tail_max = float(np.nanmax(tail)) if len(tail) else float(np.nanmax(gfit))
        empirical_cross = None
        gm = np.maximum.accumulate(gfit)
        idx = np.where(gm >= target_fraction)[0]
        if len(idx):
            empirical_cross = float(h[int(idx[0])])
        fit_ok = bool(np.isfinite(r2) and r2 >= cfg.minimum_fit_r2 and np.isfinite(range_fit))
        censored = bool(
            fit_ok and (
                range_fit > max_h * 1.02
                or empirical_tail_max < cfg.empirical_plateau_fraction
            )
        )
        if not fit_ok:
            status = "poor_fit"
            reported = None
            lower = None
        elif censored:
            status = "right_censored"
            reported = None
            lower = max_h
        else:
            status = "range_observed"
            reported = range_fit
            lower = None
        return {
            "status": status,
            "range_px": float(reported) if reported is not None else None,
            "fit_range_px": float(range_fit) if np.isfinite(range_fit) else None,
            "range_censored": bool(censored),
            "range_lower_bound_px": float(lower) if lower is not None else None,
            "fit_r2": r2,
            "nugget": nugget,
            "scale_parameter_px": scale,
            "shape_power": power,
            "largest_tested_lag_px": max_h,
            "empirical_tail_max": empirical_tail_max,
            "empirical_range_crossing_px": empirical_cross,
            "lag_px": h.tolist(),
            "gamma_normalized": g.tolist(),
            "pair_count": n.astype(int).tolist(),
            "fitted_gamma": np.asarray(pred, dtype=float).tolist(),
        }
    except Exception as exc:
        return {
            "status": "fit_failed", "range_px": None, "fit_range_px": None,
            "range_censored": False, "range_lower_bound_px": None,
            "fit_r2": None, "largest_tested_lag_px": max_h,
            "error": str(exc), "lag_px": h.tolist(),
            "gamma_normalized": g.tolist(), "pair_count": n.astype(int).tolist(),
        }


def _latent_variogram_setup(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return component coefficients for a variance-consistent latent metric.

    Each PC difference is standardized by its regional variance and then
    weighted by PCA explained variance. For PCA scores this is algebraically
    close to total squared latent distance divided by total latent variance,
    so PC importance is retained *without double-counting* its variance.
    The independent-pair normalized semivariogram therefore has sill ~1.
    """
    valid = np.asarray(mask, dtype=bool)
    z = np.asarray(latent_cube, dtype=np.float64)
    m = z.shape[2]
    variances = np.maximum(np.nanvar(z[valid], axis=0), EPS)
    explained = np.asarray(getattr(pca, "explained_variance_ratio_", np.ones(m)), dtype=float)[:m]
    explained = np.maximum(explained, 0.0)
    if np.sum(explained) <= EPS:
        explained = np.ones(m, dtype=float) / max(m, 1)
    else:
        explained = explained / np.sum(explained)
    coefficients = explained / variances
    sill = float(np.sum(explained))  # exactly ~1 after normalization
    return coefficients, variances, max(sill, EPS)


def _pair_semivariance_values(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    sill: float,
) -> np.ndarray:
    diff2 = np.square(np.asarray(a, float) - np.asarray(b, float))
    return 0.5 * (diff2 @ weights) / max(sill, EPS)


def _bootstrap_variogram_ranges(
    lag_px: np.ndarray,
    samples_by_lag: Sequence[np.ndarray],
    cfg: CorrelationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    B = max(0, int(cfg.bootstrap_iterations))
    if B <= 0 or len(lag_px) < 5:
        return np.array([], dtype=float)
    out = np.full(B, np.nan, dtype=float)
    counts = np.array([max(len(v), 1) for v in samples_by_lag], dtype=int)
    for b in range(B):
        gb = []
        for arr in samples_by_lag:
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                gb.append(np.nan)
                continue
            n = min(len(arr), max(100, int(cfg.bootstrap_max_pairs_per_lag)))
            idx = rng.integers(0, len(arr), size=n)
            gb.append(float(np.mean(arr[idx])))
        fit = _fit_normalized_variogram(lag_px, gb, counts, cfg)
        value = fit.get("fit_range_px")
        if value is not None and np.isfinite(value) and (fit.get("fit_r2") or -np.inf) >= max(0.30, cfg.minimum_fit_r2 - 0.20):
            out[b] = float(value)
    return out


def _radial_multivariate_variogram(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    coords = np.column_stack(np.where(valid))
    z = np.asarray(latent_cube, dtype=np.float64)[valid]
    weights, variances, sill = _latent_variogram_setup(latent_cube, valid, pca)
    h, w = valid.shape
    max_lag = float(min(cfg.max_lag_px, cfg.max_lag_fraction_short_side * min(h, w)))
    if len(coords) < 100 or max_lag < 4:
        return {"status": "insufficient_region", "range_px": None, "range_censored": False}

    n_pairs = max(10000, int(cfg.radial_pair_count))
    ia = rng.integers(0, len(coords), size=n_pairs)
    ib = rng.integers(0, len(coords), size=n_pairs)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    dist = np.linalg.norm(coords[ia] - coords[ib], axis=1)
    keep = (dist > 0) & (dist <= max_lag)
    ia, ib, dist = ia[keep], ib[keep], dist[keep]
    if len(dist) < cfg.min_pairs_per_bin * 5:
        return {"status": "insufficient_pairs", "range_px": None, "range_censored": False, "max_lag_px": max_lag}
    semivar = _pair_semivariance_values(z[ia], z[ib], weights, sill)

    edges = np.linspace(0.0, max_lag, int(cfg.radial_bins) + 1)
    lags: List[float] = []
    gammas: List[float] = []
    counts: List[int] = []
    samples: List[np.ndarray] = []
    for i in range(len(edges) - 1):
        sel = (dist >= edges[i]) & (dist < edges[i+1])
        n = int(sel.sum())
        if n < cfg.min_pairs_per_bin:
            continue
        vals = semivar[sel]
        lags.append(float(np.mean(dist[sel])))
        gammas.append(float(np.mean(vals)))
        counts.append(n)
        cap = min(n, int(cfg.bootstrap_max_pairs_per_lag))
        if n > cap:
            idx = rng.choice(n, cap, replace=False)
            vals = vals[idx]
        samples.append(np.asarray(vals, dtype=np.float32))

    fit = _fit_normalized_variogram(lags, gammas, counts, cfg)
    boot = _bootstrap_variogram_ranges(np.asarray(lags, float), samples, cfg, rng)
    finite_boot = boot[np.isfinite(boot)]
    if len(finite_boot) >= 20:
        ci = tuple(float(v) for v in np.quantile(finite_boot, [0.025, 0.975]))
    else:
        ci = (None, None)
    fit.update({
        "bootstrap_range_ci95_px": ci,
        "bootstrap_valid_count": int(len(finite_boot)),
        "max_requested_lag_px": max_lag,
        "latent_component_coefficients": weights.tolist(),
        "latent_component_variances": variances.tolist(),
        "latent_metric_note": "standardized PC differences weighted by PCA explained variance",
        "normalization_sill": sill,
        "accepted_random_pair_count": int(len(dist)),
    })
    return fit


def _direction_offset_for_target(angle_deg: int, target_distance_px: float) -> Tuple[int, int]:
    a = int(angle_deg) % 180
    target = max(float(target_distance_px), 1.0)
    if a == 0:
        return 0, max(1, int(round(target)))
    if a == 90:
        return max(1, int(round(target))), 0
    if a in (45, 135):
        k = max(1, int(round(target / math.sqrt(2.0))))
        return (k, k) if a == 45 else (k, -k)
    raise ValueError(f"Only 0/45/90/135 degrees are supported, got {angle_deg}.")


def _directional_multivariate_variogram(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
    rng: np.random.Generator,
    directions_deg: Sequence[int] = (0, 45, 90, 135),
) -> Dict[str, Dict[str, Any]]:
    z = np.asarray(latent_cube, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    h, w = valid.shape
    weights, _, sill = _latent_variogram_setup(latent_cube, valid, pca)
    max_common_lag = float(min(cfg.max_lag_px, cfg.max_lag_fraction_short_side * min(h, w)))
    results: Dict[str, Dict[str, Any]] = {}

    for angle in directions_deg:
        offsets: List[Tuple[int, int]] = []
        for target in cfg.directional_target_lags_px:
            dy, dx = _direction_offset_for_target(int(angle), float(target))
            actual = float(math.hypot(dy, dx))
            if actual <= max_common_lag + 1.0e-9 and (dy, dx) not in offsets:
                offsets.append((dy, dx))
        lags: List[float] = []
        gammas: List[float] = []
        counts: List[int] = []
        samples: List[np.ndarray] = []
        used_offsets: List[Tuple[int, int]] = []
        for dy, dx in offsets:
            y0a, y1a = max(0, -dy), min(h, h - dy)
            x0a, x1a = max(0, -dx), min(w, w - dx)
            y0b, y1b = y0a + dy, y1a + dy
            x0b, x1b = x0a + dx, x1a + dx
            if y1a <= y0a or x1a <= x0a:
                continue
            pm = valid[y0a:y1a, x0a:x1a] & valid[y0b:y1b, x0b:x1b]
            n = int(pm.sum())
            if n < cfg.min_pairs_per_directional_lag:
                continue
            a = z[y0a:y1a, x0a:x1a][pm]
            b = z[y0b:y1b, x0b:x1b][pm]
            vals = _pair_semivariance_values(a, b, weights, sill)
            actual = float(math.hypot(dy, dx))
            lags.append(actual)
            gammas.append(float(np.mean(vals)))
            counts.append(n)
            used_offsets.append((int(dy), int(dx)))
            cap = min(n, int(cfg.bootstrap_max_pairs_per_lag))
            if n > cap:
                idx = rng.choice(n, cap, replace=False)
                vals = vals[idx]
            samples.append(np.asarray(vals, dtype=np.float32))

        fit = _fit_normalized_variogram(lags, gammas, counts, cfg)
        boot = _bootstrap_variogram_ranges(np.asarray(lags, float), samples, cfg, rng)
        finite_boot = boot[np.isfinite(boot)]
        ci = tuple(float(v) for v in np.quantile(finite_boot, [0.025, 0.975])) if len(finite_boot) >= 20 else (None, None)
        fit.update({
            "angle_deg": int(angle),
            "offsets_dy_dx": [list(v) for v in used_offsets],
            "max_common_euclidean_lag_px": max_common_lag,
            "bootstrap_range_ci95_px": ci,
            "bootstrap_valid_count": int(len(finite_boot)),
            "_bootstrap_ranges": boot,
        })
        results[str(int(angle))] = fit
    return results


def estimate_corrected_correlation_structure(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
) -> Dict[str, Any]:
    """Estimate isotropic and directional multivariate correlation structure.

    Adaptive anisotropy is enabled only when all required directional ranges are
    observed (not right-censored), fit quality is adequate, the anisotropy-ratio
    bootstrap lower bound exceeds the pre-registered threshold, and the same
    principal direction dominates the bootstrap replicates. Otherwise the
    adaptive ellipse safely degenerates to a circle.
    """
    rng = np.random.default_rng(cfg.random_seed)
    radial = _radial_multivariate_variogram(latent_cube, mask, pca, cfg, rng)
    directional = _directional_multivariate_variogram(latent_cube, mask, pca, cfg, rng)

    finite_ranges: Dict[str, float] = {}
    reliable_angles: List[str] = []
    for angle, d in directional.items():
        v = d.get("range_px")
        reliable = bool(
            d.get("status") == "range_observed"
            and v is not None and np.isfinite(v)
            and (d.get("fit_r2") or -np.inf) >= cfg.minimum_fit_r2
        )
        if reliable:
            finite_ranges[angle] = float(v)
            reliable_angles.append(angle)

    principal = 0.0
    aspect = 1.0
    anisotropy = np.nan
    principal_freq = np.nan
    ratio_ci: Tuple[Optional[float], Optional[float]] = (None, None)
    adaptive_status = "degenerated_to_circle_insufficient_directional_evidence"

    if len(reliable_angles) >= int(cfg.adaptive_min_reliable_directions):
        point_values = np.asarray([finite_ranges[a] for a in reliable_angles], float)
        point_ratio = float(np.max(point_values) / max(np.min(point_values), EPS))
        point_principal = max(reliable_angles, key=lambda a: finite_ranges[a])

        B = max(len(np.asarray(directional[a].get("_bootstrap_ranges", []))) for a in reliable_angles)
        ratios: List[float] = []
        principals: List[str] = []
        for b in range(B):
            vals: Dict[str, float] = {}
            for a in reliable_angles:
                arr = np.asarray(directional[a].get("_bootstrap_ranges", []), dtype=float)
                if b >= len(arr) or not np.isfinite(arr[b]):
                    vals = {}
                    break
                vals[a] = float(arr[b])
            if len(vals) != len(reliable_angles):
                continue
            vv = np.asarray(list(vals.values()), float)
            ratios.append(float(np.max(vv) / max(np.min(vv), EPS)))
            principals.append(max(vals, key=vals.get))
        if len(ratios) >= 20:
            ratio_ci = tuple(float(v) for v in np.quantile(ratios, [0.025, 0.975]))
            principal_freq = float(np.mean(np.asarray(principals) == point_principal))
        lower = ratio_ci[0] if ratio_ci[0] is not None else -np.inf
        if (
            point_ratio >= cfg.adaptive_min_anisotropy_ratio
            and lower >= cfg.adaptive_min_anisotropy_ratio
            and np.isfinite(principal_freq)
            and principal_freq >= cfg.adaptive_min_principal_frequency
        ):
            principal = float(point_principal)
            aspect = float(np.clip(point_ratio, 1.0, cfg.adaptive_max_aspect_ratio))
            anisotropy = float(point_ratio - 1.0)
            adaptive_status = "anisotropy_supported"
        else:
            anisotropy = float(point_ratio - 1.0)
            adaptive_status = "degenerated_to_circle_uncertain_anisotropy"

    clean_directional: Dict[str, Dict[str, Any]] = {}
    directional_lengths: Dict[str, Optional[float]] = {}
    for angle, d in directional.items():
        clean = {k: v for k, v in d.items() if k != "_bootstrap_ranges"}
        clean_directional[angle] = clean
        directional_lengths[angle] = float(d["range_px"]) if d.get("range_px") is not None and np.isfinite(d["range_px"]) else None

    return {
        "isotropic": radial,
        "directional": clean_directional,
        "directional_ranges_px": directional_lengths,
        "anisotropy_index": float(anisotropy) if np.isfinite(anisotropy) else np.nan,
        "principal_direction_deg": principal,
        "adaptive_aspect_ratio": aspect,
        "adaptive_geometry_status": adaptive_status,
        "principal_direction_bootstrap_frequency": float(principal_freq) if np.isfinite(principal_freq) else np.nan,
        "anisotropy_ratio_ci95": ratio_ci,
    }



def fit_spectral_state_model(
    cube: np.ndarray,
    region_mask: np.ndarray,
    config: REUConfig,
) -> SpectralStateModel:
    """Fit PCA + soft spectral states and construct region-level references."""

    cube, mask = preprocess_cube(cube, region_mask, config.preprocess)
    h, w, b = cube.shape
    spectra = cube[mask].astype(np.float64)
    rng = np.random.default_rng(config.state_model.random_seed)

    if config.preprocess.l2_normalize_for_state_model:
        state_spectra = normalize(spectra, norm="l2")
    else:
        state_spectra = spectra.copy()

    fit_n = min(len(state_spectra), config.state_model.fit_sample_size)
    fit_idx = rng.choice(len(state_spectra), fit_n, replace=False)
    n_components = min(config.state_model.n_components, b, max(1, fit_n - 1))
    pca = PCA(n_components=n_components, random_state=config.state_model.random_seed)
    pca.fit(state_spectra[fit_idx])
    latent = pca.transform(state_spectra)

    n_states = min(config.state_model.n_states, max(2, fit_n // 20))
    kmeans = MiniBatchKMeans(
        n_clusters=n_states,
        random_state=config.state_model.random_seed,
        batch_size=min(4096, fit_n),
        n_init=10,
    )
    kmeans.fit(latent[fit_idx])
    distances = kmeans.transform(latent)
    nearest = np.min(distances, axis=1)
    temperature = float(np.quantile(nearest[nearest > EPS], config.state_model.soft_temperature_quantile))
    temperature = max(temperature, 1.0e-4)
    soft = _soft_assign(distances, temperature)

    latent_cube = np.full((h, w, n_components), np.nan, dtype=np.float32)
    latent_cube[mask] = latent.astype(np.float32)
    soft_cube = np.zeros((h, w, n_states), dtype=np.float32)
    soft_cube[mask] = soft.astype(np.float32)
    state_map = np.full((h, w), -1, dtype=np.int16)
    state_map[mask] = np.argmax(soft, axis=1).astype(np.int16)

    global_composition = soft.mean(axis=0)
    global_configuration, direction_strengths = compute_soft_cooccurrence(
        soft_cube,
        mask,
        config.metrics.cooccurrence_distances_px,
        config.metrics.cooccurrence_directions,
    )

    # Deterministic random projections for a sliced-Wasserstein-like distribution distance.
    projection_vectors = rng.normal(size=(config.metrics.projection_count, n_components))
    projection_vectors /= np.linalg.norm(projection_vectors, axis=1, keepdims=True) + EPS
    probs = np.linspace(0.0, 1.0, config.metrics.projection_quantiles)
    reference_sample_n = min(len(latent), config.windows.max_reference_spectra)
    ref_idx = rng.choice(len(latent), reference_sample_n, replace=False)
    reference_latent = latent[ref_idx]
    reference_proj = reference_latent @ projection_vectors.T
    reference_quantiles = np.quantile(reference_proj, probs, axis=0).T
    reference_scale = np.nanpercentile(reference_proj, 95, axis=0) - np.nanpercentile(reference_proj, 5, axis=0)
    reference_scale = np.maximum(reference_scale, EPS)

    # V2.7 corrected correlation structure: multivariate latent field,
    # Euclidean directional lag, consistent normalized sill, explicit censoring.
    correlation_structure = estimate_corrected_correlation_structure(
        latent_cube=latent_cube,
        mask=mask,
        pca=pca,
        cfg=config.correlation,
    )
    radial_corr = correlation_structure["isotropic"]
    directional_lengths = correlation_structure["directional_ranges_px"]
    principal_direction_deg = float(correlation_structure["principal_direction_deg"])
    adaptive_aspect_ratio = float(correlation_structure["adaptive_aspect_ratio"])
    anisotropy = float(correlation_structure["anisotropy_index"])
    rare_states = global_composition <= config.metrics.rare_state_probability

    return SpectralStateModel(
        pca=pca,
        kmeans=kmeans,
        temperature=temperature,
        projection_vectors=projection_vectors,
        projection_probabilities=probs,
        reference_projection_quantiles=reference_quantiles,
        reference_projection_scale=reference_scale,
        global_composition=global_composition,
        global_configuration=global_configuration,
        global_mean_spectrum=spectra.mean(axis=0),
        global_rare_states=rare_states,
        state_map=state_map,
        soft_state_cube=soft_cube,
        latent_cube=latent_cube,
        valid_mask=mask,
        correlation_length_px=(float(radial_corr["range_px"]) if radial_corr.get("range_px") is not None else None),
        correlation_length_fit_px=(float(radial_corr["fit_range_px"]) if radial_corr.get("fit_range_px") is not None else None),
        correlation_length_lower_bound_px=(float(radial_corr["range_lower_bound_px"]) if radial_corr.get("range_lower_bound_px") is not None else None),
        correlation_length_censored=bool(radial_corr.get("range_censored", False)),
        correlation_fit_r2=(float(radial_corr["fit_r2"]) if radial_corr.get("fit_r2") is not None else None),
        correlation_length_ci95_px=tuple(radial_corr.get("bootstrap_range_ci95_px", (None, None))),
        composition_entropy=_composition_entropy(global_composition),
        configuration_entropy=_configuration_entropy(global_configuration),
        anisotropy_index=anisotropy,
        directional_correlation_lengths_px=directional_lengths,
        directional_correlation_diagnostics=correlation_structure["directional"],
        principal_direction_deg=principal_direction_deg,
        adaptive_aspect_ratio=adaptive_aspect_ratio,
        adaptive_geometry_status=str(correlation_structure["adaptive_geometry_status"]),
        principal_direction_bootstrap_frequency=float(correlation_structure["principal_direction_bootstrap_frequency"]),
        anisotropy_ratio_ci95=tuple(correlation_structure["anisotropy_ratio_ci95"]),
        correlation_diagnostics=correlation_structure,
    )


# -----------------------------------------------------------------------------
# Window-level metrics
# -----------------------------------------------------------------------------

def _safe_js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def _sam_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= EPS:
        return 90.0
    cos_value = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def _spectral_distribution_error(latent: np.ndarray, model: SpectralStateModel) -> Tuple[float, np.ndarray]:
    projected = latent @ model.projection_vectors.T
    quantiles = np.quantile(projected, model.projection_probabilities, axis=0).T
    per_projection = np.mean(
        np.abs(quantiles - model.reference_projection_quantiles)
        / model.reference_projection_scale[:, None],
        axis=1,
    )
    # Smoothly maps an unbounded normalized distance to [0,1).
    error = float(np.mean(per_projection / (1.0 + per_projection)))
    return error, quantiles


def _candidate_positions(
    mask: np.ndarray,
    support_mask: np.ndarray,
    base_size_px: int,
    overlap: float,
    min_coverage: float,
    max_windows: int,
    rng: np.random.Generator,
    max_position_stride_px: Optional[int] = 8,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, int]]:
    """Generate valid translated positions for an arbitrary support mask.

    Coverage is calculated only inside the support geometry, not inside its
    rectangular bounding box. A convolution supplies the valid-pixel count for
    every translation. The translation stride is tied to the equal-area base
    scale and capped so large supports are not severely under-sampled.
    """

    region = np.asarray(mask, dtype=bool)
    support = np.asarray(support_mask, dtype=bool)
    h, w = region.shape
    sh, sw = support.shape
    if sh > h or sw > w or int(support.sum()) <= 0:
        return [], {
            "position_stride_y_px": 0,
            "position_stride_x_px": 0,
            "candidate_position_count": 0,
            "used_position_count": 0,
        }

    raw_step = max(1, int(round(float(base_size_px) * (1.0 - overlap))))
    if max_position_stride_px is None:
        step_y = step_x = raw_step
    else:
        cap = max(1, int(max_position_stride_px))
        step_y = step_x = min(raw_step, cap)

    # Cross-correlation between region validity and support mask.
    valid_counts = fftconvolve(
        region.astype(np.float32),
        support[::-1, ::-1].astype(np.float32),
        mode="valid",
    )
    support_area = float(support.sum())
    ys = np.arange(0, valid_counts.shape[0], step_y, dtype=int)
    xs = np.arange(0, valid_counts.shape[1], step_x, dtype=int)

    candidates: List[Tuple[int, int, float]] = []
    for y in ys:
        row_values = valid_counts[y, xs] / support_area
        accepted = np.where(row_values >= min_coverage)[0]
        for idx in accepted:
            x = int(xs[idx])
            coverage = float(np.clip(row_values[idx], 0.0, 1.0))
            candidates.append((int(y), x, coverage))

    full_count = len(candidates)
    if full_count > max_windows:
        chosen = np.sort(rng.choice(full_count, max_windows, replace=False))
        candidates = [candidates[int(i)] for i in chosen]

    return candidates, {
        "position_stride_y_px": int(step_y),
        "position_stride_x_px": int(step_x),
        "candidate_position_count": int(full_count),
        "used_position_count": int(len(candidates)),
    }


def analyze_multiscale_windows(
    cube: np.ndarray,
    region_mask: np.ndarray,
    model: SpectralStateModel,
    config: REUConfig,
    attribute_maps: Optional[Mapping[str, np.ndarray]] = None,
    task_loss_by_scale: Optional[Mapping[Tuple[int, str], float]] = None,
) -> pd.DataFrame:
    """Evaluate all valid positions for every equal-area support geometry.

    Square, circle, rotated rectangles and adaptive ellipses are compared using
    approximately equal support area ``base_size_px**2``. Pixels outside the
    actual support mask are excluded from all spectral, composition,
    configuration and attribute calculations.
    """

    cube = np.asarray(cube, dtype=np.float32)
    mask = np.asarray(region_mask, dtype=bool) & model.valid_mask
    rng = np.random.default_rng(config.windows.random_seed)

    attribute_maps = attribute_maps or {}
    attribute_reference: Dict[str, Tuple[float, float]] = {}
    for name, attr in attribute_maps.items():
        arr = np.asarray(attr, dtype=np.float64)
        if arr.shape != mask.shape:
            raise ValueError(f"Attribute map '{name}' has shape {arr.shape}, expected {mask.shape}.")
        values = arr[mask & np.isfinite(arr)]
        if values.size == 0:
            continue
        scale = np.quantile(values, 0.95) - np.quantile(values, 0.05)
        attribute_reference[name] = (float(np.mean(values)), float(max(scale, EPS)))

    enabled_specs = [spec for spec in config.windows.shape_specs if spec.enabled]
    if not enabled_specs:
        raise ValueError("No enabled support shape was configured.")

    records: List[Dict[str, Any]] = []
    for base_size in sorted(set(int(v) for v in config.windows.window_sizes_px)):
        for spec in enabled_specs:
            geometry = make_shape_geometry(
                base_size,
                spec,
                adaptive_angle_deg=model.principal_direction_deg,
                adaptive_aspect_ratio=model.adaptive_aspect_ratio,
            )
            support = geometry.support_mask
            win_h, win_w = support.shape
            positions, position_meta = _candidate_positions(
                mask=mask,
                support_mask=support,
                base_size_px=base_size,
                overlap=config.windows.overlap,
                min_coverage=config.windows.min_region_coverage,
                max_windows=config.windows.max_windows_per_scale,
                rng=rng,
                max_position_stride_px=config.windows.max_position_stride_px,
            )

            for window_id, (y, x, coverage) in enumerate(positions):
                patch_region = mask[y : y + win_h, x : x + win_w]
                local_mask = patch_region & support
                n_valid = int(local_mask.sum())
                if n_valid < config.windows.min_valid_pixels:
                    continue

                cube_patch = cube[y : y + win_h, x : x + win_w]
                latent_patch = model.latent_cube[y : y + win_h, x : x + win_w]
                soft_patch = model.soft_state_cube[y : y + win_h, x : x + win_w]

                local_spectra = cube_patch[local_mask].astype(np.float64)
                local_latent = latent_patch[local_mask].astype(np.float64)
                if len(local_spectra) > config.windows.max_spectra_per_window:
                    sample_idx = rng.choice(
                        len(local_spectra), config.windows.max_spectra_per_window, replace=False
                    )
                    spectral_sample = local_spectra[sample_idx]
                    latent_sample = local_latent[sample_idx]
                else:
                    spectral_sample = local_spectra
                    latent_sample = local_latent

                spectral_error, projection_quantiles = _spectral_distribution_error(
                    latent_sample, model
                )
                local_composition = soft_patch[local_mask].mean(axis=0)
                composition_js = _safe_js(local_composition, model.global_composition)
                local_configuration, _ = compute_soft_cooccurrence(
                    soft_patch,
                    local_mask,
                    config.metrics.cooccurrence_distances_px,
                    config.metrics.cooccurrence_directions,
                )
                configuration_js = _safe_js(local_configuration, model.global_configuration)
                mean_sam = _sam_deg(spectral_sample.mean(axis=0), model.global_mean_spectrum)

                rare_mask = model.global_rare_states & (model.global_composition > EPS)
                rare_total = int(rare_mask.sum())
                if rare_total == 0:
                    rare_coverage = 1.0
                    rare_proportional_recovery = 1.0
                    rare_presence_fraction = 1.0
                    global_rare_mass = 0.0
                    local_rare_mass = 0.0
                else:
                    global_rare_mass = float(model.global_composition[rare_mask].sum())
                    local_rare_mass = float(local_composition[rare_mask].sum())
                    rare_coverage = float(np.clip(
                        1.0 - abs(local_rare_mass - global_rare_mass)
                        / max(global_rare_mass, EPS),
                        0.0,
                        1.0,
                    ))
                    rare_ratio = local_composition[rare_mask] / np.maximum(
                        model.global_composition[rare_mask], EPS
                    )
                    rare_proportional_recovery = float(
                        np.mean(np.clip(rare_ratio, 0.0, 1.0))
                    )
                    detection_floor = np.maximum(
                        0.10 * model.global_composition[rare_mask],
                        1.0 / max(n_valid, 1),
                    )
                    rare_presence_fraction = float(
                        np.mean(local_composition[rare_mask] >= detection_floor)
                    )

                characteristic_size_px = geometry.characteristic_size_px
                characteristic_size_physical = (
                    characteristic_size_px * float(config.pixel_size)
                    if config.pixel_size is not None
                    else np.nan
                )
                record: Dict[str, Any] = {
                    "base_size_px": int(base_size),
                    "shape_name": geometry.name,
                    "shape_kind": geometry.kind,
                    "shape_angle_deg": geometry.angle_deg,
                    "shape_aspect_ratio": geometry.aspect_ratio,
                    "major_axis_px": geometry.major_axis_px,
                    "minor_axis_px": geometry.minor_axis_px,
                    "window_height_px": int(win_h),
                    "window_width_px": int(win_w),
                    "target_support_area_px": geometry.target_area_px,
                    "actual_support_area_px": geometry.actual_area_px,
                    "support_area_error_fraction": geometry.area_error_fraction,
                    "characteristic_size_px": characteristic_size_px,
                    "characteristic_size_physical": characteristic_size_physical,
                    "window_id": int(window_id),
                    "top": int(y),
                    "left": int(x),
                    "bottom": int(y + win_h),
                    "right": int(x + win_w),
                    "center_y": float(y + win_h / 2.0),
                    "center_x": float(x + win_w / 2.0),
                    "coverage_ratio": float(coverage),
                    "n_valid_pixels": int(n_valid),
                    "position_stride_y_px": position_meta["position_stride_y_px"],
                    "position_stride_x_px": position_meta["position_stride_x_px"],
                    "candidate_position_count": position_meta["candidate_position_count"],
                    "used_position_count": position_meta["used_position_count"],
                    "spectral_distribution_error": spectral_error,
                    "composition_js": composition_js,
                    "configuration_js": configuration_js,
                    "mean_sam_deg": mean_sam,
                    "rare_state_coverage": rare_coverage,
                    "rare_state_proportional_recovery": rare_proportional_recovery,
                    "rare_state_presence_fraction": rare_presence_fraction,
                    "rare_state_count": rare_total,
                    "global_rare_state_mass": global_rare_mass,
                    "local_rare_state_mass": local_rare_mass,
                    "local_composition_entropy": _composition_entropy(local_composition),
                    "local_configuration_entropy": _configuration_entropy(local_configuration),
                    "local_composition": json.dumps(local_composition.tolist()),
                    "local_configuration": json.dumps(local_configuration.tolist()),
                    "projection_quantiles": json.dumps(projection_quantiles.ravel().tolist()),
                    "task_loss": np.nan,
                }

                if task_loss_by_scale is not None:
                    record["task_loss"] = float(
                        task_loss_by_scale.get((base_size, geometry.name), np.nan)
                    )

                for name, (reference, scale) in attribute_reference.items():
                    attr_patch = np.asarray(attribute_maps[name], dtype=np.float64)[
                        y : y + win_h, x : x + win_w
                    ]
                    values = attr_patch[local_mask & np.isfinite(attr_patch)]
                    record[f"attribute_error__{name}"] = (
                        float(abs(np.mean(values) - reference) / scale)
                        if values.size else np.nan
                    )
                records.append(record)

    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Scale summaries and convergence decision
# -----------------------------------------------------------------------------

def _quantile_summary(values: pd.Series, prefix: str) -> Dict[str, float]:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_q05": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_width90": np.nan,
        }
    q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
    return {
        f"{prefix}_q05": float(q05),
        f"{prefix}_median": float(q50),
        f"{prefix}_q95": float(q95),
        f"{prefix}_width90": float(q95 - q05),
    }


def summarize_scales(window_df: pd.DataFrame, config: REUConfig) -> pd.DataFrame:
    """Summarize position-dependent responses at each pixel support scale."""
    if window_df.empty:
        return pd.DataFrame()

    metric_map = {
        "spectral_distribution_error": "spectral_error",
        "composition_js": "composition_js",
        "configuration_js": "configuration_js",
        "rare_state_coverage": "rare_coverage",
        "rare_state_proportional_recovery": "rare_proportional_recovery",
        "rare_state_presence_fraction": "rare_presence",
        "mean_sam_deg": "sam_deg",
        "task_loss": "task_loss",
    }
    attribute_columns = [c for c in window_df.columns if c.startswith("attribute_error__")]
    for column in attribute_columns:
        metric_map[column] = column

    conv = config.convergence
    rows: List[Dict[str, Any]] = []
    for (base_size, shape_name), group in window_df.groupby(
        ["base_size_px", "shape_name"], sort=True
    ):
        first = group.iloc[0]
        per_window_pass = (
            (group["spectral_distribution_error"] <= conv.spectral_error_q95_max)
            & (group["composition_js"] <= conv.composition_js_q95_max)
            & (group["configuration_js"] <= conv.configuration_js_q95_max)
        )
        if conv.enforce_rare_state_constraint:
            per_window_pass &= (
                group["rare_state_coverage"] >= conv.rare_state_coverage_q05_min
            )
        row: Dict[str, Any] = {
            "base_size_px": int(base_size),
            "shape_name": str(shape_name),
            "shape_kind": str(first.get("shape_kind", "rectangle")),
            "shape_angle_deg": float(first.get("shape_angle_deg", 0.0)),
            "shape_aspect_ratio": float(first.get("shape_aspect_ratio", 1.0)),
            "major_axis_px": float(first.get("major_axis_px", first["window_width_px"])),
            "minor_axis_px": float(first.get("minor_axis_px", first["window_height_px"])),
            "window_height_px": int(first["window_height_px"]),
            "window_width_px": int(first["window_width_px"]),
            "target_support_area_px": float(first.get("target_support_area_px", base_size * base_size)),
            "actual_support_area_px": int(first.get("actual_support_area_px", first["n_valid_pixels"])),
            "support_area_error_fraction": float(first.get("support_area_error_fraction", 0.0)),
            "characteristic_size_px": float(first["characteristic_size_px"]),
            "characteristic_size_physical": float(first["characteristic_size_physical"]),
            "valid_window_count": int(len(group)),
            # Number of feasible translated positions before the optional cap.
            "candidate_position_count": int(first.get("candidate_position_count", len(group))),
            "used_position_count": int(first.get("used_position_count", len(group))),
            "position_stride_y_px": int(first.get("position_stride_y_px", 0)),
            "position_stride_x_px": int(first.get("position_stride_x_px", 0)),
            # Kept as an integer for compatibility with older result tables.
            "eligible_window_count": int(len(group)),
            "full_coverage_window_count": int((group["coverage_ratio"] >= 0.999999).sum()),
            "single_window_constraints_pass_count": int(per_window_pass.sum()),
            "single_window_constraints_pass_fraction": float(per_window_pass.mean()),
        }
        for source, prefix in metric_map.items():
            if source in group.columns:
                row.update(_quantile_summary(group[source], prefix))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["shape_name", "base_size_px"]
    ).reset_index(drop=True)
    summary["enough_windows"] = (
        summary["valid_window_count"] >= config.windows.min_windows_per_scale
    )
    return summary


def _constraint_threshold_map(
    group: pd.DataFrame, config: REUConfig
) -> Dict[str, float]:
    conv = config.convergence
    result: Dict[str, float] = {
        "spectral_error_q95": conv.spectral_error_q95_max,
        "composition_js_q95": conv.composition_js_q95_max,
        "configuration_js_q95": conv.configuration_js_q95_max,
    }
    if conv.attribute_error_q95_max is not None:
        for column in group.columns:
            if column.startswith("attribute_error__") and column.endswith("_q95"):
                result[column] = conv.attribute_error_q95_max
    if conv.task_loss_q95_max is not None and "task_loss_q95" in group.columns:
        result["task_loss_q95"] = conv.task_loss_q95_max
    return result


def _forward_stability_score(
    block: pd.DataFrame, threshold_map: Mapping[str, float]
) -> float:
    """Robust plateau score over a multi-scale block.

    For every constrained error metric, calculate the range across the block
    and normalize it by that metric's acceptance threshold. This avoids the
    instability caused by dividing by a near-zero previous value.
    """
    scores: List[float] = []
    for column, threshold in threshold_map.items():
        values = block[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            return np.inf
        scores.append(float((values.max() - values.min()) / max(threshold, EPS)))
    return float(max(scores)) if scores else np.inf


def determine_representative_scales(
    summary_df: pd.DataFrame,
    config: REUConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Find strict REU and distinguish domain censoring from non-convergence.

    V2.7 decision states
    --------------------
    global_representative_scale_found
        A forward block satisfies all pre-registered constraints and stability.
    domain_censored_before_confirmation
        The support geometry can no longer be sampled at larger configured
        scales in the finite/irregular region, so absence of REU is *not* a
        scientific failure.
    no_representative_scale_within_configured_range
        The geometry remains testable through the largest configured scale but
        no strict REU is confirmed within that range.
    """
    if summary_df.empty:
        return summary_df.copy(), {}

    conv = config.convergence
    configured_max = int(max(int(v) for v in config.windows.window_sizes_px))
    out_parts: List[pd.DataFrame] = []
    decisions: Dict[str, Dict[str, Any]] = {}

    for shape_name, group in summary_df.groupby("shape_name", sort=True):
        group = group.sort_values("base_size_px").copy().reset_index(drop=True)
        group["pass_enough_windows"] = group["enough_windows"].astype(bool)
        group["pass_spectral_q95"] = group["spectral_error_q95"] <= conv.spectral_error_q95_max
        group["pass_spectral_width"] = group["spectral_error_width90"] <= conv.spectral_funnel_width_max
        group["pass_composition_q95"] = group["composition_js_q95"] <= conv.composition_js_q95_max
        group["pass_composition_width"] = group["composition_js_width90"] <= conv.composition_funnel_width_max
        group["pass_configuration_q95"] = group["configuration_js_q95"] <= conv.configuration_js_q95_max
        group["pass_configuration_width"] = group["configuration_js_width90"] <= conv.configuration_funnel_width_max
        group["pass_rare_coverage"] = group["rare_coverage_q05"] >= conv.rare_state_coverage_q05_min

        scientific_pass_columns = [
            "pass_spectral_q95", "pass_spectral_width",
            "pass_composition_q95", "pass_composition_width",
            "pass_configuration_q95", "pass_configuration_width",
        ]
        if conv.enforce_rare_state_constraint:
            scientific_pass_columns.append("pass_rare_coverage")

        attribute_q95_columns = [c for c in group.columns if c.startswith("attribute_error__") and c.endswith("_q95")]
        if conv.attribute_error_q95_max is not None:
            if not attribute_q95_columns:
                group["pass_attribute_q95"] = False
            else:
                arr = np.ones(len(group), dtype=bool)
                for column in attribute_q95_columns:
                    arr &= np.isfinite(group[column]) & (group[column] <= conv.attribute_error_q95_max)
                group["pass_attribute_q95"] = arr
            scientific_pass_columns.append("pass_attribute_q95")

        if conv.task_loss_q95_max is not None:
            task = pd.to_numeric(group.get("task_loss_q95", np.nan), errors="coerce")
            group["pass_task_q95"] = np.isfinite(task) & (task <= conv.task_loss_q95_max)
            scientific_pass_columns.append("pass_task_q95")

        group["scientific_constraints_pass"] = group[scientific_pass_columns].all(axis=1)
        group["instant_constraints_pass"] = group["scientific_constraints_pass"] & group["pass_enough_windows"]

        threshold_map = _constraint_threshold_map(group, config)
        stability_len = max(2, int(conv.stability_window_scales))
        group["stability_block_score"] = np.nan
        group["pass_stability_block"] = False
        for start_idx in range(0, len(group) - stability_len + 1):
            block = group.iloc[start_idx:start_idx + stability_len]
            score = _forward_stability_score(block, threshold_map)
            group.loc[start_idx, "stability_block_score"] = score
            group.loc[start_idx, "pass_stability_block"] = np.isfinite(score) and score <= conv.relative_stability_max
        group["max_relative_change"] = group["stability_block_score"]
        group["stable_transition_pass"] = group["pass_stability_block"]

        ratios = pd.DataFrame(index=group.index)
        ratios["spectral_q95"] = group["spectral_error_q95"] / max(conv.spectral_error_q95_max, EPS)
        ratios["spectral_width"] = group["spectral_error_width90"] / max(conv.spectral_funnel_width_max, EPS)
        ratios["composition_q95"] = group["composition_js_q95"] / max(conv.composition_js_q95_max, EPS)
        ratios["composition_width"] = group["composition_js_width90"] / max(conv.composition_funnel_width_max, EPS)
        ratios["configuration_q95"] = group["configuration_js_q95"] / max(conv.configuration_js_q95_max, EPS)
        ratios["configuration_width"] = group["configuration_js_width90"] / max(conv.configuration_funnel_width_max, EPS)
        if conv.enforce_rare_state_constraint:
            ratios["rare_deficit"] = np.maximum(0.0, conv.rare_state_coverage_q05_min - group["rare_coverage_q05"]) / max(conv.rare_state_coverage_q05_min, EPS)
        group["scale_minimax_ratio"] = ratios.max(axis=1)

        chosen_idx: Optional[int] = None
        block_len = max(1, int(conv.consecutive_scales))
        for start_idx in range(0, len(group) - block_len + 1):
            block = group.iloc[start_idx:start_idx + block_len]
            if not bool(block["instant_constraints_pass"].all()):
                continue
            if not bool(group.loc[start_idx, "pass_stability_block"]):
                continue
            chosen_idx = start_idx
            break

        group["selected_representative_scale"] = False
        if chosen_idx is not None:
            group.loc[chosen_idx, "selected_representative_scale"] = True
            row = group.loc[chosen_idx]
            decisions[shape_name] = {
                "status": "global_representative_scale_found",
                "decision_class": "success",
                "base_size_px": int(row["base_size_px"]),
                "shape_kind": str(row.get("shape_kind", "rectangle")),
                "shape_angle_deg": float(row.get("shape_angle_deg", 0.0)),
                "shape_aspect_ratio": float(row.get("shape_aspect_ratio", 1.0)),
                "major_axis_px": float(row.get("major_axis_px", row["window_width_px"])),
                "minor_axis_px": float(row.get("minor_axis_px", row["window_height_px"])),
                "window_height_px": int(row["window_height_px"]),
                "window_width_px": int(row["window_width_px"]),
                "actual_support_area_px": int(row.get("actual_support_area_px", row["window_height_px"] * row["window_width_px"])),
                "characteristic_size_px": float(row["characteristic_size_px"]),
                "stable_block_length": block_len,
                "stability_score": float(row["stability_block_score"]),
                "rare_constraint_enforced": bool(conv.enforce_rare_state_constraint),
                "shape_name": shape_name,
                "configured_max_base_size_px": configured_max,
            }
        else:
            best_idx = int(group["scale_minimax_ratio"].idxmin())
            best_row = group.loc[best_idx]
            last = group.iloc[-1]
            largest_available = int(last["base_size_px"])
            domain_censored = largest_available < configured_max
            limiting = [c.replace("pass_", "") for c in scientific_pass_columns if not bool(last[c])]
            if not bool(last["pass_enough_windows"]):
                limiting = ["enough_windows"] + limiting
            if domain_censored:
                status = "domain_censored_before_confirmation"
                decision_class = "domain_censored"
            else:
                status = "no_representative_scale_within_configured_range"
                decision_class = "range_exhausted"
            decisions[shape_name] = {
                "status": status,
                "decision_class": decision_class,
                "base_size_px": None,
                "shape_kind": str(best_row.get("shape_kind", "rectangle")),
                "shape_angle_deg": float(best_row.get("shape_angle_deg", 0.0)),
                "shape_aspect_ratio": float(best_row.get("shape_aspect_ratio", 1.0)),
                "major_axis_px": float(best_row.get("major_axis_px", best_row["window_width_px"])),
                "minor_axis_px": float(best_row.get("minor_axis_px", best_row["window_height_px"])),
                "window_height_px": None,
                "window_width_px": None,
                "characteristic_size_px": None,
                "stable_block_length": block_len,
                "configured_max_base_size_px": configured_max,
                "largest_tested_base_size_px": largest_available,
                "largest_scale_minimax_ratio": float(last["scale_minimax_ratio"]),
                "largest_scale_scientific_constraints_pass": bool(last["scientific_constraints_pass"]),
                "largest_scale_enough_windows": bool(last["pass_enough_windows"]),
                "best_candidate_base_size_px": int(best_row["base_size_px"]),
                "best_candidate_minimax_ratio": float(best_row["scale_minimax_ratio"]),
                "candidate_censored_at_upper_bound": bool(int(best_row["base_size_px"]) == largest_available),
                "censor_lower_bound_px": largest_available if domain_censored else None,
                "rare_constraint_enforced": bool(conv.enforce_rare_state_constraint),
                "diagnostic_rare_pass_at_largest_scale": bool(last["pass_rare_coverage"]),
                "limiting_constraints_at_largest_scale": limiting,
                "shape_name": shape_name,
            }
        out_parts.append(group)

    return pd.concat(out_parts, ignore_index=True), decisions


# -----------------------------------------------------------------------------
# Representative unit / unit-set selection# -----------------------------------------------------------------------------
# Representative unit / unit-set selection at an accepted scale
# -----------------------------------------------------------------------------

def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    return float(inter / (area_a + area_b - inter + EPS))


def select_representative_units(
    window_df: pd.DataFrame,
    decision: Mapping[str, Any],
    config: REUConfig,
    top_k: int = 10,
    max_iou: float = 0.30,
) -> pd.DataFrame:
    """Select nonredundant feasible units by minimax normalized error, not weighted sum."""

    if decision.get("status") != "global_representative_scale_found":
        return pd.DataFrame()
    base_size = int(decision["base_size_px"])
    candidates = window_df[window_df["base_size_px"] == base_size].copy()
    if "shape_name" in decision:
        candidates = candidates[candidates["shape_name"] == decision["shape_name"]]
    if candidates.empty:
        return pd.DataFrame()

    conv = config.convergence
    candidates["spectral_ratio"] = candidates["spectral_distribution_error"] / max(
        conv.spectral_error_q95_max, EPS
    )
    candidates["composition_ratio"] = candidates["composition_js"] / max(
        conv.composition_js_q95_max, EPS
    )
    candidates["configuration_ratio"] = candidates["configuration_js"] / max(
        conv.configuration_js_q95_max, EPS
    )
    ratio_columns = [
        "spectral_ratio",
        "composition_ratio",
        "configuration_ratio",
    ]
    if conv.enforce_rare_state_constraint:
        candidates["rare_deficit_ratio"] = np.maximum(
            0.0,
            conv.rare_state_coverage_q05_min - candidates["rare_state_coverage"],
        ) / max(conv.rare_state_coverage_q05_min, EPS)
        ratio_columns.append("rare_deficit_ratio")
    if conv.task_loss_q95_max is not None and "task_loss" in candidates:
        candidates["task_ratio"] = candidates["task_loss"] / max(conv.task_loss_q95_max, EPS)
        ratio_columns.append("task_ratio")
    candidates["minimax_error_ratio"] = candidates[ratio_columns].max(axis=1, skipna=True)
    candidates["all_window_constraints_pass"] = candidates["minimax_error_ratio"] <= 1.0
    candidates = candidates.sort_values(
        ["all_window_constraints_pass", "minimax_error_ratio", "rare_state_coverage"],
        ascending=[False, True, False],
    )

    selected: List[int] = []
    selected_boxes: List[Tuple[float, float, float, float]] = []
    for idx, row in candidates.iterrows():
        bbox = (row["top"], row["left"], row["bottom"], row["right"])
        if all(_bbox_iou(bbox, existing) <= max_iou for existing in selected_boxes):
            selected.append(idx)
            selected_boxes.append(bbox)
        if len(selected) >= top_k:
            break

    result = candidates.loc[selected].copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def select_representative_unit_set(
    window_df: pd.DataFrame,
    base_size_px: int,
    shape_name: str,
    model: SpectralStateModel,
    config: REUConfig,
    max_units: int = 8,
    max_iou: float = 0.30,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Greedy set selection for non-stationary regions.

    The set aggregates state compositions, configuration vectors and projection
    quantiles. It stops when the set-level composition/configuration/spectral
    errors and rare-state coverage meet the same scientific tolerances.
    """

    candidates = window_df[
        (window_df["base_size_px"] == int(base_size_px))
        & (window_df["shape_name"] == str(shape_name))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(), {}

    conv = config.convergence
    selected_indices: List[int] = []
    selected_boxes: List[Tuple[float, float, float, float]] = []
    remaining = list(candidates.index)
    best_metrics: Dict[str, float] = {}

    def aggregate_metrics(indices: Sequence[int]) -> Dict[str, float]:
        subset = candidates.loc[list(indices)]
        weights = subset["n_valid_pixels"].to_numpy(float)
        weights /= weights.sum() + EPS
        compositions = np.stack(subset["local_composition"].map(lambda x: np.asarray(json.loads(x))).to_list())
        configurations = np.stack(subset["local_configuration"].map(lambda x: np.asarray(json.loads(x))).to_list())
        quantiles = np.stack(subset["projection_quantiles"].map(lambda x: np.asarray(json.loads(x))).to_list())
        agg_comp = np.sum(compositions * weights[:, None], axis=0)
        agg_conf = np.sum(configurations * weights[:, None], axis=0)
        agg_quant = np.sum(quantiles * weights[:, None], axis=0).reshape(
            model.reference_projection_quantiles.shape
        )
        per_projection = np.mean(
            np.abs(agg_quant - model.reference_projection_quantiles)
            / model.reference_projection_scale[:, None],
            axis=1,
        )
        spectral_error = float(np.mean(per_projection / (1.0 + per_projection)))
        rare_mask = model.global_rare_states & (model.global_composition > EPS)
        rare_total = int(rare_mask.sum())
        if rare_total == 0:
            rare_coverage = 1.0
        else:
            global_rare_mass = float(model.global_composition[rare_mask].sum())
            local_rare_mass = float(agg_comp[rare_mask].sum())
            rare_coverage = float(np.clip(
                1.0 - abs(local_rare_mass - global_rare_mass)
                / max(global_rare_mass, EPS),
                0.0, 1.0,
            ))
        return {
            "spectral_distribution_error": spectral_error,
            "composition_js": _safe_js(agg_comp, model.global_composition),
            "configuration_js": _safe_js(agg_conf, model.global_configuration),
            "rare_state_coverage": rare_coverage,
        }

    def minimax(metrics: Mapping[str, float]) -> float:
        ratios = [
            metrics["spectral_distribution_error"] / max(conv.spectral_error_q95_max, EPS),
            metrics["composition_js"] / max(conv.composition_js_q95_max, EPS),
            metrics["configuration_js"] / max(conv.configuration_js_q95_max, EPS),
        ]
        if conv.enforce_rare_state_constraint:
            ratios.append(
                max(0.0, conv.rare_state_coverage_q05_min - metrics["rare_state_coverage"])
                / max(conv.rare_state_coverage_q05_min, EPS)
            )
        return float(max(ratios))

    for _ in range(max_units):
        best_idx: Optional[int] = None
        best_objective = np.inf
        best_candidate_metrics: Dict[str, float] = {}
        for idx in remaining:
            row = candidates.loc[idx]
            bbox = (row["top"], row["left"], row["bottom"], row["right"])
            if any(_bbox_iou(bbox, existing) > max_iou for existing in selected_boxes):
                continue
            metrics = aggregate_metrics(selected_indices + [idx])
            objective = minimax(metrics)
            if objective < best_objective:
                best_idx = idx
                best_objective = objective
                best_candidate_metrics = metrics
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        row = candidates.loc[best_idx]
        selected_boxes.append((row["top"], row["left"], row["bottom"], row["right"]))
        remaining.remove(best_idx)
        best_metrics = best_candidate_metrics
        if best_objective <= 1.0:
            break

    result = candidates.loc[selected_indices].copy().reset_index(drop=True)
    result.insert(0, "set_order", np.arange(1, len(result) + 1))
    best_metrics["minimax_error_ratio"] = minimax(best_metrics) if best_metrics else np.nan
    best_metrics["constraints_pass"] = bool(best_metrics and best_metrics["minimax_error_ratio"] <= 1.0)
    return result, best_metrics


# -----------------------------------------------------------------------------
# Region-level orchestration and outputs
# -----------------------------------------------------------------------------

def _save_json(path: Path, value: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=convert)


def analyze_region(
    cube: np.ndarray,
    region_mask: np.ndarray,
    region_id: str,
    output_dir: str | Path,
    config: REUConfig,
    attribute_maps: Optional[Mapping[str, np.ndarray]] = None,
    task_loss_by_scale: Optional[Mapping[Tuple[int, str], float]] = None,
    top_k_units: int = 10,
) -> Dict[str, Any]:
    """Complete region analysis and CSV/NPY/JSON export."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = fit_spectral_state_model(cube, region_mask, config)
    window_df = analyze_multiscale_windows(
        cube,
        region_mask,
        model,
        config,
        attribute_maps=attribute_maps,
        task_loss_by_scale=task_loss_by_scale,
    )
    summary = summarize_scales(window_df, config)
    summary_with_decision, decisions = determine_representative_scales(summary, config)

    # Attach shape name to each decision for downstream filtering.
    for shape_name in decisions:
        decisions[shape_name]["shape_name"] = shape_name

    selected_units: Dict[str, pd.DataFrame] = {}
    unit_sets: Dict[str, Dict[str, Any]] = {}
    for shape_name, decision in decisions.items():
        units = select_representative_units(
            window_df,
            decision,
            config,
            top_k=top_k_units,
        )
        selected_units[shape_name] = units
        if not units.empty:
            units.to_csv(output_dir / f"selected_units_{shape_name}.csv", index=False)
        if decision["status"] != "global_representative_scale_found":
            # Test a representative-unit set at the largest scale having enough windows.
            eligible = summary_with_decision[
                (summary_with_decision["shape_name"] == shape_name)
                & summary_with_decision["enough_windows"]
            ]
            if not eligible.empty:
                base_size = int(eligible.sort_values("base_size_px").iloc[-1]["base_size_px"])
                selected_set, set_metrics = select_representative_unit_set(
                    window_df,
                    base_size,
                    shape_name,
                    model,
                    config,
                )
                if not selected_set.empty:
                    selected_set.to_csv(
                        output_dir / f"representative_unit_set_{shape_name}.csv", index=False
                    )
                unit_sets[shape_name] = {
                    "base_size_px": base_size,
                    "metrics": set_metrics,
                    "n_units": int(len(selected_set)),
                }

    window_df.to_csv(output_dir / "window_level_metrics.csv", index=False)
    summary_with_decision.to_csv(output_dir / "scale_response_summary.csv", index=False)
    pass_columns = [c for c in summary_with_decision.columns if c.startswith("pass_")]
    for shape_name, group in summary_with_decision.groupby("shape_name", sort=True):
        export_columns = [
            "base_size_px", "shape_kind", "shape_angle_deg", "shape_aspect_ratio",
            "major_axis_px", "minor_axis_px", "window_height_px", "window_width_px",
            "actual_support_area_px", "support_area_error_fraction",
            "position_stride_y_px", "position_stride_x_px",
            "candidate_position_count", "used_position_count",
            "valid_window_count", "single_window_constraints_pass_count",
            "single_window_constraints_pass_fraction",
        ] + pass_columns + [
            "instant_constraints_pass", "stability_block_score",
            "selected_representative_scale",
        ]
        export_columns = [c for c in export_columns if c in group.columns]
        group[export_columns].to_csv(
            output_dir / f"constraint_matrix_{shape_name}.csv", index=False
        )
    np.save(output_dir / "spectral_state_map.npy", model.state_map)
    np.save(output_dir / "soft_state_cube.npy", model.soft_state_cube.astype(np.float32))
    _save_json(output_dir / "config.json", config.to_dict())
    _save_json(output_dir / "scale_decisions.json", decisions)

    # Corrected spatial-correlation diagnostics for audit and publication.
    _save_json(output_dir / "correlation_structure.json", model.correlation_diagnostics)
    iso = model.correlation_diagnostics.get("isotropic", {})
    if iso.get("lag_px"):
        pd.DataFrame({
            "lag_px": iso.get("lag_px", []),
            "gamma_normalized": iso.get("gamma_normalized", []),
            "fitted_gamma": iso.get("fitted_gamma", [np.nan] * len(iso.get("lag_px", []))),
            "pair_count": iso.get("pair_count", []),
        }).to_csv(output_dir / "correlation_variogram_isotropic.csv", index=False)
    directional_rows: List[Dict[str, Any]] = []
    for angle, diag in model.directional_correlation_diagnostics.items():
        lags = diag.get("lag_px", [])
        gammas = diag.get("gamma_normalized", [])
        fits = diag.get("fitted_gamma", [np.nan] * len(lags))
        counts = diag.get("pair_count", [np.nan] * len(lags))
        offsets = diag.get("offsets_dy_dx", [[None, None]] * len(lags))
        for i in range(len(lags)):
            directional_rows.append({
                "angle_deg": int(angle), "lag_px": lags[i],
                "gamma_normalized": gammas[i],
                "fitted_gamma": fits[i] if i < len(fits) else np.nan,
                "pair_count": counts[i] if i < len(counts) else np.nan,
                "dy": offsets[i][0] if i < len(offsets) else None,
                "dx": offsets[i][1] if i < len(offsets) else None,
                "range_status": diag.get("status"),
                "range_px": diag.get("range_px"),
                "fit_range_px": diag.get("fit_range_px"),
                "range_censored": diag.get("range_censored"),
                "fit_r2": diag.get("fit_r2"),
            })
    if directional_rows:
        pd.DataFrame(directional_rows).to_csv(output_dir / "correlation_variogram_directional.csv", index=False)

    region_summary = {
        "region_id": region_id,
        "pixel_count": int(model.valid_mask.sum()),
        "region_bbox_height_px": int(model.valid_mask.shape[0]),
        "region_bbox_width_px": int(model.valid_mask.shape[1]),
        "region_bbox_short_side_px": int(min(model.valid_mask.shape)),
        "correlation_length_px": (float(model.correlation_length_px) if model.correlation_length_px is not None else None),
        "correlation_length_fit_px": (float(model.correlation_length_fit_px) if model.correlation_length_fit_px is not None else None),
        "correlation_length_lower_bound_px": (float(model.correlation_length_lower_bound_px) if model.correlation_length_lower_bound_px is not None else None),
        "correlation_length_censored": bool(model.correlation_length_censored),
        "correlation_fit_r2": (float(model.correlation_fit_r2) if model.correlation_fit_r2 is not None else None),
        "correlation_length_ci95_px": list(model.correlation_length_ci95_px),
        "correlation_length_physical": (
            float(model.correlation_length_px * config.pixel_size)
            if config.pixel_size is not None and model.correlation_length_px is not None
            else None
        ),
        "physical_unit": config.physical_unit,
        "composition_entropy": float(model.composition_entropy),
        "configuration_entropy": float(model.configuration_entropy),
        "anisotropy_index": (float(model.anisotropy_index) if np.isfinite(model.anisotropy_index) else None),
        "directional_correlation_lengths_px": model.directional_correlation_lengths_px,
        "directional_correlation_diagnostics": model.directional_correlation_diagnostics,
        "principal_direction_deg": float(model.principal_direction_deg),
        "adaptive_aspect_ratio": float(model.adaptive_aspect_ratio),
        "adaptive_geometry_status": model.adaptive_geometry_status,
        "principal_direction_bootstrap_frequency": (float(model.principal_direction_bootstrap_frequency) if np.isfinite(model.principal_direction_bootstrap_frequency) else None),
        "anisotropy_ratio_ci95": list(model.anisotropy_ratio_ci95),
        "decisions": decisions,
        "unit_sets": unit_sets,
    }
    _save_json(output_dir / "region_summary.json", region_summary)

    return {
        "region_id": region_id,
        "model": model,
        "window_df": window_df,
        "scale_summary": summary_with_decision,
        "decisions": decisions,
        "selected_units": selected_units,
        "unit_sets": unit_sets,
        "region_summary": region_summary,
    }


def build_cross_region_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build cross-region table with corrected correlation/censor diagnostics."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        base = result["region_summary"]
        for shape_name, decision in result["decisions"].items():
            rows.append({
                "region_id": base["region_id"],
                "shape_name": shape_name,
                "shape_kind": decision.get("shape_kind"),
                "shape_angle_deg": decision.get("shape_angle_deg"),
                "shape_aspect_ratio": decision.get("shape_aspect_ratio"),
                "status": decision["status"],
                "decision_class": decision.get("decision_class"),
                "representative_base_size_px": decision.get("base_size_px"),
                "window_height_px": decision.get("window_height_px"),
                "window_width_px": decision.get("window_width_px"),
                "characteristic_size_px": decision.get("characteristic_size_px"),
                "correlation_length_px": base.get("correlation_length_px"),
                "correlation_length_fit_px": base.get("correlation_length_fit_px"),
                "correlation_length_lower_bound_px": base.get("correlation_length_lower_bound_px"),
                "correlation_length_censored": base.get("correlation_length_censored"),
                "correlation_fit_r2": base.get("correlation_fit_r2"),
                "composition_entropy": base.get("composition_entropy"),
                "configuration_entropy": base.get("configuration_entropy"),
                "anisotropy_index": base.get("anisotropy_index"),
                "principal_direction_deg": base.get("principal_direction_deg"),
                "adaptive_aspect_ratio": base.get("adaptive_aspect_ratio"),
                "adaptive_geometry_status": base.get("adaptive_geometry_status"),
                "directional_correlation_length_0_px": base.get("directional_correlation_lengths_px", {}).get("0"),
                "directional_correlation_length_45_px": base.get("directional_correlation_lengths_px", {}).get("45"),
                "directional_correlation_length_90_px": base.get("directional_correlation_lengths_px", {}).get("90"),
                "directional_correlation_length_135_px": base.get("directional_correlation_lengths_px", {}).get("135"),
                "configured_max_base_size_px": decision.get("configured_max_base_size_px"),
                "largest_tested_base_size_px": decision.get("largest_tested_base_size_px"),
                "largest_scale_minimax_ratio": decision.get("largest_scale_minimax_ratio"),
                "best_candidate_base_size_px": decision.get("best_candidate_base_size_px"),
                "best_candidate_minimax_ratio": decision.get("best_candidate_minimax_ratio"),
                "candidate_censored_at_upper_bound": decision.get("candidate_censored_at_upper_bound"),
                "censor_lower_bound_px": decision.get("censor_lower_bound_px"),
                "rare_constraint_enforced": decision.get("rare_constraint_enforced"),
            })
    return pd.DataFrame(rows)



# =============================================================================
# NC-REU V2.7 robust-scale inference overrides
# =============================================================================
# V2.7 intentionally keeps the V2.6 strict REU definition.  The changes below
# affect (i) deterministic sampling, (ii) robust spatial-structure inference,
# and (iii) richer output diagnostics.  Later definitions override the V2.6
# functions above without changing their public names.

import hashlib as _hashlib_v27


def _stable_seed_v27(base_seed: int, *tokens: Any) -> int:
    """Stable 32-bit seed independent of Python's randomized hash()."""
    payload = "|".join([str(int(base_seed))] + [str(v) for v in tokens]).encode("utf-8")
    digest = _hashlib_v27.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _integral_correlation_scale_v27(
    lag_px: Sequence[float],
    gamma_normalized: Sequence[float],
) -> float:
    """Integral positive-correlation scale, L_I = integral max(1-gamma,0) dh.

    The origin (0,0) is added explicitly.  The positive part prevents noisy
    semivariogram overshoot above the sill from cancelling genuine short-range
    correlation.  Values are clipped to [0,1] only for the integral diagnostic;
    the exported semivariogram itself remains unmodified.
    """
    h = np.asarray(lag_px, dtype=float)
    g = np.asarray(gamma_normalized, dtype=float)
    finite = np.isfinite(h) & np.isfinite(g) & (h > 0)
    h, g = h[finite], g[finite]
    if len(h) < 2:
        return np.nan
    order = np.argsort(h)
    h, g = h[order], g[order]
    # Collapse duplicate lags by mean to keep integration well-defined.
    uh = np.unique(h)
    if len(uh) != len(h):
        gg = np.array([np.mean(g[h == v]) for v in uh], dtype=float)
        h, g = uh, gg
    h = np.r_[0.0, h]
    g = np.r_[0.0, g]
    rho_plus = np.clip(1.0 - g, 0.0, 1.0)
    # NumPy compatibility: np.trapezoid was introduced in NumPy 2.0.
    # The user's DL environment may still use NumPy 1.x, where np.trapz is the
    # equivalent trapezoidal-rule implementation. Keep both paths numerically
    # equivalent and avoid forcing an environment upgrade.
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(rho_plus, h)
    else:
        area = np.trapz(rho_plus, h)
    return float(area)


def _robustify_variogram_fit_v27(fit: Mapping[str, Any], cfg: CorrelationConfig) -> Dict[str, Any]:
    """Augment a parametric fit with practical and integral correlation scales.

    A poor global R2 no longer automatically deletes a physically stable range.
    If the parametric range and empirical first 95%-sill crossing agree within
    a pre-registered relative tolerance and the far-lag tail reaches the sill,
    the range is retained with an explicit low-R2 agreement status.  Large
    fit/crossing disagreement is labelled ambiguous rather than forced into a
    single correlation length.
    """
    d = dict(fit)
    original_status = str(d.get("status", "unknown"))
    original_range = d.get("range_px")
    fit_range = d.get("fit_range_px")
    empirical = d.get("empirical_range_crossing_px")
    tail = d.get("empirical_tail_max")
    r2 = d.get("fit_r2")

    integral = _integral_correlation_scale_v27(d.get("lag_px", []), d.get("gamma_normalized", []))
    practical = None
    pstatus = original_status
    disagreement = None

    fr = float(fit_range) if fit_range is not None and np.isfinite(fit_range) else None
    er = float(empirical) if empirical is not None and np.isfinite(empirical) else None
    if fr is not None and er is not None:
        disagreement = float(abs(fr - er) / max(0.5 * (abs(fr) + abs(er)), EPS))

    if original_status == "range_observed" and original_range is not None and np.isfinite(original_range):
        practical = float(original_range)
        pstatus = "range_observed_parametric"
    elif original_status == "right_censored":
        practical = None
        pstatus = "right_censored"
    elif original_status == "poor_fit":
        tail_ok = tail is not None and np.isfinite(tail) and float(tail) >= float(getattr(cfg, "practical_range_tail_min", 0.90))
        agree_ok = disagreement is not None and disagreement <= float(getattr(cfg, "practical_range_low_r2_agreement_tolerance", 0.30))
        if fr is not None and er is not None and tail_ok and agree_ok:
            # Median is robust and symmetric between the model and empirical crossing.
            practical = float(np.median([fr, er]))
            pstatus = "range_observed_fit_empirical_agreement_despite_low_r2"
        else:
            pstatus = "ambiguous_multiscale_or_poor_fit"
    elif original_status in {"fit_failed", "insufficient_lags", "insufficient_pairs", "insufficient_region"}:
        pstatus = original_status
    elif original_range is not None and np.isfinite(original_range):
        practical = float(original_range)

    d["parametric_fit_status"] = original_status
    d["parametric_range_px"] = fr
    d["practical_range_px"] = practical
    d["practical_range_status"] = pstatus
    d["fit_empirical_range_relative_disagreement"] = disagreement
    d["integral_scale_px"] = float(integral) if np.isfinite(integral) else None

    # Compatibility: the historical correlation_length_px now means the robust
    # practical 95%-sill range.  Integral scale is stored separately and is the
    # preferred structural predictor in V2.7 cross-region analyses.
    d["range_px"] = practical
    if practical is not None:
        d["status"] = "range_observed"
        d["range_censored"] = False
        d["range_lower_bound_px"] = None
    elif pstatus == "right_censored":
        d["status"] = "right_censored"
        d["range_censored"] = True
    elif pstatus == "ambiguous_multiscale_or_poor_fit":
        d["status"] = "ambiguous_multiscale_or_poor_fit"
        d["range_censored"] = False
    return d


def _bootstrap_variogram_dual_metrics_v27(
    lag_px: np.ndarray,
    samples_by_lag: Sequence[np.ndarray],
    cfg: CorrelationConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    B = max(0, int(cfg.bootstrap_iterations))
    fit_ranges = np.full(B, np.nan, dtype=float)
    practical_ranges = np.full(B, np.nan, dtype=float)
    integral_scales = np.full(B, np.nan, dtype=float)
    if B <= 0 or len(lag_px) < 5:
        return {"fit_range": fit_ranges, "practical_range": practical_ranges, "integral_scale": integral_scales}
    counts = np.array([max(len(v), 1) for v in samples_by_lag], dtype=int)
    for b in range(B):
        gb: List[float] = []
        for arr in samples_by_lag:
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                gb.append(np.nan)
                continue
            n = min(len(arr), max(100, int(cfg.bootstrap_max_pairs_per_lag)))
            idx = rng.integers(0, len(arr), size=n)
            gb.append(float(np.mean(arr[idx])))
        fit0 = _fit_normalized_variogram(lag_px, gb, counts, cfg)
        fit = _robustify_variogram_fit_v27(fit0, cfg)
        fr = fit.get("fit_range_px")
        pr = fit.get("practical_range_px")
        li = fit.get("integral_scale_px")
        if fr is not None and np.isfinite(fr): fit_ranges[b] = float(fr)
        if pr is not None and np.isfinite(pr): practical_ranges[b] = float(pr)
        if li is not None and np.isfinite(li): integral_scales[b] = float(li)
    return {"fit_range": fit_ranges, "practical_range": practical_ranges, "integral_scale": integral_scales}


def _ci95_v27(values: Sequence[float], minimum: int = 20) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < minimum:
        return (None, None)
    q = np.quantile(arr, [0.025, 0.975])
    return float(q[0]), float(q[1])


def _radial_multivariate_variogram_v27(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
) -> Dict[str, Any]:
    rng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "correlation", "radial"))
    valid = np.asarray(mask, dtype=bool)
    coords = np.column_stack(np.where(valid))
    z = np.asarray(latent_cube, dtype=np.float64)[valid]
    weights, variances, sill = _latent_variogram_setup(latent_cube, valid, pca)
    h, w = valid.shape
    max_lag = float(min(cfg.max_lag_px, cfg.max_lag_fraction_short_side * min(h, w)))
    if len(coords) < 100 or max_lag < 4:
        return {"status": "insufficient_region", "range_px": None, "integral_scale_px": None, "range_censored": False}

    n_pairs = max(10000, int(cfg.radial_pair_count))
    ia = rng.integers(0, len(coords), size=n_pairs)
    ib = rng.integers(0, len(coords), size=n_pairs)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    dist = np.linalg.norm(coords[ia] - coords[ib], axis=1)
    keep = (dist > 0) & (dist <= max_lag)
    ia, ib, dist = ia[keep], ib[keep], dist[keep]
    if len(dist) < cfg.min_pairs_per_bin * 5:
        return {"status": "insufficient_pairs", "range_px": None, "integral_scale_px": None, "range_censored": False, "max_lag_px": max_lag}
    semivar = _pair_semivariance_values(z[ia], z[ib], weights, sill)

    edges = np.linspace(0.0, max_lag, int(cfg.radial_bins) + 1)
    lags: List[float] = []; gammas: List[float] = []; counts: List[int] = []; samples: List[np.ndarray] = []
    for i in range(len(edges) - 1):
        sel = (dist >= edges[i]) & (dist < edges[i+1])
        n = int(sel.sum())
        if n < cfg.min_pairs_per_bin: continue
        vals = semivar[sel]
        lags.append(float(np.mean(dist[sel]))); gammas.append(float(np.mean(vals))); counts.append(n)
        cap = min(n, int(cfg.bootstrap_max_pairs_per_lag))
        if n > cap:
            srng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "radial-bin", i))
            vals = vals[srng.choice(n, cap, replace=False)]
        samples.append(np.asarray(vals, dtype=np.float32))

    fit = _robustify_variogram_fit_v27(_fit_normalized_variogram(lags, gammas, counts, cfg), cfg)
    brng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "correlation", "radial-bootstrap"))
    boot = _bootstrap_variogram_dual_metrics_v27(np.asarray(lags, float), samples, cfg, brng)
    fit.update({
        "bootstrap_range_ci95_px": _ci95_v27(boot["practical_range"]),
        "bootstrap_fit_range_ci95_px": _ci95_v27(boot["fit_range"]),
        "bootstrap_integral_scale_ci95_px": _ci95_v27(boot["integral_scale"]),
        "bootstrap_practical_range_valid_count": int(np.isfinite(boot["practical_range"]).sum()),
        "bootstrap_integral_scale_valid_count": int(np.isfinite(boot["integral_scale"]).sum()),
        "max_requested_lag_px": max_lag,
        "latent_component_coefficients": weights.tolist(),
        "latent_component_variances": variances.tolist(),
        "latent_metric_note": "standardized PC differences weighted by PCA explained variance",
        "normalization_sill": sill,
        "accepted_random_pair_count": int(len(dist)),
    })
    return fit


def _directional_multivariate_variogram_v27(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
    directions_deg: Sequence[int] = (0, 45, 90, 135),
) -> Dict[str, Dict[str, Any]]:
    z = np.asarray(latent_cube, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    h, w = valid.shape
    weights, _, sill = _latent_variogram_setup(latent_cube, valid, pca)
    max_common_lag = float(min(cfg.max_lag_px, cfg.max_lag_fraction_short_side * min(h, w)))
    results: Dict[str, Dict[str, Any]] = {}

    for angle in directions_deg:
        rng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "correlation", "direction", int(angle)))
        offsets: List[Tuple[int, int]] = []
        for target in cfg.directional_target_lags_px:
            dy, dx = _direction_offset_for_target(int(angle), float(target))
            actual = float(math.hypot(dy, dx))
            if actual <= max_common_lag + 1e-9 and (dy, dx) not in offsets:
                offsets.append((dy, dx))
        lags: List[float] = []; gammas: List[float] = []; counts: List[int] = []; samples: List[np.ndarray] = []; used_offsets=[]
        for oi, (dy, dx) in enumerate(offsets):
            y0a, y1a = max(0, -dy), min(h, h - dy); x0a, x1a = max(0, -dx), min(w, w - dx)
            y0b, y1b = y0a + dy, y1a + dy; x0b, x1b = x0a + dx, x1a + dx
            if y1a <= y0a or x1a <= x0a: continue
            pm = valid[y0a:y1a, x0a:x1a] & valid[y0b:y1b, x0b:x1b]
            n = int(pm.sum())
            if n < cfg.min_pairs_per_directional_lag: continue
            a = z[y0a:y1a, x0a:x1a][pm]; b = z[y0b:y1b, x0b:x1b][pm]
            vals = _pair_semivariance_values(a, b, weights, sill)
            actual = float(math.hypot(dy, dx))
            lags.append(actual); gammas.append(float(np.mean(vals))); counts.append(n); used_offsets.append((int(dy), int(dx)))
            cap = min(n, int(cfg.bootstrap_max_pairs_per_lag))
            if n > cap:
                srng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "direction-subsample", int(angle), oi))
                vals = vals[srng.choice(n, cap, replace=False)]
            samples.append(np.asarray(vals, dtype=np.float32))

        fit = _robustify_variogram_fit_v27(_fit_normalized_variogram(lags, gammas, counts, cfg), cfg)
        brng = np.random.default_rng(_stable_seed_v27(cfg.random_seed, "direction-bootstrap", int(angle)))
        boot = _bootstrap_variogram_dual_metrics_v27(np.asarray(lags, float), samples, cfg, brng)
        fit.update({
            "angle_deg": int(angle),
            "offsets_dy_dx": [list(v) for v in used_offsets],
            "max_common_euclidean_lag_px": max_common_lag,
            "bootstrap_range_ci95_px": _ci95_v27(boot["practical_range"]),
            "bootstrap_integral_scale_ci95_px": _ci95_v27(boot["integral_scale"]),
            "bootstrap_practical_range_valid_count": int(np.isfinite(boot["practical_range"]).sum()),
            "bootstrap_integral_scale_valid_count": int(np.isfinite(boot["integral_scale"]).sum()),
            "_bootstrap_practical_ranges": boot["practical_range"],
            "_bootstrap_integral_scales": boot["integral_scale"],
        })
        results[str(int(angle))] = fit
    return results


def estimate_corrected_correlation_structure(
    latent_cube: np.ndarray,
    mask: np.ndarray,
    pca: PCA,
    cfg: CorrelationConfig,
) -> Dict[str, Any]:
    """V2.7 dual-scale spatial-structure inference.

    The practical 95%-sill range and the empirical integral correlation scale
    are both reported.  Adaptive anisotropy requires agreement between them;
    parametric-fit anisotropy alone is insufficient.
    """
    radial = _radial_multivariate_variogram_v27(latent_cube, mask, pca, cfg)
    directional = _directional_multivariate_variogram_v27(latent_cube, mask, pca, cfg)

    integral = {a: float(d["integral_scale_px"]) for a,d in directional.items()
                if d.get("integral_scale_px") is not None and np.isfinite(d.get("integral_scale_px"))}
    practical = {a: float(d["practical_range_px"]) for a,d in directional.items()
                 if d.get("practical_range_px") is not None and np.isfinite(d.get("practical_range_px"))}
    censored_angles = {a for a,d in directional.items() if d.get("practical_range_status") == "right_censored"}

    integral_ratio = np.nan; practical_ratio = np.nan; principal_int = None; principal_pr = None
    ratio_ci=(None,None); principal_freq=np.nan
    if len(integral) >= 4:
        iv=np.asarray(list(integral.values()),float)
        integral_ratio=float(np.max(iv)/max(np.min(iv),EPS))
        principal_int=max(integral,key=integral.get)
        arrays={a:np.asarray(directional[a].get("_bootstrap_integral_scales",[]),float) for a in integral}
        B=min([len(v) for v in arrays.values()] or [0])
        ratios=[]; principals=[]
        for b in range(B):
            vals={a:arrays[a][b] for a in arrays if b<len(arrays[a]) and np.isfinite(arrays[a][b])}
            if len(vals)!=len(arrays): continue
            vv=np.asarray(list(vals.values()),float)
            ratios.append(float(np.max(vv)/max(np.min(vv),EPS)))
            principals.append(max(vals,key=vals.get))
        if len(ratios)>=20:
            ratio_ci=tuple(float(v) for v in np.quantile(ratios,[0.025,0.975]))
            principal_freq=float(np.mean(np.asarray(principals)==principal_int))
    if len(practical)>=2:
        pv=np.asarray(list(practical.values()),float)
        practical_ratio=float(np.max(pv)/max(np.min(pv),EPS))
        principal_pr=max(practical,key=practical.get)

    principal=0.0; aspect=1.0
    status="degenerated_to_circle_insufficient_integral_evidence"
    supported=False
    lower = ratio_ci[0] if ratio_ci[0] is not None else -np.inf
    if len(integral) >= 4:
        if not np.isfinite(integral_ratio) or integral_ratio < float(cfg.adaptive_integral_ratio_min):
            status="degenerated_to_circle_integral_anisotropy_below_threshold"
        elif lower < float(cfg.adaptive_integral_ci_lower_min):
            status="degenerated_to_circle_integral_anisotropy_uncertain"
        elif not np.isfinite(principal_freq) or principal_freq < float(cfg.adaptive_min_principal_frequency):
            status="degenerated_to_circle_principal_direction_unstable"
        elif principal_int in censored_angles:
            status="degenerated_to_circle_practical_principal_censored"
        elif len(practical) < int(cfg.adaptive_min_practical_reliable_directions):
            status="degenerated_to_circle_insufficient_practical_range_evidence"
        elif not np.isfinite(practical_ratio) or practical_ratio < float(cfg.adaptive_practical_ratio_min):
            status="degenerated_to_circle_practical_anisotropy_below_threshold"
        elif bool(cfg.adaptive_require_principal_concordance) and principal_pr != principal_int:
            status="degenerated_to_circle_dual_scale_direction_disagreement"
        else:
            supported=True
            status="anisotropy_supported_dual_scale"
            principal=float(principal_int)
            aspect=float(np.clip(integral_ratio,1.0,cfg.adaptive_max_aspect_ratio))

    clean_directional={}
    for a,d in directional.items():
        clean_directional[a]={k:v for k,v in d.items() if not str(k).startswith("_bootstrap_")}

    anis_index=float(integral_ratio-1.0) if np.isfinite(integral_ratio) else np.nan
    return {
        "isotropic": radial,
        "directional": clean_directional,
        "directional_ranges_px": {a:(float(d.get("practical_range_px")) if d.get("practical_range_px") is not None and np.isfinite(d.get("practical_range_px")) else None) for a,d in directional.items()},
        "directional_integral_scales_px": {a:(float(d.get("integral_scale_px")) if d.get("integral_scale_px") is not None and np.isfinite(d.get("integral_scale_px")) else None) for a,d in directional.items()},
        "anisotropy_index": anis_index,
        "anisotropy_supported": bool(supported),
        "integral_anisotropy_ratio": float(integral_ratio) if np.isfinite(integral_ratio) else None,
        "practical_anisotropy_ratio": float(practical_ratio) if np.isfinite(practical_ratio) else None,
        "integral_principal_direction_deg": float(principal_int) if principal_int is not None else None,
        "practical_principal_direction_deg": float(principal_pr) if principal_pr is not None else None,
        "principal_direction_deg": principal,
        "adaptive_aspect_ratio": aspect,
        "adaptive_geometry_status": status,
        "principal_direction_bootstrap_frequency": float(principal_freq) if np.isfinite(principal_freq) else np.nan,
        "anisotropy_ratio_ci95": ratio_ci,
        "anisotropy_policy": "adaptive ellipse requires integral-scale anisotropy + bootstrap stability + practical-range concordance",
    }


def analyze_multiscale_windows(
    cube: np.ndarray,
    region_mask: np.ndarray,
    model: SpectralStateModel,
    config: REUConfig,
    attribute_maps: Optional[Mapping[str, np.ndarray]] = None,
    task_loss_by_scale: Optional[Mapping[Tuple[int, str], float]] = None,
) -> pd.DataFrame:
    """V2.7 deterministic multi-scale sampling.

    Position and within-window spectrum RNG streams are keyed by geometry,
    scale and location.  Adding/removing another geometry therefore cannot
    change the sampling of an unchanged geometry.
    """
    cube=np.asarray(cube,dtype=np.float32)
    mask=np.asarray(region_mask,dtype=bool)&model.valid_mask
    attribute_maps=attribute_maps or {}
    attribute_reference={}
    for name,attr in attribute_maps.items():
        arr=np.asarray(attr,dtype=np.float64)
        if arr.shape!=mask.shape: raise ValueError(f"Attribute map '{name}' has shape {arr.shape}, expected {mask.shape}.")
        values=arr[mask & np.isfinite(arr)]
        if values.size:
            scale=np.quantile(values,0.95)-np.quantile(values,0.05)
            attribute_reference[name]=(float(np.mean(values)),float(max(scale,EPS)))
    specs=[s for s in config.windows.shape_specs if s.enabled]
    if not specs: raise ValueError("No enabled support shape was configured.")

    records=[]
    for base_size in sorted(set(int(v) for v in config.windows.window_sizes_px)):
        for spec in specs:
            geometry=make_shape_geometry(base_size,spec,adaptive_angle_deg=model.principal_direction_deg,adaptive_aspect_ratio=model.adaptive_aspect_ratio)
            support=geometry.support_mask; win_h,win_w=support.shape
            prng=np.random.default_rng(_stable_seed_v27(config.windows.random_seed,"position",geometry.name,int(base_size)))
            positions,meta=_candidate_positions(mask,support,base_size,config.windows.overlap,config.windows.min_region_coverage,
                                                config.windows.max_windows_per_scale,prng,config.windows.max_position_stride_px)
            for window_id,(y,x,coverage) in enumerate(positions):
                local_mask=mask[y:y+win_h,x:x+win_w]&support
                n_valid=int(local_mask.sum())
                if n_valid<config.windows.min_valid_pixels: continue
                cube_patch=cube[y:y+win_h,x:x+win_w]
                latent_patch=model.latent_cube[y:y+win_h,x:x+win_w]
                soft_patch=model.soft_state_cube[y:y+win_h,x:x+win_w]
                local_spectra=cube_patch[local_mask].astype(np.float64)
                local_latent=latent_patch[local_mask].astype(np.float64)
                if len(local_spectra)>config.windows.max_spectra_per_window:
                    srng=np.random.default_rng(_stable_seed_v27(config.windows.random_seed,"spectra",geometry.name,int(base_size),int(y),int(x)))
                    si=srng.choice(len(local_spectra),config.windows.max_spectra_per_window,replace=False)
                    spectral_sample=local_spectra[si]; latent_sample=local_latent[si]
                else:
                    spectral_sample=local_spectra; latent_sample=local_latent
                spectral_error,projection_quantiles=_spectral_distribution_error(latent_sample,model)
                local_composition=soft_patch[local_mask].mean(axis=0)
                composition_js=_safe_js(local_composition,model.global_composition)
                local_configuration,_=compute_soft_cooccurrence(soft_patch,local_mask,config.metrics.cooccurrence_distances_px,config.metrics.cooccurrence_directions)
                configuration_js=_safe_js(local_configuration,model.global_configuration)
                mean_sam=_sam_deg(spectral_sample.mean(axis=0),model.global_mean_spectrum)
                rare_mask=model.global_rare_states & (model.global_composition>EPS); rare_total=int(rare_mask.sum())
                if rare_total==0:
                    rare_coverage=rare_proportional_recovery=rare_presence_fraction=1.0; global_rare_mass=local_rare_mass=0.0
                else:
                    global_rare_mass=float(model.global_composition[rare_mask].sum()); local_rare_mass=float(local_composition[rare_mask].sum())
                    rare_coverage=float(np.clip(1.0-abs(local_rare_mass-global_rare_mass)/max(global_rare_mass,EPS),0.0,1.0))
                    rr=local_composition[rare_mask]/np.maximum(model.global_composition[rare_mask],EPS)
                    rare_proportional_recovery=float(np.mean(np.clip(rr,0.0,1.0)))
                    floor=np.maximum(0.10*model.global_composition[rare_mask],1.0/max(n_valid,1))
                    rare_presence_fraction=float(np.mean(local_composition[rare_mask]>=floor))
                characteristic_size_px=geometry.characteristic_size_px
                characteristic_size_physical=characteristic_size_px*float(config.pixel_size) if config.pixel_size is not None else np.nan
                rec={
                    "base_size_px":int(base_size),"shape_name":geometry.name,"shape_kind":geometry.kind,
                    "shape_angle_deg":geometry.angle_deg,"shape_aspect_ratio":geometry.aspect_ratio,
                    "major_axis_px":geometry.major_axis_px,"minor_axis_px":geometry.minor_axis_px,
                    "window_height_px":int(win_h),"window_width_px":int(win_w),"target_support_area_px":geometry.target_area_px,
                    "actual_support_area_px":geometry.actual_area_px,"support_area_error_fraction":geometry.area_error_fraction,
                    "characteristic_size_px":characteristic_size_px,"characteristic_size_physical":characteristic_size_physical,
                    "window_id":int(window_id),"top":int(y),"left":int(x),"bottom":int(y+win_h),"right":int(x+win_w),
                    "center_y":float(y+win_h/2.0),"center_x":float(x+win_w/2.0),"coverage_ratio":float(coverage),"n_valid_pixels":n_valid,
                    "position_stride_y_px":meta["position_stride_y_px"],"position_stride_x_px":meta["position_stride_x_px"],
                    "candidate_position_count":meta["candidate_position_count"],"used_position_count":meta["used_position_count"],
                    "spectral_distribution_error":spectral_error,"composition_js":composition_js,"configuration_js":configuration_js,"mean_sam_deg":mean_sam,
                    "rare_state_coverage":rare_coverage,"rare_state_proportional_recovery":rare_proportional_recovery,"rare_state_presence_fraction":rare_presence_fraction,
                    "rare_state_count":rare_total,"global_rare_state_mass":global_rare_mass,"local_rare_state_mass":local_rare_mass,
                    "local_composition_entropy":_composition_entropy(local_composition),"local_configuration_entropy":_configuration_entropy(local_configuration),
                    "local_composition":json.dumps(local_composition.tolist()),"local_configuration":json.dumps(local_configuration.tolist()),
                    "projection_quantiles":json.dumps(projection_quantiles.ravel().tolist()),"task_loss":np.nan,
                    "deterministic_position_seed":_stable_seed_v27(config.windows.random_seed,"position",geometry.name,int(base_size)),
                }
                if task_loss_by_scale is not None: rec["task_loss"]=float(task_loss_by_scale.get((base_size,geometry.name),np.nan))
                for name,(reference,scale) in attribute_reference.items():
                    ap=np.asarray(attribute_maps[name],dtype=np.float64)[y:y+win_h,x:x+win_w]
                    values=ap[local_mask & np.isfinite(ap)]
                    rec[f"attribute_error__{name}"]=float(abs(np.mean(values)-reference)/scale) if values.size else np.nan
                records.append(rec)
    return pd.DataFrame.from_records(records)


# Wrap the V2.6 region export so V2.7 structural diagnostics are promoted into
# region_summary.json without duplicating the long, validated REU orchestration.
_analyze_region_v26_base = analyze_region

def analyze_region(
    cube: np.ndarray,
    region_mask: np.ndarray,
    region_id: str,
    output_dir: str | Path,
    config: REUConfig,
    attribute_maps: Optional[Mapping[str, np.ndarray]] = None,
    task_loss_by_scale: Optional[Mapping[Tuple[int, str], float]] = None,
    top_k_units: int = 10,
) -> Dict[str, Any]:
    result=_analyze_region_v26_base(cube,region_mask,region_id,output_dir,config,attribute_maps,task_loss_by_scale,top_k_units)
    rs=result["region_summary"]; corr=result["model"].correlation_diagnostics; iso=corr.get("isotropic",{})
    rs.update({
        "core_version":CORE_VERSION,
        "deterministic_window_sampling":True,
        "window_sampling_seed_policy":"stable BLAKE2b key by geometry/scale/location; geometry order independent",
        "correlation_practical_range_px":iso.get("practical_range_px"),
        "correlation_practical_range_status":iso.get("practical_range_status"),
        "correlation_integral_scale_px":iso.get("integral_scale_px"),
        "correlation_integral_scale_ci95_px":list(iso.get("bootstrap_integral_scale_ci95_px",(None,None))),
        "directional_integral_scales_px":corr.get("directional_integral_scales_px",{}),
        "integral_anisotropy_ratio":corr.get("integral_anisotropy_ratio"),
        "practical_anisotropy_ratio":corr.get("practical_anisotropy_ratio"),
        "anisotropy_supported":corr.get("anisotropy_supported",False),
        "integral_principal_direction_deg":corr.get("integral_principal_direction_deg"),
        "practical_principal_direction_deg":corr.get("practical_principal_direction_deg"),
    })
    # Compatibility alias: robust practical range.
    rs["correlation_length_px"]=iso.get("practical_range_px")
    rs["correlation_length_censored"]=bool(iso.get("practical_range_status")=="right_censored")
    _save_json(Path(output_dir)/"region_summary.json",rs)

    rows=[{
        "scope":"isotropic","angle_deg":np.nan,"practical_range_px":iso.get("practical_range_px"),
        "practical_range_status":iso.get("practical_range_status"),"integral_scale_px":iso.get("integral_scale_px"),
        "integral_ci95_low":(iso.get("bootstrap_integral_scale_ci95_px") or [None,None])[0],
        "integral_ci95_high":(iso.get("bootstrap_integral_scale_ci95_px") or [None,None])[1],
        "fit_range_px":iso.get("fit_range_px"),"fit_r2":iso.get("fit_r2")
    }]
    for a,d in corr.get("directional",{}).items():
        ci=d.get("bootstrap_integral_scale_ci95_px") or [None,None]
        rows.append({"scope":"directional","angle_deg":float(a),"practical_range_px":d.get("practical_range_px"),
                     "practical_range_status":d.get("practical_range_status"),"integral_scale_px":d.get("integral_scale_px"),
                     "integral_ci95_low":ci[0],"integral_ci95_high":ci[1],"fit_range_px":d.get("fit_range_px"),"fit_r2":d.get("fit_r2")})
    pd.DataFrame(rows).to_csv(Path(output_dir)/"correlation_scale_summary.csv",index=False)
    return result


def build_cross_region_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows=[]
    for result in results:
        base=result["region_summary"]
        for shape_name,decision in result["decisions"].items():
            rows.append({
                "region_id":base["region_id"],"shape_name":shape_name,"shape_kind":decision.get("shape_kind"),
                "shape_angle_deg":decision.get("shape_angle_deg"),"shape_aspect_ratio":decision.get("shape_aspect_ratio"),
                "status":decision.get("status"),"decision_class":decision.get("decision_class"),
                "representative_base_size_px":decision.get("base_size_px"),"characteristic_size_px":decision.get("characteristic_size_px"),
                "correlation_practical_range_px":base.get("correlation_practical_range_px"),
                "correlation_practical_range_status":base.get("correlation_practical_range_status"),
                "correlation_integral_scale_px":base.get("correlation_integral_scale_px"),
                "correlation_length_px":base.get("correlation_length_px"),
                "composition_entropy":base.get("composition_entropy"),"configuration_entropy":base.get("configuration_entropy"),
                "anisotropy_index":base.get("anisotropy_index"),"integral_anisotropy_ratio":base.get("integral_anisotropy_ratio"),
                "anisotropy_supported":base.get("anisotropy_supported"),"principal_direction_deg":base.get("principal_direction_deg"),
                "adaptive_aspect_ratio":base.get("adaptive_aspect_ratio"),"adaptive_geometry_status":base.get("adaptive_geometry_status"),
                "largest_tested_base_size_px":decision.get("largest_tested_base_size_px"),"censor_lower_bound_px":decision.get("censor_lower_bound_px"),
                "rare_constraint_enforced":decision.get("rare_constraint_enforced")
            })
    return pd.DataFrame(rows)
