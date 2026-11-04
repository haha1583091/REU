# -*- coding: utf-8 -*-
"""
Synthetic Ground-Truth Validation for NC-REU
============================================
Purpose
-------
This script tests a single scientific claim:

    A larger intrinsic spatial heterogeneity scale should require a larger
    spatial support before a hyperspectral field becomes statistically representative.

The synthetic data are NOT used to tune REU thresholds.  The same three primary
convergence tolerances used in the REU framework are fixed before simulation:

    spectral-distribution q95 <= 0.10
    composition JS q95       <= 0.10
    configuration JS q95     <= 0.15

The primary synthetic experiment uses isotropic Gaussian random fields with a
known e-folding spatial correlation scale Lc.  White Gaussian noise is filtered
with a Gaussian kernel of sigma=Lc/2.  For an infinite continuous field the
resulting autocorrelation is approximately

    C(r) = exp[-r^2 / Lc^2],

so C(Lc)=exp(-1).  Several independent latent fields with the same Lc are mapped
through a softmax to endmember abundances and then mixed into a hyperspectral
cube.  Spectral endmembers, abundance contrast and sensor noise are identical
for every Lc.  Therefore the controlled variable is the spatial structure scale.

A second anisotropic experiment uses different known e-folding scales in x and y
and checks whether the V2.7 directional correlation analysis recovers the known
principal direction.

Important design choices
------------------------
1. Synthetic truth is used for causal validation, not threshold tuning.
2. Isotropic REU recovery uses the exact V2.7 state-space representation and
   the exact V2.7 spectral/composition/configuration metrics.
3. Correlation estimation is disabled during isotropic REU recovery because Lc
   is already imposed by construction and correlation is not part of the three
   REU acceptance metrics.  This avoids a circular validation and makes the run
   much faster.
4. The anisotropy experiment separately restores the original V2.7 correlation
   estimator and evaluates directional recovery.
5. No power-law or universal scaling law is fitted.  The main test is monotonic
   response of estimated REU to the known input spatial scale.

Expected files
--------------
<OUTPUT_ROOT>/
    synthetic_validation_config.json
    synthetic_endmembers.csv
    isotropic_realization_results.csv
    isotropic_group_summary.csv
    isotropic_scale_metrics_all.csv
    threshold_sensitivity.csv
    monotonicity_test.json
    anisotropy_validation.json
    Fig_1_synthetic_input_fields.jpg
    Fig_2_REU_ground_truth_recovery.jpg
    Fig_3_normalized_convergence.jpg
    Fig_4_anisotropy_validation.jpg
    lc_XX_seed_YYYY/
        dominant_endmember_map.npy
        abundance_cube.npy              (optional)
        scale_metrics.csv
        result.json
        convergence.jpg

Run
---
Place this file in the same directory as:
    reu_nc_core_v2_7.py
    reu_nc_geometry_v2_7.py

Then run, for example:
    D:\\Anaconda\\envs\\DL\\python.exe D:\\LS_REU\\260811\\reu_nc_synthetic_ground_truth_validation.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Locate V2.7 core next to this script.
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import reu_nc_core_v2_7 as core
    from reu_nc_core_v2_7 import (
        REUConfig,
        PreprocessConfig,
        StateModelConfig,
        WindowConfig,
        MetricConfig,
        CorrelationConfig,
        ConvergenceConfig,
    )
    from reu_nc_geometry_v2_7 import ShapeSpec
except Exception as exc:
    raise RuntimeError(
        "Cannot import V2.7 core. Put reu_nc_core_v2_7.py and "
        "reu_nc_geometry_v2_7.py in the same folder as this script."
    ) from exc


# =============================================================================
# 0. USER SETTINGS — keep these fixed for the reported experiment
# =============================================================================

OUTPUT_ROOT = Path(r"D:\LS_REU\results\nc_reu_synthetic_ground_truth")

# Four known isotropic spatial scales spanning one order of magnitude.
# The default maximum Lc=32 px keeps the synthetic domain >=10*Lc, reducing
# finite-domain censoring.  If you later want Lc=64 px, increase FIELD_SIZE to
# at least 640 px rather than simply appending 64 to this list.
TRUE_LC_VALUES_PX: Tuple[int, ...] = (4, 8, 16, 32)

# Independent realizations. Three are sufficient for the main causal check;
# set to five for a stronger final supplementary analysis if runtime permits.
REALIZATION_SEEDS: Tuple[int, ...] = (3407, 2027, 5909)

FIELD_SIZE = 320
N_BANDS = 64
N_ENDMEMBERS = 4
WAVELENGTH_MIN_NM = 1000.0
WAVELENGTH_MAX_NM = 2500.0

# Fixed synthetic physics/statistics across all Lc.
ABUNDANCE_CONTRAST = 0.55
SENSOR_NOISE_STD = 0.003
SAVE_ABUNDANCE_CUBES = False

# Candidate spatial supports.  Dense small scales are included, but the list
# also extends far enough to resolve the Lc=32 px case in a 320x320 field.
WINDOW_SCALES_PX: Tuple[int, ...] = (
    4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 80, 96, 112, 128, 144,
    160, 176, 192, 208, 224, 240, 256,
)

# Window-position uncertainty.  A q95 is estimated from many spatial positions.
MIN_WINDOWS_PER_SCALE = 12
MAX_WINDOWS_PER_SCALE = 60
MAX_SPECTRA_PER_WINDOW = 700
MAX_REFERENCE_SPECTRA = 6000
MAX_POSITION_STRIDE_PX = 8

# Primary REU criteria — fixed, not tuned on synthetic outcomes.
SPECTRAL_Q95_MAX = 0.10
COMPOSITION_Q95_MAX = 0.10
CONFIGURATION_Q95_MAX = 0.15
CONSECUTIVE_PASSING_SCALES = 3

# Automatic supplementary sensitivity; this does NOT rerun the spectral model.
THRESHOLD_MULTIPLIERS: Tuple[float, ...] = (0.8, 1.0, 1.2)

# Anisotropic ground truth.  Major direction is y, i.e. 90 degrees measured
# counter-clockwise from the image x-axis, matching the V2.7 convention.
RUN_ANISOTROPY_VALIDATION = True
ANISO_LC_X_PX = 8
ANISO_LC_Y_PX = 32
ANISO_TRUE_MAJOR_DIRECTION_DEG = 90.0
ANISO_SEED = 8849

# Full V2.7 directional correlation is used only once, here.
ANISO_CORRELATION_BOOTSTRAPS = 40

# Reproducibility / rerun behavior.
OVERWRITE = True


# =============================================================================
# 1. GENERAL UTILITIES
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
    """Small deterministic hash independent of Python's randomized hash()."""
    x = 2166136261
    for token in tokens:
        for byte in str(token).encode("utf-8"):
            x ^= int(byte)
            x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def setup_plot_fonts() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600


# =============================================================================
# 2. SYNTHETIC HYPERSPECTRAL FIELD
# =============================================================================

def make_endmembers(
    n_bands: int = N_BANDS,
    n_endmembers: int = N_ENDMEMBERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create smooth, deterministic, non-pathological synthetic endmembers.

    These are deliberately generic rather than labelled as real minerals.  The
    purpose is to test spatial-scale recovery without introducing a geological
    prior or a library-spectrum advantage.
    """
    if n_endmembers != 4:
        raise ValueError("This compact synthetic design currently expects 4 endmembers.")

    wl = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n_bands)
    definitions = [
        # baseline, slope, [(centre_nm, depth, width_nm), ...]
        (0.34, +3.0e-5, [(1400, 0.050, 45), (2200, 0.090, 34)]),
        (0.42, -2.0e-5, [(1900, 0.080, 58), (2340, 0.075, 38)]),
        (0.30, +1.0e-5, [(1450, 0.045, 42), (2250, 0.085, 31), (2410, 0.045, 34)]),
        (0.50, -1.0e-5, [(1200, 0.040, 50), (2100, 0.065, 46)]),
    ]

    endmembers: List[np.ndarray] = []
    for j, (baseline, slope, absorptions) in enumerate(definitions):
        e = baseline + slope * (wl - 1750.0)
        e = e + 0.010 * np.sin((wl - 1000.0) / 210.0 + 0.65 * j)
        for center, depth, width in absorptions:
            e = e - depth * np.exp(-0.5 * ((wl - center) / width) ** 2)
        endmembers.append(np.clip(e, 0.03, 0.85))

    return wl.astype(np.float64), np.asarray(endmembers, dtype=np.float32)


def _standardize_field(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return (z - np.mean(z)) / max(float(np.std(z)), 1.0e-12)


def generate_isotropic_cube(
    lc_px: float,
    seed: int,
    field_size: int = FIELD_SIZE,
    n_bands: int = N_BANDS,
) -> Dict[str, np.ndarray]:
    """Generate an isotropic hyperspectral field with known e-fold scale Lc.

    White noise is convolved with a Gaussian kernel of sigma=Lc/2.  The
    autocorrelation of the filtered Gaussian field is approximately
    exp(-r^2/Lc^2), so Lc is the imposed 1/e spatial correlation distance.
    """
    rng = np.random.default_rng(int(seed))
    wl, endmembers = make_endmembers(n_bands=n_bands)

    sigma = float(lc_px) / 2.0
    latent_fields: List[np.ndarray] = []
    for _ in range(N_ENDMEMBERS):
        white = rng.normal(0.0, 1.0, size=(field_size, field_size))
        smooth = gaussian_filter(white, sigma=sigma, mode="reflect")
        latent_fields.append(_standardize_field(smooth))

    z = np.stack(latent_fields, axis=-1)
    biases = np.asarray([0.24, 0.08, -0.08, -0.24], dtype=np.float64)
    logits = ABUNDANCE_CONTRAST * z + biases[None, None, :]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    abundance = np.exp(logits)
    abundance = abundance / np.sum(abundance, axis=-1, keepdims=True)

    cube = abundance @ endmembers
    cube = cube + rng.normal(0.0, SENSOR_NOISE_STD, size=cube.shape)
    cube = np.clip(cube, 0.02, 0.90).astype(np.float32)

    return {
        "cube": cube,
        "abundance": abundance.astype(np.float32),
        "dominant_map": np.argmax(abundance, axis=-1).astype(np.int16),
        "latent_fields": z.astype(np.float32),
        "wavelength_nm": wl,
        "endmembers": endmembers,
    }


def generate_anisotropic_cube(
    lc_x_px: float,
    lc_y_px: float,
    seed: int,
    field_size: int = FIELD_SIZE,
    n_bands: int = N_BANDS,
) -> Dict[str, np.ndarray]:
    """Generate axis-aligned anisotropic synthetic field.

    For each latent channel, Gaussian filtering uses
        sigma_y = Lc_y / 2,
        sigma_x = Lc_x / 2,
    giving an approximate covariance
        C(dx,dy)=exp[-dx^2/Lc_x^2 - dy^2/Lc_y^2].
    """
    rng = np.random.default_rng(int(seed))
    wl, endmembers = make_endmembers(n_bands=n_bands)

    sigma_y = float(lc_y_px) / 2.0
    sigma_x = float(lc_x_px) / 2.0
    latent_fields: List[np.ndarray] = []
    for _ in range(N_ENDMEMBERS):
        white = rng.normal(0.0, 1.0, size=(field_size, field_size))
        smooth = gaussian_filter(white, sigma=(sigma_y, sigma_x), mode="reflect")
        latent_fields.append(_standardize_field(smooth))

    z = np.stack(latent_fields, axis=-1)
    biases = np.asarray([0.24, 0.08, -0.08, -0.24], dtype=np.float64)
    logits = ABUNDANCE_CONTRAST * z + biases[None, None, :]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    abundance = np.exp(logits)
    abundance = abundance / np.sum(abundance, axis=-1, keepdims=True)

    cube = abundance @ endmembers
    cube = cube + rng.normal(0.0, SENSOR_NOISE_STD, size=cube.shape)
    cube = np.clip(cube, 0.02, 0.90).astype(np.float32)

    return {
        "cube": cube,
        "abundance": abundance.astype(np.float32),
        "dominant_map": np.argmax(abundance, axis=-1).astype(np.int16),
        "latent_fields": z.astype(np.float32),
        "wavelength_nm": wl,
        "endmembers": endmembers,
    }


# =============================================================================
# 3. V2.7 STATE MODEL + LIGHTWEIGHT SQUARE-SUPPORT METRICS
# =============================================================================

def make_v27_config(
    correlation_bootstraps: int = 0,
    max_lag_px: int = 128,
) -> REUConfig:
    """Configuration aligned with the validated V2.7 metric definitions.

    The window evaluator below is a lightweight square-only evaluator, but the
    PCA/soft-state model and all three REU metrics are taken directly from the
    V2.7 core.
    """
    return REUConfig(
        pixel_size=None,
        physical_unit="pixel",
        preprocess=PreprocessConfig(
            clip_percentiles=(0.5, 99.5),
            savgol_enabled=False,
            savgol_window=9,
            savgol_polyorder=2,
            l2_normalize_for_state_model=True,
        ),
        state_model=StateModelConfig(
            n_components=6,
            n_states=8,
            fit_sample_size=30000,
            soft_temperature_quantile=0.50,
            random_seed=3407,
        ),
        windows=WindowConfig(
            window_sizes_px=list(WINDOW_SCALES_PX),
            shape_specs=[ShapeSpec("square", "rectangle", 1.0, 0.0)],
            overlap=0.75,
            max_position_stride_px=MAX_POSITION_STRIDE_PX,
            min_region_coverage=0.98,
            min_valid_pixels=16,
            min_windows_per_scale=MIN_WINDOWS_PER_SCALE,
            max_windows_per_scale=MAX_WINDOWS_PER_SCALE,
            max_spectra_per_window=MAX_SPECTRA_PER_WINDOW,
            max_reference_spectra=MAX_REFERENCE_SPECTRA,
            random_seed=3407,
        ),
        metrics=MetricConfig(
            projection_count=24,
            projection_quantiles=33,
            cooccurrence_distances_px=[1, 2, 4],
            cooccurrence_directions=["horizontal", "vertical", "diag_down", "diag_up"],
            rare_state_probability=0.08,
        ),
        correlation=CorrelationConfig(
            max_lag_fraction_short_side=0.45,
            max_lag_px=max_lag_px,
            radial_pair_count=80000,
            radial_bins=20,
            min_pairs_per_bin=80,
            directional_target_lags_px=[2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128],
            min_pairs_per_directional_lag=80,
            bootstrap_iterations=int(correlation_bootstraps),
            bootstrap_max_pairs_per_lag=2500,
            adaptive_min_anisotropy_ratio=1.15,
            adaptive_min_principal_frequency=0.60,
            adaptive_min_reliable_directions=4,
            adaptive_max_aspect_ratio=3.0,
            practical_range_low_r2_agreement_tolerance=0.30,
            practical_range_tail_min=0.90,
            adaptive_integral_ratio_min=1.25,
            adaptive_integral_ci_lower_min=1.15,
            adaptive_practical_ratio_min=1.15,
            adaptive_min_practical_reliable_directions=3,
            adaptive_require_principal_concordance=True,
            random_seed=3407,
        ),
        convergence=ConvergenceConfig(
            spectral_error_q95_max=SPECTRAL_Q95_MAX,
            composition_js_q95_max=COMPOSITION_Q95_MAX,
            configuration_js_q95_max=CONFIGURATION_Q95_MAX,
            spectral_funnel_width_max=0.10,
            composition_funnel_width_max=0.10,
            configuration_funnel_width_max=0.15,
            rare_state_coverage_q05_min=0.80,
            enforce_rare_state_constraint=False,
            relative_stability_max=0.20,
            stability_window_scales=3,
            consecutive_scales=3,
        ),
    )


def _dummy_correlation_structure(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Minimal structure needed by fit_spectral_state_model.

    This is deliberately used only in the isotropic REU recovery experiment.
    REU q95 metrics do not use the correlation estimate.  The anisotropy test
    restores the original V2.7 estimator.
    """
    directional = {}
    for angle in (0, 45, 90, 135):
        directional[angle] = {
            "status": "disabled_for_non_circular_synthetic_reu_test",
            "range_px": None,
            "fit_range_px": None,
            "range_censored": False,
            "fit_r2": None,
            "lag_px": [],
            "gamma_normalized": [],
            "pair_count": [],
        }
    return {
        "isotropic": {
            "status": "disabled_for_non_circular_synthetic_reu_test",
            "range_px": None,
            "fit_range_px": None,
            "range_lower_bound_px": None,
            "range_censored": False,
            "fit_r2": None,
            "bootstrap_range_ci95_px": (None, None),
            "lag_px": [],
            "gamma_normalized": [],
            "pair_count": [],
        },
        "directional_ranges_px": {},
        "principal_direction_deg": float("nan"),
        "adaptive_aspect_ratio": 1.0,
        "anisotropy_index": float("nan"),
        "directional": directional,
        "adaptive_geometry_status": "disabled_for_isotropic_synthetic_reu_test",
        "principal_direction_bootstrap_frequency": float("nan"),
        "anisotropy_ratio_ci95": (None, None),
    }


@contextmanager
def correlation_disabled_for_isotropic_reu() -> Iterable[None]:
    original = core.estimate_corrected_correlation_structure
    core.estimate_corrected_correlation_structure = _dummy_correlation_structure
    try:
        yield
    finally:
        core.estimate_corrected_correlation_structure = original


def _sample_square_positions(
    h: int,
    w: int,
    size: int,
    max_windows: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """Deterministic, spatially distributed square-window positions.

    Candidate positions are generated on a bounded stride grid; if there are
    too many, a deterministic random subset is used.  No position is selected
    according to spectral similarity, so there is no best-window bias.
    """
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

    positions = [(int(y), int(x)) for y in ys for x in xs]
    if len(positions) <= max_windows:
        return positions

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(positions), size=max_windows, replace=False))
    return [positions[int(i)] for i in idx]


def evaluate_square_scales(
    cube: np.ndarray,
    model: Any,
    config: REUConfig,
    realization_seed: int,
) -> pd.DataFrame:
    """Evaluate exact V2.7 q95 ingredients using only square windows.

    This function is intentionally much lighter than the full geometry engine.
    The three per-window metrics call V2.7 core functions directly:
      - core._spectral_distribution_error
      - core._safe_js
      - core.compute_soft_cooccurrence
    """
    cube = np.asarray(cube, dtype=np.float32)
    h, w, _ = cube.shape
    rows: List[Dict[str, Any]] = []

    for size in WINDOW_SCALES_PX:
        if size > min(h, w):
            continue

        positions = _sample_square_positions(
            h=h,
            w=w,
            size=int(size),
            max_windows=MAX_WINDOWS_PER_SCALE,
            seed=stable_seed("positions", realization_seed, size),
        )
        if not positions:
            continue

        spectral_errors: List[float] = []
        composition_errors: List[float] = []
        configuration_errors: List[float] = []

        for window_id, (y, x) in enumerate(positions):
            local_latent = model.latent_cube[y:y + size, x:x + size].reshape(-1, model.latent_cube.shape[-1])
            soft_patch = model.soft_state_cube[y:y + size, x:x + size]
            local_mask = np.ones((size, size), dtype=bool)

            # Deterministic spectral subsampling within large supports.
            if len(local_latent) > MAX_SPECTRA_PER_WINDOW:
                rng = np.random.default_rng(stable_seed("spectra", realization_seed, size, window_id))
                idx = rng.choice(len(local_latent), MAX_SPECTRA_PER_WINDOW, replace=False)
                latent_sample = local_latent[idx].astype(np.float64)
            else:
                latent_sample = local_latent.astype(np.float64)

            spectral_error, _ = core._spectral_distribution_error(latent_sample, model)
            local_composition = soft_patch.reshape(-1, soft_patch.shape[-1]).mean(axis=0)
            composition_js = core._safe_js(local_composition, model.global_composition)
            local_configuration, _ = core.compute_soft_cooccurrence(
                soft_patch,
                local_mask,
                config.metrics.cooccurrence_distances_px,
                config.metrics.cooccurrence_directions,
            )
            configuration_js = core._safe_js(local_configuration, model.global_configuration)

            spectral_errors.append(float(spectral_error))
            composition_errors.append(float(composition_js))
            configuration_errors.append(float(configuration_js))

        n = len(spectral_errors)
        def q(arr: Sequence[float], p: float) -> float:
            return float(np.quantile(np.asarray(arr, dtype=float), p))

        rows.append({
            "base_size_px": int(size),
            "valid_window_count": int(n),
            "spectral_error_q05": q(spectral_errors, 0.05),
            "spectral_error_q50": q(spectral_errors, 0.50),
            "spectral_error_q95": q(spectral_errors, 0.95),
            "composition_js_q05": q(composition_errors, 0.05),
            "composition_js_q50": q(composition_errors, 0.50),
            "composition_js_q95": q(composition_errors, 0.95),
            "configuration_js_q05": q(configuration_errors, 0.05),
            "configuration_js_q50": q(configuration_errors, 0.50),
            "configuration_js_q95": q(configuration_errors, 0.95),
        })

    return pd.DataFrame(rows).sort_values("base_size_px").reset_index(drop=True)


def decide_reu(
    scale_df: pd.DataFrame,
    threshold_multiplier: float = 1.0,
    consecutive: int = CONSECUTIVE_PASSING_SCALES,
) -> Dict[str, Any]:
    """First of N consecutive supported scales satisfying all three q95 tests."""
    if scale_df.empty:
        return {
            "status": "no_evaluable_scale",
            "reu_px": None,
            "convergence_onset_px": None,
            "largest_supported_scale_px": None,
        }

    df = scale_df.sort_values("base_size_px").copy()
    df["enough_windows"] = df["valid_window_count"] >= MIN_WINDOWS_PER_SCALE
    df["pass_spectral"] = df["spectral_error_q95"] <= SPECTRAL_Q95_MAX * threshold_multiplier
    df["pass_composition"] = df["composition_js_q95"] <= COMPOSITION_Q95_MAX * threshold_multiplier
    df["pass_configuration"] = df["configuration_js_q95"] <= CONFIGURATION_Q95_MAX * threshold_multiplier
    df["three_constraints_pass"] = (
        df["enough_windows"]
        & df["pass_spectral"]
        & df["pass_composition"]
        & df["pass_configuration"]
    )

    supported = df[df["enough_windows"]].reset_index(drop=True)
    largest_supported = None if supported.empty else int(supported.iloc[-1]["base_size_px"])
    passing_supported = supported[supported["three_constraints_pass"]]
    onset = None if passing_supported.empty else int(passing_supported.iloc[0]["base_size_px"])

    reu = None
    if len(supported) >= consecutive:
        flags = supported["three_constraints_pass"].to_numpy(dtype=bool)
        for i in range(0, len(flags) - consecutive + 1):
            if bool(np.all(flags[i:i + consecutive])):
                reu = int(supported.iloc[i]["base_size_px"])
                break

    if reu is not None:
        status = "reu_found"
    elif onset is not None and largest_supported == onset:
        status = "right_censored_after_convergence_onset"
    elif onset is not None:
        status = "convergence_seen_but_not_consecutively_confirmed"
    else:
        status = "no_reu_within_supported_range"

    return {
        "status": status,
        "reu_px": reu,
        "convergence_onset_px": onset,
        "largest_supported_scale_px": largest_supported,
        "threshold_multiplier": float(threshold_multiplier),
        "consecutive_scales": int(consecutive),
    }


def add_decision_columns(scale_df: pd.DataFrame, multiplier: float = 1.0) -> pd.DataFrame:
    df = scale_df.copy()
    df["enough_windows"] = df["valid_window_count"] >= MIN_WINDOWS_PER_SCALE
    df["pass_spectral"] = df["spectral_error_q95"] <= SPECTRAL_Q95_MAX * multiplier
    df["pass_composition"] = df["composition_js_q95"] <= COMPOSITION_Q95_MAX * multiplier
    df["pass_configuration"] = df["configuration_js_q95"] <= CONFIGURATION_Q95_MAX * multiplier
    df["three_constraints_pass"] = (
        df["enough_windows"]
        & df["pass_spectral"]
        & df["pass_composition"]
        & df["pass_configuration"]
    )
    df["spectral_ratio"] = df["spectral_error_q95"] / SPECTRAL_Q95_MAX
    df["composition_ratio"] = df["composition_js_q95"] / COMPOSITION_Q95_MAX
    df["configuration_ratio"] = df["configuration_js_q95"] / CONFIGURATION_Q95_MAX
    df["minimax_ratio"] = df[["spectral_ratio", "composition_ratio", "configuration_ratio"]].max(axis=1)
    return df


# =============================================================================
# 4. INDIVIDUAL ISOTROPIC RUN
# =============================================================================

def plot_case_convergence(df: pd.DataFrame, lc_px: int, seed: int, out_path: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    x = df["base_size_px"].to_numpy()
    ax.plot(x, df["spectral_ratio"], marker="o", markersize=4, linewidth=1.3, label="Spectral q95 / threshold")
    ax.plot(x, df["composition_ratio"], marker="s", markersize=4, linewidth=1.3, label="Composition q95 / threshold")
    ax.plot(x, df["configuration_ratio"], marker="^", markersize=4, linewidth=1.3, label="Configuration q95 / threshold")
    ax.axhline(1.0, linewidth=1.0, linestyle="--")
    decision = decide_reu(df)
    if decision["reu_px"] is not None:
        ax.axvline(decision["reu_px"], linewidth=1.0, linestyle=":")
        ax.text(decision["reu_px"], 0.03, f"REU={decision['reu_px']} px", rotation=90, va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("Spatial support L (px)")
    ax.set_ylabel("Normalized q95 error")
    ax.set_title(f"Synthetic field: $L_c$={lc_px} px, seed={seed}")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_isotropic_realization(lc_px: int, seed: int, output_root: Path) -> Dict[str, Any]:
    case_dir = output_root / f"lc_{int(lc_px):03d}_seed_{int(seed)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_json = case_dir / "result.json"

    if result_json.exists() and not OVERWRITE:
        with result_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    synthetic = generate_isotropic_cube(lc_px=float(lc_px), seed=int(seed))
    cube = synthetic["cube"]
    mask = np.ones(cube.shape[:2], dtype=bool)
    config = make_v27_config(correlation_bootstraps=0)

    # Fit the exact V2.7 PCA + soft-state representation while deliberately
    # skipping correlation estimation for this non-circular recovery test.
    with correlation_disabled_for_isotropic_reu():
        model = core.fit_spectral_state_model(cube, mask, config)

    scale_df = evaluate_square_scales(cube, model, config, realization_seed=seed)
    scale_df = add_decision_columns(scale_df, multiplier=1.0)
    decision = decide_reu(scale_df, threshold_multiplier=1.0)

    scale_df.to_csv(case_dir / "scale_metrics.csv", index=False)
    np.save(case_dir / "dominant_endmember_map.npy", synthetic["dominant_map"])
    if SAVE_ABUNDANCE_CUBES:
        np.save(case_dir / "abundance_cube.npy", synthetic["abundance"].astype(np.float32))

    plot_case_convergence(scale_df, int(lc_px), int(seed), case_dir / "convergence.jpg")

    result = {
        "true_lc_px": int(lc_px),
        "construction": "isotropic Gaussian random field; sigma=Lc/2; approximate C(r)=exp(-r^2/Lc^2)",
        "seed": int(seed),
        "reu_px": decision["reu_px"],
        "reu_to_lc_ratio": (
            float(decision["reu_px"] / lc_px) if decision["reu_px"] is not None else None
        ),
        "status": decision["status"],
        "convergence_onset_px": decision["convergence_onset_px"],
        "largest_supported_scale_px": decision["largest_supported_scale_px"],
        "runtime_seconds": float(time.time() - t0),
    }
    save_json(result_json, result)
    return result


# =============================================================================
# 5. AGGREGATION, MONOTONICITY AND SENSITIVITY
# =============================================================================

def aggregate_isotropic_results(output_root: Path, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(results).sort_values(["true_lc_px", "seed"]).reset_index(drop=True)
    df.to_csv(output_root / "isotropic_realization_results.csv", index=False)

    grouped_rows: List[Dict[str, Any]] = []
    for lc, g in df.groupby("true_lc_px", sort=True):
        valid = pd.to_numeric(g["reu_px"], errors="coerce").dropna().to_numpy(dtype=float)
        ratios = pd.to_numeric(g["reu_to_lc_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
        grouped_rows.append({
            "true_lc_px": int(lc),
            "n_realizations": int(len(g)),
            "n_reu_found": int(len(valid)),
            "reu_found_fraction": float(len(valid) / max(len(g), 1)),
            "reu_median_px": float(np.median(valid)) if len(valid) else np.nan,
            "reu_min_px": float(np.min(valid)) if len(valid) else np.nan,
            "reu_max_px": float(np.max(valid)) if len(valid) else np.nan,
            "reu_q25_px": float(np.quantile(valid, 0.25)) if len(valid) else np.nan,
            "reu_q75_px": float(np.quantile(valid, 0.75)) if len(valid) else np.nan,
            "reu_to_lc_ratio_median": float(np.median(ratios)) if len(ratios) else np.nan,
        })
    group_df = pd.DataFrame(grouped_rows)
    group_df.to_csv(output_root / "isotropic_group_summary.csv", index=False)
    return df, group_df


def collect_all_scale_metrics(output_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for lc in TRUE_LC_VALUES_PX:
        for seed in REALIZATION_SEEDS:
            p = output_root / f"lc_{int(lc):03d}_seed_{int(seed)}" / "scale_metrics.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d.insert(0, "seed", int(seed))
            d.insert(0, "true_lc_px", int(lc))
            rows.append(d)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(output_root / "isotropic_scale_metrics_all.csv", index=False)
    return out


def monotonicity_test(group_df: pd.DataFrame) -> Dict[str, Any]:
    valid = group_df[np.isfinite(group_df["reu_median_px"])].sort_values("true_lc_px")
    if len(valid) < 3:
        result = {
            "status": "insufficient_valid_Lc_levels",
            "n_valid_levels": int(len(valid)),
        }
        save_json(OUTPUT_ROOT / "monotonicity_test.json", result)
        return result

    x = valid["true_lc_px"].to_numpy(dtype=float)
    y = valid["reu_median_px"].to_numpy(dtype=float)
    rho, p = spearmanr(x, y)
    diffs = np.diff(y)
    result = {
        "status": "evaluated",
        "n_valid_levels": int(len(valid)),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p),
        "strictly_monotonic_increase": bool(np.all(diffs > 0)),
        "nondecreasing": bool(np.all(diffs >= 0)),
        "true_lc_px": x.tolist(),
        "median_reu_px": y.tolist(),
        "interpretation": (
            "Primary success criterion is a positive monotonic response of REU to the imposed spatial scale; "
            "no universal functional law is assumed."
        ),
    }
    save_json(OUTPUT_ROOT / "monotonicity_test.json", result)
    return result


def threshold_sensitivity(output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lc in TRUE_LC_VALUES_PX:
        for seed in REALIZATION_SEEDS:
            p = output_root / f"lc_{int(lc):03d}_seed_{int(seed)}" / "scale_metrics.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            for multiplier in THRESHOLD_MULTIPLIERS:
                decision = decide_reu(d, threshold_multiplier=float(multiplier))
                rows.append({
                    "true_lc_px": int(lc),
                    "seed": int(seed),
                    "threshold_multiplier": float(multiplier),
                    "reu_px": decision["reu_px"],
                    "status": decision["status"],
                })
    out = pd.DataFrame(rows)
    out.to_csv(output_root / "threshold_sensitivity.csv", index=False)
    return out


# =============================================================================
# 6. ANISOTROPY GROUND-TRUTH VALIDATION
# =============================================================================

def orientation_error_deg(estimated: float, truth: float) -> float:
    if not np.isfinite(estimated):
        return float("nan")
    # Undirected axis: theta and theta+180 are equivalent.
    delta = abs((estimated - truth) % 180.0)
    return float(min(delta, 180.0 - delta))


def run_anisotropy_validation(output_root: Path) -> Dict[str, Any]:
    out_dir = output_root / "anisotropy_case"
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic = generate_anisotropic_cube(
        lc_x_px=ANISO_LC_X_PX,
        lc_y_px=ANISO_LC_Y_PX,
        seed=ANISO_SEED,
    )
    cube = synthetic["cube"]
    mask = np.ones(cube.shape[:2], dtype=bool)

    config = make_v27_config(
        correlation_bootstraps=ANISO_CORRELATION_BOOTSTRAPS,
        max_lag_px=min(128, FIELD_SIZE // 2),
    )

    # Do NOT monkeypatch here. This is the independent V2.7 direction-recovery test.
    model = core.fit_spectral_state_model(cube, mask, config)
    diag = model.correlation_diagnostics

    directional_rows: List[Dict[str, Any]] = []
    for angle, d in model.directional_correlation_diagnostics.items():
        directional_rows.append({
            "angle_deg": int(angle),
            "practical_range_px": d.get("practical_range_px", d.get("range_px")),
            "fit_range_px": d.get("fit_range_px"),
            "integral_scale_px": d.get("integral_scale_px"),
            "fit_r2": d.get("fit_r2"),
            "status": d.get("practical_range_status", d.get("status")),
        })
    directional_df = pd.DataFrame(directional_rows).sort_values("angle_deg")
    directional_df.to_csv(out_dir / "directional_recovery.csv", index=False)

    # V2.7 deliberately degenerates the adaptive geometry to a circle when its
    # bootstrap gate is not met; in that case model.principal_direction_deg may
    # be reset to 0.  For a ground-truth direction test, the unbiased quantity
    # is therefore the direction with the largest raw directional integral scale.
    finite_dir = directional_df[np.isfinite(pd.to_numeric(directional_df["integral_scale_px"], errors="coerce"))].copy()
    if len(finite_dir):
        estimated_direction = float(finite_dir.loc[pd.to_numeric(finite_dir["integral_scale_px"], errors="coerce").idxmax(), "angle_deg"])
    else:
        estimated_direction = float("nan")
    angular_error = orientation_error_deg(estimated_direction, ANISO_TRUE_MAJOR_DIRECTION_DEG)
    np.save(out_dir / "dominant_endmember_map.npy", synthetic["dominant_map"])

    result = {
        "true_lc_x_px": int(ANISO_LC_X_PX),
        "true_lc_y_px": int(ANISO_LC_Y_PX),
        "true_major_direction_deg": float(ANISO_TRUE_MAJOR_DIRECTION_DEG),
        "true_anisotropy_ratio": float(max(ANISO_LC_X_PX, ANISO_LC_Y_PX) / min(ANISO_LC_X_PX, ANISO_LC_Y_PX)),
        "estimated_raw_integral_principal_direction_deg": estimated_direction if np.isfinite(estimated_direction) else None,
        "raw_integral_direction_error_deg": angular_error if np.isfinite(angular_error) else None,
        "v27_gated_principal_direction_deg": (
            float(model.principal_direction_deg) if np.isfinite(model.principal_direction_deg) else None
        ),
        "estimated_anisotropy_index": float(model.anisotropy_index) if np.isfinite(model.anisotropy_index) else None,
        "adaptive_aspect_ratio": float(model.adaptive_aspect_ratio),
        "adaptive_geometry_status": str(model.adaptive_geometry_status),
        "principal_direction_bootstrap_frequency": (
            float(model.principal_direction_bootstrap_frequency)
            if np.isfinite(model.principal_direction_bootstrap_frequency) else None
        ),
        "anisotropy_ratio_ci95": list(model.anisotropy_ratio_ci95),
        "direction_recovered_within_22_5_deg": (
            bool(angular_error <= 22.5) if np.isfinite(angular_error) else False
        ),
        "note": "Direction recovery is a separate structural validation and is not used to tune the isotropic REU thresholds.",
    }
    save_json(output_root / "anisotropy_validation.json", result)
    return result


# =============================================================================
# 7. PUBLICATION FIGURES
# =============================================================================

def plot_input_fields(output_root: Path) -> None:
    setup_plot_fonts()
    n = len(TRUE_LC_VALUES_PX)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    for j, lc in enumerate(TRUE_LC_VALUES_PX):
        p = output_root / f"lc_{int(lc):03d}_seed_{int(REALIZATION_SEEDS[0])}" / "dominant_endmember_map.npy"
        ax = axes[0, j]
        if p.exists():
            arr = np.load(p)
            ax.imshow(arr, interpolation="nearest")
        ax.set_title(f"$L_c$ = {lc} px")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(output_root / "Fig_1_synthetic_input_fields.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_reu_recovery(real_df: pd.DataFrame, group_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    for lc in sorted(real_df["true_lc_px"].unique()):
        g = real_df[real_df["true_lc_px"] == lc]
        y = pd.to_numeric(g["reu_px"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y):
            xj = np.full(len(y), float(lc))
            ax.scatter(xj, y, s=28, alpha=0.75)

    valid = group_df[np.isfinite(group_df["reu_median_px"])].sort_values("true_lc_px")
    if not valid.empty:
        x = valid["true_lc_px"].to_numpy(dtype=float)
        y = valid["reu_median_px"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - valid["reu_min_px"].to_numpy(dtype=float),
            valid["reu_max_px"].to_numpy(dtype=float) - y,
        ])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=4, label="Median REU (range)")

    ax.set_xlabel("Imposed spatial correlation scale $L_c$ (px)")
    ax.set_ylabel("Estimated REU (px)")
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_root / "Fig_2_REU_ground_truth_recovery.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_normalized_convergence(all_scale_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    if all_scale_df.empty:
        plt.close(fig)
        return

    for lc, g_lc in all_scale_df.groupby("true_lc_px", sort=True):
        curves: List[pd.DataFrame] = []
        for seed, g in g_lc.groupby("seed", sort=True):
            q = g[["base_size_px", "minimax_ratio"]].copy()
            q["normalized_support"] = q["base_size_px"] / float(lc)
            curves.append(q)
        merged = pd.concat(curves, ignore_index=True)
        # Round only for aggregation; raw CSV retains exact scales.
        merged["lambda_round"] = merged["normalized_support"].round(3)
        agg = merged.groupby("lambda_round", as_index=False)["minimax_ratio"].median()
        ax.plot(
            agg["lambda_round"],
            agg["minimax_ratio"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=f"$L_c$={int(lc)} px",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Normalized support $L/L_c$")
    ax.set_ylabel("Median maximum normalized q95 error")
    ax.set_yscale("log")
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_root / "Fig_3_normalized_convergence.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_anisotropy(output_root: Path) -> None:
    p = output_root / "anisotropy_case" / "directional_recovery.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    metric = None
    for candidate in ("integral_scale_px", "practical_range_px", "fit_range_px"):
        if candidate in d.columns and pd.to_numeric(d[candidate], errors="coerce").notna().any():
            metric = candidate
            break
    if metric is None:
        plt.close(fig)
        return
    x = d["angle_deg"].to_numpy(dtype=float)
    y = pd.to_numeric(d[metric], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=1.4)
    ax.axvline(ANISO_TRUE_MAJOR_DIRECTION_DEG, linestyle="--", linewidth=1.0, label="True major direction")
    ax.set_xticks([0, 45, 90, 135])
    ax.set_xlabel("Direction (deg)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_root / "Fig_4_anisotropy_validation.jpg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_plot_fonts()

    # Save exact synthetic experiment design before analysis.
    config_record = {
        "script": Path(__file__).name,
        "core_version": getattr(core, "CORE_VERSION", "unknown"),
        "scientific_question": "Does estimated REU increase when the only controlled change is the intrinsic spatial heterogeneity scale?",
        "true_lc_values_px": list(TRUE_LC_VALUES_PX),
        "realization_seeds": list(REALIZATION_SEEDS),
        "field_size": int(FIELD_SIZE),
        "n_bands": int(N_BANDS),
        "n_endmembers": int(N_ENDMEMBERS),
        "abundance_contrast": float(ABUNDANCE_CONTRAST),
        "sensor_noise_std": float(SENSOR_NOISE_STD),
        "window_scales_px": list(WINDOW_SCALES_PX),
        "min_windows_per_scale": int(MIN_WINDOWS_PER_SCALE),
        "max_windows_per_scale": int(MAX_WINDOWS_PER_SCALE),
        "thresholds": {
            "spectral_q95_max": SPECTRAL_Q95_MAX,
            "composition_q95_max": COMPOSITION_Q95_MAX,
            "configuration_q95_max": CONFIGURATION_Q95_MAX,
        },
        "consecutive_passing_scales": int(CONSECUTIVE_PASSING_SCALES),
        "primary_success_criterion": "positive monotonic relation between imposed Lc and estimated REU; no universal law assumed",
        "anisotropy_truth": {
            "lc_x_px": ANISO_LC_X_PX,
            "lc_y_px": ANISO_LC_Y_PX,
            "major_direction_deg": ANISO_TRUE_MAJOR_DIRECTION_DEG,
        },
    }
    save_json(OUTPUT_ROOT / "synthetic_validation_config.json", config_record)

    wl, endmembers = make_endmembers()
    em_df = pd.DataFrame({"wavelength_nm": wl})
    for i in range(endmembers.shape[0]):
        em_df[f"endmember_{i+1}"] = endmembers[i]
    em_df.to_csv(OUTPUT_ROOT / "synthetic_endmembers.csv", index=False)

    print("=" * 100)
    print("NC-REU Synthetic Ground-Truth Validation")
    print(f"Core: {getattr(core, 'CORE_VERSION', 'unknown')}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Known isotropic Lc values: {list(TRUE_LC_VALUES_PX)} px")
    print(f"Realizations: {len(REALIZATION_SEEDS)}")
    print("Primary test: larger imposed spatial scale -> larger estimated REU")
    print("No threshold tuning and no universal scaling-law fit.")
    print("=" * 100)

    all_results: List[Dict[str, Any]] = []
    total = len(TRUE_LC_VALUES_PX) * len(REALIZATION_SEEDS)
    k = 0
    for lc in TRUE_LC_VALUES_PX:
        for seed in REALIZATION_SEEDS:
            k += 1
            print("-" * 100)
            print(f"[{k}/{total}] isotropic ground truth: Lc={lc} px, seed={seed}")
            result = run_isotropic_realization(int(lc), int(seed), OUTPUT_ROOT)
            all_results.append(result)
            print(
                f"  status={result['status']}; REU={result['reu_px']}; "
                f"onset={result['convergence_onset_px']}; runtime={result['runtime_seconds']:.1f}s"
            )

    real_df, group_df = aggregate_isotropic_results(OUTPUT_ROOT, all_results)
    all_scale_df = collect_all_scale_metrics(OUTPUT_ROOT)
    sensitivity_df = threshold_sensitivity(OUTPUT_ROOT)
    monotonic = monotonicity_test(group_df)

    print("\nIsotropic group summary")
    print(group_df.to_string(index=False))
    if monotonic.get("status") == "evaluated":
        print(
            f"\nMonotonicity: Spearman rho={monotonic['spearman_rho']:.4f}, "
            f"p={monotonic['spearman_p_value']:.4g}, "
            f"nondecreasing={monotonic['nondecreasing']}"
        )

    anisotropy_result: Optional[Dict[str, Any]] = None
    if RUN_ANISOTROPY_VALIDATION:
        print("\n" + "=" * 100)
        print(
            f"Anisotropy truth: Lc_x={ANISO_LC_X_PX} px, Lc_y={ANISO_LC_Y_PX} px, "
            f"major direction={ANISO_TRUE_MAJOR_DIRECTION_DEG:.0f} deg"
        )
        anisotropy_result = run_anisotropy_validation(OUTPUT_ROOT)
        print(
            "Recovered principal direction: "
            f"{anisotropy_result.get('estimated_raw_integral_principal_direction_deg')} deg; "
            f"error={anisotropy_result.get('raw_integral_direction_error_deg')} deg"
        )

    plot_input_fields(OUTPUT_ROOT)
    plot_reu_recovery(real_df, group_df, OUTPUT_ROOT)
    plot_normalized_convergence(all_scale_df, OUTPUT_ROOT)
    if RUN_ANISOTROPY_VALIDATION:
        plot_anisotropy(OUTPUT_ROOT)

    final_summary = {
        "isotropic_monotonicity": monotonic,
        "anisotropy_validation": anisotropy_result,
        "recommended_interpretation": [
            "The primary causal result is whether estimated REU increases with the imposed intrinsic spatial scale.",
            "The REU/Lc ratio is descriptive and must not be presented as a universal constant from this experiment alone.",
            "Threshold sensitivity is supplementary robustness evidence, not a basis for selecting thresholds.",
            "Anisotropy recovery is an independent structural check and is not used in the isotropic REU decision.",
        ],
    }
    save_json(OUTPUT_ROOT / "synthetic_validation_summary.json", final_summary)

    print("\n" + "=" * 100)
    print("FINISHED")
    print(f"Primary results: {OUTPUT_ROOT / 'isotropic_group_summary.csv'}")
    print(f"Monotonicity:    {OUTPUT_ROOT / 'monotonicity_test.json'}")
    print(f"Sensitivity:     {OUTPUT_ROOT / 'threshold_sensitivity.csv'}")
    if RUN_ANISOTROPY_VALIDATION:
        print(f"Anisotropy:      {OUTPUT_ROOT / 'anisotropy_validation.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
