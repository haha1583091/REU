# -*- coding: utf-8 -*-
"""
Resolution-Perturbation Validation for NC-REU
=============================================

Scientific question
-------------------
Does the representative elementary unit (REU) behave as a property of the
underlying hyperspectral spatial field rather than as a fixed number of pixels?

This script performs a controlled spatial-resolution perturbation experiment on
synthetic hyperspectral fields.  The SAME high-resolution abundance field is
observed at three pixel sizes (1x, 2x and 4x) by block-averaging the noiseless
spectral field.  REU is then re-estimated independently at each resolution.

The key test is deliberately simple:

    pixel REU should change with pixel size,
    while physical REU = pixel REU * pixel size should remain comparatively stable.

Important design choices
------------------------
1. Only spatial resolution is perturbed.  Spectral bands/endmembers, abundance
   field, REU thresholds and the physical field extent are held fixed.
2. The resolution-validation fields are generated at 640 x 640 base pixels.
   This reduces the finite-domain issue that can occur for the largest Lc.
3. Sensor noise is omitted in this particular controlled experiment so that
   spatial resolution is the only observation variable.  Noise robustness has
   already been exercised in the preceding synthetic ground-truth validation.
4. Spatial configuration is compared at the SAME PHYSICAL lags at all
   resolutions.  The physical lags [4, 8, 16] base-pixel units become native
   pixel lags [4,8,16], [2,4,8] and [1,2,4] at 1x, 2x and 4x resolution.
5. Candidate supports are also defined in COMMON PHYSICAL units and converted
   exactly to native pixels.  All physical supports are multiples of 4, so no
   rounding is needed for factors 1, 2 and 4.
6. The three primary REU thresholds remain fixed:
       spectral-distribution q95 <= 0.10
       composition JS q95       <= 0.10
       configuration JS q95     <= 0.15
   and three consecutive supported scales are required.
7. No equivalence threshold is imposed and no post-hoc tuning is performed.
   Stability is reported descriptively by ratios, absolute relative deviations
   and within-realization coefficients of variation.

Outputs
-------
<OUTPUT_ROOT>/
    resolution_validation_config.json
    resolution_realization_results.csv
    resolution_group_summary.csv
    physical_reu_stability_by_realization.csv
    physical_reu_stability_summary.csv
    all_resolution_scale_metrics.csv
    Fig_1_resolution_perturbation_example.jpg
    Fig_2_pixel_REU_vs_resolution.jpg
    Fig_3_physical_REU_vs_resolution.jpg
    Fig_4_physical_REU_ratio_to_native.jpg
    Fig_5_physical_scale_convergence_alignment.jpg
    lc_XXX_seed_YYYY/factor_Xx/
        scale_metrics.csv
        result.json

Required files in the same folder
---------------------------------
    reu_nc_synthetic_ground_truth_validation.py
    reu_nc_core_v2_7.py
    reu_nc_geometry_v2_7.py

Example
-------
D:\\Anaconda\\envs\\DL\\python.exe \
D:\\LS_REU\\260811_synthetic\\reu_nc_resolution_perturbation_validation.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import the exact synthetic generator and V2.7 metric engine used previously.
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import reu_nc_synthetic_ground_truth_validation as synth
    import reu_nc_core_v2_7 as core
except Exception as exc:
    raise RuntimeError(
        "Put reu_nc_synthetic_ground_truth_validation.py, reu_nc_core_v2_7.py "
        "and reu_nc_geometry_v2_7.py in the same folder as this script."
    ) from exc


# =============================================================================
# 0. FIXED EXPERIMENT SETTINGS
# =============================================================================

OUTPUT_ROOT = Path(r"D:\LS_REU\results\nc_reu_resolution_perturbation_validation")

# A larger field than the previous 320x320 synthetic experiment is used here so
# the Lc=32 case is not close to the domain boundary.
HIGHRES_FIELD_SIZE = 640
N_BANDS = 64

# Use three intrinsic scales spanning fine to coarse structure while remaining
# well resolved at the coarsest 4x pixel size.
TRUE_LC_VALUES_BASE_PX: Tuple[int, ...] = (8, 16, 32)
REALIZATION_SEEDS: Tuple[int, ...] = (3407, 2027, 5909)

# Pixel size relative to the finest synthetic grid.
RESOLUTION_FACTORS: Tuple[int, ...] = (1, 2, 4)
BASE_PIXEL_SIZE_PHYSICAL = 1.0
PHYSICAL_UNIT = "base-pixel unit"

# Spatial block averaging is the ONLY perturbation.  Sensor noise is disabled
# here to isolate resolution.  The previous synthetic validation already used
# SENSOR_NOISE_STD=0.003.
ADD_SENSOR_NOISE = False
SENSOR_NOISE_STD = 0.003

# Common physical candidate supports.  All are divisible by 4 so they map
# exactly to integer native-pixel windows at 1x/2x/4x.
PHYSICAL_SUPPORTS: Tuple[int, ...] = (
    24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
    72, 80, 88, 96, 104, 112, 128, 144, 160,
    176, 192, 208, 224, 240, 256, 288, 320,
)

# The spatial-configuration metric uses the same PHYSICAL lags at every
# resolution.  Native lags are PHYSICAL_CONFIGURATION_LAGS / factor.
PHYSICAL_CONFIGURATION_LAGS: Tuple[int, ...] = (4, 8, 16)

# Position sampling is also matched approximately in physical units.
MAX_POSITION_STRIDE_PHYSICAL = 8

MIN_WINDOWS_PER_SCALE = 12
MAX_WINDOWS_PER_SCALE = 60
MAX_SPECTRA_PER_WINDOW = 700

SPECTRAL_Q95_MAX = 0.10
COMPOSITION_Q95_MAX = 0.10
CONFIGURATION_Q95_MAX = 0.15
CONSECUTIVE_PASSING_SCALES = 3

SAVE_OBSERVED_CUBES = False
OVERWRITE = True


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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600


def block_average_cube(cube: np.ndarray, factor: int) -> np.ndarray:
    """Area-average a hyperspectral cube by an integer spatial factor."""
    cube = np.asarray(cube, dtype=np.float32)
    if factor == 1:
        return cube.copy()
    h, w, b = cube.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Cube shape {(h, w)} is not divisible by factor={factor}.")
    out = cube.reshape(h // factor, factor, w // factor, factor, b).mean(axis=(1, 3))
    return out.astype(np.float32)


def native_window_sizes_for_factor(factor: int) -> Tuple[int, ...]:
    sizes = []
    for physical_size in PHYSICAL_SUPPORTS:
        if physical_size % factor != 0:
            raise ValueError("All PHYSICAL_SUPPORTS must divide exactly by resolution factor.")
        size = physical_size // factor
        sizes.append(int(size))
    return tuple(sizes)


def native_configuration_lags_for_factor(factor: int) -> Tuple[int, ...]:
    lags = []
    for physical_lag in PHYSICAL_CONFIGURATION_LAGS:
        if physical_lag % factor != 0:
            raise ValueError("Physical configuration lags must divide exactly by factor.")
        lag = physical_lag // factor
        if lag < 1:
            raise ValueError("A native configuration lag became <1 pixel.")
        lags.append(int(lag))
    return tuple(lags)


def make_clean_highres_field(lc_px: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate the exact previous abundance field, then reconstruct noiseless HSI.

    The previous generator returns abundance and deterministic endmembers.  We
    deliberately reconstruct abundance @ endmembers here rather than using its
    noisy cube, so spatial resolution is the only perturbed observation factor.
    """
    generated = synth.generate_isotropic_cube(
        lc_px=float(lc_px),
        seed=int(seed),
        field_size=int(HIGHRES_FIELD_SIZE),
        n_bands=int(N_BANDS),
    )
    abundance = np.asarray(generated["abundance"], dtype=np.float32)
    endmembers = np.asarray(generated["endmembers"], dtype=np.float32)
    clean_cube = (abundance @ endmembers).astype(np.float32)
    return {
        "cube": clean_cube,
        "abundance": abundance,
        "dominant_map": np.argmax(abundance, axis=-1).astype(np.int16),
        "wavelength_nm": np.asarray(generated["wavelength_nm"], dtype=np.float64),
        "endmembers": endmembers,
    }


def make_observed_cube(clean_highres_cube: np.ndarray, factor: int, lc_px: int, seed: int) -> np.ndarray:
    observed = block_average_cube(clean_highres_cube, int(factor))
    if ADD_SENSOR_NOISE:
        rng = np.random.default_rng(stable_seed("resolution_noise", lc_px, seed, factor))
        observed = observed + rng.normal(0.0, SENSOR_NOISE_STD, size=observed.shape)
    return np.clip(observed, 0.02, 0.90).astype(np.float32)


# =============================================================================
# 2. RESOLUTION-COMPARABLE V2.7 CONFIGURATION
# =============================================================================

def make_resolution_config(factor: int) -> Any:
    """Create V2.7 config with physically matched configuration lags."""
    config = synth.make_v27_config(correlation_bootstraps=0)

    native_sizes = native_window_sizes_for_factor(factor)
    native_lags = native_configuration_lags_for_factor(factor)

    config.windows.window_sizes_px = list(native_sizes)
    config.windows.min_windows_per_scale = int(MIN_WINDOWS_PER_SCALE)
    config.windows.max_windows_per_scale = int(MAX_WINDOWS_PER_SCALE)
    config.windows.max_spectra_per_window = int(MAX_SPECTRA_PER_WINDOW)
    config.windows.max_position_stride_px = max(1, int(MAX_POSITION_STRIDE_PHYSICAL // factor))

    config.metrics.cooccurrence_distances_px = list(native_lags)

    config.convergence.spectral_error_q95_max = float(SPECTRAL_Q95_MAX)
    config.convergence.composition_js_q95_max = float(COMPOSITION_Q95_MAX)
    config.convergence.configuration_js_q95_max = float(CONFIGURATION_Q95_MAX)
    config.convergence.consecutive_scales = int(CONSECUTIVE_PASSING_SCALES)
    return config


# =============================================================================
# 3. PARAMETERIZED SQUARE-SUPPORT EVALUATION
# =============================================================================

def sample_square_positions(
    h: int,
    w: int,
    size: int,
    max_windows: int,
    max_stride_px: int,
    seed: int,
) -> List[Tuple[int, int]]:
    if size > h or size > w:
        return []

    max_y = h - size
    max_x = w - size
    stride = max(1, min(int(max_stride_px), max(1, size // 4)))

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


def evaluate_resolution_scales(
    cube: np.ndarray,
    model: Any,
    config: Any,
    factor: int,
    lc_px: int,
    realization_seed: int,
) -> pd.DataFrame:
    cube = np.asarray(cube, dtype=np.float32)
    h, w, _ = cube.shape
    native_sizes = native_window_sizes_for_factor(factor)
    rows: List[Dict[str, Any]] = []

    for physical_size, size in zip(PHYSICAL_SUPPORTS, native_sizes):
        if size > min(h, w):
            continue

        # Require room for the largest configured physical-lag co-occurrence.
        max_lag = max(int(v) for v in config.metrics.cooccurrence_distances_px)
        if size <= max_lag:
            continue

        positions = sample_square_positions(
            h=h,
            w=w,
            size=int(size),
            max_windows=int(MAX_WINDOWS_PER_SCALE),
            max_stride_px=max(1, int(MAX_POSITION_STRIDE_PHYSICAL // factor)),
            seed=stable_seed("positions", lc_px, realization_seed, factor, physical_size),
        )
        if not positions:
            continue

        spectral_errors: List[float] = []
        composition_errors: List[float] = []
        configuration_errors: List[float] = []

        for window_id, (y, x) in enumerate(positions):
            local_latent = model.latent_cube[y:y + size, x:x + size].reshape(
                -1, model.latent_cube.shape[-1]
            )
            soft_patch = model.soft_state_cube[y:y + size, x:x + size]
            local_mask = np.ones((size, size), dtype=bool)

            if len(local_latent) > MAX_SPECTRA_PER_WINDOW:
                rng = np.random.default_rng(
                    stable_seed("spectra", lc_px, realization_seed, factor, physical_size, window_id)
                )
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

        def q(values: Sequence[float], p: float) -> float:
            return float(np.quantile(np.asarray(values, dtype=float), p))

        rows.append({
            "true_lc_base_px": int(lc_px),
            "seed": int(realization_seed),
            "resolution_factor": int(factor),
            "pixel_size_physical": float(BASE_PIXEL_SIZE_PHYSICAL * factor),
            "base_size_px": int(size),
            "physical_support": float(physical_size * BASE_PIXEL_SIZE_PHYSICAL),
            "valid_window_count": int(len(spectral_errors)),
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

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("physical_support").reset_index(drop=True)


def add_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["enough_windows"] = out["valid_window_count"] >= MIN_WINDOWS_PER_SCALE
    out["pass_spectral"] = out["spectral_error_q95"] <= SPECTRAL_Q95_MAX
    out["pass_composition"] = out["composition_js_q95"] <= COMPOSITION_Q95_MAX
    out["pass_configuration"] = out["configuration_js_q95"] <= CONFIGURATION_Q95_MAX
    out["three_constraints_pass"] = (
        out["enough_windows"]
        & out["pass_spectral"]
        & out["pass_composition"]
        & out["pass_configuration"]
    )
    out["spectral_ratio"] = out["spectral_error_q95"] / SPECTRAL_Q95_MAX
    out["composition_ratio"] = out["composition_js_q95"] / COMPOSITION_Q95_MAX
    out["configuration_ratio"] = out["configuration_js_q95"] / CONFIGURATION_Q95_MAX
    out["minimax_ratio"] = out[["spectral_ratio", "composition_ratio", "configuration_ratio"]].max(axis=1)
    return out


def decide_reu(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "status": "no_evaluable_scale",
            "reu_px": None,
            "reu_physical": None,
            "convergence_onset_px": None,
            "convergence_onset_physical": None,
            "largest_supported_scale_px": None,
            "largest_supported_physical": None,
        }

    d = add_decision_columns(df).sort_values("physical_support").reset_index(drop=True)
    supported = d[d["enough_windows"]].reset_index(drop=True)
    if supported.empty:
        return {
            "status": "no_evaluable_scale",
            "reu_px": None,
            "reu_physical": None,
            "convergence_onset_px": None,
            "convergence_onset_physical": None,
            "largest_supported_scale_px": None,
            "largest_supported_physical": None,
        }

    passing = supported[supported["three_constraints_pass"]]
    onset_px = None if passing.empty else int(passing.iloc[0]["base_size_px"])
    onset_phys = None if passing.empty else float(passing.iloc[0]["physical_support"])

    reu_px = None
    reu_phys = None
    flags = supported["three_constraints_pass"].to_numpy(dtype=bool)
    if len(flags) >= CONSECUTIVE_PASSING_SCALES:
        for i in range(len(flags) - CONSECUTIVE_PASSING_SCALES + 1):
            if bool(np.all(flags[i:i + CONSECUTIVE_PASSING_SCALES])):
                reu_px = int(supported.iloc[i]["base_size_px"])
                reu_phys = float(supported.iloc[i]["physical_support"])
                break

    largest_px = int(supported.iloc[-1]["base_size_px"])
    largest_phys = float(supported.iloc[-1]["physical_support"])

    if reu_px is not None:
        status = "reu_found"
    elif onset_px is not None and onset_phys == largest_phys:
        status = "right_censored_after_convergence_onset"
    elif onset_px is not None:
        status = "convergence_seen_but_not_consecutively_confirmed"
    else:
        status = "no_reu_within_supported_range"

    return {
        "status": status,
        "reu_px": reu_px,
        "reu_physical": reu_phys,
        "convergence_onset_px": onset_px,
        "convergence_onset_physical": onset_phys,
        "largest_supported_scale_px": largest_px,
        "largest_supported_physical": largest_phys,
    }


# =============================================================================
# 4. ONE FIELD AT ONE RESOLUTION
# =============================================================================

def run_one_resolution(
    clean_highres_cube: np.ndarray,
    lc_px: int,
    seed: int,
    factor: int,
    output_root: Path,
) -> Dict[str, Any]:
    case_dir = output_root / f"lc_{lc_px:03d}_seed_{seed}" / f"factor_{factor}x"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"

    if result_path.exists() and not OVERWRITE:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    cube = make_observed_cube(clean_highres_cube, factor, lc_px, seed)
    mask = np.ones(cube.shape[:2], dtype=bool)
    config = make_resolution_config(factor)

    # Correlation is not one of the three acceptance metrics and is disabled to
    # keep this test focused on spatial-resolution perturbation.
    with synth.correlation_disabled_for_isotropic_reu():
        model = core.fit_spectral_state_model(cube, mask, config)

    scale_df = evaluate_resolution_scales(
        cube=cube,
        model=model,
        config=config,
        factor=factor,
        lc_px=lc_px,
        realization_seed=seed,
    )
    scale_df = add_decision_columns(scale_df)
    decision = decide_reu(scale_df)
    scale_df.to_csv(case_dir / "scale_metrics.csv", index=False)

    if SAVE_OBSERVED_CUBES:
        np.save(case_dir / "observed_cube.npy", cube.astype(np.float32))

    result = {
        "true_lc_base_px": int(lc_px),
        "seed": int(seed),
        "resolution_factor": int(factor),
        "pixel_size_physical": float(BASE_PIXEL_SIZE_PHYSICAL * factor),
        "observed_height_px": int(cube.shape[0]),
        "observed_width_px": int(cube.shape[1]),
        "configuration_lags_native_px": list(native_configuration_lags_for_factor(factor)),
        "configuration_lags_physical": list(PHYSICAL_CONFIGURATION_LAGS),
        "status": decision["status"],
        "reu_px": decision["reu_px"],
        "reu_physical": decision["reu_physical"],
        "convergence_onset_px": decision["convergence_onset_px"],
        "convergence_onset_physical": decision["convergence_onset_physical"],
        "largest_supported_scale_px": decision["largest_supported_scale_px"],
        "largest_supported_physical": decision["largest_supported_physical"],
        "runtime_seconds": float(time.time() - t0),
    }
    save_json(result_path, result)
    return result


# =============================================================================
# 5. AGGREGATION AND DESCRIPTIVE STABILITY METRICS
# =============================================================================

def aggregate_results(output_root: Path, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(results).sort_values(
        ["true_lc_base_px", "seed", "resolution_factor"]
    ).reset_index(drop=True)
    df.to_csv(output_root / "resolution_realization_results.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for (lc, factor), g in df.groupby(["true_lc_base_px", "resolution_factor"], sort=True):
        reu_px = pd.to_numeric(g["reu_px"], errors="coerce").dropna().to_numpy(dtype=float)
        reu_phys = pd.to_numeric(g["reu_physical"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append({
            "true_lc_base_px": int(lc),
            "resolution_factor": int(factor),
            "pixel_size_physical": float(BASE_PIXEL_SIZE_PHYSICAL * factor),
            "n_realizations": int(len(g)),
            "n_reu_found": int(len(reu_phys)),
            "reu_found_fraction": float(len(reu_phys) / max(len(g), 1)),
            "reu_px_median": float(np.median(reu_px)) if len(reu_px) else np.nan,
            "reu_px_min": float(np.min(reu_px)) if len(reu_px) else np.nan,
            "reu_px_max": float(np.max(reu_px)) if len(reu_px) else np.nan,
            "reu_physical_median": float(np.median(reu_phys)) if len(reu_phys) else np.nan,
            "reu_physical_min": float(np.min(reu_phys)) if len(reu_phys) else np.nan,
            "reu_physical_max": float(np.max(reu_phys)) if len(reu_phys) else np.nan,
        })
    group_df = pd.DataFrame(rows)
    group_df.to_csv(output_root / "resolution_group_summary.csv", index=False)
    return df, group_df


def build_stability_tables(output_root: Path, realization_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for (lc, seed), g in realization_df.groupby(["true_lc_base_px", "seed"], sort=True):
        g = g.sort_values("resolution_factor")
        valid = g.dropna(subset=["reu_physical"]).copy()
        row: Dict[str, Any] = {
            "true_lc_base_px": int(lc),
            "seed": int(seed),
            "n_resolutions_with_reu": int(len(valid)),
        }

        by_factor = {int(r["resolution_factor"]): r for _, r in g.iterrows()}
        base_phys = None
        if 1 in by_factor and pd.notna(by_factor[1]["reu_physical"]):
            base_phys = float(by_factor[1]["reu_physical"])
        row["reu_physical_1x"] = base_phys

        for factor in RESOLUTION_FACTORS:
            r = by_factor.get(int(factor))
            phys = None if r is None or pd.isna(r["reu_physical"]) else float(r["reu_physical"])
            px = None if r is None or pd.isna(r["reu_px"]) else float(r["reu_px"])
            row[f"reu_px_{factor}x"] = px
            row[f"reu_physical_{factor}x"] = phys
            if base_phys is not None and phys is not None and base_phys > 0:
                row[f"physical_ratio_to_1x_{factor}x"] = float(phys / base_phys)
                row[f"physical_abs_relative_deviation_{factor}x"] = float(abs(phys - base_phys) / base_phys)
            else:
                row[f"physical_ratio_to_1x_{factor}x"] = np.nan
                row[f"physical_abs_relative_deviation_{factor}x"] = np.nan

        phys_values = pd.to_numeric(valid["reu_physical"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(phys_values) >= 2 and np.mean(phys_values) > 0:
            row["physical_reu_mean"] = float(np.mean(phys_values))
            row["physical_reu_sd"] = float(np.std(phys_values, ddof=1))
            row["physical_reu_cv"] = float(np.std(phys_values, ddof=1) / np.mean(phys_values))
            row["physical_reu_range"] = float(np.max(phys_values) - np.min(phys_values))
        else:
            row["physical_reu_mean"] = np.nan
            row["physical_reu_sd"] = np.nan
            row["physical_reu_cv"] = np.nan
            row["physical_reu_range"] = np.nan
        rows.append(row)

    stability_df = pd.DataFrame(rows).sort_values(["true_lc_base_px", "seed"]).reset_index(drop=True)
    stability_df.to_csv(output_root / "physical_reu_stability_by_realization.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    for lc, g in stability_df.groupby("true_lc_base_px", sort=True):
        cvs = pd.to_numeric(g["physical_reu_cv"], errors="coerce").dropna().to_numpy(dtype=float)
        row: Dict[str, Any] = {
            "true_lc_base_px": int(lc),
            "n_realizations": int(len(g)),
            "n_complete_1x_2x_4x": int((g["n_resolutions_with_reu"] == len(RESOLUTION_FACTORS)).sum()),
            "physical_reu_cv_median": float(np.median(cvs)) if len(cvs) else np.nan,
            "physical_reu_cv_q25": float(np.quantile(cvs, 0.25)) if len(cvs) else np.nan,
            "physical_reu_cv_q75": float(np.quantile(cvs, 0.75)) if len(cvs) else np.nan,
        }
        for factor in (2, 4):
            ratios = pd.to_numeric(g[f"physical_ratio_to_1x_{factor}x"], errors="coerce").dropna().to_numpy(dtype=float)
            dev = pd.to_numeric(g[f"physical_abs_relative_deviation_{factor}x"], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"physical_ratio_to_1x_{factor}x_median"] = float(np.median(ratios)) if len(ratios) else np.nan
            row[f"physical_ratio_to_1x_{factor}x_q25"] = float(np.quantile(ratios, 0.25)) if len(ratios) else np.nan
            row[f"physical_ratio_to_1x_{factor}x_q75"] = float(np.quantile(ratios, 0.75)) if len(ratios) else np.nan
            row[f"physical_abs_relative_deviation_{factor}x_median"] = float(np.median(dev)) if len(dev) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "physical_reu_stability_summary.csv", index=False)
    return stability_df, summary_df


def collect_scale_metrics(output_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for lc in TRUE_LC_VALUES_BASE_PX:
        for seed in REALIZATION_SEEDS:
            for factor in RESOLUTION_FACTORS:
                p = output_root / f"lc_{lc:03d}_seed_{seed}" / f"factor_{factor}x" / "scale_metrics.csv"
                if p.exists():
                    frames.append(pd.read_csv(p))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(output_root / "all_resolution_scale_metrics.csv", index=False)
    return out


# =============================================================================
# 6. FIGURES
# =============================================================================

def plot_resolution_example(output_root: Path) -> None:
    setup_plot_fonts()
    lc = 16
    seed = REALIZATION_SEEDS[0]
    clean = make_clean_highres_field(lc, seed)["cube"]
    band_idx = int(clean.shape[-1] // 2)

    # A single figure with three panels is used only for the conceptual input
    # illustration; quantitative results are plotted in separate figures below.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, factor in zip(axes, RESOLUTION_FACTORS):
        obs = make_observed_cube(clean, factor, lc, seed)
        ax.imshow(
            obs[:, :, band_idx],
            origin="upper",
            extent=(0, HIGHRES_FIELD_SIZE, HIGHRES_FIELD_SIZE, 0),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"{factor}x pixel size ({obs.shape[0]}×{obs.shape[1]})")
        ax.set_xlabel("Physical x (base-pixel units)")
        ax.set_ylabel("Physical y")
        ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_root / "Fig_1_resolution_perturbation_example.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_pixel_reu(group_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for lc, g in group_df.groupby("true_lc_base_px", sort=True):
        g = g.sort_values("resolution_factor")
        ax.plot(
            g["resolution_factor"],
            g["reu_px_median"],
            marker="o",
            linewidth=1.4,
            label=f"$L_c$={int(lc)}",
        )
        ax.fill_between(
            g["resolution_factor"].to_numpy(dtype=float),
            g["reu_px_min"].to_numpy(dtype=float),
            g["reu_px_max"].to_numpy(dtype=float),
            alpha=0.15,
        )
    ax.set_xlabel("Relative pixel size")
    ax.set_ylabel("REU in native pixels")
    ax.set_xticks(list(RESOLUTION_FACTORS))
    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_root / "Fig_2_pixel_REU_vs_resolution.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_physical_reu(group_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for lc, g in group_df.groupby("true_lc_base_px", sort=True):
        g = g.sort_values("resolution_factor")
        ax.plot(
            g["resolution_factor"],
            g["reu_physical_median"],
            marker="o",
            linewidth=1.4,
            label=f"$L_c$={int(lc)}",
        )
        ax.fill_between(
            g["resolution_factor"].to_numpy(dtype=float),
            g["reu_physical_min"].to_numpy(dtype=float),
            g["reu_physical_max"].to_numpy(dtype=float),
            alpha=0.15,
        )
    ax.set_xlabel("Relative pixel size")
    ax.set_ylabel(f"Physical REU ({PHYSICAL_UNIT})")
    ax.set_xticks(list(RESOLUTION_FACTORS))
    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_root / "Fig_3_physical_REU_vs_resolution.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_physical_ratio(stability_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for lc, g in stability_df.groupby("true_lc_base_px", sort=True):
        xs: List[int] = []
        med: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        for factor in RESOLUTION_FACTORS:
            col = f"physical_ratio_to_1x_{factor}x"
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
            if factor == 1 and len(vals) == 0:
                vals = np.ones(len(g), dtype=float)
            if len(vals):
                xs.append(int(factor))
                med.append(float(np.median(vals)))
                q25.append(float(np.quantile(vals, 0.25)))
                q75.append(float(np.quantile(vals, 0.75)))
        ax.plot(xs, med, marker="o", linewidth=1.4, label=f"$L_c$={int(lc)}")
        if xs:
            ax.fill_between(np.asarray(xs), np.asarray(q25), np.asarray(q75), alpha=0.15)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Relative pixel size")
    ax.set_ylabel("Physical REU / native-resolution physical REU")
    ax.set_xticks(list(RESOLUTION_FACTORS))
    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_root / "Fig_4_physical_REU_ratio_to_native.jpg", bbox_inches="tight")
    plt.close(fig)


def plot_convergence_alignment(scale_df: pd.DataFrame, output_root: Path) -> None:
    setup_plot_fonts()
    lc = 16
    seed = REALIZATION_SEEDS[0]
    d = scale_df[(scale_df["true_lc_base_px"] == lc) & (scale_df["seed"] == seed)].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for factor, g in d.groupby("resolution_factor", sort=True):
        g = g.sort_values("physical_support")
        ax.plot(
            g["physical_support"],
            g["minimax_ratio"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=f"{int(factor)}x pixel size",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Physical support L ({PHYSICAL_UNIT})")
    ax.set_ylabel("Maximum normalized q95 error")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_root / "Fig_5_physical_scale_convergence_alignment.jpg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    config_record = {
        "scientific_question": "Does REU remain stable in physical units when spatial resolution is perturbed?",
        "highres_field_size": HIGHRES_FIELD_SIZE,
        "n_bands": N_BANDS,
        "wavelength_range_nm": [synth.WAVELENGTH_MIN_NM, synth.WAVELENGTH_MAX_NM],
        "true_lc_values_base_px": TRUE_LC_VALUES_BASE_PX,
        "realization_seeds": REALIZATION_SEEDS,
        "resolution_factors": RESOLUTION_FACTORS,
        "base_pixel_size_physical": BASE_PIXEL_SIZE_PHYSICAL,
        "physical_unit": PHYSICAL_UNIT,
        "physical_supports": PHYSICAL_SUPPORTS,
        "physical_configuration_lags": PHYSICAL_CONFIGURATION_LAGS,
        "max_position_stride_physical": MAX_POSITION_STRIDE_PHYSICAL,
        "min_windows_per_scale": MIN_WINDOWS_PER_SCALE,
        "max_windows_per_scale": MAX_WINDOWS_PER_SCALE,
        "spectral_q95_max": SPECTRAL_Q95_MAX,
        "composition_q95_max": COMPOSITION_Q95_MAX,
        "configuration_q95_max": CONFIGURATION_Q95_MAX,
        "consecutive_passing_scales": CONSECUTIVE_PASSING_SCALES,
        "sensor_noise_added": ADD_SENSOR_NOISE,
        "sensor_noise_std_if_enabled": SENSOR_NOISE_STD,
        "note": (
            "All resolution factors observe the same underlying high-resolution abundance field. "
            "Candidate supports and configuration lags are matched in physical units. "
            "No post-hoc equivalence threshold is applied."
        ),
    }
    save_json(OUTPUT_ROOT / "resolution_validation_config.json", config_record)

    print("=" * 100)
    print("NC-REU resolution-perturbation validation")
    print("Same field -> 1x / 2x / 4x pixels; physical supports and configuration lags matched")
    print(f"Field: {HIGHRES_FIELD_SIZE}x{HIGHRES_FIELD_SIZE}; Lc={TRUE_LC_VALUES_BASE_PX}")
    print(f"Physical supports: {PHYSICAL_SUPPORTS}")
    print(f"Physical configuration lags: {PHYSICAL_CONFIGURATION_LAGS}")
    print(f"Noise added: {ADD_SENSOR_NOISE}")
    print("=" * 100)

    results: List[Dict[str, Any]] = []
    total = len(TRUE_LC_VALUES_BASE_PX) * len(REALIZATION_SEEDS) * len(RESOLUTION_FACTORS)
    run_id = 0

    for lc in TRUE_LC_VALUES_BASE_PX:
        for seed in REALIZATION_SEEDS:
            field = make_clean_highres_field(int(lc), int(seed))
            clean_highres_cube = field["cube"]
            for factor in RESOLUTION_FACTORS:
                run_id += 1
                native_shape = HIGHRES_FIELD_SIZE // factor
                print("\n" + "-" * 100)
                print(
                    f"[{run_id}/{total}] Lc={lc} base px | seed={seed} | "
                    f"pixel size={factor}x | observed={native_shape}x{native_shape}"
                )
                r = run_one_resolution(
                    clean_highres_cube=clean_highres_cube,
                    lc_px=int(lc),
                    seed=int(seed),
                    factor=int(factor),
                    output_root=OUTPUT_ROOT,
                )
                print(
                    f"status={r['status']}; REU={r['reu_px']} native px; "
                    f"physical REU={r['reu_physical']}; runtime={r['runtime_seconds']:.1f}s"
                )
                results.append(r)

    realization_df, group_df = aggregate_results(OUTPUT_ROOT, results)
    stability_df, stability_summary_df = build_stability_tables(OUTPUT_ROOT, realization_df)
    scale_df = collect_scale_metrics(OUTPUT_ROOT)

    plot_resolution_example(OUTPUT_ROOT)
    plot_pixel_reu(group_df, OUTPUT_ROOT)
    plot_physical_reu(group_df, OUTPUT_ROOT)
    plot_physical_ratio(stability_df, OUTPUT_ROOT)
    plot_convergence_alignment(scale_df, OUTPUT_ROOT)

    print("\n" + "=" * 100)
    print("Resolution group summary")
    if not group_df.empty:
        print(group_df.to_string(index=False))

    print("\nPhysical-REU stability summary")
    if not stability_summary_df.empty:
        print(stability_summary_df.to_string(index=False))

    complete = stability_df[stability_df["n_resolutions_with_reu"] == len(RESOLUTION_FACTORS)]
    cvs = pd.to_numeric(complete["physical_reu_cv"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(cvs):
        print(
            f"\nAcross complete realizations: median within-realization physical-REU CV = "
            f"{np.median(cvs):.4f}"
        )
    else:
        print("\nNo realization produced REU at all three resolutions; inspect censoring before interpretation.")

    print("\nFINISHED")
    print(f"Results: {OUTPUT_ROOT}")
    print("Primary files:")
    print("  resolution_realization_results.csv")
    print("  resolution_group_summary.csv")
    print("  physical_reu_stability_by_realization.csv")
    print("  physical_reu_stability_summary.csv")
    print("  Fig_2_pixel_REU_vs_resolution.jpg")
    print("  Fig_3_physical_REU_vs_resolution.jpg")
    print("  Fig_5_physical_scale_convergence_alignment.jpg")
    print("=" * 100)


if __name__ == "__main__":
    main()
