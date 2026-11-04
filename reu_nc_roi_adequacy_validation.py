# -*- coding: utf-8 -*-
"""
Minimum Observation Support / ROI Adequacy Validation for NC-REU
=================================================================

Scientific question
-------------------
Does the estimated REU mark the transition at which a local hyperspectral
observation becomes sufficiently representative of the larger heterogeneous
field?

This script is deliberately a VALIDATION layer, not a new REU algorithm.
It does not modify REU thresholds or refit a new scale criterion.  Instead, it
uses the already estimated synthetic REU values and asks whether independent,
ground-truth-based ROI errors decrease strongly below the REU and show
substantially smaller marginal gains once the support reaches / exceeds REU.

Why this is not circular
------------------------
REU itself was defined from the V2.7 learned spectral-state metrics:
    1) spectral-distribution q95,
    2) soft-state composition JS q95,
    3) soft-state configuration JS q95.

ROI adequacy is evaluated here using INDEPENDENT quantities that were not used
as REU acceptance criteria:
    A) raw-reflectance mean-spectrum spectral angle (SAM),
    B) raw-reflectance mean-spectrum relative RMSE,
    C) true synthetic endmember-abundance composition error (total variation),
    D) true dominant-endmember fraction error (total variation),
    E) true dominant-endmember adjacency/configuration JS divergence.

The abundance and dominant-endmember fields are known ground truth from the
synthetic generator.  Therefore this experiment asks whether REU predicts when
independent estimates of spectral, compositional and spatial properties become
less location-sensitive and more field-representative.

Primary scale coordinate
------------------------
    lambda = L / L_REU

Default supports:
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00 x REU

The primary inference window is lambda <= 1.5.  lambda=2.0 is retained as a
supplementary point when the finite synthetic domain permits enough distinct
ROI positions.

Expected result
---------------
The code does NOT force a plateau.  It reports what the data show.  The desired
validation pattern is:
    - large ROI-to-field error and strong position dependence at lambda < 1,
    - a marked reduction as lambda approaches 1,
    - smaller marginal gains beyond lambda ~= 1.

This supports the interpretation of REU as a minimum observation support and
as the boundary between information-deficient and information-sufficient
spatial support.

Inputs
------
1) Previous synthetic REU result folder, by default:
       D:\\LS_REU\\results\\nc_reu_synthetic_ground_truth
   It must contain:
       synthetic_validation_config.json
       isotropic_realization_results.csv
       lc_XXX_seed_YYYY/result.json

2) The previous synthetic generator script in the same folder as this script:
       reu_nc_synthetic_ground_truth_validation.py
   together with:
       reu_nc_core_v2_7.py
       reu_nc_geometry_v2_7.py

Outputs
-------
<OUTPUT_ROOT>/
    roi_adequacy_config.json
    roi_position_metrics.csv
    roi_scale_summary.csv
    roi_realization_summary.csv
    roi_group_summary.csv
    diminishing_returns_statistics.csv
    statistical_tests.json
    Fig_1_REU_normalized_observation_support.jpg
    Fig_2_spectral_representativeness.jpg
    Fig_3_abundance_representativeness.jpg
    Fig_4_configuration_representativeness.jpg
    Fig_5_example_observation_supports.jpg

Run
---
One-click default:
    D:\\Anaconda\\envs\\DL\\python.exe D:\\LS_REU\\260811_synthetic\\reu_nc_roi_adequacy_validation.py

Optional command-line paths:
    python reu_nc_roi_adequacy_validation.py \
        --synthetic-root D:\\LS_REU\\results\\nc_reu_synthetic_ground_truth \
        --output-root D:\\LS_REU\\results\\nc_reu_roi_adequacy_validation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# =============================================================================
# 0. USER SETTINGS — freeze these for the reported experiment
# =============================================================================

DEFAULT_SYNTHETIC_ROOT = Path(r"D:\LS_REU\results\nc_reu_synthetic_ground_truth")
DEFAULT_OUTPUT_ROOT = Path(r"D:\LS_REU\results\nc_reu_roi_adequacy_validation")

# REU-normalized spatial supports.
ROI_LAMBDAS: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)

# lambda <= 1.5 is the primary observation-support analysis.
# lambda=2.0 is supplementary because the largest REUs approach the 320-px
# synthetic-domain boundary.
PRIMARY_MAX_LAMBDA = 1.50

# Spatial-position uncertainty: q95 is estimated from repeated ROI locations.
MIN_ROIS_FOR_PRIMARY_SUMMARY = 12
MAX_ROIS_PER_SUPPORT = 80
MAX_POSITION_STRIDE_PX = 8

# Independent metrics used for the main composite display.
PRIMARY_INDEPENDENT_METRICS: Tuple[str, ...] = (
    "mean_spectrum_sam_deg",
    "abundance_tv",
    "transition_js",
)

# Numerical safety.
EPS = 1.0e-12

# Reproducibility.
OVERWRITE = True


# =============================================================================
# 1. IMPORT THE EXACT SYNTHETIC GENERATOR USED IN THE PREVIOUS VALIDATION
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import reu_nc_synthetic_ground_truth_validation as synth
except Exception as exc:
    raise RuntimeError(
        "Cannot import reu_nc_synthetic_ground_truth_validation.py. "
        "Put this ROI validation script in the same folder as the previous "
        "synthetic validation script and its V2.7 core files."
    ) from exc


# =============================================================================
# 2. GENERAL UTILITIES
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
    """Deterministic FNV-like hash independent of Python randomized hash()."""
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


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def relative_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    ref = float(np.sqrt(np.mean(b ** 2)))
    return rmse / max(ref, EPS)


def safe_js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.size != q.size or p.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps = float(p.sum())
    qs = float(q.sum())
    if ps <= EPS or qs <= EPS:
        return float("nan")
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    p2 = np.clip(p, EPS, None)
    q2 = np.clip(q, EPS, None)
    m2 = np.clip(m, EPS, None)
    kl_pm = float(np.sum(p * np.log(p2 / m2)))
    kl_qm = float(np.sum(q * np.log(q2 / m2)))
    return 0.5 * (kl_pm + kl_qm)


def qvalue(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


# =============================================================================
# 3. FAST RECTANGULAR-SUPPORT STATISTICS USING INTEGRAL IMAGES
# =============================================================================

def integral_image(arr: np.ndarray) -> np.ndarray:
    """Integral image with leading zero row/column.

    Supports 2-D or HxWxC arrays.  Accumulation is float64 for numerical
    stability because many spectra are summed.
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 2:
        out = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
        out[1:, 1:] = x.cumsum(axis=0).cumsum(axis=1)
        return out
    if x.ndim == 3:
        out = np.zeros((x.shape[0] + 1, x.shape[1] + 1, x.shape[2]), dtype=np.float64)
        out[1:, 1:, :] = x.cumsum(axis=0).cumsum(axis=1)
        return out
    raise ValueError("integral_image expects a 2-D or 3-D array")


def rect_sum(ii: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    y0, x0 = int(y), int(x)
    y1, x1 = y0 + int(h), x0 + int(w)
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def rect_mean(ii: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    return rect_sum(ii, y, x, h, w) / float(max(1, int(h) * int(w)))


class FastROIReference:
    """Precomputed field reference and integral images for fast ROI queries."""

    def __init__(self, cube: np.ndarray, abundance: np.ndarray, dominant_map: np.ndarray):
        self.cube = np.asarray(cube, dtype=np.float32)
        self.abundance = np.asarray(abundance, dtype=np.float32)
        self.dominant = np.asarray(dominant_map, dtype=np.int16)
        self.h, self.w, self.bands = self.cube.shape
        self.k = int(self.abundance.shape[-1])

        if self.abundance.shape[:2] != (self.h, self.w):
            raise ValueError("abundance shape does not match cube")
        if self.dominant.shape != (self.h, self.w):
            raise ValueError("dominant_map shape does not match cube")

        # Raw-field and true-abundance references.
        self.global_mean_spectrum = self.cube.mean(axis=(0, 1), dtype=np.float64)
        self.global_mean_abundance = self.abundance.mean(axis=(0, 1), dtype=np.float64)

        class_counts = np.bincount(self.dominant.ravel(), minlength=self.k).astype(np.float64)
        self.global_dominant_fraction = class_counts / max(class_counts.sum(), 1.0)

        self.cube_ii = integral_image(self.cube)
        self.abundance_ii = integral_image(self.abundance)

        onehot = np.eye(self.k, dtype=np.float32)[self.dominant]
        self.class_ii = integral_image(onehot)
        del onehot

        # Ground-truth dominant-endmember adjacency distribution.
        # Horizontal and vertical edges are kept separately so ROI queries do
        # not accidentally include edges crossing the ROI boundary.
        pair_count = self.k * self.k
        hpair = self.dominant[:, :-1].astype(np.int32) * self.k + self.dominant[:, 1:].astype(np.int32)
        vpair = self.dominant[:-1, :].astype(np.int32) * self.k + self.dominant[1:, :].astype(np.int32)
        h_onehot = np.eye(pair_count, dtype=np.float32)[hpair]
        v_onehot = np.eye(pair_count, dtype=np.float32)[vpair]
        self.hpair_ii = integral_image(h_onehot)
        self.vpair_ii = integral_image(v_onehot)
        del h_onehot, v_onehot, hpair, vpair

        global_h = rect_sum(self.hpair_ii, 0, 0, self.h, max(self.w - 1, 1)) if self.w > 1 else np.zeros(pair_count)
        global_v = rect_sum(self.vpair_ii, 0, 0, max(self.h - 1, 1), self.w) if self.h > 1 else np.zeros(pair_count)
        global_pairs = np.asarray(global_h + global_v, dtype=np.float64)
        self.global_transition = global_pairs / max(float(global_pairs.sum()), 1.0)

    def query(self, y: int, x: int, size: int) -> Dict[str, float]:
        size = int(size)
        mean_spectrum = np.asarray(rect_mean(self.cube_ii, y, x, size, size), dtype=np.float64)
        mean_abundance = np.asarray(rect_mean(self.abundance_ii, y, x, size, size), dtype=np.float64)
        class_counts = np.asarray(rect_sum(self.class_ii, y, x, size, size), dtype=np.float64)
        dominant_fraction = class_counts / max(float(class_counts.sum()), 1.0)

        if size >= 2:
            h_counts = rect_sum(self.hpair_ii, y, x, size, size - 1)
            v_counts = rect_sum(self.vpair_ii, y, x, size - 1, size)
            pair_counts = np.asarray(h_counts + v_counts, dtype=np.float64)
            transition = pair_counts / max(float(pair_counts.sum()), 1.0)
            transition_js = safe_js(transition, self.global_transition)
        else:
            transition_js = float("nan")

        return {
            "mean_spectrum_sam_deg": spectral_angle_deg(mean_spectrum, self.global_mean_spectrum),
            "mean_spectrum_rrmse": relative_rmse(mean_spectrum, self.global_mean_spectrum),
            "abundance_tv": 0.5 * float(np.sum(np.abs(mean_abundance - self.global_mean_abundance))),
            "dominant_fraction_tv": 0.5 * float(np.sum(np.abs(dominant_fraction - self.global_dominant_fraction))),
            "transition_js": float(transition_js),
        }


# =============================================================================
# 4. DETERMINISTIC ROI POSITION SAMPLING
# =============================================================================

def sample_roi_positions(
    h: int,
    w: int,
    size: int,
    max_rois: int,
    seed: int,
) -> List[Tuple[int, int]]:
    size = int(size)
    if size <= 0 or size > h or size > w:
        return []

    max_y = h - size
    max_x = w - size
    stride = int(min(MAX_POSITION_STRIDE_PX, max(1, size // 4)))

    ys = list(range(0, max_y + 1, stride))
    xs = list(range(0, max_x + 1, stride))
    if ys[-1] != max_y:
        ys.append(max_y)
    if xs[-1] != max_x:
        xs.append(max_x)

    positions = [(int(y), int(x)) for y in ys for x in xs]
    if len(positions) <= int(max_rois):
        return positions

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(positions), size=int(max_rois), replace=False))
    return [positions[int(i)] for i in idx]


# =============================================================================
# 5. LOAD PREVIOUS SYNTHETIC REU RESULTS
# =============================================================================

def load_synthetic_design(synthetic_root: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    cfg_path = synthetic_root / "synthetic_validation_config.json"
    results_path = synthetic_root / "isotropic_realization_results.csv"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing: {results_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    results = pd.read_csv(results_path)

    required = {"true_lc_px", "seed", "reu_px", "status"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"isotropic_realization_results.csv missing columns: {sorted(missing)}")

    results = results[results["status"].astype(str) == "reu_found"].copy()
    results = results[np.isfinite(pd.to_numeric(results["reu_px"], errors="coerce"))].copy()
    results["true_lc_px"] = pd.to_numeric(results["true_lc_px"]).astype(int)
    results["seed"] = pd.to_numeric(results["seed"]).astype(int)
    results["reu_px"] = pd.to_numeric(results["reu_px"]).astype(int)
    results = results.sort_values(["true_lc_px", "seed"]).reset_index(drop=True)

    if results.empty:
        raise RuntimeError("No confirmed synthetic REU realization is available for ROI adequacy validation.")
    return config, results


# =============================================================================
# 6. ONE SYNTHETIC REALIZATION: ROI ADEQUACY
# =============================================================================

def summarize_metric(values: Sequence[float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_q05": qvalue(values, 0.05),
        f"{prefix}_q25": qvalue(values, 0.25),
        f"{prefix}_q50": qvalue(values, 0.50),
        f"{prefix}_q75": qvalue(values, 0.75),
        f"{prefix}_q95": qvalue(values, 0.95),
    }


def evaluate_realization(
    lc_px: int,
    seed: int,
    reu_px: int,
    field_size: int,
    n_bands: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    generated = synth.generate_isotropic_cube(
        lc_px=float(lc_px),
        seed=int(seed),
        field_size=int(field_size),
        n_bands=int(n_bands),
    )
    ref = FastROIReference(
        generated["cube"],
        generated["abundance"],
        generated["dominant_map"],
    )

    position_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []

    for lam in ROI_LAMBDAS:
        side = int(max(2, round(float(lam) * float(reu_px))))
        actual_lambda = float(side) / float(reu_px)
        positions = sample_roi_positions(
            h=field_size,
            w=field_size,
            size=side,
            max_rois=MAX_ROIS_PER_SUPPORT,
            seed=stable_seed("roi", lc_px, seed, reu_px, lam, side),
        )

        if not positions:
            scale_rows.append({
                "true_lc_px": int(lc_px),
                "seed": int(seed),
                "reu_px": int(reu_px),
                "requested_lambda": float(lam),
                "roi_side_px": int(side),
                "actual_lambda": float(actual_lambda),
                "roi_count": 0,
                "enough_positions": False,
                "domain_censored": True,
            })
            continue

        per_metric: Dict[str, List[float]] = {
            "mean_spectrum_sam_deg": [],
            "mean_spectrum_rrmse": [],
            "abundance_tv": [],
            "dominant_fraction_tv": [],
            "transition_js": [],
        }

        for roi_id, (y, x) in enumerate(positions):
            metrics = ref.query(y=y, x=x, size=side)
            row = {
                "true_lc_px": int(lc_px),
                "seed": int(seed),
                "reu_px": int(reu_px),
                "requested_lambda": float(lam),
                "roi_side_px": int(side),
                "actual_lambda": float(actual_lambda),
                "roi_id": int(roi_id),
                "y0": int(y),
                "x0": int(x),
            }
            row.update(metrics)
            position_rows.append(row)
            for key, value in metrics.items():
                if np.isfinite(value):
                    per_metric[key].append(float(value))

        srow: Dict[str, Any] = {
            "true_lc_px": int(lc_px),
            "seed": int(seed),
            "reu_px": int(reu_px),
            "requested_lambda": float(lam),
            "roi_side_px": int(side),
            "actual_lambda": float(actual_lambda),
            "roi_count": int(len(positions)),
            "enough_positions": bool(len(positions) >= MIN_ROIS_FOR_PRIMARY_SUMMARY),
            "domain_censored": bool(side > field_size),
        }
        for key, vals in per_metric.items():
            srow.update(summarize_metric(vals, key))
            q05 = srow.get(f"{key}_q05", np.nan)
            q95 = srow.get(f"{key}_q95", np.nan)
            srow[f"{key}_q95_minus_q05"] = (
                float(q95 - q05) if np.isfinite(q05) and np.isfinite(q95) else np.nan
            )
        scale_rows.append(srow)

    scale_df = pd.DataFrame(scale_rows).sort_values("requested_lambda").reset_index(drop=True)
    position_df = pd.DataFrame(position_rows)

    # Independent error normalization is for visualization only.  It is NOT a
    # new adequacy threshold.  The main composite is normalized to the 0.25-REU
    # baseline so the location of lambda=1 is not mathematically forced to 1.
    baseline = scale_df[np.isclose(scale_df["requested_lambda"], 0.25)]
    reu_row = scale_df[np.isclose(scale_df["requested_lambda"], 1.00)]

    for metric in PRIMARY_INDEPENDENT_METRICS:
        q95col = f"{metric}_q95"
        base_val = float(baseline.iloc[0][q95col]) if len(baseline) and q95col in baseline.columns else np.nan
        reu_val = float(reu_row.iloc[0][q95col]) if len(reu_row) and q95col in reu_row.columns else np.nan
        scale_df[f"{metric}_relative_to_025reu"] = (
            scale_df[q95col] / base_val if np.isfinite(base_val) and base_val > EPS else np.nan
        )
        scale_df[f"{metric}_relative_to_reu"] = (
            scale_df[q95col] / reu_val if np.isfinite(reu_val) and reu_val > EPS else np.nan
        )

    rel_cols = [f"{m}_relative_to_025reu" for m in PRIMARY_INDEPENDENT_METRICS]
    scale_df["composite_residual_error"] = scale_df[rel_cols].median(axis=1, skipna=True)

    return position_df, scale_df


# =============================================================================
# 7. REALIZATION-LEVEL AND GROUP-LEVEL INFERENCE
# =============================================================================

def get_scale_value(df: pd.DataFrame, lam: float, col: str) -> float:
    row = df[np.isclose(df["requested_lambda"].astype(float), float(lam))]
    if row.empty or col not in row.columns:
        return float("nan")
    value = float(row.iloc[0][col])
    return value if np.isfinite(value) else float("nan")


def make_realization_summary(scale_df: pd.DataFrame) -> Dict[str, Any]:
    first = scale_df.iloc[0]
    out: Dict[str, Any] = {
        "true_lc_px": int(first["true_lc_px"]),
        "seed": int(first["seed"]),
        "reu_px": int(first["reu_px"]),
    }

    for metric in PRIMARY_INDEPENDENT_METRICS:
        col = f"{metric}_q95"
        e05 = get_scale_value(scale_df, 0.50, col)
        e10 = get_scale_value(scale_df, 1.00, col)
        e15 = get_scale_value(scale_df, 1.50, col)
        out[f"{metric}_q95_lambda_0p5"] = e05
        out[f"{metric}_q95_lambda_1p0"] = e10
        out[f"{metric}_q95_lambda_1p5"] = e15

        out[f"{metric}_sub_to_reu_ratio"] = e05 / e10 if np.isfinite(e05) and np.isfinite(e10) and e10 > EPS else np.nan
        out[f"{metric}_reu_to_super_ratio"] = e10 / e15 if np.isfinite(e10) and np.isfinite(e15) and e15 > EPS else np.nan

        # Compare equal absolute support increments: 0.5->1.0 REU and
        # 1.0->1.5 REU are both +0.5 REU.  This avoids a misleading log-
        # elasticity comparison whose denominators differ.  Diminishing
        # returns means the fractional (and typically absolute) error reduction
        # over the second +0.5 REU increment is smaller than over the first.
        if np.isfinite(e05) and np.isfinite(e10) and e05 > EPS:
            pre_frac = (e05 - e10) / e05
            pre_abs = e05 - e10
        else:
            pre_frac = np.nan
            pre_abs = np.nan
        if np.isfinite(e10) and np.isfinite(e15) and e10 > EPS:
            post_frac = (e10 - e15) / e10
            post_abs = e10 - e15
        else:
            post_frac = np.nan
            post_abs = np.nan
        out[f"{metric}_pre_reu_fractional_reduction"] = pre_frac
        out[f"{metric}_post_reu_fractional_reduction"] = post_frac
        out[f"{metric}_pre_reu_absolute_reduction"] = pre_abs
        out[f"{metric}_post_reu_absolute_reduction"] = post_abs
        out[f"{metric}_diminishing_returns"] = (
            bool(post_frac < pre_frac) if np.isfinite(pre_frac) and np.isfinite(post_frac) else None
        )

    out["composite_at_0p5"] = get_scale_value(scale_df, 0.50, "composite_residual_error")
    out["composite_at_1p0"] = get_scale_value(scale_df, 1.00, "composite_residual_error")
    out["composite_at_1p5"] = get_scale_value(scale_df, 1.50, "composite_residual_error")
    return out


def aggregate_group_summary(all_scale_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lam, g in all_scale_df.groupby("requested_lambda", sort=True):
        # Primary group statistics require enough spatial positions in each
        # realization; finite-domain supplementary points remain in raw CSVs.
        supported = g[g["enough_positions"].astype(bool)].copy()
        row: Dict[str, Any] = {
            "requested_lambda": float(lam),
            "n_realizations_total": int(len(g)),
            "n_realizations_supported": int(len(supported)),
            "supported_fraction": float(len(supported) / max(len(g), 1)),
            "primary_range": bool(float(lam) <= PRIMARY_MAX_LAMBDA),
        }
        for metric in PRIMARY_INDEPENDENT_METRICS:
            col = f"{metric}_q95"
            vals = pd.to_numeric(supported[col], errors="coerce").dropna().to_numpy(dtype=float) if col in supported else np.asarray([])
            if vals.size:
                row[f"{metric}_q95_median"] = float(np.median(vals))
                row[f"{metric}_q95_q25"] = float(np.quantile(vals, 0.25))
                row[f"{metric}_q95_q75"] = float(np.quantile(vals, 0.75))
            else:
                row[f"{metric}_q95_median"] = np.nan
                row[f"{metric}_q95_q25"] = np.nan
                row[f"{metric}_q95_q75"] = np.nan

        vals = pd.to_numeric(supported.get("composite_residual_error", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            row["composite_residual_error_median"] = float(np.median(vals))
            row["composite_residual_error_q25"] = float(np.quantile(vals, 0.25))
            row["composite_residual_error_q75"] = float(np.quantile(vals, 0.75))
        else:
            row["composite_residual_error_median"] = np.nan
            row["composite_residual_error_q25"] = np.nan
            row["composite_residual_error_q75"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("requested_lambda").reset_index(drop=True)


def make_diminishing_returns_summary(real_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in PRIMARY_INDEPENDENT_METRICS:
        pre_frac_col = f"{metric}_pre_reu_fractional_reduction"
        post_frac_col = f"{metric}_post_reu_fractional_reduction"
        pre_abs_col = f"{metric}_pre_reu_absolute_reduction"
        post_abs_col = f"{metric}_post_reu_absolute_reduction"
        dim_col = f"{metric}_diminishing_returns"
        pre_frac = pd.to_numeric(real_df[pre_frac_col], errors="coerce")
        post_frac = pd.to_numeric(real_df[post_frac_col], errors="coerce")
        pre_abs = pd.to_numeric(real_df[pre_abs_col], errors="coerce")
        post_abs = pd.to_numeric(real_df[post_abs_col], errors="coerce")
        valid = pre_frac.notna() & post_frac.notna()
        dim = real_df.loc[valid, dim_col].dropna().astype(bool)
        rows.append({
            "metric": metric,
            "n_valid_realizations": int(valid.sum()),
            "median_pre_reu_fractional_reduction": float(np.median(pre_frac[valid])) if valid.any() else np.nan,
            "median_post_reu_fractional_reduction": float(np.median(post_frac[valid])) if valid.any() else np.nan,
            "median_pre_reu_absolute_reduction": float(np.median(pre_abs[valid])) if valid.any() else np.nan,
            "median_post_reu_absolute_reduction": float(np.median(post_abs[valid])) if valid.any() else np.nan,
            "diminishing_returns_fraction": float(dim.mean()) if len(dim) else np.nan,
        })
    return pd.DataFrame(rows)


def paired_statistics(real_df: pd.DataFrame) -> Dict[str, Any]:
    """Paired descriptive/Wilcoxon statistics across independent realizations.

    P values are supplementary; effect direction and ratios are the primary
    interpretation because the experiment contains only 12 realizations.
    """
    out: Dict[str, Any] = {}
    for metric in PRIMARY_INDEPENDENT_METRICS:
        c05 = f"{metric}_q95_lambda_0p5"
        c10 = f"{metric}_q95_lambda_1p0"
        c15 = f"{metric}_q95_lambda_1p5"
        a = pd.to_numeric(real_df[c05], errors="coerce")
        b = pd.to_numeric(real_df[c10], errors="coerce")
        c = pd.to_numeric(real_df[c15], errors="coerce")

        valid_pre = a.notna() & b.notna()
        valid_post = b.notna() & c.notna()

        metric_out: Dict[str, Any] = {
            "n_pre_pair": int(valid_pre.sum()),
            "n_post_pair": int(valid_post.sum()),
            "median_q95_lambda_0p5": float(np.median(a[valid_pre])) if valid_pre.any() else None,
            "median_q95_lambda_1p0": float(np.median(b[valid_pre])) if valid_pre.any() else None,
            "median_q95_lambda_1p5": float(np.median(c[valid_post])) if valid_post.any() else None,
            "median_sub_to_reu_ratio": None,
            "median_reu_to_super_ratio": None,
            "wilcoxon_0p5_vs_1p0_p": None,
            "wilcoxon_1p0_vs_1p5_p": None,
        }
        if valid_pre.any():
            ratios = a[valid_pre].to_numpy(dtype=float) / np.maximum(b[valid_pre].to_numpy(dtype=float), EPS)
            metric_out["median_sub_to_reu_ratio"] = float(np.median(ratios))
            try:
                metric_out["wilcoxon_0p5_vs_1p0_p"] = float(
                    wilcoxon(a[valid_pre].to_numpy(dtype=float), b[valid_pre].to_numpy(dtype=float), alternative="greater").pvalue
                )
            except Exception:
                pass
        if valid_post.any():
            ratios = b[valid_post].to_numpy(dtype=float) / np.maximum(c[valid_post].to_numpy(dtype=float), EPS)
            metric_out["median_reu_to_super_ratio"] = float(np.median(ratios))
            try:
                metric_out["wilcoxon_1p0_vs_1p5_p"] = float(
                    wilcoxon(b[valid_post].to_numpy(dtype=float), c[valid_post].to_numpy(dtype=float), alternative="greater").pvalue
                )
            except Exception:
                pass
        out[metric] = metric_out
    return out


# =============================================================================
# 8. FIGURES
# =============================================================================

def _draw_iqr_curve(ax: Any, group_df: pd.DataFrame, median_col: str, q25_col: str, q75_col: str) -> None:
    d = group_df[group_df["requested_lambda"] <= PRIMARY_MAX_LAMBDA].copy()
    x = d["requested_lambda"].to_numpy(dtype=float)
    med = d[median_col].to_numpy(dtype=float)
    q25 = d[q25_col].to_numpy(dtype=float)
    q75 = d[q75_col].to_numpy(dtype=float)
    ax.plot(x, med, marker="o", linewidth=1.5, markersize=5)
    ax.fill_between(x, q25, q75, alpha=0.18)
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.tick_params(direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_normalized_observation_support(all_scale_df: pd.DataFrame, group_df: pd.DataFrame, out_path: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for (_, _), g in all_scale_df.groupby(["true_lc_px", "seed"], sort=True):
        d = g[(g["requested_lambda"] <= PRIMARY_MAX_LAMBDA) & g["enough_positions"].astype(bool)].sort_values("requested_lambda")
        if len(d):
            ax.plot(
                d["requested_lambda"],
                d["composite_residual_error"],
                linewidth=0.7,
                alpha=0.28,
            )

    d = group_df[group_df["requested_lambda"] <= PRIMARY_MAX_LAMBDA].copy()
    ax.plot(
        d["requested_lambda"],
        d["composite_residual_error_median"],
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="Median across realizations",
    )
    ax.fill_between(
        d["requested_lambda"].to_numpy(dtype=float),
        d["composite_residual_error_q25"].to_numpy(dtype=float),
        d["composite_residual_error_q75"].to_numpy(dtype=float),
        alpha=0.20,
        label="Interquartile range",
    )
    ax.axvline(1.0, linestyle="--", linewidth=1.1, label="$L=L_{REU}$")
    ax.set_xlabel(r"REU-normalized observation support, $\lambda=L/L_{REU}$")
    ax.set_ylabel("Residual independent representation error\n(relative to 0.25 REU)")
    ax.set_xlim(0.20, PRIMARY_MAX_LAMBDA + 0.05)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric(group_df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    setup_plot_fonts()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    _draw_iqr_curve(
        ax,
        group_df,
        f"{metric}_q95_median",
        f"{metric}_q95_q25",
        f"{metric}_q95_q75",
    )
    ax.set_xlabel(r"REU-normalized observation support, $\lambda=L/L_{REU}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.20, PRIMARY_MAX_LAMBDA + 0.05)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_example_supports(
    lc_px: int,
    seed: int,
    reu_px: int,
    field_size: int,
    n_bands: int,
    out_path: Path,
) -> None:
    setup_plot_fonts()
    generated = synth.generate_isotropic_cube(
        lc_px=float(lc_px), seed=int(seed), field_size=int(field_size), n_bands=int(n_bands)
    )
    dom = generated["dominant_map"]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.imshow(dom, interpolation="nearest")
    cy = field_size // 2
    cx = field_size // 2
    line_styles = ["--", "-", ":"]
    labels = [r"$0.5L_{REU}$", r"$1.0L_{REU}$", r"$1.5L_{REU}$"]
    for lam, ls, label in zip((0.5, 1.0, 1.5), line_styles, labels):
        side = int(round(lam * reu_px))
        side = min(side, field_size)
        y0 = max(0, min(field_size - side, cy - side // 2))
        x0 = max(0, min(field_size - side, cx - side // 2))
        rect = plt.Rectangle((x0, y0), side, side, fill=False, linewidth=1.6, linestyle=ls, label=label)
        ax.add_patch(rect)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Observation supports on synthetic field ($L_c$={lc_px} px, REU={reu_px} px)")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NC-REU minimum observation support / ROI adequacy validation")
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_root = Path(args.synthetic_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("NC-REU minimum observation support / ROI adequacy validation")
    print("Independent raw-spectrum + true-abundance + true-configuration validation")
    print(f"Synthetic results: {synthetic_root}")
    print(f"Output:            {output_root}")
    print(f"lambda:            {list(ROI_LAMBDAS)}")
    print(f"Primary range:     lambda <= {PRIMARY_MAX_LAMBDA}")
    print("=" * 100)

    design, reu_results = load_synthetic_design(synthetic_root)
    field_size = int(design.get("field_size", synth.FIELD_SIZE))
    n_bands = int(design.get("n_bands", synth.N_BANDS))

    config_out = {
        "script": Path(__file__).name,
        "scientific_question": (
            "Does REU identify the minimum observation support at which independent spectral, "
            "true-compositional and true-spatial estimates become field-representative and less location-sensitive?"
        ),
        "synthetic_result_root": synthetic_root,
        "output_root": output_root,
        "field_size": field_size,
        "n_bands": n_bands,
        "roi_lambdas": ROI_LAMBDAS,
        "primary_max_lambda": PRIMARY_MAX_LAMBDA,
        "min_rois_for_primary_summary": MIN_ROIS_FOR_PRIMARY_SUMMARY,
        "max_rois_per_support": MAX_ROIS_PER_SUPPORT,
        "independent_metrics": {
            "mean_spectrum_sam_deg": "SAM between raw ROI mean reflectance and full-field mean reflectance",
            "mean_spectrum_rrmse": "relative RMSE between raw ROI mean reflectance and full-field mean reflectance",
            "abundance_tv": "total-variation distance between true mean abundance vectors",
            "dominant_fraction_tv": "total-variation distance between true dominant-endmember fractions",
            "transition_js": "JS divergence between true dominant-endmember horizontal+vertical adjacency distributions",
        },
        "primary_composite_metrics": PRIMARY_INDEPENDENT_METRICS,
        "circularity_control": (
            "None of the independent ROI metrics is one of the three learned soft-state q95 criteria used to define REU."
        ),
    }
    save_json(output_root / "roi_adequacy_config.json", config_out)

    all_position: List[pd.DataFrame] = []
    all_scale: List[pd.DataFrame] = []
    realization_rows: List[Dict[str, Any]] = []

    total = len(reu_results)
    for i, row in reu_results.iterrows():
        lc = int(row["true_lc_px"])
        seed = int(row["seed"])
        reu = int(row["reu_px"])
        print("-" * 100)
        print(f"[{i + 1}/{total}] Lc={lc} px, seed={seed}, previously confirmed REU={reu} px")
        t0 = time.time()
        pos_df, scale_df = evaluate_realization(
            lc_px=lc,
            seed=seed,
            reu_px=reu,
            field_size=field_size,
            n_bands=n_bands,
        )
        all_position.append(pos_df)
        all_scale.append(scale_df)
        realization_rows.append(make_realization_summary(scale_df))

        at05 = get_scale_value(scale_df, 0.50, "composite_residual_error")
        at10 = get_scale_value(scale_df, 1.00, "composite_residual_error")
        at15 = get_scale_value(scale_df, 1.50, "composite_residual_error")
        print(
            "  composite residual error (relative to 0.25 REU): "
            f"lambda=0.5 {at05:.3f}; lambda=1.0 {at10:.3f}; lambda=1.5 {at15:.3f}"
        )
        print(f"  runtime={time.time() - t0:.1f}s")

    position_df = pd.concat(all_position, ignore_index=True) if all_position else pd.DataFrame()
    scale_df_all = pd.concat(all_scale, ignore_index=True) if all_scale else pd.DataFrame()
    real_df = pd.DataFrame(realization_rows).sort_values(["true_lc_px", "seed"]).reset_index(drop=True)
    group_df = aggregate_group_summary(scale_df_all)
    diminishing_df = make_diminishing_returns_summary(real_df)
    stats = paired_statistics(real_df)

    position_df.to_csv(output_root / "roi_position_metrics.csv", index=False)
    scale_df_all.to_csv(output_root / "roi_scale_summary.csv", index=False)
    real_df.to_csv(output_root / "roi_realization_summary.csv", index=False)
    group_df.to_csv(output_root / "roi_group_summary.csv", index=False)
    diminishing_df.to_csv(output_root / "diminishing_returns_statistics.csv", index=False)
    save_json(output_root / "statistical_tests.json", stats)

    plot_normalized_observation_support(
        scale_df_all,
        group_df,
        output_root / "Fig_1_REU_normalized_observation_support.jpg",
    )
    plot_metric(
        group_df,
        "mean_spectrum_sam_deg",
        "95th-percentile mean-spectrum SAM (deg)",
        output_root / "Fig_2_spectral_representativeness.jpg",
    )
    plot_metric(
        group_df,
        "abundance_tv",
        "95th-percentile true abundance TV distance",
        output_root / "Fig_3_abundance_representativeness.jpg",
    )
    plot_metric(
        group_df,
        "transition_js",
        "95th-percentile true adjacency JS divergence",
        output_root / "Fig_4_configuration_representativeness.jpg",
    )

    # Use the middle Lc group for an interpretable observation-support example.
    example = reu_results.iloc[len(reu_results) // 2]
    plot_example_supports(
        lc_px=int(example["true_lc_px"]),
        seed=int(example["seed"]),
        reu_px=int(example["reu_px"]),
        field_size=field_size,
        n_bands=n_bands,
        out_path=output_root / "Fig_5_example_observation_supports.jpg",
    )

    print("=" * 100)
    print("ROI adequacy validation finished")
    print("Primary scientific outputs:")
    print("  roi_group_summary.csv")
    print("  diminishing_returns_statistics.csv")
    print("  statistical_tests.json")
    print("  Fig_1_REU_normalized_observation_support.jpg")
    print("Interpretation rule:")
    print("  Do NOT tune REU from these curves.  Ask whether independent errors fall strongly below REU")
    print("  and whether their marginal reduction becomes smaller at/above REU.")
    print("=" * 100)


if __name__ == "__main__":
    main()
