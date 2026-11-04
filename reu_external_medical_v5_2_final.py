# -*- coding: utf-8 -*-
"""
reu_external_medical_v5_2_final.py
==================================
FINAL pathology cross-domain REU validation runner for the preprocessed
MHSI Choledoch dataset.

PRIMARY VALIDATION
------------------
Whole released MHSI field:
    GT is NOT used to define REU.
    GT is used only for an independent tumor-fraction audit.

SECONDARY STRATIFIED ANALYSIS
-----------------------------
Tumor-core and normal-core connected components are considered only if the
component has >=6 stride-1-supported scales. This eligibility is based only on
GT geometry, never HSI values or REU outcomes.

INTERFACE
---------
The thin tumor-normal interface is diagnostic only. No strict square-support
REU is claimed for a narrow interface band.

The canonical scientific status is read directly from the exact V2.7
scale_decisions.json.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from reu_external_v5_2_common import (
    DPI,
    WINDOW_SIZES_PX,
    MIN_REGION_COVERAGE,
    MIN_WINDOWS_PER_SCALE,
    MIN_SUPPORTED_SCALES_CONFIRMATORY,
    SPECTRAL_Q95_MAX,
    COMPOSITION_Q95_MAX,
    CONFIGURATION_Q95_MAX,
    RELATIVE_STABILITY_MAX,
    STABILITY_BLOCK_LENGTH,
    RANDOM_SEED,
    analyze_unit_v27,
    component_is_confirmatory_eligible,
    component_records,
    crop_component,
    engine_provenance,
    ensure_dir,
    info,
    integral_image,
    plot_unit_status,
    safe_float,
    save_json,
)

# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_ROOT = Path(r"D:\LS_REU\data\choledoch_HSI\archive")
MHSI_DIR = DATA_ROOT / "MHSI" / "MHSI"
MASK_DIR = DATA_ROOT / "Mask"
SPLIT_JSON = DATA_ROOT / "train_val_test.json"

OUTPUT_ROOT = DATA_ROOT.parent / "_REU_external_medical_v5_2_final"

RUN_MODE = "confirmatory"          # "pilot" or "confirmatory"
ANALYSIS_SPLIT = "test"
ONLY_MIXED_SCENES = True
PILOT_SCENES = 1

RUN_SECONDARY_TUMOR_NORMAL = True
PILOT_COMPONENTS_PER_REGION = 1
CORE_MARGIN_PX = 8
INTERFACE_HALF_WIDTH_PX = 8   # display/diagnostic only

KEEP_HEAVY_ARRAYS = False
GT_AUDIT_MAX_WINDOWS_PER_SCALE = 300

ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


# =============================================================================
# ENVI IO
# =============================================================================

def parse_envi_header(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.lower() == "envi" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if v.startswith("{") and "}" not in v:
            parts = [v]
            while i < len(lines):
                parts.append(lines[i].strip())
                if "}" in lines[i]:
                    i += 1
                    break
                i += 1
            v = " ".join(parts)
        out[k] = v
    return out


def parse_num_list(value: Optional[str]) -> List[float]:
    if not value:
        return []
    s = value.strip().strip("{}")
    vals = []
    for token in re.split(r"[,\s]+", s):
        if token:
            try:
                vals.append(float(token))
            except ValueError:
                pass
    return vals


def resolve_data_file(hdr: Path, meta: Mapping[str, str]) -> Path:
    if "data file" in meta:
        p = Path(meta["data file"].strip().strip("{}\"'"))
        if not p.is_absolute():
            p = hdr.parent / p
        if p.exists():
            return p
    for ext in [".img", ".raw", ".dat", ".bin", ".bsq", ".bil", ".bip"]:
        p = hdr.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No ENVI binary paired with {hdr}")


@dataclass
class Scene:
    stem: str
    hdr_path: Path
    img_path: Path
    mask_path: Path
    split: str
    tumor_fraction: float


def load_envi_cube(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    m = parse_envi_header(hdr_path)
    lines = int(float(m["lines"]))
    samples = int(float(m["samples"]))
    bands = int(float(m["bands"]))
    dtype_code = int(float(m.get("data type", 4)))
    interleave = m.get("interleave", "bsq").strip().lower()
    byte_order = int(float(m.get("byte order", 0)))
    offset = int(float(m.get("header offset", 0)))

    if dtype_code not in ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type {dtype_code}")

    dtype = np.dtype(ENVI_DTYPES[dtype_code]).newbyteorder(
        ">" if byte_order == 1 else "<"
    )
    data_path = resolve_data_file(hdr_path, m)
    count = lines * samples * bands
    raw = np.memmap(
        data_path, dtype=dtype, mode="r", offset=offset, shape=(count,)
    )

    if interleave == "bip":
        cube = np.asarray(raw.reshape(lines, samples, bands), dtype=np.float32)
    elif interleave == "bsq":
        cube = np.asarray(
            raw.reshape(bands, lines, samples).transpose(1, 2, 0),
            dtype=np.float32,
        )
    elif interleave == "bil":
        cube = np.asarray(
            raw.reshape(lines, bands, samples).transpose(0, 2, 1),
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported interleave {interleave!r}")

    wl = np.asarray(parse_num_list(m.get("wavelength")), dtype=float)
    if wl.size != bands:
        wl = np.arange(bands, dtype=float)
    return cube, wl


def read_gt(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    if arr.shape != expected_shape:
        raise ValueError(
            f"GT shape mismatch {path}: {arr.shape} vs {expected_shape}"
        )
    return arr > 127


# =============================================================================
# SPLIT / SCENE DISCOVERY
# =============================================================================

def flatten_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, Mapping):
        for v in obj.values():
            out.extend(flatten_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_strings(v))
    return out


def split_stems() -> Dict[str, set]:
    if not SPLIT_JSON.exists():
        return {}
    obj = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    result: Dict[str, set] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in {"train", "val", "valid", "validation", "test"}:
                name = "val" if kl in {"valid", "validation"} else kl
                result[name] = {Path(s).stem for s in flatten_strings(v)}
    return result


def discover_scenes() -> List[Scene]:
    splits = split_stems()
    requested = splits.get(ANALYSIS_SPLIT, set())
    scenes: List[Scene] = []

    for hdr in sorted(MHSI_DIR.glob("*.hdr")):
        stem = hdr.stem
        if requested and stem not in requested:
            continue

        mask_path = MASK_DIR / f"{stem}.png"
        if not mask_path.exists():
            continue

        meta = parse_envi_header(hdr)
        h = int(float(meta["lines"]))
        w = int(float(meta["samples"]))
        gt = read_gt(mask_path, (h, w))
        frac = float(gt.mean())

        if ONLY_MIXED_SCENES and not (0.0 < frac < 1.0):
            continue

        scenes.append(Scene(
            stem=stem,
            hdr_path=hdr,
            img_path=resolve_data_file(hdr, meta),
            mask_path=mask_path,
            split=ANALYSIS_SPLIT if requested else "unknown",
            tumor_fraction=frac,
        ))

    if RUN_MODE.lower() == "pilot":
        # Debug-only scene choice by GT balance; no HSI/REU outcome is used.
        scenes.sort(key=lambda s: (abs(s.tumor_fraction - 0.5), s.stem))
        return scenes[:max(1, int(PILOT_SCENES))]
    if RUN_MODE.lower() == "confirmatory":
        return sorted(scenes, key=lambda s: s.stem)
    raise ValueError("RUN_MODE must be 'pilot' or 'confirmatory'")


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def robust_rgb(cube: np.ndarray) -> np.ndarray:
    b = cube.shape[2]
    ids = [
        min(b - 1, int(round(x)))
        for x in (0.15 * b, 0.50 * b, 0.85 * b)
    ]
    rgb = cube[:, :, ids].astype(float)
    out = np.zeros_like(rgb, dtype=float)
    for c in range(3):
        x = rgb[:, :, c]
        valid = x[np.isfinite(x)]
        lo, hi = np.nanpercentile(valid, [1, 99])
        out[:, :, c] = np.clip(
            (x - lo) / max(hi - lo, 1e-12), 0, 1
        )
    return out


def save_scene_audit(
    outdir: Path,
    cube: np.ndarray,
    tumor: np.ndarray,
    tumor_core: np.ndarray,
    normal_core: np.ndarray,
    interface: np.ndarray,
) -> None:
    rgb = robust_rgb(cube)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    axes[0].imshow(rgb)
    axes[0].set_title("Released MHSI view")
    axes[1].imshow(tumor.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Tumor GT ({tumor.mean():.1%})")
    axes[2].imshow(tumor_core.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Tumor core")
    axes[3].imshow(normal_core.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Normal core")
    axes[4].imshow(interface.astype(float), cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Interface diagnostic")
    axes[5].imshow(rgb)
    axes[5].contour(tumor.astype(float), levels=[0.5], linewidths=1.0)
    axes[5].set_title("Tumor boundary on MHSI")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(outdir / "00_scene_audit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def tumor_fraction_audit(
    tumor: np.ndarray,
    output_dir: Path,
    reference_fraction: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    ii = integral_image(tumor)
    h, w = tumor.shape
    rows = []

    for L in WINDOW_SIZES_PX:
        if L > h or L > w:
            rows.append({
                "base_size_px": L,
                "n_positions_total": 0,
                "n_positions_sampled": 0,
                "tumor_fraction_abs_error_q95": np.nan,
                "tumor_fraction_abs_error_median": np.nan,
            })
            continue

        counts = (
            ii[L:, L:] - ii[:-L, L:]
            - ii[L:, :-L] + ii[:-L, :-L]
        ).astype(float)
        frac = counts.ravel() / float(L * L)

        if frac.size > GT_AUDIT_MAX_WINDOWS_PER_SCALE:
            idx = rng.choice(
                frac.size,
                size=GT_AUDIT_MAX_WINDOWS_PER_SCALE,
                replace=False,
            )
            frac = frac[idx]

        err = np.abs(frac - reference_fraction)
        rows.append({
            "base_size_px": L,
            "n_positions_total":
                int((h-L+1) * (w-L+1)),
            "n_positions_sampled": int(err.size),
            "tumor_fraction_abs_error_q95":
                float(np.quantile(err, 0.95))
                if err.size else np.nan,
            "tumor_fraction_abs_error_median":
                float(np.median(err))
                if err.size else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        output_dir / "whole_tissue_GT_tumor_fraction_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    return df


def plot_tumor_fraction_audit(
    df: pd.DataFrame,
    output_path: Path,
    strict_reu_px: Any,
    ref_fraction: float,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = pd.to_numeric(df["base_size_px"], errors="coerce")
    y = pd.to_numeric(
        df["tumor_fraction_abs_error_q95"], errors="coerce"
    )
    ax.plot(x, y, marker="o", ms=4, lw=1.2)
    reu = safe_float(strict_reu_px)
    if np.isfinite(reu):
        ax.axvline(
            reu, ls="--", lw=1.0,
            label=f"Strict REU = {reu:g} px"
        )
        ax.legend(frameon=False)
    ax.set_xlabel("Observation support on released grid (px)")
    ax.set_ylabel("q95 |local tumor fraction - scene fraction|")
    ax.set_title(
        f"Independent pathology-label audit "
        f"(scene fraction={ref_fraction:.3f})"
    )
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_ROOT)

    info("=" * 112)
    info("FINAL strict cross-domain REU | MHSI Choledoch pathology | v5.2")
    info(
        f"RUN_MODE={RUN_MODE}; split={ANALYSIS_SPLIT}; "
        f"only_mixed={ONLY_MIXED_SCENES}"
    )
    info(f"supports={WINDOW_SIZES_PX}")
    info(
        f"stride-1 support audit; coverage>={MIN_REGION_COVERAGE:.2f}; "
        f"minimum valid placements={MIN_WINDOWS_PER_SCALE}"
    )
    info(
        f"secondary component eligibility >= "
        f"{MIN_SUPPORTED_SCALES_CONFIRMATORY} supported scales"
    )
    info(
        f"V2.7 thresholds={SPECTRAL_Q95_MAX:.2f}/"
        f"{COMPOSITION_Q95_MAX:.2f}/{CONFIGURATION_Q95_MAX:.2f}; "
        f"stability<={RELATIVE_STABILITY_MAX:.2f}; "
        f"block={STABILITY_BLOCK_LENGTH}"
    )
    info("=" * 112)

    scenes = discover_scenes()
    if not scenes:
        raise RuntimeError("No paired scenes found for requested mode/split.")

    pd.DataFrame([
        {
            "scene": s.stem,
            "split": s.split,
            "tumor_fraction": s.tumor_fraction,
            "hdr": str(s.hdr_path),
            "img": str(s.img_path),
            "mask": str(s.mask_path),
        }
        for s in scenes
    ]).to_csv(
        OUTPUT_ROOT / "scene_selection_audit.csv",
        index=False, encoding="utf-8-sig",
    )

    save_json(OUTPUT_ROOT / "analysis_design.json", {
        "domain": "pathology",
        "dataset": "MHSI Choledoch preprocessed",
        "run_mode": RUN_MODE,
        "analysis_split": ANALYSIS_SPLIT,
        "primary_validation":
            "whole released field; GT independent",
        "secondary_validation":
            "geometry-eligible tumor-core/normal-core connected components",
        "interface":
            "diagnostic only; no strict square-support REU",
        "secondary_component_min_supported_scales":
            MIN_SUPPORTED_SCALES_CONFIRMATORY,
        "support_sizes_px": WINDOW_SIZES_PX,
        "minimum_region_coverage": MIN_REGION_COVERAGE,
        "minimum_windows_per_scale": MIN_WINDOWS_PER_SCALE,
        "strict_decision_source": "exact V2.7 scale_decisions.json",
        "engine": engine_provenance(),
    })

    primary_rows: List[Dict[str, Any]] = []
    secondary_rows: List[Dict[str, Any]] = []
    secondary_eligibility_rows: List[Dict[str, Any]] = []

    for si, scene in enumerate(scenes, start=1):
        info("\n" + "-" * 112)
        info(
            f"[{si}/{len(scenes)}] SCENE {scene.stem} | "
            f"GT tumor fraction={scene.tumor_fraction:.4f}"
        )

        scene_dir = ensure_dir(OUTPUT_ROOT / scene.stem)
        cube, wavelengths = load_envi_cube(scene.hdr_path)
        tumor = read_gt(scene.mask_path, cube.shape[:2])
        normal = ~tumor

        # Strictly mutually exclusive stratified zones.
        tumor_dist = ndi.distance_transform_edt(tumor)
        normal_dist = ndi.distance_transform_edt(normal)
        tumor_core = tumor & (tumor_dist > CORE_MARGIN_PX)
        normal_core = normal & (normal_dist > CORE_MARGIN_PX)
        interface = ~(tumor_core | normal_core)

        save_scene_audit(
            scene_dir,
            cube,
            tumor,
            tumor_core,
            normal_core,
            interface,
        )

        # ------------------------------------------------------------------
        # PRIMARY: whole field
        # ------------------------------------------------------------------
        whole = np.ones(cube.shape[:2], dtype=bool)
        whole_dir = ensure_dir(scene_dir / "whole_tissue")

        try:
            result = analyze_unit_v27(
                cube=cube,
                mask=whole,
                unit_id=f"{scene.stem}_whole_tissue",
                output_dir=whole_dir,
                keep_heavy=KEEP_HEAVY_ARRAYS,
                minimum_supported_scales=3,
            )
            info(
                f"  whole_tissue -> "
                f"{result.get('scientific_status')} | "
                f"REU={result.get('strict_reu_px')} | "
                f"core={result.get('core_status')}"
            )
            primary_rows.append({
                "scene": scene.stem,
                "tumor_fraction": float(tumor.mean()),
                "pixel_count": int(whole.sum()),
                "strict_reu_px": result.get("strict_reu_px"),
                "first_joint_pass_px":
                    result.get("first_joint_pass_px"),
                "largest_dense_supported_scale_px":
                    result.get("largest_dense_supported_scale_px"),
                "scientific_status":
                    result.get("scientific_status"),
                "core_status": result.get("core_status"),
                "core_decision_class":
                    result.get("core_decision_class"),
                "censor_lower_bound_px":
                    result.get("censor_lower_bound_px"),
                "largest_scale_minimax_ratio":
                    result.get("largest_scale_minimax_ratio"),
                "label": f"{scene.stem} | whole",
            })

            gt_df = tumor_fraction_audit(
                tumor,
                whole_dir,
                float(tumor.mean()),
            )
            plot_tumor_fraction_audit(
                gt_df,
                whole_dir /
                "whole_tissue_GT_tumor_fraction_audit.png",
                result.get("strict_reu_px"),
                float(tumor.mean()),
            )

        except Exception as exc:
            info(f"  whole_tissue FAILED: {exc}")
            traceback.print_exc()
            primary_rows.append({
                "scene": scene.stem,
                "tumor_fraction": float(tumor.mean()),
                "pixel_count": int(whole.sum()),
                "strict_reu_px": np.nan,
                "scientific_status":
                    f"analysis_failed: {type(exc).__name__}",
                "core_status": "",
                "label": f"{scene.stem} | whole",
            })

        # ------------------------------------------------------------------
        # SECONDARY: tumor / normal core components
        # ------------------------------------------------------------------
        if RUN_SECONDARY_TUMOR_NORMAL:
            for region_name, region_mask in [
                ("tumor_core", tumor_core),
                ("normal_core", normal_core),
            ]:
                records = component_records(region_mask)
                eligible = [
                    r for r in records
                    if component_is_confirmatory_eligible(r)
                ]

                secondary_eligibility_rows.append({
                    "scene": scene.stem,
                    "region_type": region_name,
                    "region_pixel_count": int(region_mask.sum()),
                    "n_components_ge300px": int(len(records)),
                    "n_confirmatory_eligible_components":
                        int(len(eligible)),
                    "confirmatory_eligible":
                        bool(len(eligible) > 0),
                    "best_n_supported_scales":
                        max(
                            [r["n_supported_scales"] for r in records],
                            default=0,
                        ),
                    "eligibility_reason":
                        "eligible"
                        if eligible else
                        "extent_insufficient_not_estimable",
                })

                if not eligible:
                    info(
                        f"  {region_name}: excluded from strict REU "
                        f"estimation; no component has >=6 supported scales"
                    )
                    continue

                if RUN_MODE.lower() == "pilot":
                    selected = eligible[
                        :max(1, int(PILOT_COMPONENTS_PER_REGION))
                    ]
                else:
                    selected = eligible

                region_dir = ensure_dir(scene_dir / region_name)

                for rank, rec in enumerate(selected, start=1):
                    comp_id = int(rec["component_id"])
                    info(
                        f"  {region_name} C{comp_id}: "
                        f"pixels={rec['pixel_count']}; "
                        f"supported={rec['supported_scales']}"
                    )

                    comp_dir = ensure_dir(
                        region_dir / f"component_{comp_id:03d}"
                    )
                    rec["support_audit"].to_csv(
                        comp_dir /
                        "dense_support_audit_stride1_PRE.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    crop_cube, crop_mask = crop_component(cube, rec)

                    try:
                        result = analyze_unit_v27(
                            cube=crop_cube,
                            mask=crop_mask,
                            unit_id=(
                                f"{scene.stem}_{region_name}_C{comp_id}"
                            ),
                            output_dir=comp_dir,
                            keep_heavy=KEEP_HEAVY_ARRAYS,
                            minimum_supported_scales=
                                MIN_SUPPORTED_SCALES_CONFIRMATORY,
                        )
                        info(
                            f"    -> "
                            f"{result.get('scientific_status')} | "
                            f"REU={result.get('strict_reu_px')} | "
                            f"core={result.get('core_status')}"
                        )
                        secondary_rows.append({
                            "scene": scene.stem,
                            "region_type": region_name,
                            "component_id": comp_id,
                            "geometry_rank": rank,
                            "component_pixels":
                                int(rec["pixel_count"]),
                            "n_dense_supported_scales":
                                result.get(
                                    "n_dense_supported_scales"
                                ),
                            "strict_reu_px":
                                result.get("strict_reu_px"),
                            "first_joint_pass_px":
                                result.get("first_joint_pass_px"),
                            "largest_dense_supported_scale_px":
                                result.get(
                                    "largest_dense_supported_scale_px"
                                ),
                            "scientific_status":
                                result.get("scientific_status"),
                            "core_status":
                                result.get("core_status"),
                            "core_decision_class":
                                result.get("core_decision_class"),
                            "censor_lower_bound_px":
                                result.get("censor_lower_bound_px"),
                            "largest_scale_minimax_ratio":
                                result.get(
                                    "largest_scale_minimax_ratio"
                                ),
                            "label":
                                f"{scene.stem} | "
                                f"{region_name} C{comp_id}",
                        })
                    except Exception as exc:
                        info(f"    FAILED: {exc}")
                        traceback.print_exc()
                        secondary_rows.append({
                            "scene": scene.stem,
                            "region_type": region_name,
                            "component_id": comp_id,
                            "component_pixels":
                                int(rec["pixel_count"]),
                            "strict_reu_px": np.nan,
                            "scientific_status":
                                f"analysis_failed: "
                                f"{type(exc).__name__}",
                            "core_status": "",
                            "label":
                                f"{scene.stem} | "
                                f"{region_name} C{comp_id}",
                        })

        Image.fromarray(
            interface.astype(np.uint8) * 255
        ).save(scene_dir / "interface_diagnostic_mask.png")

    primary = pd.DataFrame(primary_rows)
    secondary = pd.DataFrame(secondary_rows)
    eligibility = pd.DataFrame(secondary_eligibility_rows)

    primary.to_csv(
        OUTPUT_ROOT / "PRIMARY_whole_tissue_REU_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    secondary.to_csv(
        OUTPUT_ROOT / "SECONDARY_tumor_normal_component_REU_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    eligibility.to_csv(
        OUTPUT_ROOT / "SECONDARY_component_geometry_eligibility.csv",
        index=False, encoding="utf-8-sig",
    )

    if not primary.empty:
        plot_unit_status(
            primary,
            OUTPUT_ROOT / "PRIMARY_whole_tissue_REU_status.png",
            "MHSI Choledoch: primary whole-field REU status",
        )
    if not secondary.empty:
        plot_unit_status(
            secondary,
            OUTPUT_ROOT /
            "SECONDARY_tumor_normal_component_REU_status.png",
            "MHSI Choledoch: secondary stratified REU status",
        )

    # Transparent aggregation.
    agg_rows = []
    for label, df in [
        ("whole_tissue_primary", primary),
        ("tumor_normal_secondary", secondary),
    ]:
        if df.empty:
            continue
        statuses = df["scientific_status"].astype(str)
        strict = pd.to_numeric(df["strict_reu_px"], errors="coerce")
        confirmed = strict[np.isfinite(strict)]
        agg_rows.append({
            "analysis_level": label,
            "n_analyzed_units": int(len(df)),
            "n_confirmed_strict_reu":
                int((statuses == "confirmed_strict_reu").sum()),
            "n_domain_censored":
                int(
                    (
                        statuses ==
                        "domain_censored_before_confirmation"
                    ).sum()
                ),
            "n_range_exhausted":
                int(
                    (
                        statuses ==
                        "configured_range_exhausted_without_strict_reu"
                    ).sum()
                ),
            "n_qc_failures":
                int(statuses.str.contains(
                    "implementation_qc_failure", regex=False
                ).sum()),
            "median_confirmed_reu_px":
                float(np.median(confirmed))
                if confirmed.size else np.nan,
        })

    pd.DataFrame(agg_rows).to_csv(
        OUTPUT_ROOT / "medical_REU_aggregate.csv",
        index=False, encoding="utf-8-sig",
    )

    info("\n" + "=" * 112)
    info("FINISHED")
    info(f"Results: {OUTPUT_ROOT}")
    info("PRIMARY science = whole_tissue test scenes.")
    info(
        "Tumor/normal components are secondary and only geometry-eligible "
        "components enter strict REU estimation."
    )
    info("Interface remains diagnostic only.")
    info(
        "For final manuscript results, set RUN_MODE='confirmatory'."
    )
    info("=" * 112)


if __name__ == "__main__":
    main()
