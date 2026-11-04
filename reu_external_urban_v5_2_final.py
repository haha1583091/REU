# -*- coding: utf-8 -*-
"""
reu_external_urban_v5_2_final.py
======================================
FINAL urban cross-domain REU validation for Pavia University.

V5.2 PRINCIPLE
--------------
Not every labelled class is required to yield an REU.

A class/component enters confirmatory REU estimation ONLY if its GT geometry,
evaluated independently of HSI values and REU outcomes, provides at least
MIN_SUPPORTED_SCALES_CONFIRMATORY (=6) supported scales on the stride-1 grid.

Small/narrow classes are therefore reported as:
    extent_insufficient_not_estimable
rather than being counted as REU failures.

The exact V2.7 core decision in scale_decisions.json is canonical.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

from reu_external_v5_2_common import (
    WINDOW_SIZES_PX,
    MIN_REGION_COVERAGE,
    MIN_WINDOWS_PER_SCALE,
    MIN_SUPPORTED_SCALES_CONFIRMATORY,
    SPECTRAL_Q95_MAX,
    COMPOSITION_Q95_MAX,
    CONFIGURATION_Q95_MAX,
    RELATIVE_STABILITY_MAX,
    STABILITY_BLOCK_LENGTH,
    analyze_unit_v27,
    component_is_confirmatory_eligible,
    component_records,
    crop_component,
    engine_provenance,
    ensure_dir,
    info,
    plot_unit_status,
    safe_float,
    save_json,
)

# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_PATH = Path(r"D:\LS_REU\data\PaviaU\PaviaU.mat")
GT_PATH = Path(r"D:\LS_REU\data\PaviaU\PaviaU_gt.mat")
DATA_KEY = "paviaU"
GT_KEY = "paviaU_gt"

OUTPUT_ROOT = Path(r"D:\LS_REU\results\REU_external_urban_v5_2_final")

# pilot = one geometry-ranked eligible component per eligible class.
# confirmatory = ALL geometry-eligible components in ALL eligible classes.
RUN_MODE = "pilot"
PILOT_COMPONENTS_PER_CLASS = 1

# Keep None for final science. If used for debugging, the restriction is saved.
SELECTED_CLASS_IDS: Optional[List[int]] = None
KEEP_HEAVY_ARRAYS = False

CLASS_NAMES = {
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare soil",
    7: "Bitumen",
    8: "Self-blocking bricks",
    9: "Shadows",
}


def load_mat_exact(path: Path, key: str, ndim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = loadmat(path)
    if key not in obj:
        public = [k for k in obj.keys() if not k.startswith("__")]
        raise KeyError(
            f"{path}: key {key!r} not found. Available keys: {public}"
        )
    arr = np.asarray(obj[key])
    if arr.ndim != ndim:
        raise ValueError(
            f"{path}:{key} expected {ndim}D, got {arr.shape}"
        )
    return arr


def main() -> None:
    ensure_dir(OUTPUT_ROOT)

    info("=" * 112)
    info("FINAL strict cross-domain REU | urban | Pavia University | v5.2")
    info(f"RUN_MODE={RUN_MODE}")
    info(f"supports={WINDOW_SIZES_PX}")
    info(
        f"stride-1 support audit; coverage>={MIN_REGION_COVERAGE:.2f}; "
        f"minimum valid placements={MIN_WINDOWS_PER_SCALE}"
    )
    info(
        f"confirmatory geometry eligibility >= "
        f"{MIN_SUPPORTED_SCALES_CONFIRMATORY} supported scales"
    )
    info(
        f"V2.7 thresholds={SPECTRAL_Q95_MAX:.2f}/"
        f"{COMPOSITION_Q95_MAX:.2f}/{CONFIGURATION_Q95_MAX:.2f}; "
        f"stability<={RELATIVE_STABILITY_MAX:.2f}; "
        f"block={STABILITY_BLOCK_LENGTH}"
    )
    info("=" * 112)

    cube = load_mat_exact(DATA_PATH, DATA_KEY, 3).astype(np.float32)
    gt = load_mat_exact(GT_PATH, GT_KEY, 2).astype(np.int32)
    if gt.shape != cube.shape[:2]:
        raise ValueError(
            f"Cube/GT mismatch: cube={cube.shape}, gt={gt.shape}"
        )
    info(f"cube shape={cube.shape}; GT shape={gt.shape}")

    class_ids = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    if SELECTED_CLASS_IDS is not None:
        allowed = {int(v) for v in SELECTED_CLASS_IDS}
        class_ids = [v for v in class_ids if v in allowed]

    design = {
        "domain": "urban",
        "dataset": "Pavia University",
        "data_path": str(DATA_PATH),
        "gt_path": str(GT_PATH),
        "data_key": DATA_KEY,
        "gt_key": GT_KEY,
        "run_mode": RUN_MODE,
        "selected_class_ids_debug_only": SELECTED_CLASS_IDS,
        "selection_uses_hsi_values": False,
        "selection_uses_reu_outcomes": False,
        "eligibility_rule":
            "GT 8-connected component with >=6 stride-1-supported scales",
        "why_six_scales":
            "2 x strict three-scale persistent block length",
        "support_sizes_px": WINDOW_SIZES_PX,
        "minimum_region_coverage": MIN_REGION_COVERAGE,
        "minimum_windows_per_scale": MIN_WINDOWS_PER_SCALE,
        "strict_decision_source": "exact V2.7 scale_decisions.json",
        "engine": engine_provenance(),
    }
    save_json(OUTPUT_ROOT / "analysis_design.json", design)

    component_rows: List[Dict[str, Any]] = []
    eligibility_rows: List[Dict[str, Any]] = []
    dense_audits: List[pd.DataFrame] = []

    for ci, class_id in enumerate(class_ids, start=1):
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        info("\n" + "-" * 112)
        info(f"[{ci}/{len(class_ids)}] class {class_id}: {class_name}")

        records = component_records(gt == class_id)
        eligible = [r for r in records if component_is_confirmatory_eligible(r)]

        # Class-level eligibility audit is independent of HSI and REU outcomes.
        eligibility_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "gt_pixel_count": int((gt == class_id).sum()),
            "n_components_ge300px": int(len(records)),
            "n_confirmatory_eligible_components": int(len(eligible)),
            "class_confirmatory_eligible": bool(len(eligible) > 0),
            "best_n_supported_scales":
                max([r["n_supported_scales"] for r in records], default=0),
            "best_largest_supported_scale_px":
                max(
                    [safe_float(r["largest_supported_scale_px"], -1)
                     for r in records],
                    default=np.nan,
                ),
            "eligibility_reason":
                "eligible"
                if eligible else
                "extent_insufficient_not_estimable",
        })

        for r in records:
            ar = r["support_audit"].copy()
            ar.insert(0, "class_id", class_id)
            ar.insert(1, "class_name", class_name)
            ar.insert(2, "component_id", r["component_id"])
            ar.insert(3, "component_pixels", r["pixel_count"])
            ar.insert(4, "n_supported_scales", r["n_supported_scales"])
            ar.insert(
                5, "confirmatory_eligible",
                component_is_confirmatory_eligible(r)
            )
            dense_audits.append(ar)

        if not eligible:
            info(
                "  excluded from strict REU estimation: "
                "no component has >=6 supported scales"
            )
            continue

        info(
            "  confirmatory-eligible components: "
            + ", ".join(
                f"C{r['component_id']}"
                f"(n={r['n_supported_scales']}, "
                f"Lmax={r['largest_supported_scale_px']})"
                for r in eligible
            )
        )

        if RUN_MODE.lower() == "pilot":
            selected = eligible[:max(1, int(PILOT_COMPONENTS_PER_CLASS))]
        elif RUN_MODE.lower() == "confirmatory":
            selected = eligible
        else:
            raise ValueError("RUN_MODE must be 'pilot' or 'confirmatory'")

        class_dir = ensure_dir(
            OUTPUT_ROOT /
            f"class_{class_id:02d}_{class_name.replace(' ', '_')}"
        )

        for rank, rec in enumerate(selected, start=1):
            comp_id = int(rec["component_id"])
            info(
                f"  analyze C{comp_id}: pixels={rec['pixel_count']}, "
                f"bbox={rec['bbox_height_px']}x{rec['bbox_width_px']}, "
                f"supported={rec['supported_scales']}"
            )

            comp_dir = ensure_dir(class_dir / f"component_{comp_id:03d}")
            rec["support_audit"].to_csv(
                comp_dir / "dense_support_audit_stride1_PRE.csv",
                index=False, encoding="utf-8-sig",
            )

            crop_cube, crop_mask = crop_component(cube, rec)

            try:
                result = analyze_unit_v27(
                    cube=crop_cube,
                    mask=crop_mask,
                    unit_id=(
                        "PaviaU_"
                        f"class_{class_id:02d}_component_{comp_id:03d}"
                    ),
                    output_dir=comp_dir,
                    keep_heavy=KEEP_HEAVY_ARRAYS,
                    minimum_supported_scales=
                        MIN_SUPPORTED_SCALES_CONFIRMATORY,
                )
                info(
                    f"    -> {result.get('scientific_status')} | "
                    f"REU={result.get('strict_reu_px')} | "
                    f"core={result.get('core_status')}"
                )

                component_rows.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "component_id": comp_id,
                    "geometry_rank": rank,
                    "component_pixels": int(rec["pixel_count"]),
                    "bbox_height_px": int(rec["bbox_height_px"]),
                    "bbox_width_px": int(rec["bbox_width_px"]),
                    "bbox_fill_fraction": float(rec["bbox_fill_fraction"]),
                    "n_dense_supported_scales":
                        result.get("n_dense_supported_scales"),
                    "strict_reu_px": result.get("strict_reu_px"),
                    "first_joint_pass_px":
                        result.get("first_joint_pass_px"),
                    "largest_dense_supported_scale_px":
                        result.get("largest_dense_supported_scale_px"),
                    "largest_core_supported_scale_px":
                        result.get("largest_core_supported_scale_px"),
                    "scientific_status":
                        result.get("scientific_status"),
                    "core_status": result.get("core_status"),
                    "core_decision_class":
                        result.get("core_decision_class"),
                    "censor_lower_bound_px":
                        result.get("censor_lower_bound_px"),
                    "largest_scale_minimax_ratio":
                        result.get("largest_scale_minimax_ratio"),
                    "largest_scale_scientific_constraints_pass":
                        result.get(
                            "largest_scale_scientific_constraints_pass"
                        ),
                    "limiting_constraints_at_largest_scale":
                        "|".join(
                            result.get(
                                "limiting_constraints_at_largest_scale", []
                            ) or []
                        ),
                    "label": f"{class_name} | C{comp_id}",
                })
            except Exception as exc:
                info(f"    FAILED: {exc}")
                traceback.print_exc()
                component_rows.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "component_id": comp_id,
                    "component_pixels": int(rec["pixel_count"]),
                    "strict_reu_px": np.nan,
                    "scientific_status":
                        f"analysis_failed: {type(exc).__name__}",
                    "core_status": "",
                    "label": f"{class_name} | C{comp_id}",
                })

    eligibility = pd.DataFrame(eligibility_rows)
    eligibility.to_csv(
        OUTPUT_ROOT / "class_geometry_eligibility.csv",
        index=False, encoding="utf-8-sig",
    )

    if dense_audits:
        pd.concat(dense_audits, ignore_index=True).to_csv(
            OUTPUT_ROOT / "ALL_component_dense_support_audit_stride1.csv",
            index=False, encoding="utf-8-sig",
        )

    summary = pd.DataFrame(component_rows)
    summary.to_csv(
        OUTPUT_ROOT / "eligible_component_REU_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    # Aggregate ONLY analyzed geometry-eligible units.
    aggregate_rows = []
    if not summary.empty:
        for class_id, q in summary.groupby("class_id", sort=True):
            statuses = q["scientific_status"].astype(str)
            strict = pd.to_numeric(q["strict_reu_px"], errors="coerce")
            confirmed = strict[np.isfinite(strict)]
            aggregate_rows.append({
                "class_id": int(class_id),
                "class_name": str(q.iloc[0]["class_name"]),
                "n_analyzed_eligible_components": int(len(q)),
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
                "confirmed_fraction_among_analyzed":
                    float(
                        (statuses == "confirmed_strict_reu").sum()
                        / max(1, len(q))
                    ),
                "median_confirmed_component_reu_px":
                    float(np.median(confirmed))
                    if confirmed.size else np.nan,
                "evidence_level":
                    "replicated_components"
                    if len(q) >= 2 else "single_component",
            })

    pd.DataFrame(aggregate_rows).to_csv(
        OUTPUT_ROOT / "eligible_class_REU_aggregate.csv",
        index=False, encoding="utf-8-sig",
    )

    if not summary.empty:
        plot_unit_status(
            summary,
            OUTPUT_ROOT / "eligible_component_REU_status.png",
            "Pavia University: geometry-eligible strict REU units",
        )

    n_eligible_classes = int(
        eligibility["class_confirmatory_eligible"].sum()
    ) if not eligibility.empty else 0

    info("\n" + "=" * 112)
    info("FINISHED")
    info(f"Results: {OUTPUT_ROOT}")
    info(
        f"Confirmatory-eligible classes: {n_eligible_classes}/"
        f"{len(eligibility)}"
    )
    info(
        "Classes excluded by geometry are NOT REU failures and are NOT included "
        "in the strict-REU success denominator."
    )
    info(
        "For final manuscript results, set RUN_MODE='confirmatory' and retain "
        "SELECTED_CLASS_IDS=None."
    )
    info("=" * 112)


if __name__ == "__main__":
    main()
