# -*- coding: utf-8 -*-
"""Geometry utilities for equal-area representative-support analysis.

All support geometries use ``base_size_px`` as an equal-area scale:

    target support area = base_size_px ** 2

This makes square, circle, ellipse and rotated rectangle comparisons fair.
Angles are measured counter-clockwise from the positive image x-axis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Union
import math

import numpy as np

GEOMETRY_VERSION = "2.7-robust-scale-inference"

EPS = 1.0e-12
NumericOrAdaptive = Union[float, str]


@dataclass(frozen=True)
class ShapeSpec:
    """Definition of one candidate spatial-support geometry.

    Parameters
    ----------
    name:
        Unique output-safe name.
    kind:
        ``rectangle`` or ``ellipse``.
    aspect_ratio:
        Major/minor axis ratio, or ``"adaptive"``.
    angle_deg:
        Counter-clockwise angle from horizontal, or ``"adaptive"``.
    enabled:
        Whether the shape is analysed.
    """

    name: str
    kind: str
    aspect_ratio: NumericOrAdaptive = 1.0
    angle_deg: NumericOrAdaptive = 0.0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShapeGeometry:
    """Resolved discrete geometry for one scale and shape."""

    name: str
    kind: str
    base_size_px: int
    aspect_ratio: float
    angle_deg: float
    major_axis_px: float
    minor_axis_px: float
    bbox_height_px: int
    bbox_width_px: int
    target_area_px: float
    actual_area_px: int
    area_error_fraction: float
    support_mask: np.ndarray

    @property
    def characteristic_size_px(self) -> float:
        return float(math.sqrt(max(self.actual_area_px, 1)))


def resolve_shape_spec(
    spec: ShapeSpec,
    adaptive_angle_deg: float = 0.0,
    adaptive_aspect_ratio: float = 1.0,
) -> tuple[float, float]:
    """Resolve adaptive fields into finite numeric values."""

    if str(spec.aspect_ratio).lower() == "adaptive":
        aspect = float(adaptive_aspect_ratio)
    else:
        aspect = float(spec.aspect_ratio)
    if str(spec.angle_deg).lower() == "adaptive":
        angle = float(adaptive_angle_deg)
    else:
        angle = float(spec.angle_deg)

    aspect = float(np.clip(aspect, 1.0, 4.0))
    angle = float(angle % 180.0)
    return aspect, angle


def _continuous_axes(base_size_px: int, kind: str, aspect_ratio: float) -> tuple[float, float]:
    """Return continuous major/minor full-axis lengths for equal area."""

    base = float(base_size_px)
    ratio = max(float(aspect_ratio), 1.0)
    target_area = base * base
    kind = kind.lower()

    if kind == "rectangle":
        major = base * math.sqrt(ratio)
        minor = base / math.sqrt(ratio)
    elif kind == "ellipse":
        # pi * (major/2) * (minor/2) = base^2 and major/minor = ratio
        major = 2.0 * base * math.sqrt(ratio / math.pi)
        minor = 2.0 * base / math.sqrt(math.pi * ratio)
    else:
        raise ValueError(f"Unsupported shape kind: {kind!r}")

    if major * minor <= EPS or target_area <= EPS:
        raise ValueError("Degenerate support geometry.")
    return major, minor


def _rasterize(
    kind: str,
    major: float,
    minor: float,
    angle_deg: float,
    target_pixel_count: int,
) -> np.ndarray:
    """Rasterize a centered rotated shape with an exact pixel-count target.

    Pixels are ranked by a normalized implicit-shape score and the closest
    ``target_pixel_count`` pixels are retained. This removes the severe
    equal-area error that ordinary binary rasterization can produce for small
    45-degree supports.
    """

    theta = math.radians(float(angle_deg))
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    # Extra margin guarantees enough candidate pixels after rotation.
    bbox_w = max(5, int(math.ceil(c * major + s * minor)) + 6)
    bbox_h = max(5, int(math.ceil(s * major + c * minor)) + 6)

    yy, xx = np.mgrid[0:bbox_h, 0:bbox_w]
    x = xx + 0.5 - bbox_w / 2.0
    y = yy + 0.5 - bbox_h / 2.0
    ct, st = math.cos(theta), math.sin(theta)
    u = x * ct + y * st
    v = -x * st + y * ct

    if kind.lower() == "rectangle":
        score = np.maximum(
            np.abs(u) / max(major / 2.0, EPS),
            np.abs(v) / max(minor / 2.0, EPS),
        )
    elif kind.lower() == "ellipse":
        score = (u / max(major / 2.0, EPS)) ** 2 + (v / max(minor / 2.0, EPS)) ** 2
    else:
        raise ValueError(f"Unsupported shape kind: {kind!r}")

    n = int(max(1, target_pixel_count))
    if score.size < n:
        raise ValueError("Rasterization grid is smaller than target support area.")
    # Deterministic tie-breaking preserves symmetry as much as the pixel grid allows.
    tie_break = 1.0e-10 * (x * x + y * y) + 1.0e-12 * (yy * bbox_w + xx)
    ranked_score = score + tie_break
    selected = np.argpartition(ranked_score.ravel(), n - 1)[:n]
    inside = np.zeros(score.size, dtype=bool)
    inside[selected] = True
    inside = inside.reshape(score.shape)

    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    return inside[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def make_shape_geometry(
    base_size_px: int,
    spec: ShapeSpec,
    adaptive_angle_deg: float = 0.0,
    adaptive_aspect_ratio: float = 1.0,
    max_area_error_fraction: float = 0.035,
) -> ShapeGeometry:
    """Create an approximately equal-area discrete support mask.

    A short area-correction loop rescales the continuous axes so that the
    discrete support area is close to ``base_size_px**2``. For very small
    supports exact equality is impossible; the achieved error is recorded.
    """

    base_size_px = int(base_size_px)
    if base_size_px < 3:
        raise ValueError("base_size_px must be at least 3.")
    if not spec.enabled:
        raise ValueError(f"Shape {spec.name!r} is disabled.")

    aspect, angle = resolve_shape_spec(spec, adaptive_angle_deg, adaptive_aspect_ratio)
    major, minor = _continuous_axes(base_size_px, spec.kind, aspect)
    target = float(base_size_px * base_size_px)

    target_count = int(round(target))
    support = _rasterize(spec.kind, major, minor, angle, target_count)
    actual_int = int(support.sum())
    return ShapeGeometry(
        name=str(spec.name),
        kind=str(spec.kind).lower(),
        base_size_px=base_size_px,
        aspect_ratio=float(aspect),
        angle_deg=float(angle),
        major_axis_px=float(major),
        minor_axis_px=float(minor),
        bbox_height_px=int(support.shape[0]),
        bbox_width_px=int(support.shape[1]),
        target_area_px=target,
        actual_area_px=actual_int,
        area_error_fraction=float(abs(actual_int - target) / target),
        support_mask=support.astype(bool),
    )


def shape_specs_to_dicts(specs: list[ShapeSpec]) -> list[Dict[str, Any]]:
    return [spec.to_dict() for spec in specs]


def geometry_scale_robustness(reu_scales):
    """Return geometry dispersion diagnostics for finite representative scales.

    Returns success-independent dispersion measures. The success fraction is
    calculated in the scaling module because it requires the full decision set.
    """
    values = np.asarray(list(reu_scales), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "cv": np.nan, "dispersion_robustness": np.nan}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    cv = float(std / max(abs(mean), EPS))
    return {
        "n": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "cv": cv,
        "dispersion_robustness": float(np.clip(1.0 - cv, 0.0, 1.0)),
    }



def geometry_bbox_fits_region(
    base_size_px: int,
    spec: ShapeSpec,
    region_shape: tuple[int, int],
    adaptive_angle_deg: float = 0.0,
    adaptive_aspect_ratio: float = 1.0,
) -> bool:
    """Return whether the resolved support bounding box can fit in the region bbox.

    This is a geometric feasibility diagnostic only. Actual feasibility still
    depends on the irregular region mask and ``min_region_coverage``.
    """
    geom = make_shape_geometry(
        base_size_px,
        spec,
        adaptive_angle_deg=adaptive_angle_deg,
        adaptive_aspect_ratio=adaptive_aspect_ratio,
    )
    h, w = map(int, region_shape)
    return bool(geom.bbox_height_px <= h and geom.bbox_width_px <= w)
