"""
Two-pass void detection with morphological reconstruction.

This module implements the core innovation of porosity-detect: a two-pass
approach that eliminates the shadow artifacts common in single-threshold
void detection on optical micrographs.

The Problem:
    Single-threshold void detection faces a tradeoff:
    - Low threshold: misses lighter-edged voids (elongated resin-starved regions)
    - High threshold: captures shadows adjacent to voids as false positives
    
    These shadows arise from polishing relief, resin-rich pockets near voids,
    and optical artifacts at void boundaries. They are a consistent source of
    inter-operator variability in manual analysis.

The Solution:
    Pass 1 (strict threshold): Identifies definite void cores — the darkest
    pixels that are unambiguously void interior. These serve as seeds.
    
    Pass 2 (moderate threshold): Identifies candidate regions — everything
    that *might* be a void, including shadows and lighter void edges.
    
    Morphological Reconstruction: Keeps only candidate regions that are
    physically connected to a void core. Disconnected shadow patches are
    automatically eliminated because they form separate connected components
    from the true void regions at the moderate threshold level.

    This encodes the domain knowledge that real voids have dark cores while
    shadows do not — without requiring ML training data.

Usage:
    from porosity_detect.two_pass import TwoPassDetector
    
    detector = TwoPassDetector()
    results = detector.detect(gray_image)
    print(f"Porosity: {results['porosity_pct']:.3f}%")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class TwoPassParams:
    """Parameters for two-pass void detection.

    Defaults are tuned for high-magnification optical micrographs
    of carbon fiber/epoxy composite cross-sections.

    Attributes
    ----------
    strict_threshold : float
        Intensity threshold for definite void cores (Pass 1).
        Only the darkest pixels pass this. Default 0.15.
    moderate_threshold : float
        Intensity threshold for candidate regions (Pass 2).
        Includes void edges and some shadows. Default 0.25.
    gaussian_sigma : float
        Gaussian smoothing sigma before thresholding.
    min_void_area : int
        Minimum void area in pixels to retain.
    max_void_area : int
        Maximum void area in pixels.
    min_contrast : float
        Minimum boundary contrast (void vs. surrounding matrix).
    min_dark_fraction : float
        Minimum fraction of void pixels below strict threshold.
        Regions with no dark core are rejected as shadows.
    morph_open_iter : int
        Morphological opening iterations for noise removal.
    morph_close_iter : int
        Morphological closing iterations for gap filling.
    """

    strict_threshold: float = 0.15
    moderate_threshold: float = 0.25
    gaussian_sigma: float = 1.2
    min_void_area: int = 20
    max_void_area: int = 200000
    min_contrast: float = 0.06
    min_dark_fraction: float = 0.05
    morph_open_iter: int = 1
    morph_close_iter: int = 1


# ── Material presets ──────────────────────────────────────────

PRESETS = {
    "composite_high_mag": TwoPassParams(
        strict_threshold=0.15,
        moderate_threshold=0.25,
        gaussian_sigma=1.2,
        min_void_area=20,
        min_contrast=0.06,
    ),
    "composite_low_mag": TwoPassParams(
        strict_threshold=0.18,
        moderate_threshold=0.30,
        gaussian_sigma=1.0,
        min_void_area=10,
        min_contrast=0.05,
    ),
    "fabric_cross_section": TwoPassParams(
        strict_threshold=0.25,
        moderate_threshold=0.40,
        gaussian_sigma=0.8,
        min_void_area=5,
        min_contrast=0.03,
        min_dark_fraction=0.02,
    ),
    "am_metal": TwoPassParams(
        strict_threshold=0.12,
        moderate_threshold=0.22,
        gaussian_sigma=1.0,
        min_void_area=15,
        min_contrast=0.08,
    ),
    "sensitive": TwoPassParams(
        strict_threshold=0.22,
        moderate_threshold=0.35,
        gaussian_sigma=0.8,
        min_void_area=5,
        min_contrast=0.03,
        min_dark_fraction=0.02,
    ),
    "conservative": TwoPassParams(
        strict_threshold=0.12,
        moderate_threshold=0.20,
        gaussian_sigma=1.2,
        min_void_area=30,
        min_contrast=0.08,
        min_dark_fraction=0.10,
    ),
}


class TwoPassDetector:
    """Two-pass void detector with morphological reconstruction.

    Parameters
    ----------
    params : TwoPassParams, optional
        Detection parameters. If None, uses defaults for high-mag composites.
    preset : str, optional
        Use a named preset: 'composite_high_mag', 'composite_low_mag',
        'am_metal', 'sensitive', 'conservative'.
    """

    def __init__(
        self,
        params: Optional[TwoPassParams] = None,
        preset: Optional[str] = None,
    ):
        if preset is not None:
            if preset not in PRESETS:
                available = ", ".join(PRESETS.keys())
                raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
            self.params = PRESETS[preset]
        elif params is not None:
            self.params = params
        else:
            self.params = TwoPassParams()

    def detect(
        self,
        gray: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run two-pass void detection on a grayscale image.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image in [0, 1] range.
        roi_mask : np.ndarray, optional
            Binary ROI mask (True = analyze). If None, entire image is used.

        Returns
        -------
        dict with keys:
            labels : np.ndarray — integer label map of detected voids
            voids : list of dict — per-void measurements
            porosity_pct : float — porosity as percentage of ROI area
            porosity_fraction : float — porosity as fraction
            void_count : int
            roi_area_px : int
            total_void_area_px : int
            params : dict — parameters used
        """
        from scipy.ndimage import (
            gaussian_filter, label, binary_opening, binary_closing,
            binary_dilation, binary_erosion, find_objects,
        )

        p = self.params
        h, w = gray.shape

        if roi_mask is None:
            roi_mask = np.ones((h, w), dtype=bool)

        smoothed = gaussian_filter(gray, sigma=p.gaussian_sigma)
        struct = np.ones((3, 3), dtype=bool)
        struct_small = np.ones((2, 2), dtype=bool)

        # ── Pass 1: Strict cores ──
        cores = (smoothed < p.strict_threshold) & roi_mask
        cores = binary_opening(cores, structure=struct, iterations=p.morph_open_iter)

        # ── Pass 2: Moderate candidates ──
        candidates = (smoothed < p.moderate_threshold) & roi_mask
        candidates = binary_opening(candidates, structure=struct_small, iterations=1)
        candidates = binary_closing(candidates, structure=struct, iterations=p.morph_close_iter)

        # ── Morphological reconstruction ──
        # Label candidate regions
        cand_labeled, n_cand = label(candidates)

        # Find which candidate labels contain at least one core pixel
        core_labels = set(np.unique(cand_labeled[cores])) - {0}

        # Keep only connected-to-core candidates
        reconstructed = np.zeros_like(candidates)
        for lbl in core_labels:
            reconstructed[cand_labeled == lbl] = True

        reconstructed = binary_opening(reconstructed, structure=struct, iterations=1)

        # ── Label and filter ──
        labeled_img, n_feat = label(reconstructed)
        regions = find_objects(labeled_img)

        voids = []
        final_labels = np.zeros_like(labeled_img)
        vid = 0

        for i, slc in enumerate(regions):
            if slc is None:
                continue

            region = labeled_img[slc] == (i + 1)
            area = int(region.sum())

            if area < p.min_void_area or area > p.max_void_area:
                continue

            ys, xs = np.where(region)
            gy = ys + slc[0].start
            gx = xs + slc[1].start

            # Verify within ROI
            if not roi_mask[gy, gx].all():
                in_roi = roi_mask[gy, gx]
                area = int(in_roi.sum())
                if area < p.min_void_area:
                    continue

            # Intensity
            mean_int = float(np.mean(smoothed[gy, gx]))
            dark_fraction = float(np.mean(smoothed[gy, gx] < p.strict_threshold))

            # Boundary contrast
            dilated = binary_dilation(region, iterations=5)
            ring = dilated & ~region
            ry, rx = np.where(ring)
            rgy = np.clip(ry + slc[0].start, 0, h - 1)
            rgx = np.clip(rx + slc[1].start, 0, w - 1)
            valid_ring = roi_mask[rgy, rgx]

            if valid_ring.sum() > 0:
                contrast = float(
                    np.mean(smoothed[rgy[valid_ring], rgx[valid_ring]])
                ) - mean_int
            else:
                contrast = 0.0

            # Filter: must have contrast and dark core
            if contrast < p.min_contrast:
                continue
            if dark_fraction < p.min_dark_fraction and mean_int > p.strict_threshold:
                continue

            # Shape metrics
            bbox_h = ys.max() - ys.min() + 1
            bbox_w = xs.max() - xs.min() + 1
            eroded = binary_erosion(region)
            boundary = region & ~eroded
            perim = max(boundary.sum(), 1)
            circ = min(4 * np.pi * area / (perim ** 2), 1.0)
            aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
            eq_diam = np.sqrt(4 * area / np.pi)

            vid += 1
            final_labels[slc][region] = vid

            voids.append({
                "id": vid,
                "area_px": area,
                "circularity": round(float(circ), 4),
                "aspect_ratio": round(float(aspect), 3),
                "eq_diameter_px": round(float(eq_diam), 2),
                "mean_intensity": round(float(mean_int), 4),
                "dark_fraction": round(float(dark_fraction), 4),
                "boundary_contrast": round(float(contrast), 4),
                "centroid_x": round(float(np.mean(gx)), 1),
                "centroid_y": round(float(np.mean(gy)), 1),
                "bbox": [int(slc[1].start), int(slc[0].start), int(bbox_w), int(bbox_h)],
            })

        roi_area = int(roi_mask.sum())
        total_void_area = sum(v["area_px"] for v in voids)
        porosity_frac = total_void_area / roi_area if roi_area > 0 else 0.0

        return {
            "labels": final_labels,
            "voids": voids,
            "porosity_pct": round(porosity_frac * 100, 4),
            "porosity_fraction": round(porosity_frac, 8),
            "void_count": len(voids),
            "roi_area_px": roi_area,
            "total_void_area_px": total_void_area,
            "n_candidates": n_cand,
            "n_core_regions": len(core_labels),
            "params": {
                "strict_threshold": p.strict_threshold,
                "moderate_threshold": p.moderate_threshold,
                "gaussian_sigma": p.gaussian_sigma,
                "min_void_area": p.min_void_area,
                "min_contrast": p.min_contrast,
                "min_dark_fraction": p.min_dark_fraction,
            },
        }
