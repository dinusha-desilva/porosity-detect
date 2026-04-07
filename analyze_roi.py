"""
Porosity analysis with user-defined ROI from ImageJ.

Workflow:
    1. Open micrograph in ImageJ/FIJI
    2. Draw ROI (rectangle, freehand, polygon — any shape)
    3. Export ROI as binary mask:
         Edit → Selection → Create Mask
         File → Save As → PNG (save as mask.png)
       OR save ROI coordinates:
         Analyze → Tools → ROI Manager → More → Save
    4. Run this script:
         python analyze_roi.py micrograph.tif mask.png
         python analyze_roi.py micrograph.tif mask.png --pixel-size 0.65

The mask should be a binary image where:
    WHITE (255) = region to analyze
    BLACK (0)   = region to ignore

Output:
    - Porosity percentage within the ROI
    - Void count, size statistics
    - Detection overlay image
    - JSON results file

Usage:
    python analyze_roi.py <image> <mask> [options]

    Options:
        --pixel-size    Micrometers per pixel (default: 1.0)
        --threshold     Intensity threshold for void detection (default: auto)
        --min-area      Minimum void area in pixels (default: 8)
        --output        Output directory (default: ./roi_results)
        --no-viz        Skip visualization output
"""

import argparse
import json
import sys
import os
import time
import numpy as np


def load_image(path):
    """Load image as grayscale float [0, 1]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

    img = plt.imread(path)
    if img.ndim == 3:
        gray = np.mean(img[..., :3], axis=2)
    else:
        gray = img.copy()
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray.astype(np.float64), img


def load_mask(path, target_shape):
    """Load ROI mask from ImageJ export.

    Handles:
        - Binary PNG (white = ROI)
        - Grayscale with threshold at 128
        - Inverted masks (auto-detect)
        - Size mismatch (resize to match image)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isfile(path):
        print(f"Error: Mask not found: {path}")
        sys.exit(1)

    mask_img = plt.imread(path)

    # Convert to single channel
    if mask_img.ndim == 3:
        mask_gray = np.mean(mask_img[..., :3], axis=2)
    else:
        mask_gray = mask_img.copy()

    # Normalize to [0, 1]
    if mask_gray.max() > 1.0:
        mask_gray = mask_gray / 255.0

    # Binarize at 0.5
    mask = mask_gray > 0.5

    # Auto-detect inverted mask
    # If mask covers >90% of image, it's probably inverted
    coverage = mask.sum() / mask.size
    if coverage > 0.9:
        print("  Note: Mask covers >90% of image — assuming inverted, flipping.")
        mask = ~mask

    # Handle size mismatch
    if mask.shape != target_shape:
        print(f"  Resizing mask from {mask.shape} to {target_shape}")
        from scipy.ndimage import zoom

        zoom_y = target_shape[0] / mask.shape[0]
        zoom_x = target_shape[1] / mask.shape[1]
        mask = zoom(mask.astype(float), (zoom_y, zoom_x), order=0) > 0.5

    return mask


def auto_threshold(gray, roi_mask):
    """Automatically determine void intensity threshold.

    Uses Otsu's method within the ROI, then takes the lower
    portion of the bimodal distribution as the void threshold.
    """
    roi_pixels = gray[roi_mask]

    # Histogram within ROI
    hist, bin_edges = np.histogram(roi_pixels, bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu threshold
    total = hist.sum()
    best_thresh = 0
    best_var = 0
    w0 = 0
    sum0 = 0
    total_sum = np.sum(bin_centers * hist)

    for i in range(256):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += bin_centers[i] * hist[i]
        m0 = sum0 / w0
        m1 = (total_sum - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = bin_centers[i]

    # For voids, we want things darker than the Otsu threshold
    # Use a fraction of Otsu to be conservative
    void_thresh = best_thresh * 0.6

    # But also check percentiles — voids should be in the darkest tail
    p5 = np.percentile(roi_pixels, 5)
    p15 = np.percentile(roi_pixels, 15)

    # Use the more conservative of the two approaches
    threshold = min(void_thresh, p15)

    return float(threshold)


def detect_voids(gray, roi_mask, threshold, min_area=8, max_area=50000):
    """Detect voids within the ROI using intensity thresholding.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image [0, 1].
    roi_mask : np.ndarray
        Binary ROI mask (True = analyze).
    threshold : float
        Intensity threshold. Pixels darker than this are void candidates.
    min_area : int
        Minimum void area in pixels.
    max_area : int
        Maximum void area in pixels.

    Returns
    -------
    labels : np.ndarray
        Label image of confirmed voids.
    voids : list of dict
        Per-void measurements.
    stats : dict
        Detection statistics.
    """
    from scipy.ndimage import (
        gaussian_filter, label, find_objects,
        binary_opening, binary_closing, binary_erosion,
        binary_dilation
    )

    h, w = gray.shape

    # Smooth
    smoothed = gaussian_filter(gray, sigma=0.8)

    # Threshold within ROI only
    binary = (smoothed < threshold) & roi_mask

    # Morphological cleanup
    struct = np.ones((2, 2), dtype=bool)
    cleaned = binary_opening(binary, structure=struct, iterations=1)
    cleaned = binary_closing(cleaned, structure=struct, iterations=1)

    # Label connected components
    labeled, n_features = label(cleaned)
    regions = find_objects(labeled)

    # Filter and measure
    final_labels = np.zeros_like(labeled)
    voids = []
    vid = 0

    for i, slc in enumerate(regions):
        if slc is None:
            continue

        region = labeled[slc] == (i + 1)
        area = region.sum()

        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(region)
        global_ys = ys + slc[0].start
        global_xs = xs + slc[1].start

        # Verify all pixels are within ROI
        if not roi_mask[global_ys, global_xs].all():
            # Trim to ROI
            in_roi = roi_mask[global_ys, global_xs]
            if in_roi.sum() < min_area:
                continue
            area = int(in_roi.sum())

        # Shape metrics
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        eroded = binary_erosion(region)
        boundary = region & ~eroded
        perim = max(boundary.sum(), 1)
        circ = min(4 * np.pi * area / (perim ** 2), 1.0)
        aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
        eq_diam = np.sqrt(4 * area / np.pi)

        # Intensity
        region_int = smoothed[global_ys, global_xs]
        mean_int = float(np.mean(region_int))
        std_int = float(np.std(region_int))

        # Boundary contrast
        dilated = binary_dilation(region, iterations=3)
        ring = dilated & ~region
        ring_ys, ring_xs = np.where(ring)
        ring_gy = np.clip(ring_ys + slc[0].start, 0, h - 1)
        ring_gx = np.clip(ring_xs + slc[1].start, 0, w - 1)
        valid = roi_mask[ring_gy, ring_gx]
        if valid.sum() > 0:
            contrast = float(np.mean(smoothed[ring_gy[valid], ring_gx[valid]])) - mean_int
        else:
            contrast = 0.0

        vid += 1
        final_labels[slc][region] = vid

        voids.append({
            "id": vid,
            "area_px": int(area),
            "circularity": round(float(circ), 4),
            "aspect_ratio": round(float(aspect), 3),
            "eq_diameter_px": round(float(eq_diam), 2),
            "mean_intensity": round(float(mean_int), 4),
            "std_intensity": round(float(std_int), 4),
            "boundary_contrast": round(float(contrast), 4),
            "centroid_x": round(float(np.mean(global_xs)), 1),
            "centroid_y": round(float(np.mean(global_ys)), 1),
        })

    roi_area = int(roi_mask.sum())
    total_void_area = sum(v["area_px"] for v in voids)
    porosity = total_void_area / roi_area if roi_area > 0 else 0.0

    stats = {
        "roi_area_px": roi_area,
        "total_void_area_px": total_void_area,
        "porosity_fraction": round(porosity, 8),
        "porosity_percent": round(porosity * 100, 4),
        "void_count": len(voids),
        "threshold_used": round(threshold, 4),
    }

    if voids:
        areas = [v["area_px"] for v in voids]
        stats["mean_void_area_px"] = round(float(np.mean(areas)), 2)
        stats["std_void_area_px"] = round(float(np.std(areas)), 2)
        stats["max_void_area_px"] = int(np.max(areas))
        stats["min_void_area_px"] = int(np.min(areas))
        stats["mean_circularity"] = round(float(np.mean([v["circularity"] for v in voids])), 4)
        stats["mean_eq_diameter_px"] = round(float(np.mean([v["eq_diameter_px"] for v in voids])), 2)

    return final_labels, voids, stats


def save_visualization(gray, roi_mask, labels, voids, stats, output_path, pixel_size=1.0):
    """Save analysis visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    h, w = gray.shape
    porosity = stats["porosity_percent"]
    n_voids = stats["void_count"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Porosity Analysis (ROI) — {porosity:.3f}% porosity, {n_voids} voids",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: Original with ROI outline
    from scipy.ndimage import binary_dilation
    roi_outline = binary_dilation(roi_mask, iterations=1) & ~roi_mask
    overlay_roi = np.stack([gray, gray, gray], axis=-1).copy()
    overlay_roi[roi_outline] = [0, 0.9, 0]
    overlay_roi[~roi_mask] *= 0.4  # dim outside ROI
    axes[0].imshow(overlay_roi)
    axes[0].set_title("Image + ROI (green outline)")
    axes[0].axis("off")

    # Panel 2: Detection overlay within ROI
    overlay = np.stack([gray, gray, gray], axis=-1).copy()
    overlay[~roi_mask] *= 0.4  # dim outside ROI
    overlay[labels > 0] = [0.9, 0.15, 0.15]
    for v in voids:
        c = Circle(
            (v["centroid_x"], v["centroid_y"]),
            radius=max(v["eq_diameter_px"] / 2 + 3, 5),
            fill=False, edgecolor="red", linewidth=1.5,
        )
        axes[1].add_patch(c)
    axes[1].imshow(overlay)
    axes[1].set_title(f"Detected voids: {n_voids}")
    axes[1].axis("off")

    # Panel 3: Void size distribution
    if voids:
        areas = [v["area_px"] for v in voids]
        if pixel_size != 1.0:
            areas_um2 = [a * pixel_size ** 2 for a in areas]
            axes[2].hist(areas_um2, bins=max(10, len(areas) // 3),
                         color="#c44e52", edgecolor="black", alpha=0.8)
            axes[2].set_xlabel("Void area (µm²)")
            axes[2].axvline(np.mean(areas_um2), color="black", linestyle="--",
                            label=f"Mean={np.mean(areas_um2):.1f} µm²")
        else:
            axes[2].hist(areas, bins=max(10, len(areas) // 3),
                         color="#c44e52", edgecolor="black", alpha=0.8)
            axes[2].set_xlabel("Void area (px²)")
            axes[2].axvline(np.mean(areas), color="black", linestyle="--",
                            label=f"Mean={np.mean(areas):.0f} px²")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Void size distribution")
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "No voids detected", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title("Void size distribution")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        prog="analyze_roi",
        description="Porosity analysis within ImageJ-defined ROI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ImageJ ROI Export Instructions:
  1. Open image in ImageJ/FIJI
  2. Draw ROI (rectangle, freehand, polygon)
  3. Edit → Selection → Create Mask
  4. File → Save As → PNG (save as mask.png)
  5. Run: python analyze_roi.py image.tif mask.png

Examples:
  python analyze_roi.py micrograph.tif roi_mask.png
  python analyze_roi.py micrograph.tif roi_mask.png --pixel-size 0.65
  python analyze_roi.py micrograph.tif roi_mask.png --threshold 0.25 --output results/
        """,
    )
    parser.add_argument("image", help="Path to micrograph image")
    parser.add_argument("mask", nargs="?", default=None,
                        help="Path to ROI mask (binary PNG from ImageJ). If omitted, entire image is used.")
    parser.add_argument(
        "--pixel-size", type=float, default=1.0,
        help="Pixel size in micrometers (default: 1.0)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Intensity threshold for voids (default: auto-detect)",
    )
    parser.add_argument(
        "--strict", type=float, default=None,
        help="Strict threshold for void cores (Pass 1). Overrides preset.",
    )
    parser.add_argument(
        "--moderate", type=float, default=None,
        help="Moderate threshold for candidate regions (Pass 2). Overrides preset.",
    )
    parser.add_argument(
        "--min-area", type=int, default=8,
        help="Minimum void area in pixels (default: 8)",
    )
    parser.add_argument(
        "--min-contrast", type=float, default=None,
        help="Minimum boundary contrast (default: from preset)",
    )
    parser.add_argument(
        "--preset", default=None,
        choices=["composite_high_mag", "composite_low_mag", "fabric_cross_section", "am_metal", "sensitive", "conservative"],
        help="Use a material-specific preset for detection parameters",
    )
    parser.add_argument(
        "--output", "-o", default="./roi_results",
        help="Output directory (default: ./roi_results)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization output",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────
    print("=" * 60)
    print("POROSITY ANALYSIS — ImageJ ROI Mode")
    print("=" * 60)

    print(f"\nLoading image: {args.image}")
    gray, img_raw = load_image(args.image)
    h, w = gray.shape
    print(f"  Size: {w} x {h} px")

    if args.mask:
        print(f"Loading ROI mask: {args.mask}")
        roi_mask = load_mask(args.mask, gray.shape)
    else:
        print("No mask provided — using entire image as ROI")
        roi_mask = np.ones((h, w), dtype=bool)

    roi_area = roi_mask.sum()
    roi_pct = roi_area / (h * w) * 100
    print(f"  ROI area: {roi_area} px ({roi_pct:.1f}% of image)")

    if roi_area == 0:
        print("Error: ROI mask is empty. Check your mask file.")
        sys.exit(1)

    # ── Detect voids ───────────────────────────────────────────
    print("Detecting voids within ROI (two-pass + reconstruction)...")
    t0 = time.time()

    try:
        from porosity_detect.two_pass import TwoPassDetector, TwoPassParams

        if args.preset:
            detector = TwoPassDetector(preset=args.preset)
            print(f"  Preset: {args.preset}")
        elif args.threshold is not None:
            tp_params = TwoPassParams()
            tp_params.moderate_threshold = args.threshold
            tp_params.strict_threshold = args.threshold * 0.6
            detector = TwoPassDetector(params=tp_params)
        else:
            detector = TwoPassDetector(params=TwoPassParams())

        # Apply command-line overrides on top of preset/defaults
        if args.strict is not None:
            detector.params.strict_threshold = args.strict
        if args.moderate is not None:
            detector.params.moderate_threshold = args.moderate
        if args.min_contrast is not None:
            detector.params.min_contrast = args.min_contrast
        if args.min_area > detector.params.min_void_area:
            detector.params.min_void_area = args.min_area

        tp = detector.params
        print(f"  Thresholds: strict={tp.strict_threshold}, moderate={tp.moderate_threshold}")
        print(f"  Min area: {tp.min_void_area}, min contrast: {tp.min_contrast}")

        result = detector.detect(gray, roi_mask=roi_mask)

        labels = result["labels"]
        voids = result["voids"]
        stats = {
            "roi_area_px": result["roi_area_px"],
            "total_void_area_px": result["total_void_area_px"],
            "porosity_fraction": result["porosity_fraction"],
            "porosity_percent": result["porosity_pct"],
            "void_count": result["void_count"],
            "threshold_used": f"strict={tp.strict_threshold}, moderate={tp.moderate_threshold}",
            "method": "two_pass_reconstruction",
        }
        if voids:
            areas = [v["area_px"] for v in voids]
            stats["mean_void_area_px"] = round(float(np.mean(areas)), 2)
            stats["std_void_area_px"] = round(float(np.std(areas)), 2)
            stats["max_void_area_px"] = int(np.max(areas))
            stats["min_void_area_px"] = int(np.min(areas))
            stats["mean_circularity"] = round(float(np.mean([v["circularity"] for v in voids])), 4)
            stats["mean_eq_diameter_px"] = round(float(np.mean([v["eq_diameter_px"] for v in voids])), 2)

    except ImportError:
        # Fallback to old single-threshold method
        print("  (Falling back to single-threshold method)")
        if args.threshold is not None:
            threshold = args.threshold
        else:
            threshold = auto_threshold(gray, roi_mask)
        labels, voids, stats = detect_voids(
            gray, roi_mask, threshold, min_area=args.min_area,
        )

    elapsed = time.time() - t0

    # Add physical units if pixel size provided
    px = args.pixel_size
    if px != 1.0:
        stats["pixel_size_um"] = px
        stats["roi_area_um2"] = round(roi_area * px ** 2, 2)
        stats["total_void_area_um2"] = round(stats["total_void_area_px"] * px ** 2, 2)
        if voids:
            stats["mean_void_area_um2"] = round(stats["mean_void_area_px"] * px ** 2, 2)
            stats["max_void_area_um2"] = round(stats["max_void_area_px"] * px ** 2, 2)
            stats["mean_eq_diameter_um"] = round(stats["mean_eq_diameter_px"] * px, 2)
            for v in voids:
                v["area_um2"] = round(v["area_px"] * px ** 2, 2)
                v["eq_diameter_um"] = round(v["eq_diameter_px"] * px, 2)

    # ── Print results ──────────────────────────────────────────
    print(f"\nAnalysis complete in {elapsed:.2f}s")
    print(f"\n{'─' * 40}")
    print(f"  POROSITY:  {stats['porosity_percent']:.4f}%")
    print(f"  VOIDS:     {stats['void_count']}")
    print(f"{'─' * 40}")

    if voids:
        print(f"\n  ROI area:          {stats['roi_area_px']} px²", end="")
        if px != 1.0:
            print(f" ({stats['roi_area_um2']:.0f} µm²)")
        else:
            print()
        print(f"  Total void area:   {stats['total_void_area_px']} px²", end="")
        if px != 1.0:
            print(f" ({stats['total_void_area_um2']:.0f} µm²)")
        else:
            print()
        print(f"  Mean void area:    {stats['mean_void_area_px']:.1f} px²", end="")
        if px != 1.0:
            print(f" ({stats['mean_void_area_um2']:.1f} µm²)")
        else:
            print()
        print(f"  Max void area:     {stats['max_void_area_px']} px²", end="")
        if px != 1.0:
            print(f" ({stats['max_void_area_um2']:.0f} µm²)")
        else:
            print()
        print(f"  Mean circularity:  {stats['mean_circularity']:.3f}")
        print(f"  Mean eq. diameter: {stats['mean_eq_diameter_px']:.1f} px", end="")
        if px != 1.0:
            print(f" ({stats['mean_eq_diameter_um']:.1f} µm)")
        else:
            print()

    # ── Save outputs ───────────────────────────────────────────
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Get the base name of the image for naming outputs
    image_basename = os.path.splitext(os.path.basename(args.image))[0]

    # 1. Full-resolution overlay image (same dimensions as input image)
    overlay_full = np.stack([gray, gray, gray], axis=-1).copy()
    overlay_full[labels > 0] = [0.9, 0.15, 0.15]  # red voids
    # Draw ROI outline in green
    from scipy.ndimage import binary_dilation as bd_
    roi_outline_ = bd_(roi_mask, iterations=2) & ~roi_mask
    overlay_full[roi_outline_] = [0, 0.9, 0]
    # Dim outside ROI
    overlay_full[~roi_mask] *= 0.4

    overlay_path = os.path.join(output_dir, f"{image_basename}_void_overlay.png")
    from PIL import Image as PILImage
    overlay_uint8 = (np.clip(overlay_full, 0, 1) * 255).astype(np.uint8)
    PILImage.fromarray(overlay_uint8).save(overlay_path)
    print(f"\n  Overlay image: {overlay_path}")
    print(f"    (Same resolution as input: {w} x {h})")

    # 2. Text report file with all results
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("POROSITY-DETECT ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Image:          {args.image}")
    report_lines.append(f"ROI Mask:       {args.mask}")
    report_lines.append(f"Image size:     {w} x {h} pixels")
    report_lines.append(f"Pixel size:     {px} um/px" if px != 1.0 else "Pixel size:     Not specified")
    report_lines.append(f"Preset:         {args.preset if args.preset else 'default'}")
    report_lines.append(f"Method:         {stats.get('method', 'two_pass_reconstruction')}")
    report_lines.append(f"Analysis time:  {elapsed:.2f} seconds")
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("RESULTS")
    report_lines.append("-" * 60)
    report_lines.append(f"  Porosity:          {stats['porosity_percent']:.4f}%")
    report_lines.append(f"  Void count:        {stats['void_count']}")
    report_lines.append(f"  ROI area:          {stats['roi_area_px']} px²")
    if px != 1.0:
        report_lines.append(f"                     ({stats.get('roi_area_um2', 'N/A')} µm²)")
    report_lines.append(f"  Total void area:   {stats['total_void_area_px']} px²")
    if px != 1.0:
        report_lines.append(f"                     ({stats.get('total_void_area_um2', 'N/A')} µm²)")

    if voids:
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("VOID STATISTICS")
        report_lines.append("-" * 60)
        report_lines.append(f"  Mean void area:    {stats['mean_void_area_px']:.1f} px²")
        if px != 1.0:
            report_lines.append(f"                     ({stats.get('mean_void_area_um2', 'N/A')} µm²)")
        report_lines.append(f"  Std void area:     {stats.get('std_void_area_px', 'N/A')} px²")
        report_lines.append(f"  Max void area:     {stats['max_void_area_px']} px²")
        if px != 1.0:
            report_lines.append(f"                     ({stats.get('max_void_area_um2', 'N/A')} µm²)")
        report_lines.append(f"  Min void area:     {stats['min_void_area_px']} px²")
        report_lines.append(f"  Mean circularity:  {stats['mean_circularity']:.4f}")
        report_lines.append(f"  Mean eq diameter:  {stats['mean_eq_diameter_px']:.1f} px")
        if px != 1.0:
            report_lines.append(f"                     ({stats.get('mean_eq_diameter_um', 'N/A')} µm)")

        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("INDIVIDUAL VOIDS (sorted by area, largest first)")
        report_lines.append("-" * 60)
        report_lines.append(f"  {'ID':>4s}  {'Area(px²)':>10s}  {'Circ':>6s}  {'Aspect':>7s}  {'EqDiam(px)':>11s}  {'DarkFrac':>9s}  {'Contrast':>9s}  {'CentX':>7s}  {'CentY':>7s}")
        for v in sorted(voids, key=lambda x: x["area_px"], reverse=True):
            report_lines.append(
                f"  {v['id']:>4d}  {v['area_px']:>10d}  {v.get('circularity', 0):>6.3f}  "
                f"{v.get('aspect_ratio', 0):>7.2f}  {v['eq_diameter_px']:>11.1f}  "
                f"{v.get('dark_fraction', 0):>9.4f}  {v.get('boundary_contrast', 0):>9.4f}  "
                f"{v['centroid_x']:>7.1f}  {v['centroid_y']:>7.1f}"
            )

    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("DETECTION PARAMETERS")
    report_lines.append("-" * 60)
    if "threshold_used" in stats:
        report_lines.append(f"  Threshold:         {stats['threshold_used']}")
    for key in ["strict_threshold", "moderate_threshold", "gaussian_sigma", "min_void_area", "min_contrast", "min_dark_fraction"]:
        # Try to get from nested params dict
        pass

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 60)

    report_path = os.path.join(output_dir, f"{image_basename}_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Text report:   {report_path}")

    # 3. JSON results (existing)
    results = {
        "image": args.image,
        "mask": args.mask,
        "image_shape": [h, w],
        "pixel_size_um": px,
        "stats": stats,
        "voids": voids,
    }
    json_path = os.path.join(output_dir, f"{image_basename}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON results:  {json_path}")

    # 4. Matplotlib visualization (existing, optional)
    if not args.no_viz:
        viz_path = os.path.join(output_dir, f"{image_basename}_analysis_plot.png")
        save_visualization(gray, roi_mask, labels, voids, stats, viz_path, px)
        print(f"  Analysis plot:  {viz_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
