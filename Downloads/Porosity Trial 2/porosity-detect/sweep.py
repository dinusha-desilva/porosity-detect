"""
Threshold sweep tool for calibrating porosity-detect on your images.

Run this on your local machine with full-resolution images to find
the thresholds that match your ImageJ measurement.

Usage:
    python sweep.py image.jpg mask.png
    python sweep.py image.tif mask.png --target 3.24

It will test many threshold combinations and show which ones
produce porosity closest to your ImageJ target value.
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for porosity-detect calibration")
    parser.add_argument("image", help="Path to micrograph image")
    parser.add_argument("mask", help="Path to ROI mask (white=ROI)")
    parser.add_argument("--target", type=float, default=None,
                        help="Target porosity %% from ImageJ (highlights closest match)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from porosity_detect.two_pass import TwoPassDetector, TwoPassParams

    # Load image
    img = plt.imread(args.image)
    if img.ndim == 2:
        gray = img.copy()
    else:
        gray = np.mean(img[:, :, :3], axis=2)
    if gray.max() > 1.0:
        gray = gray / 255.0
    h, w = gray.shape

    # Load mask
    mask_img = plt.imread(args.mask)
    if mask_img.ndim == 3:
        mask_gray = np.mean(mask_img[:, :, :3], axis=2)
    else:
        mask_gray = mask_img.copy()
    if mask_gray.max() > 1.0:
        mask_gray = mask_gray / 255.0
    roi = mask_gray > 0.5
    if roi.sum() < roi.size * 0.05:
        roi = mask_gray < 0.5  # auto-invert if needed

    print(f"Image: {w} x {h}")
    print(f"ROI:   {roi.sum()} pixels ({roi.sum()/(h*w)*100:.1f}%)")
    if args.target:
        print(f"Target: {args.target:.3f}% (from ImageJ)")
    print()

    # Sweep
    configs = []
    for strict in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        for moderate in [strict + 0.08, strict + 0.10, strict + 0.12, strict + 0.15, strict + 0.18]:
            if moderate > 0.50:
                continue
            for sigma in [0.5, 0.8, 1.0]:
                for min_a in [3, 5, 10]:
                    for min_c in [0.03, 0.05, 0.08]:
                        configs.append((strict, moderate, sigma, min_a, min_c))

    print(f"Testing {len(configs)} configurations...")
    print()

    results = []
    for i, (strict, moderate, sigma, min_a, min_c) in enumerate(configs):
        params = TwoPassParams(
            strict_threshold=strict, moderate_threshold=moderate,
            gaussian_sigma=sigma, min_void_area=min_a,
            min_contrast=min_c, min_dark_fraction=0.02
        )
        det = TwoPassDetector(params=params)
        res = det.detect(gray, roi_mask=roi)
        results.append((strict, moderate, sigma, min_a, min_c, res["void_count"], res["porosity_pct"]))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(configs)} done...")

    # Sort by closeness to target (or by porosity)
    if args.target:
        results.sort(key=lambda x: abs(x[6] - args.target))
        print(f"\nTOP 20 CLOSEST TO {args.target:.3f}%:")
    else:
        results.sort(key=lambda x: x[6])
        print(f"\nALL RESULTS (sorted by porosity):")

    print(f"{'Strict':>7s} {'Moderate':>9s} {'Sigma':>6s} {'MinA':>5s} {'MinC':>5s} {'Voids':>6s} {'Porosity':>10s}  {'Diff':>8s}")
    print("-" * 70)

    shown = results[:20] if args.target else results
    for strict, moderate, sigma, min_a, min_c, voids, pct in shown:
        diff = f"{pct - args.target:+.3f}%" if args.target else ""
        print(f"{strict:>7.2f} {moderate:>9.2f} {sigma:>6.1f} {min_a:>5d} {min_c:>5.2f} {voids:>6d} {pct:>9.4f}%  {diff:>8s}")

    # Show the best match
    if args.target:
        best = results[0]
        print(f"\nBEST MATCH:")
        print(f"  --strict {best[0]} --moderate {best[1]} --min-area {best[3]} --min-contrast {best[4]}")
        print(f"  Porosity: {best[6]:.4f}% (target: {args.target:.3f}%, diff: {best[6]-args.target:+.4f}%)")
        print(f"\nRun with:")
        print(f"  python analyze_roi.py {args.image} {args.mask} --strict {best[0]} --moderate {best[1]} --min-area {best[3]} --min-contrast {best[4]} --output results")


if __name__ == "__main__":
    main()
