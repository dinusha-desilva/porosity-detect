"""
Command-line interface for porosity-detect.

Provides a professional CLI for running void/porosity detection on
optical microscopy images. Supports both real images and synthetic
demo mode.

Usage:
    # Analyze a real micrograph
    porosity-detect analyze image.png --output results/ --pixel-size 0.5

    # Run demo with synthetic image
    porosity-detect demo --output demo_results/

    # Validate with ground truth comparison
    porosity-detect validate --output validation/
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="porosity-detect",
        description=(
            "Hybrid Classical + ML Void/Porosity Detection for "
            "Optical Microscopy of Aerospace Materials"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  porosity-detect analyze micrograph.png
  porosity-detect analyze micrograph.tif --pixel-size 0.5 --output results/
  porosity-detect demo --output demo_results/
  porosity-detect validate --output validation/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Analyze command ---
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a microscopy image for porosity"
    )
    analyze_parser.add_argument(
        "image", type=str, help="Path to microscopy image"
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./porosity_results",
        help="Output directory (default: ./porosity_results)",
    )
    analyze_parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="Pixel size in micrometers (default: 1.0)",
    )
    analyze_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ML confidence threshold (default: 0.5)",
    )
    analyze_parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Use classical detection only (no ML filtering)",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format (default: both)",
    )

    # --- Demo command ---
    demo_parser = subparsers.add_parser(
        "demo", help="Run demo with synthetic microscopy image"
    )
    demo_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./demo_results",
        help="Output directory",
    )
    demo_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    demo_parser.add_argument(
        "--width", type=int, default=1024, help="Image width"
    )
    demo_parser.add_argument(
        "--height", type=int, default=768, help="Image height"
    )

    # --- Validate command ---
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate detection accuracy using synthetic ground truth",
    )
    validate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./validation_results",
        help="Output directory",
    )
    validate_parser.add_argument(
        "--n-images", type=int, default=5, help="Number of test images"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "validate":
        cmd_validate(args)


def cmd_analyze(args):
    """Run porosity analysis on an input image."""
    from porosity_detect.hybrid import HybridPipeline, HybridConfig
    from porosity_detect.classical import ClassicalParams
    from porosity_detect.metrics import PorosityMetrics
    from porosity_detect import visualization as viz

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {image_path}")
    image = _load_image(str(image_path))

    # Configure pipeline
    config = HybridConfig(
        pixel_size_um=args.pixel_size,
        confidence_threshold=args.threshold,
        auto_train=not args.no_ml,
    )

    print("Initializing hybrid pipeline...")
    pipeline = HybridPipeline(config)

    # Run analysis
    print("Running porosity analysis...")
    t0 = time.time()
    results = pipeline.analyze(image)
    elapsed = time.time() - t0

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = results["pipeline_metadata"]
    print(f"\nAnalysis complete in {elapsed:.2f}s")
    print(f"  Classical candidates: {meta.get('n_classical_candidates', 0)}")
    print(f"  Confirmed voids:     {meta.get('n_confirmed_voids', 0)}")
    print(f"  Rejected artifacts:  {meta.get('n_rejected_artifacts', 0)}")

    pm = results["porosity_metrics"]
    print(f"\n  Porosity: {pm['porosity_percent']:.4f}%")
    print(f"  Void count: {pm['void_count']}")

    # Save visualizations
    viz.save_detection_overlay(
        image,
        results["labels"],
        str(output_dir / "detection_overlay.png"),
        void_props=results["void_properties"],
    )
    viz.save_analysis_dashboard(
        image,
        results["labels"],
        results["void_properties"],
        pm,
        str(output_dir / "analysis_dashboard.png"),
    )
    print(f"\nVisualizations saved to {output_dir}/")

    # Save metrics
    _save_metrics(results, output_dir, args.format)


def cmd_demo(args):
    """Run demonstration with synthetic microscopy image."""
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig
    from porosity_detect.hybrid import HybridPipeline, HybridConfig
    from porosity_detect import visualization as viz

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POROSITY-DETECT DEMONSTRATION")
    print("Hybrid Classical + ML Void Detection Pipeline")
    print("=" * 60)

    # Generate synthetic image
    print("\n[1/4] Generating synthetic micrograph...")
    syn_config = SyntheticConfig(
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    generator = SyntheticMicrograph(syn_config)
    image, ground_truth, gt_meta = generator.generate()

    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"  True voids: {gt_meta['n_true_voids']}")
    print(f"  Artifacts:  {gt_meta['n_artifacts']}")
    print(f"  True porosity: {gt_meta['true_porosity_percent']:.4f}%")

    # Save synthetic image
    _save_image(image, str(output_dir / "synthetic_micrograph.png"))

    # Run hybrid pipeline
    print("\n[2/4] Running hybrid detection pipeline...")
    config = HybridConfig(auto_train=True, confidence_threshold=0.5)
    pipeline = HybridPipeline(config)

    t0 = time.time()
    results = pipeline.analyze(image)
    elapsed = time.time() - t0

    meta = results["pipeline_metadata"]
    pm = results["porosity_metrics"]

    print(f"  Analysis time: {elapsed:.2f}s")
    print(f"  Classical candidates: {meta.get('n_classical_candidates', 0)}")
    print(f"  Confirmed voids: {meta.get('n_confirmed_voids', 0)}")
    print(f"  Rejected artifacts: {meta.get('n_rejected_artifacts', 0)}")
    print(f"  Detected porosity: {pm['porosity_percent']:.4f}%")

    # Save visualizations
    print("\n[3/4] Generating visualizations...")
    viz.save_detection_overlay(
        image,
        results["labels"],
        str(output_dir / "detection_overlay.png"),
        void_props=results["void_properties"],
    )
    viz.save_analysis_dashboard(
        image,
        results["labels"],
        results["void_properties"],
        pm,
        str(output_dir / "analysis_dashboard.png"),
    )
    validation_metrics = viz.save_ground_truth_comparison(
        image,
        ground_truth,
        results["labels"],
        str(output_dir / "ground_truth_comparison.png"),
    )

    # Print validation metrics
    print("\n[4/4] Validation against ground truth:")
    print(f"  Pixel Precision: {validation_metrics['pixel_precision']:.3f}")
    print(f"  Pixel Recall:    {validation_metrics['pixel_recall']:.3f}")
    print(f"  Pixel IoU:       {validation_metrics['pixel_iou']:.3f}")
    print(
        f"  Porosity Error:  {validation_metrics['porosity_error_percent']:.4f}%"
    )

    # Save all results
    _save_metrics(results, output_dir, "both")

    # Save porosity report
    from porosity_detect.metrics import PorosityMetrics
    metrics_calc = PorosityMetrics()
    report = metrics_calc.format_report(pm)
    (output_dir / "porosity_report.txt").write_text(report)

    print(f"\nAll results saved to {output_dir}/")
    print("=" * 60)


def cmd_validate(args):
    """Run validation across multiple synthetic images."""
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig
    from porosity_detect.hybrid import HybridPipeline, HybridConfig
    from porosity_detect import visualization as viz

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POROSITY-DETECT VALIDATION")
    print(f"Testing on {args.n_images} synthetic images")
    print("=" * 60)

    pipeline = HybridPipeline(HybridConfig(auto_train=True))

    all_metrics = []

    for i in range(args.n_images):
        seed = 42 + i * 7
        print(f"\n--- Image {i + 1}/{args.n_images} (seed={seed}) ---")

        config = SyntheticConfig(seed=seed)
        gen = SyntheticMicrograph(config)
        image, gt, gt_meta = gen.generate()

        results = pipeline.analyze(image)

        val = viz.save_ground_truth_comparison(
            image,
            gt,
            results["labels"],
            str(output_dir / f"validation_{i + 1}.png"),
        )

        val["seed"] = seed
        val["n_true_voids"] = gt_meta["n_true_voids"]
        val["n_detected"] = results["porosity_metrics"]["void_count"]
        all_metrics.append(val)

        print(f"  IoU: {val['pixel_iou']:.3f}, "
              f"Precision: {val['pixel_precision']:.3f}, "
              f"Recall: {val['pixel_recall']:.3f}, "
              f"Porosity Error: {val['porosity_error_percent']:.4f}%")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    ious = [m["pixel_iou"] for m in all_metrics]
    precisions = [m["pixel_precision"] for m in all_metrics]
    recalls = [m["pixel_recall"] for m in all_metrics]
    errors = [m["porosity_error_percent"] for m in all_metrics]

    print(f"  Mean IoU:           {np.mean(ious):.3f} ± {np.std(ious):.3f}")
    print(f"  Mean Precision:     {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"  Mean Recall:        {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"  Mean Porosity Error: {np.mean(errors):.4f}% ± {np.std(errors):.4f}%")

    # Save summary
    summary = {
        "n_images": args.n_images,
        "per_image_metrics": all_metrics,
        "summary": {
            "mean_iou": float(np.mean(ious)),
            "std_iou": float(np.std(ious)),
            "mean_precision": float(np.mean(precisions)),
            "std_precision": float(np.std(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "std_recall": float(np.std(recalls)),
            "mean_porosity_error": float(np.mean(errors)),
            "std_porosity_error": float(np.std(errors)),
        },
    }
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def _load_image(path: str) -> np.ndarray:
    """Load an image file as numpy array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(path)
    if img.ndim == 3:
        gray = np.mean(img[..., :3], axis=2)
    else:
        gray = img.copy()
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray.astype(np.float64)


def _save_image(image: np.ndarray, path: str):
    """Save a grayscale image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(path, image, cmap="gray", vmin=0, vmax=1)


def _save_metrics(results: dict, output_dir: Path, fmt: str):
    """Save analysis metrics to file."""
    pm = results["porosity_metrics"]

    if fmt in ("json", "both"):
        # Serialize (remove non-serializable items)
        serializable = {
            "porosity_metrics": pm,
            "pipeline_metadata": results["pipeline_metadata"],
            "void_properties": [
                {k: v for k, v in vp.items() if k != "label"}
                for vp in results.get("void_properties", [])
            ],
        }
        with open(output_dir / "porosity_metrics.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    if fmt in ("text", "both"):
        from porosity_detect.metrics import PorosityMetrics

        calculator = PorosityMetrics()
        report = calculator.format_report(pm)
        (output_dir / "porosity_report.txt").write_text(report)


if __name__ == "__main__":
    main()
