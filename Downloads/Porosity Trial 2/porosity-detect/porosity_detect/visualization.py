"""
Visualization utilities for porosity analysis results.

Generates publication-quality figures for:
- Detection overlay on original micrograph
- Void size distribution histograms
- Porosity summary dashboards
- Ground truth comparison (for validation)

Uses only matplotlib (no additional dependencies).
"""

import numpy as np
from typing import Optional
from pathlib import Path


def create_colormap(n_labels: int, seed: int = 42) -> np.ndarray:
    """Create a colormap for void labels."""
    rng = np.random.RandomState(seed)
    colors = np.zeros((n_labels + 1, 3))
    for i in range(1, n_labels + 1):
        colors[i] = rng.uniform(0.3, 1.0, size=3)
    return colors


def save_detection_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Void Detection Results",
    void_props: Optional[list] = None,
):
    """Save detection overlay visualization.

    Shows original image with detected voids highlighted in color.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Micrograph", fontsize=12)
    axes[0].axis("off")

    # Binary detection mask
    binary = labels > 0
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*image.shape[:2], 4))
    overlay[binary, 0] = 1.0  # Red
    overlay[binary, 3] = 0.5  # Semi-transparent
    axes[1].imshow(overlay)
    axes[1].set_title(f"Detected Voids (n={labels.max()})", fontsize=12)
    axes[1].axis("off")

    # Labeled voids with type coloring
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    if void_props:
        type_colors = {
            "gas_porosity": [0, 1, 0, 0.6],       # Green
            "shrinkage_porosity": [1, 0.5, 0, 0.6],  # Orange
            "delamination": [1, 0, 0, 0.6],        # Red
            "micro_void": [0, 0.7, 1, 0.6],        # Cyan
            "unclassified": [1, 1, 0, 0.6],        # Yellow
        }
        colored_overlay = np.zeros((*image.shape[:2], 4))
        for props in void_props:
            lbl = props["label"]
            vtype = props.get("void_type", "unclassified")
            color = type_colors.get(vtype, [1, 1, 0, 0.6])
            mask = labels == lbl
            colored_overlay[mask] = color
        axes[2].imshow(colored_overlay)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c[:3], alpha=0.6, label=t.replace("_", " ").title())
            for t, c in type_colors.items()
            if any(
                p.get("void_type") == t for p in void_props
            )
        ]
        axes[2].legend(
            handles=legend_elements, loc="lower right", fontsize=8
        )
    axes[2].set_title("Void Classification", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_analysis_dashboard(
    image: np.ndarray,
    labels: np.ndarray,
    void_props: list,
    metrics: dict,
    output_path: str,
):
    """Save comprehensive analysis dashboard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 12))

    # Layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)  # Original
    ax2 = fig.add_subplot(2, 3, 2)  # Detection overlay
    ax3 = fig.add_subplot(2, 3, 3)  # Size distribution
    ax4 = fig.add_subplot(2, 3, 4)  # Circularity distribution
    ax5 = fig.add_subplot(2, 3, 5)  # Type distribution
    ax6 = fig.add_subplot(2, 3, 6)  # Metrics summary

    # 1. Original image
    ax1.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Original Micrograph")
    ax1.axis("off")

    # 2. Detection overlay
    ax2.imshow(image, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*image.shape[:2], 4))
    binary = labels > 0
    overlay[binary] = [1, 0, 0, 0.5]
    ax2.imshow(overlay)
    ax2.set_title(f"Detected Voids (n={len(void_props)})")
    ax2.axis("off")

    # 3. Void size distribution
    if void_props:
        areas = [v["area_px"] for v in void_props]
        ax3.hist(areas, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
        ax3.set_xlabel("Void Area (pixels)")
        ax3.set_ylabel("Count")
        ax3.set_title("Void Size Distribution")
        ax3.axvline(
            np.mean(areas),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(areas):.0f} px",
        )
        ax3.legend(fontsize=8)

    # 4. Circularity distribution
    if void_props:
        circs = [v.get("circularity", 0) for v in void_props]
        ax4.hist(circs, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax4.set_xlabel("Circularity")
        ax4.set_ylabel("Count")
        ax4.set_title("Circularity Distribution")
        ax4.axvline(0.7, color="red", linestyle="--", label="Gas/Shrinkage threshold")
        ax4.legend(fontsize=8)

    # 5. Void type distribution
    type_dist = metrics.get("void_type_distribution", {})
    if type_dist:
        types = list(type_dist.keys())
        counts = list(type_dist.values())
        colors = ["#4CAF50", "#FF9800", "#F44336", "#00BCD4", "#FFEB3B"]
        ax5.bar(
            range(len(types)),
            counts,
            color=colors[: len(types)],
            edgecolor="white",
        )
        ax5.set_xticks(range(len(types)))
        ax5.set_xticklabels(
            [t.replace("_", "\n") for t in types], fontsize=8
        )
        ax5.set_ylabel("Count")
        ax5.set_title("Void Type Distribution")

    # 6. Metrics summary text
    ax6.axis("off")
    summary_text = (
        f"POROSITY ANALYSIS SUMMARY\n"
        f"{'=' * 35}\n\n"
        f"Porosity:     {metrics.get('porosity_percent', 0):.4f}%\n"
        f"Void Count:   {metrics.get('void_count', 0)}\n"
        f"Void Area:    {metrics.get('total_void_area_px', 0)} px\n"
        f"Image Area:   {metrics.get('image_area_px', 0)} px\n\n"
    )

    size_stats = metrics.get("size_statistics", {})
    if size_stats:
        summary_text += (
            f"Mean Void Area:  {size_stats.get('mean_void_area_px', 0):.1f} px\n"
            f"Max Void Area:   {size_stats.get('max_void_area_px', 0):.0f} px\n"
            f"Mean Diameter:   {size_stats.get('mean_equivalent_diameter_px', 0):.1f} px\n\n"
        )

    shape_stats = metrics.get("shape_statistics", {})
    if shape_stats:
        summary_text += (
            f"Mean Circularity:   {shape_stats.get('mean_circularity', 0):.3f}\n"
            f"Mean Aspect Ratio:  {shape_stats.get('mean_aspect_ratio', 0):.2f}\n"
        )

    spatial = metrics.get("spatial_distribution", {})
    if spatial:
        summary_text += (
            f"\nClustering Index:   {spatial.get('clustering_index', 0):.2f}\n"
            f"(<1=clustered, 1=random, >1=dispersed)\n"
        )

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        "Porosity Detection — Hybrid Classical + ML Pipeline",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ground_truth_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    detected: np.ndarray,
    output_path: str,
    gt_metadata: Optional[dict] = None,
    det_metadata: Optional[dict] = None,
):
    """Save comparison of detection vs ground truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Original
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    gt_overlay = np.zeros((*image.shape[:2], 4))
    gt_overlay[ground_truth, 1] = 1.0
    gt_overlay[ground_truth, 3] = 0.5
    axes[1].imshow(gt_overlay)
    gt_pct = ground_truth.sum() / ground_truth.size * 100
    axes[1].set_title(f"Ground Truth ({gt_pct:.3f}%)")
    axes[1].axis("off")

    # Detected
    det_binary = detected > 0
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    det_overlay = np.zeros((*image.shape[:2], 4))
    det_overlay[det_binary, 0] = 1.0
    det_overlay[det_binary, 3] = 0.5
    axes[2].imshow(det_overlay)
    det_pct = det_binary.sum() / det_binary.size * 100
    axes[2].set_title(f"Detected ({det_pct:.3f}%)")
    axes[2].axis("off")

    # Comparison overlay
    axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    comp_overlay = np.zeros((*image.shape[:2], 4))
    tp = ground_truth & det_binary
    fp = ~ground_truth & det_binary
    fn = ground_truth & ~det_binary
    comp_overlay[tp] = [0, 1, 0, 0.5]   # Green = true positive
    comp_overlay[fp] = [1, 0, 0, 0.5]   # Red = false positive
    comp_overlay[fn] = [0, 0, 1, 0.5]   # Blue = false negative
    axes[3].imshow(comp_overlay)

    # Pixel-level metrics
    tp_count = tp.sum()
    fp_count = fp.sum()
    fn_count = fn.sum()
    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    iou = tp_count / max(tp_count + fp_count + fn_count, 1)

    axes[3].set_title(
        f"Comparison (IoU={iou:.3f})\n"
        f"Green=TP, Red=FP, Blue=FN",
        fontsize=10,
    )
    axes[3].axis("off")

    fig.suptitle(
        f"Detection Validation — Precision: {precision:.3f}, Recall: {recall:.3f}, IoU: {iou:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_iou": float(iou),
        "true_porosity_percent": float(gt_pct),
        "detected_porosity_percent": float(det_pct),
        "porosity_error_percent": float(abs(gt_pct - det_pct)),
    }
