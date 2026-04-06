"""
Void Annotation Tool for porosity-detect.

Interactive tool for labeling candidate void regions in real micrographs.
Click candidate regions to mark them as TRUE VOID or ARTIFACT, building
a labeled training dataset for the Random Forest classifier.

Usage:
    python annotate.py my_micrograph.tif
    python annotate.py my_micrograph.tif --output annotations/
    python annotate.py my_micrograph.tif --resume annotations/labels.json

Controls:
    LEFT CLICK on a region  → Mark as TRUE VOID (turns green)
    RIGHT CLICK on a region → Mark as ARTIFACT (turns red)
    MIDDLE CLICK            → Undo last annotation
    Press 'S'               → Save annotations and export training data
    Press 'R'               → Reset all annotations
    Press 'Q'               → Quit (prompts to save)
    Press 'T'               → Train classifier on current annotations
"""

import argparse
import json
import sys
import os
import numpy as np

os.environ["MPLBACKEND"] = "TkAgg"


def main():
    parser = argparse.ArgumentParser(
        description="Interactive void annotation tool for porosity-detect"
    )
    parser.add_argument("image", help="Path to micrograph image")
    parser.add_argument(
        "--output", "-o", default="./annotations",
        help="Output directory for annotations (default: ./annotations)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to existing labels.json to resume annotation"
    )
    parser.add_argument(
        "--block-size", type=int, default=25,
        help="Adaptive threshold block size (default: 25)"
    )
    parser.add_argument(
        "--offset", type=float, default=15.0,
        help="Adaptive threshold offset (default: 15.0)"
    )
    parser.add_argument(
        "--min-area", type=int, default=10,
        help="Minimum candidate area in pixels (default: 10)"
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch
    from scipy.ndimage import (
        gaussian_filter, label, binary_opening, binary_closing,
        find_objects, binary_erosion, binary_dilation
    )

    # ── Load image ──────────────────────────────────────────────
    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print(f"Loading: {args.image}")
    img_raw = plt.imread(args.image)
    if img_raw.ndim == 3:
        gray = np.mean(img_raw[..., :3], axis=2)
    else:
        gray = img_raw.copy()
    if gray.max() > 1.0:
        gray = gray / 255.0
    gray = gray.astype(np.float64)
    h, w = gray.shape
    print(f"Image size: {w} x {h} px")

    # ── Run classical detection to get candidates ───────────────
    print("Detecting candidate regions...")
    smoothed = gaussian_filter(gray, sigma=0.8)

    # Adaptive thresholding
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(smoothed, size=args.block_size)
    binary = smoothed < (local_mean - args.offset / 255.0)

    # Morphological cleanup
    struct = np.ones((3, 3), dtype=bool)
    cleaned = binary_opening(binary, structure=struct, iterations=1)
    cleaned = binary_closing(cleaned, structure=struct, iterations=1)

    # Label connected components
    labeled, n_features = label(cleaned)
    regions = find_objects(labeled)

    # Extract candidate info
    candidates = []
    candidate_labels = np.zeros_like(labeled)
    cand_id = 0

    for i, slc in enumerate(regions):
        if slc is None:
            continue
        region = labeled[slc] == (i + 1)
        area = region.sum()

        if area < args.min_area or area > 10000:
            continue

        ys, xs = np.where(region)
        global_ys = ys + slc[0].start
        global_xs = xs + slc[1].start

        # Shape
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
        dilated_region = binary_dilation(region, iterations=3)
        ring = dilated_region & ~region
        ring_ys, ring_xs = np.where(ring)
        ring_gy = np.clip(ring_ys + slc[0].start, 0, h - 1)
        ring_gx = np.clip(ring_xs + slc[1].start, 0, w - 1)
        if len(ring_gy) > 0:
            boundary_contrast = float(np.mean(smoothed[ring_gy, ring_gx])) - mean_int
        else:
            boundary_contrast = 0.0

        cand_id += 1
        candidate_labels[slc][region] = cand_id

        candidates.append({
            "id": cand_id,
            "area_px": int(area),
            "circularity": round(circ, 3),
            "aspect_ratio": round(aspect, 2),
            "eq_diameter_px": round(eq_diam, 1),
            "mean_intensity": round(mean_int, 3),
            "std_intensity": round(std_int, 3),
            "boundary_contrast": round(boundary_contrast, 3),
            "centroid_y": round(float(np.mean(global_ys)), 1),
            "centroid_x": round(float(np.mean(global_xs)), 1),
            "bbox": [int(slc[1].start), int(slc[0].start), int(bbox_w), int(bbox_h)],
            "label": None,  # None=unlabeled, 1=void, 0=artifact
        })

    print(f"Found {len(candidates)} candidate regions")

    if len(candidates) == 0:
        print("No candidates found. Try lowering --offset or --min-area.")
        sys.exit(1)

    # ── Resume from saved annotations if provided ───────────────
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from: {args.resume}")
        with open(args.resume) as f:
            saved = json.load(f)
        saved_labels = {item["id"]: item["label"] for item in saved.get("candidates", [])}
        for c in candidates:
            if c["id"] in saved_labels:
                c["label"] = saved_labels[c["id"]]
        n_labeled = sum(1 for c in candidates if c["label"] is not None)
        print(f"Restored {n_labeled} annotations")

    # ── Setup interactive figure ────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.canvas.manager.set_window_title("porosity-detect — Void Annotation Tool")

    ax.imshow(gray, cmap="gray")
    ax.set_title(
        f"LEFT click = Void | RIGHT click = Artifact | "
        f"S = Save | T = Train | Q = Quit\n"
        f"0 / {len(candidates)} labeled",
        fontsize=11,
    )
    ax.axis("off")

    # Draw candidate circles
    circle_patches = {}
    for c in candidates:
        r = max(c["eq_diameter_px"] / 2 + 2, 4)
        if c["label"] == 1:
            color = "lime"
        elif c["label"] == 0:
            color = "red"
        else:
            color = "cyan"

        circ_patch = Circle(
            (c["centroid_x"], c["centroid_y"]),
            radius=r,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(circ_patch)
        circle_patches[c["id"]] = circ_patch

    annotation_history = []

    def update_title():
        n_voids = sum(1 for c in candidates if c["label"] == 1)
        n_artifacts = sum(1 for c in candidates if c["label"] == 0)
        n_total = n_voids + n_artifacts
        ax.set_title(
            f"LEFT click = Void | RIGHT click = Artifact | "
            f"S = Save | T = Train | Q = Quit\n"
            f"{n_total} / {len(candidates)} labeled "
            f"({n_voids} voids, {n_artifacts} artifacts)",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def find_nearest_candidate(x, y):
        """Find the candidate closest to click coordinates."""
        best = None
        best_dist = float("inf")
        for c in candidates:
            d = np.sqrt((c["centroid_x"] - x) ** 2 + (c["centroid_y"] - y) ** 2)
            max_r = max(c["eq_diameter_px"] / 2 + 5, 8)
            if d < max_r and d < best_dist:
                best = c
                best_dist = d
        return best

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        cand = find_nearest_candidate(event.xdata, event.ydata)
        if cand is None:
            return

        old_label = cand["label"]

        if event.button == 1:  # Left click → void
            cand["label"] = 1
            circle_patches[cand["id"]].set_edgecolor("lime")
            circle_patches[cand["id"]].set_linewidth(2.5)
        elif event.button == 3:  # Right click → artifact
            cand["label"] = 0
            circle_patches[cand["id"]].set_edgecolor("red")
            circle_patches[cand["id"]].set_linewidth(2.5)
        elif event.button == 2:  # Middle click → undo
            cand["label"] = None
            circle_patches[cand["id"]].set_edgecolor("cyan")
            circle_patches[cand["id"]].set_linewidth(1.5)

        annotation_history.append((cand["id"], old_label))
        update_title()

    def on_key(event):
        if event.key == "s":
            save_annotations()
        elif event.key == "t":
            train_on_annotations()
        elif event.key == "r":
            for c in candidates:
                c["label"] = None
                circle_patches[c["id"]].set_edgecolor("cyan")
                circle_patches[c["id"]].set_linewidth(1.5)
            annotation_history.clear()
            update_title()
            print("All annotations reset.")
        elif event.key == "q":
            n_labeled = sum(1 for c in candidates if c["label"] is not None)
            if n_labeled > 0:
                save_annotations()
            plt.close(fig)

    def save_annotations():
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        # Save labels JSON
        labels_path = os.path.join(output_dir, "labels.json")
        data = {
            "image": args.image,
            "image_shape": [h, w],
            "n_candidates": len(candidates),
            "n_labeled": sum(1 for c in candidates if c["label"] is not None),
            "n_voids": sum(1 for c in candidates if c["label"] == 1),
            "n_artifacts": sum(1 for c in candidates if c["label"] == 0),
            "candidates": candidates,
        }
        with open(labels_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved annotations: {labels_path}")

        # Export training-ready CSV
        export_training_data(output_dir)

    def export_training_data(output_dir):
        """Export labeled data as numpy arrays for training."""
        labeled_cands = [c for c in candidates if c["label"] is not None]
        if len(labeled_cands) == 0:
            print("No labeled candidates to export.")
            return

        # Build feature matrix + labels
        # Use the full feature extractor for proper features
        try:
            from porosity_detect.features import FeatureExtractor

            extractor = FeatureExtractor()
            preprocessed = smoothed

            features = []
            labels_arr = []

            for c in labeled_cands:
                mask = candidate_labels == c["id"]
                feat = extractor.extract_single(preprocessed, mask, candidate_labels)
                features.append(feat)
                labels_arr.append(c["label"])

            X = np.vstack(features)
            y = np.array(labels_arr)

            np.save(os.path.join(output_dir, "X_train.npy"), X)
            np.save(os.path.join(output_dir, "y_train.npy"), y)
            print(f"Exported training data: X_train.npy ({X.shape}), y_train.npy ({y.shape})")
            print(f"  Voids: {(y == 1).sum()}, Artifacts: {(y == 0).sum()}")

        except ImportError:
            # Fallback: save basic features as CSV
            import csv
            csv_path = os.path.join(output_dir, "training_data.csv")
            fields = [
                "label", "area_px", "circularity", "aspect_ratio",
                "eq_diameter_px", "mean_intensity", "std_intensity",
                "boundary_contrast",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for c in labeled_cands:
                    writer.writerow({k: c[k] for k in fields})
            print(f"Exported training CSV: {csv_path}")

    def train_on_annotations():
        """Train the classifier on current annotations."""
        labeled_cands = [c for c in candidates if c["label"] is not None]
        n_voids = sum(1 for c in labeled_cands if c["label"] == 1)
        n_artifacts = sum(1 for c in labeled_cands if c["label"] == 0)

        if n_voids < 3 or n_artifacts < 3:
            print(f"Need at least 3 voids and 3 artifacts to train.")
            print(f"Currently: {n_voids} voids, {n_artifacts} artifacts.")
            return

        try:
            from porosity_detect.features import FeatureExtractor
            from porosity_detect.ml_model import MLDetector

            extractor = FeatureExtractor()

            features = []
            labels_arr = []
            for c in labeled_cands:
                mask = candidate_labels == c["id"]
                feat = extractor.extract_single(smoothed, mask, candidate_labels)
                features.append(feat)
                labels_arr.append(c["label"])

            X = np.vstack(features)
            y = np.array(labels_arr)

            detector = MLDetector()
            metrics = detector.train(X, y)

            print(f"\nTraining complete on {len(y)} samples:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall:    {metrics.get('recall', 0):.3f}")
            print(f"  F1 Score:  {metrics.get('f1_score', 0):.3f}")

            # Save model
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "trained_model.json")
            detector.save_model(model_path)
            print(f"  Model saved: {model_path}")

        except ImportError as e:
            print(f"Cannot train: {e}")
            print("Make sure porosity-detect is installed (pip install -e .)")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_title()
    print("\nAnnotation tool ready.")
    print("  LEFT CLICK  = mark as TRUE VOID (green)")
    print("  RIGHT CLICK = mark as ARTIFACT (red)")
    print("  MIDDLE CLICK = undo")
    print("  S = save  |  T = train  |  R = reset  |  Q = quit")
    plt.show()


if __name__ == "__main__":
    main()
