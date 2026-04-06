"""
Porosity quantification metrics for aerospace materials.

Computes standardized porosity measurements aligned with aerospace
quality standards (ASTM E2015, ASTM E1245) for automated image analysis
of materials microstructure.

Key metrics:
- Area fraction porosity (most common in aerospace specs)
- Void size distribution (important for fatigue life prediction)
- Void density (voids per unit area)
- Largest void characterization (often the critical defect)
- Void type distribution (gas vs shrinkage vs delamination)
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Configuration for porosity metrics computation."""

    # Size bins for void size distribution (in pixels)
    size_bins_px: tuple = (0, 50, 200, 500, 2000, 10000, 100000)

    # Size bin labels
    size_bin_labels: tuple = (
        "micro (<50 px)",
        "small (50-200 px)",
        "medium (200-500 px)",
        "large (500-2000 px)",
        "very_large (2000-10000 px)",
        "macro (>10000 px)",
    )


class PorosityMetrics:
    """Compute standardized porosity metrics from detection results.

    Produces metrics that are directly relevant for aerospace
    material qualification and process control.
    """

    def __init__(
        self,
        pixel_size_um: float = 1.0,
        config: Optional[MetricsConfig] = None,
    ):
        self.pixel_size_um = pixel_size_um
        self.config = config or MetricsConfig()

    def compute(
        self,
        labels: np.ndarray,
        void_props: list[dict],
        image_shape: tuple,
    ) -> dict:
        """Compute comprehensive porosity metrics.

        Args:
            labels: Label image of confirmed voids.
            void_props: List of per-void property dicts.
            image_shape: (height, width) of the analyzed image.

        Returns:
            Dictionary of porosity metrics.
        """
        h, w = image_shape[:2]
        total_area_px = h * w
        total_area_um2 = total_area_px * (self.pixel_size_um ** 2)

        if len(void_props) == 0:
            return self._empty_metrics(total_area_px, total_area_um2)

        # Basic porosity metrics
        areas = np.array([v["area_px"] for v in void_props])
        total_void_area_px = areas.sum()
        porosity_fraction = total_void_area_px / total_area_px

        # Size statistics
        size_stats = {
            "mean_void_area_px": float(areas.mean()),
            "median_void_area_px": float(np.median(areas)),
            "std_void_area_px": float(areas.std()),
            "min_void_area_px": float(areas.min()),
            "max_void_area_px": float(areas.max()),
            "mean_void_area_um2": float(
                areas.mean() * self.pixel_size_um ** 2
            ),
        }

        # Equivalent diameter statistics
        eq_diams = np.array(
            [v.get("equivalent_diameter_px", 0) for v in void_props]
        )
        if len(eq_diams) > 0:
            size_stats["mean_equivalent_diameter_px"] = float(eq_diams.mean())
            size_stats["max_equivalent_diameter_px"] = float(eq_diams.max())
            size_stats["mean_equivalent_diameter_um"] = float(
                eq_diams.mean() * self.pixel_size_um
            )

        # Size distribution
        size_dist = self._size_distribution(areas)

        # Shape statistics
        circularities = np.array(
            [v.get("circularity", 0) for v in void_props]
        )
        aspect_ratios = np.array(
            [v.get("aspect_ratio", 1) for v in void_props]
        )

        shape_stats = {
            "mean_circularity": float(circularities.mean()),
            "std_circularity": float(circularities.std()),
            "mean_aspect_ratio": float(aspect_ratios.mean()),
            "std_aspect_ratio": float(aspect_ratios.std()),
        }

        # Void type distribution
        type_dist = {}
        for v in void_props:
            vtype = v.get("void_type", "unclassified")
            type_dist[vtype] = type_dist.get(vtype, 0) + 1

        # Spatial distribution metrics
        spatial = self._spatial_metrics(void_props, h, w)

        # Largest void characterization (often critical for fatigue)
        largest_idx = np.argmax(areas)
        largest_void = {
            "area_px": int(areas[largest_idx]),
            "area_um2": float(areas[largest_idx] * self.pixel_size_um ** 2),
            "equivalent_diameter_px": float(eq_diams[largest_idx]),
            "equivalent_diameter_um": float(
                eq_diams[largest_idx] * self.pixel_size_um
            ),
            "circularity": float(circularities[largest_idx]),
            "centroid": (
                float(void_props[largest_idx].get("centroid_y", 0)),
                float(void_props[largest_idx].get("centroid_x", 0)),
            ),
        }

        return {
            "porosity_area_fraction": float(porosity_fraction),
            "porosity_percent": float(porosity_fraction * 100),
            "void_count": len(void_props),
            "total_void_area_px": int(total_void_area_px),
            "total_void_area_um2": float(
                total_void_area_px * self.pixel_size_um ** 2
            ),
            "image_area_px": int(total_area_px),
            "image_area_um2": float(total_area_um2),
            "void_density_per_mm2": float(
                len(void_props) / (total_area_um2 / 1e6)
                if total_area_um2 > 0
                else 0
            ),
            "size_statistics": size_stats,
            "shape_statistics": shape_stats,
            "void_size_distribution": size_dist,
            "void_type_distribution": type_dist,
            "spatial_distribution": spatial,
            "largest_void": largest_void,
        }

    def _size_distribution(self, areas: np.ndarray) -> dict:
        """Compute void size distribution histogram."""
        bins = self.config.size_bins_px
        labels = self.config.size_bin_labels

        counts, _ = np.histogram(areas, bins=bins)

        dist = {}
        for i, label in enumerate(labels):
            if i < len(counts):
                dist[label] = {
                    "count": int(counts[i]),
                    "fraction": float(counts[i] / max(len(areas), 1)),
                }

        return dist

    def _spatial_metrics(
        self, void_props: list[dict], h: int, w: int
    ) -> dict:
        """Compute spatial distribution metrics."""
        if len(void_props) < 2:
            return {
                "clustering_index": 0.0,
                "nearest_neighbor_mean_px": 0.0,
            }

        # Centroids
        centroids = np.array(
            [
                [v.get("centroid_y", 0), v.get("centroid_x", 0)]
                for v in void_props
            ]
        )

        # Nearest neighbor distances
        nn_dists = []
        for i, c in enumerate(centroids):
            dists = np.sqrt(np.sum((centroids - c) ** 2, axis=1))
            dists[i] = np.inf  # exclude self
            nn_dists.append(dists.min())

        nn_dists = np.array(nn_dists)

        # Expected nearest neighbor distance for random distribution
        density = len(void_props) / (h * w)
        expected_nn = 0.5 / np.sqrt(max(density, 1e-10))

        # Clustering index (< 1 = clustered, 1 = random, > 1 = dispersed)
        clustering = float(nn_dists.mean() / max(expected_nn, 1e-6))

        return {
            "clustering_index": clustering,
            "nearest_neighbor_mean_px": float(nn_dists.mean()),
            "nearest_neighbor_std_px": float(nn_dists.std()),
            "nearest_neighbor_mean_um": float(
                nn_dists.mean() * self.pixel_size_um
            ),
        }

    def _empty_metrics(
        self, total_area_px: int, total_area_um2: float
    ) -> dict:
        """Return zero metrics when no voids found."""
        return {
            "porosity_area_fraction": 0.0,
            "porosity_percent": 0.0,
            "void_count": 0,
            "total_void_area_px": 0,
            "total_void_area_um2": 0.0,
            "image_area_px": total_area_px,
            "image_area_um2": total_area_um2,
            "void_density_per_mm2": 0.0,
            "size_statistics": {},
            "shape_statistics": {},
            "void_size_distribution": {},
            "void_type_distribution": {},
            "spatial_distribution": {},
            "largest_void": {},
        }

    def format_report(self, metrics: dict) -> str:
        """Format metrics as a human-readable report.

        Returns a text report suitable for documentation or
        quality control records.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("POROSITY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append("OVERALL POROSITY")
        lines.append(f"  Area fraction:  {metrics['porosity_area_fraction']:.6f}")
        lines.append(f"  Porosity:       {metrics['porosity_percent']:.4f}%")
        lines.append(f"  Void count:     {metrics['void_count']}")
        lines.append(f"  Void density:   {metrics.get('void_density_per_mm2', 0):.1f} /mm²")
        lines.append("")

        if metrics.get("size_statistics"):
            stats = metrics["size_statistics"]
            lines.append("VOID SIZE STATISTICS")
            lines.append(
                f"  Mean area:      {stats.get('mean_void_area_um2', 0):.1f} µm²"
            )
            lines.append(
                f"  Mean Ø:         {stats.get('mean_equivalent_diameter_um', 0):.1f} µm"
            )
            lines.append(
                f"  Max Ø:          {stats.get('max_equivalent_diameter_px', 0):.1f} px"
            )
            lines.append("")

        if metrics.get("shape_statistics"):
            shape = metrics["shape_statistics"]
            lines.append("VOID SHAPE STATISTICS")
            lines.append(
                f"  Mean circularity:   {shape.get('mean_circularity', 0):.3f}"
            )
            lines.append(
                f"  Mean aspect ratio:  {shape.get('mean_aspect_ratio', 0):.2f}"
            )
            lines.append("")

        if metrics.get("void_type_distribution"):
            lines.append("VOID TYPE DISTRIBUTION")
            for vtype, count in metrics["void_type_distribution"].items():
                lines.append(f"  {vtype}: {count}")
            lines.append("")

        if metrics.get("largest_void"):
            lv = metrics["largest_void"]
            lines.append("LARGEST VOID")
            lines.append(f"  Area:      {lv.get('area_um2', 0):.1f} µm²")
            lines.append(f"  Diameter:  {lv.get('equivalent_diameter_um', 0):.1f} µm")
            lines.append(f"  Circularity: {lv.get('circularity', 0):.3f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
