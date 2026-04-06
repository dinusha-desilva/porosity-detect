"""
Feature extraction for ML-based void classification.

Extracts physics-informed features from candidate void regions that encode
materials science domain knowledge:

1. Morphological features — void shape correlates with formation mechanism:
   - Gas porosity → spherical (high circularity)
   - Shrinkage porosity → dendritic/irregular (low circularity)
   - Delamination → elongated (high aspect ratio)

2. Intensity features — void interior vs surrounding matrix contrast:
   - True voids have consistent low intensity
   - Artifacts (scratches, stains) have different intensity profiles

3. Texture features — local texture around void boundary:
   - True voids have sharp, well-defined boundaries
   - Preparation artifacts have diffuse boundaries

4. Context features — spatial relationship to other features:
   - Porosity often clusters near specific microstructural features
   - Isolated dark regions may be inclusions, not voids
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Intensity analysis
    boundary_width: int = 5  # pixels around void for boundary analysis
    intensity_percentiles: tuple = (10, 25, 50, 75, 90)

    # Texture analysis
    glcm_distances: tuple = (1, 3, 5)
    glcm_angles: tuple = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    # Context analysis
    context_radius: int = 20  # pixels around void for context features


class FeatureExtractor:
    """Extract physics-informed features from candidate void regions.

    Combines image processing with materials science domain knowledge
    to create feature vectors that capture both visual appearance and
    physical plausibility of candidate voids.
    """

    FEATURE_NAMES = [
        # Morphological (8 features)
        "area_px",
        "perimeter_px",
        "circularity",
        "aspect_ratio",
        "equivalent_diameter_px",
        "solidity",
        "extent",
        "compactness",
        # Intensity (10 features)
        "mean_intensity",
        "std_intensity",
        "intensity_p10",
        "intensity_p25",
        "intensity_p50",
        "intensity_p75",
        "intensity_p90",
        "boundary_contrast",
        "intensity_range",
        "intensity_skewness",
        # Texture (4 features)
        "boundary_gradient_mean",
        "boundary_gradient_std",
        "interior_homogeneity",
        "edge_sharpness",
        # Context (3 features)
        "local_density",
        "relative_brightness",
        "distance_to_edge",
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def extract_single(
        self, image: np.ndarray, mask: np.ndarray, all_labels: np.ndarray
    ) -> np.ndarray:
        """Extract feature vector for a single void candidate.

        Args:
            image: Preprocessed grayscale image [0, 1].
            mask: Binary mask of the void region.
            all_labels: Full label image (for context features).

        Returns:
            Feature vector of length len(FEATURE_NAMES).
        """
        features = []

        # Morphological features
        features.extend(self._morphological_features(mask))

        # Intensity features
        features.extend(self._intensity_features(image, mask))

        # Texture features
        features.extend(self._texture_features(image, mask))

        # Context features
        features.extend(self._context_features(image, mask, all_labels))

        return np.array(features, dtype=np.float64)

    def extract_batch(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        void_props: list[dict],
    ) -> np.ndarray:
        """Extract features for all detected void candidates.

        Args:
            image: Preprocessed grayscale image.
            labels: Label image from segmentation.
            void_props: List of void property dicts from classical detection.

        Returns:
            Feature matrix of shape (n_voids, n_features).
        """
        features_list = []

        for props in void_props:
            lbl = props["label"]
            mask = labels == lbl
            feat = self.extract_single(image, mask, labels)
            features_list.append(feat)

        if len(features_list) == 0:
            return np.zeros((0, len(self.FEATURE_NAMES)))

        return np.vstack(features_list)

    def _morphological_features(self, mask: np.ndarray) -> list[float]:
        """Extract shape/morphology features encoding void formation physics."""
        from scipy.ndimage import binary_erosion

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return [0.0] * 8

        area = float(mask.sum())

        # Perimeter
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        perimeter = float(boundary.sum())

        # Circularity
        circularity = (
            4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
        )
        circularity = min(circularity, 1.0)

        # Bounding box
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        major = max(h, w)
        minor = max(min(h, w), 1)
        aspect_ratio = major / minor

        # Equivalent diameter
        eq_diameter = np.sqrt(4 * area / np.pi)

        # Solidity (area / convex hull area approximation)
        bbox_area = h * w
        solidity = area / bbox_area if bbox_area > 0 else 0.0

        # Extent (area / bounding box area)
        extent = area / bbox_area if bbox_area > 0 else 0.0

        # Compactness (perimeter² / area)
        compactness = (perimeter ** 2) / area if area > 0 else 0.0

        return [
            area,
            perimeter,
            circularity,
            aspect_ratio,
            eq_diameter,
            solidity,
            extent,
            compactness,
        ]

    def _intensity_features(
        self, image: np.ndarray, mask: np.ndarray
    ) -> list[float]:
        """Extract intensity features capturing void-matrix contrast."""
        from scipy.ndimage import binary_dilation, binary_erosion

        void_pixels = image[mask]
        if len(void_pixels) == 0:
            return [0.0] * 10

        # Interior intensity statistics
        mean_int = float(np.mean(void_pixels))
        std_int = float(np.std(void_pixels))

        percentiles = np.percentile(
            void_pixels, self.config.intensity_percentiles
        )

        # Intensity range
        int_range = float(np.ptp(void_pixels))

        # Skewness
        if std_int > 0:
            skewness = float(
                np.mean(((void_pixels - mean_int) / std_int) ** 3)
            )
        else:
            skewness = 0.0

        # Boundary contrast: mean intensity of boundary neighborhood
        bw = self.config.boundary_width
        dilated = binary_dilation(mask, iterations=bw)
        boundary_ring = dilated & ~mask
        if boundary_ring.any():
            boundary_mean = float(np.mean(image[boundary_ring]))
            contrast = boundary_mean - mean_int
        else:
            contrast = 0.0

        return [
            mean_int,
            std_int,
            float(percentiles[0]),
            float(percentiles[1]),
            float(percentiles[2]),
            float(percentiles[3]),
            float(percentiles[4]),
            contrast,
            int_range,
            skewness,
        ]

    def _texture_features(
        self, image: np.ndarray, mask: np.ndarray
    ) -> list[float]:
        """Extract texture features for boundary characterization."""
        from scipy.ndimage import sobel, binary_dilation, binary_erosion

        # Gradient magnitude at void boundary
        grad_y = sobel(image, axis=0)
        grad_x = sobel(image, axis=1)
        grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)

        # Boundary region
        dilated = binary_dilation(mask, iterations=2)
        eroded = binary_erosion(mask, iterations=1)
        boundary = dilated & ~eroded

        if boundary.any():
            boundary_grad = grad_mag[boundary]
            grad_mean = float(np.mean(boundary_grad))
            grad_std = float(np.std(boundary_grad))
        else:
            grad_mean = 0.0
            grad_std = 0.0

        # Interior homogeneity (low std = uniform = likely real void)
        void_pixels = image[mask]
        if len(void_pixels) > 1:
            homogeneity = 1.0 / (1.0 + float(np.std(void_pixels)))
        else:
            homogeneity = 0.0

        # Edge sharpness (high gradient at boundary = sharp edge)
        if boundary.any():
            edge_sharpness = float(np.percentile(grad_mag[boundary], 90))
        else:
            edge_sharpness = 0.0

        return [grad_mean, grad_std, homogeneity, edge_sharpness]

    def _context_features(
        self, image: np.ndarray, mask: np.ndarray, all_labels: np.ndarray
    ) -> list[float]:
        """Extract spatial context features."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return [0.0] * 3

        cy, cx = ys.mean(), xs.mean()
        h, w = image.shape[:2]

        # Local void density (how many other voids nearby)
        r = self.config.context_radius
        y_min = max(0, int(cy - r))
        y_max = min(h, int(cy + r))
        x_min = max(0, int(cx - r))
        x_max = min(w, int(cx + r))

        local_labels = all_labels[y_min:y_max, x_min:x_max]
        unique_nearby = len(set(local_labels[local_labels > 0]) - {0})
        local_density = unique_nearby / max(
            (y_max - y_min) * (x_max - x_min) / 10000, 1
        )

        # Relative brightness (void intensity vs image mean)
        void_mean = float(np.mean(image[mask]))
        image_mean = float(np.mean(image))
        relative_brightness = void_mean / max(image_mean, 1e-6)

        # Distance to nearest image edge (normalized)
        dist_to_edge = min(cy, h - cy, cx, w - cx)
        distance_norm = dist_to_edge / max(min(h, w) / 2, 1)

        return [local_density, relative_brightness, float(distance_norm)]
