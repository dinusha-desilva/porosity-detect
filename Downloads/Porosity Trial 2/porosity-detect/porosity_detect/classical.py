"""
Classical image processing pipeline for void/porosity detection.

Implements traditional materials science microscopy analysis techniques:
- Adaptive thresholding (accounts for uneven illumination common in optical microscopy)
- Morphological operations (noise removal, void boundary refinement)
- Watershed segmentation (separates touching/overlapping voids)
- Connected component analysis (individual void identification)

These methods encode domain knowledge about how voids appear in cross-section
optical micrographs: dark regions against lighter matrix material, roughly
elliptical shapes, size ranges typical of manufacturing-induced porosity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class ClassicalParams:
    """Parameters for classical detection pipeline.

    These defaults are tuned for typical aerospace composite/metal
    optical microscopy at 50x-200x magnification.
    """

    # Preprocessing
    gaussian_sigma: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Thresholding
    adaptive_block_size: int = 51
    adaptive_offset: float = 10.0
    use_otsu: bool = False

    # Morphological operations
    morph_kernel_size: int = 3
    morph_open_iterations: int = 2
    morph_close_iterations: int = 1

    # Filtering
    min_void_area_px: int = 25
    max_void_area_px: int = 50000
    min_circularity: float = 0.1
    max_aspect_ratio: float = 10.0

    # Watershed
    use_watershed: bool = True
    watershed_min_distance: int = 10


class ClassicalDetector:
    """Classical image processing pipeline for void detection.

    Uses adaptive thresholding, morphological operations, and watershed
    segmentation to detect voids in optical microscopy images.

    This approach encodes materials science domain knowledge:
    - Voids appear as dark regions in reflected-light optical microscopy
    - Manufacturing-induced porosity has characteristic size distributions
    - Void morphology correlates with formation mechanism (gas vs shrinkage)
    """

    def __init__(self, params: Optional[ClassicalParams] = None):
        self.params = params or ClassicalParams()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess microscopy image for void detection.

        Converts to grayscale, applies Gaussian blur for noise reduction,
        and CLAHE for contrast enhancement to handle uneven illumination
        common in optical microscopy.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) color.

        Returns:
            Preprocessed grayscale image normalized to [0, 1].
        """
        from scipy.ndimage import gaussian_filter

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image[..., :3], axis=2)
        else:
            gray = image.copy()

        # Normalize to [0, 1]
        gray = gray.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0

        # Gaussian blur for noise reduction
        gray = gaussian_filter(gray, sigma=self.params.gaussian_sigma)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = self._apply_clahe(gray)

        return gray

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement.

        Divides image into tiles and equalizes each independently,
        handling the uneven illumination typical of optical microscopy.
        """
        grid = self.params.clahe_grid_size
        clip = self.params.clahe_clip_limit
        h, w = image.shape
        tile_h = max(h // grid, 1)
        tile_w = max(w // grid, 1)
        result = image.copy()

        for i in range(0, h, tile_h):
            for j in range(0, w, tile_w):
                tile = image[i : i + tile_h, j : j + tile_w]
                if tile.size == 0:
                    continue
                # Local histogram equalization with clipping
                hist, bins = np.histogram(tile.ravel(), bins=256, range=(0, 1))
                clip_val = clip * tile.size / 256
                excess = np.sum(np.maximum(hist - clip_val, 0))
                hist = np.minimum(hist, clip_val)
                hist += int(excess / 256)
                cdf = np.cumsum(hist).astype(np.float64)
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                indices = np.clip((tile * 255).astype(int), 0, 255)
                result[i : i + tile_h, j : j + tile_w] = cdf[indices]

        return result

    def threshold(self, preprocessed: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to segment dark void regions.

        Uses local adaptive thresholding which is more robust than
        global methods for microscopy images with uneven illumination.
        Optionally uses Otsu's method for global thresholding.

        Args:
            preprocessed: Preprocessed grayscale image in [0, 1].

        Returns:
            Binary mask where True = potential void pixel.
        """
        if self.params.use_otsu:
            return self._otsu_threshold(preprocessed)

        block = self.params.adaptive_block_size
        offset = self.params.adaptive_offset / 255.0

        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(preprocessed, size=block)
        binary = preprocessed < (local_mean - offset)

        return binary

    def _otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Otsu's method for automatic global thresholding."""
        hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
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

        return image < best_thresh

    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up binary mask.

        Opening removes small noise (false positive pixels).
        Closing fills small gaps in void boundaries.

        These operations encode knowledge about void morphology —
        real voids have smooth, continuous boundaries.

        Args:
            binary: Binary mask from thresholding.

        Returns:
            Cleaned binary mask.
        """
        from scipy.ndimage import binary_opening, binary_closing

        k = self.params.morph_kernel_size
        struct = np.ones((k, k), dtype=bool)

        cleaned = binary_opening(
            binary, structure=struct, iterations=self.params.morph_open_iterations
        )
        cleaned = binary_closing(
            cleaned, structure=struct, iterations=self.params.morph_close_iterations
        )

        return cleaned

    def segment_voids(self, binary: np.ndarray) -> np.ndarray:
        """Segment individual voids using connected components + watershed.

        Connected component analysis identifies separate void regions.
        Optional watershed segmentation separates touching voids.

        Args:
            binary: Cleaned binary mask.

        Returns:
            Label image where each void has a unique integer label.
        """
        from scipy.ndimage import label, distance_transform_edt
        from scipy.ndimage import maximum_filter

        labels, n_features = label(binary)

        if self.params.use_watershed and n_features > 0:
            labels = self._watershed_split(binary, labels)

        return labels

    def _watershed_split(
        self, binary: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Watershed segmentation to separate touching voids.

        Uses distance transform to find void centers, then grows
        regions from these seeds to split merged voids.
        """
        from scipy.ndimage import (
            distance_transform_edt,
            maximum_filter,
            label as ndlabel,
        )

        dist = distance_transform_edt(binary)
        min_dist = self.params.watershed_min_distance

        # Find local maxima of distance transform (void centers)
        local_max = maximum_filter(dist, size=min_dist * 2 + 1)
        seeds = (dist == local_max) & (dist > 0)
        markers, _ = ndlabel(seeds)

        # Simple marker-controlled watershed via iterative dilation
        result = markers.copy()
        changed = True
        while changed:
            changed = False
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    shifted = np.roll(np.roll(result, di, axis=0), dj, axis=1)
                    mask = (result == 0) & (shifted > 0) & binary
                    if mask.any():
                        result[mask] = shifted[mask]
                        changed = True

        return result

    def filter_voids(
        self, labels: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Filter detected voids by size and shape criteria.

        Applies materials science domain knowledge:
        - Minimum size filter removes noise artifacts
        - Maximum size filter removes background segmentation errors
        - Circularity filter removes elongated scratches/cracks
        - Aspect ratio filter removes preparation artifacts

        Args:
            labels: Label image from segmentation.

        Returns:
            Tuple of (filtered label image, list of void property dicts).
        """
        from scipy.ndimage import find_objects

        void_props = []
        filtered = np.zeros_like(labels)
        new_label = 0

        regions = find_objects(labels)
        for i, slc in enumerate(regions):
            if slc is None:
                continue
            region_mask = labels[slc] == (i + 1)
            area = region_mask.sum()

            # Area filter
            if area < self.params.min_void_area_px:
                continue
            if area > self.params.max_void_area_px:
                continue

            # Compute shape properties
            props = self._compute_region_props(region_mask, slc)
            props["area_px"] = int(area)

            # Circularity filter
            if props["circularity"] < self.params.min_circularity:
                continue

            # Aspect ratio filter
            if props["aspect_ratio"] > self.params.max_aspect_ratio:
                continue

            new_label += 1
            filtered[slc][region_mask] = new_label
            props["label"] = new_label
            void_props.append(props)

        return filtered, void_props

    def _compute_region_props(
        self, mask: np.ndarray, slc: tuple
    ) -> dict:
        """Compute geometric properties of a void region."""
        ys, xs = np.where(mask)

        # Bounding box dimensions
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1

        area = mask.sum()
        perimeter = self._compute_perimeter(mask)

        # Circularity: 4π * area / perimeter²
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0.0

        # Aspect ratio
        major = max(h, w)
        minor = max(min(h, w), 1)
        aspect_ratio = major / minor

        # Centroid (in full image coordinates)
        cy = ys.mean() + slc[0].start
        cx = xs.mean() + slc[1].start

        # Equivalent diameter
        eq_diameter = np.sqrt(4 * area / np.pi)

        return {
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "bbox_h": int(h),
            "bbox_w": int(w),
            "perimeter_px": float(perimeter),
            "circularity": float(np.clip(circularity, 0, 1)),
            "aspect_ratio": float(aspect_ratio),
            "equivalent_diameter_px": float(eq_diameter),
        }

    def _compute_perimeter(self, mask: np.ndarray) -> float:
        """Estimate perimeter of a binary region."""
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        return float(boundary.sum())

    def detect(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, list[dict], dict]:
        """Run full classical detection pipeline.

        Args:
            image: Input microscopy image.

        Returns:
            Tuple of (label image, void properties list, pipeline metadata).
        """
        preprocessed = self.preprocess(image)
        binary = self.threshold(preprocessed)
        cleaned = self.morphological_cleanup(binary)
        labels = self.segment_voids(cleaned)
        filtered_labels, void_props = self.filter_voids(labels)

        metadata = {
            "method": "classical",
            "n_voids_detected": len(void_props),
            "total_void_area_px": sum(v["area_px"] for v in void_props),
            "image_area_px": int(image.shape[0] * image.shape[1]),
            "porosity_fraction": (
                sum(v["area_px"] for v in void_props)
                / (image.shape[0] * image.shape[1])
            ),
            "params": {
                k: v
                for k, v in self.params.__dict__.items()
            },
        }

        return filtered_labels, void_props, metadata
