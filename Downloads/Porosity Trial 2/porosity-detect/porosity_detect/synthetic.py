"""
Synthetic microscopy image generator for testing and demonstration.

Generates realistic-looking optical microscopy cross-section images with
known void populations. This allows:
- Testing the detection pipeline without real (often proprietary) data
- Validating detection accuracy against ground truth
- Demonstrating the tool's capabilities

The synthetic images model key features of real micrographs:
- Matrix material with grain-like texture
- Uneven illumination (common in optical microscopy)
- Various void types (gas, shrinkage, delamination)
- Common artifacts (scratches, staining)
- Realistic noise levels
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticConfig:
    """Configuration for synthetic image generation."""

    width: int = 1024
    height: int = 768
    seed: int = 42

    # Matrix properties
    matrix_mean_intensity: float = 0.65
    matrix_texture_scale: float = 0.05
    grain_density: int = 200

    # Void population
    n_gas_pores: int = 15
    n_shrinkage_voids: int = 8
    n_delaminations: int = 3
    n_micro_voids: int = 25

    # Artifacts (to test false positive rejection)
    n_scratches: int = 3
    n_debris: int = 10
    n_stains: int = 2

    # Illumination
    illumination_gradient: float = 0.15

    # Noise
    noise_std: float = 0.02


class SyntheticMicrograph:
    """Generate synthetic optical microscopy images with known porosity.

    Creates images that mimic reflected-light optical microscopy of
    polished cross-sections of aerospace materials (composites, metals).
    """

    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate a synthetic micrograph with ground truth.

        Returns:
            Tuple of:
            - image: Synthetic grayscale micrograph (H, W) in [0, 1]
            - ground_truth: Binary mask of true voids
            - metadata: Dict with void population details
        """
        c = self.config
        h, w = c.height, c.width

        # Start with matrix material
        image = self._generate_matrix(h, w)

        # Ground truth mask
        gt_mask = np.zeros((h, w), dtype=bool)
        void_records = []

        # Add voids of different types
        for _ in range(c.n_gas_pores):
            mask, record = self._add_gas_pore(image, h, w)
            gt_mask |= mask
            void_records.append(record)

        for _ in range(c.n_shrinkage_voids):
            mask, record = self._add_shrinkage_void(image, h, w)
            gt_mask |= mask
            void_records.append(record)

        for _ in range(c.n_delaminations):
            mask, record = self._add_delamination(image, h, w)
            gt_mask |= mask
            void_records.append(record)

        for _ in range(c.n_micro_voids):
            mask, record = self._add_micro_void(image, h, w)
            gt_mask |= mask
            void_records.append(record)

        # Apply void darkening to image
        void_intensity = self.rng.uniform(0.02, 0.12, size=(h, w))
        image[gt_mask] = void_intensity[gt_mask]

        # Add artifacts (should NOT be detected as voids)
        artifact_mask = np.zeros((h, w), dtype=bool)
        for _ in range(c.n_scratches):
            artifact_mask |= self._add_scratch(image, h, w)

        for _ in range(c.n_debris):
            artifact_mask |= self._add_debris(image, h, w)

        for _ in range(c.n_stains):
            artifact_mask |= self._add_stain(image, h, w)

        # Add illumination gradient
        image = self._add_illumination(image, h, w)

        # Add noise
        noise = self.rng.normal(0, c.noise_std, size=(h, w))
        image = np.clip(image + noise, 0, 1)

        # Compute ground truth porosity
        true_porosity = gt_mask.sum() / (h * w)

        metadata = {
            "image_size": (h, w),
            "true_porosity_fraction": float(true_porosity),
            "true_porosity_percent": float(true_porosity * 100),
            "n_true_voids": len(void_records),
            "void_records": void_records,
            "n_artifacts": c.n_scratches + c.n_debris + c.n_stains,
            "seed": c.seed,
        }

        return image.astype(np.float64), gt_mask, metadata

    def _generate_matrix(self, h: int, w: int) -> np.ndarray:
        """Generate base matrix material with grain-like texture."""
        c = self.config
        base = np.full((h, w), c.matrix_mean_intensity)

        # Add grain-like texture using random Voronoi-ish pattern
        for _ in range(c.grain_density):
            cy = self.rng.randint(0, h)
            cx = self.rng.randint(0, w)
            radius = self.rng.randint(10, 40)
            intensity_offset = self.rng.uniform(
                -c.matrix_texture_scale, c.matrix_texture_scale
            )

            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
            grain_mask = dist < radius
            # Soft edge
            weight = np.clip(1 - dist / radius, 0, 1)
            base += weight * intensity_offset

        return np.clip(base, 0, 1)

    def _add_gas_pore(
        self, image: np.ndarray, h: int, w: int
    ) -> tuple[np.ndarray, dict]:
        """Add a spherical gas pore (high circularity)."""
        margin = 30
        cy = self.rng.randint(margin, h - margin)
        cx = self.rng.randint(margin, w - margin)
        radius = self.rng.uniform(5, 25)

        Y, X = np.ogrid[:h, :w]
        # Slight ellipticity
        a = radius * self.rng.uniform(0.85, 1.15)
        b = radius * self.rng.uniform(0.85, 1.15)
        angle = self.rng.uniform(0, np.pi)

        dx = X - cx
        dy = Y - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)

        mask = (rx / a) ** 2 + (ry / b) ** 2 <= 1

        return mask, {
            "type": "gas_pore",
            "center": (int(cy), int(cx)),
            "radius": float(radius),
            "area_px": int(mask.sum()),
        }

    def _add_shrinkage_void(
        self, image: np.ndarray, h: int, w: int
    ) -> tuple[np.ndarray, dict]:
        """Add an irregular shrinkage void (low circularity)."""
        margin = 40
        cy = self.rng.randint(margin, h - margin)
        cx = self.rng.randint(margin, w - margin)

        # Create irregular shape by combining multiple ellipses
        mask = np.zeros((h, w), dtype=bool)
        n_lobes = self.rng.randint(3, 7)

        for _ in range(n_lobes):
            lobe_cy = cy + self.rng.randint(-15, 16)
            lobe_cx = cx + self.rng.randint(-15, 16)
            a = self.rng.uniform(5, 20)
            b = self.rng.uniform(3, 12)
            angle = self.rng.uniform(0, np.pi)

            Y, X = np.ogrid[:h, :w]
            dx = X - lobe_cx
            dy = Y - lobe_cy
            rx = dx * np.cos(angle) + dy * np.sin(angle)
            ry = -dx * np.sin(angle) + dy * np.cos(angle)
            lobe = (rx / max(a, 1)) ** 2 + (ry / max(b, 1)) ** 2 <= 1
            mask |= lobe

        return mask, {
            "type": "shrinkage_void",
            "center": (int(cy), int(cx)),
            "area_px": int(mask.sum()),
        }

    def _add_delamination(
        self, image: np.ndarray, h: int, w: int
    ) -> tuple[np.ndarray, dict]:
        """Add an elongated delamination/disbond."""
        margin = 30
        cy = self.rng.randint(margin, h - margin)
        cx = self.rng.randint(margin, w - margin)

        length = self.rng.uniform(30, 80)
        width = self.rng.uniform(3, 10)
        angle = self.rng.uniform(0, np.pi)

        Y, X = np.ogrid[:h, :w]
        dx = X - cx
        dy = Y - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)

        mask = (rx / (length / 2)) ** 2 + (ry / (width / 2)) ** 2 <= 1

        return mask, {
            "type": "delamination",
            "center": (int(cy), int(cx)),
            "length": float(length),
            "width": float(width),
            "area_px": int(mask.sum()),
        }

    def _add_micro_void(
        self, image: np.ndarray, h: int, w: int
    ) -> tuple[np.ndarray, dict]:
        """Add a very small micro-void."""
        margin = 10
        cy = self.rng.randint(margin, h - margin)
        cx = self.rng.randint(margin, w - margin)
        radius = self.rng.uniform(2, 5)

        Y, X = np.ogrid[:h, :w]
        mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2

        return mask, {
            "type": "micro_void",
            "center": (int(cy), int(cx)),
            "radius": float(radius),
            "area_px": int(mask.sum()),
        }

    def _add_scratch(
        self, image: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Add a sample preparation scratch artifact."""
        y0 = self.rng.randint(0, h)
        x0 = self.rng.randint(0, w)
        angle = self.rng.uniform(0, np.pi)
        length = self.rng.uniform(100, 300)
        width = self.rng.uniform(1, 3)

        y1 = int(y0 + length * np.sin(angle))
        x1 = int(x0 + length * np.cos(angle))

        mask = np.zeros((h, w), dtype=bool)

        # Draw line using Bresenham-like approach
        n_points = max(abs(y1 - y0), abs(x1 - x0), 1)
        ys = np.linspace(y0, y1, n_points).astype(int)
        xs = np.linspace(x0, x1, n_points).astype(int)

        for y, x in zip(ys, xs):
            for dy in range(-int(width), int(width) + 1):
                for dx in range(-int(width), int(width) + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        mask[ny, nx] = True
                        # Scratches are darker but not as dark as voids
                        image[ny, nx] *= self.rng.uniform(0.4, 0.7)

        return mask

    def _add_debris(
        self, image: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Add polishing debris artifact."""
        cy = self.rng.randint(5, h - 5)
        cx = self.rng.randint(5, w - 5)
        radius = self.rng.uniform(2, 6)

        Y, X = np.ogrid[:h, :w]
        mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2

        # Debris is very dark, small, irregular
        image[mask] *= self.rng.uniform(0.2, 0.5)

        return mask

    def _add_stain(
        self, image: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Add staining artifact (diffuse dark region)."""
        cy = self.rng.randint(30, h - 30)
        cx = self.rng.randint(30, w - 30)
        radius = self.rng.uniform(20, 60)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        mask = dist < radius

        # Stains have gradual edges (key difference from voids)
        weight = np.clip(1 - dist / radius, 0, 1) ** 2
        image -= weight * self.rng.uniform(0.1, 0.25)
        image = np.clip(image, 0, 1)

        return mask

    def _add_illumination(
        self, image: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Add uneven illumination gradient."""
        grad = self.config.illumination_gradient
        Y, X = np.meshgrid(
            np.linspace(-grad, grad, h),
            np.linspace(-grad / 2, grad / 2, w),
            indexing="ij",
        )
        angle = self.rng.uniform(0, 2 * np.pi)
        gradient = Y * np.cos(angle) + X * np.sin(angle)
        return np.clip(image + gradient, 0, 1)
