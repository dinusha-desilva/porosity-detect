"""
Hybrid Classical + ML Pipeline for Void/Porosity Detection.

This is the core innovation: combining classical image processing (which
encodes physics-based domain knowledge) with machine learning (which learns
complex decision boundaries from data).

Pipeline Architecture:
1. Classical Detection → Candidate regions (high recall, lower precision)
2. Feature Extraction → Physics-informed feature vectors
3. ML Classification → True void vs artifact discrimination (high precision)
4. Metrics Computation → Quantitative porosity analysis

This hybrid approach mirrors best practices in materials informatics:
- Use domain knowledge to constrain the problem space
- Use data-driven methods to handle complexity within that space
- Produce quantitative, reproducible results
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from porosity_detect.classical import ClassicalDetector, ClassicalParams
from porosity_detect.features import FeatureExtractor, FeatureConfig
from porosity_detect.ml_model import MLDetector, MLParams
from porosity_detect.metrics import PorosityMetrics


@dataclass
class HybridConfig:
    """Configuration for the hybrid pipeline."""

    classical_params: Optional[ClassicalParams] = None
    feature_config: Optional[FeatureConfig] = None
    ml_params: Optional[MLParams] = None

    # ML classification threshold
    confidence_threshold: float = 0.5

    # Pixel-to-micron conversion (depends on magnification)
    pixel_size_um: float = 1.0  # micrometers per pixel

    # Auto-train on synthetic data if no model provided
    auto_train: bool = True


class HybridPipeline:
    """Hybrid Classical + ML void/porosity detection pipeline.

    Combines the strengths of both approaches:
    - Classical: Physics-based, interpretable, no training data needed
    - ML: Learns complex patterns, reduces false positives

    Usage:
        pipeline = HybridPipeline()
        results = pipeline.analyze(image)
        print(f"Porosity: {results['porosity_percent']:.2f}%")
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()

        self.classical = ClassicalDetector(
            self.config.classical_params or ClassicalParams()
        )
        self.extractor = FeatureExtractor(
            self.config.feature_config or FeatureConfig()
        )
        self.ml = MLDetector(self.config.ml_params or MLParams())

        if self.config.auto_train and not self.ml.is_trained:
            self._auto_train()

    def _auto_train(self):
        """Train ML model on synthetic data generated from domain knowledge."""
        X_syn, y_syn = self.ml.generate_synthetic_training_data(
            n_voids=300, n_artifacts=300
        )
        metrics = self.ml.train(X_syn, y_syn)
        self._training_metrics = metrics

    def analyze(self, image: np.ndarray) -> dict:
        """Run full hybrid analysis pipeline.

        Args:
            image: Input microscopy image (grayscale or color).

        Returns:
            Dictionary containing:
            - labels: Label image of confirmed voids
            - void_properties: List of dicts with per-void measurements
            - porosity_metrics: Overall porosity statistics
            - pipeline_metadata: Processing details
        """
        # Step 1: Classical detection (high recall)
        preprocessed = self.classical.preprocess(image)
        classical_labels, classical_props, classical_meta = (
            self.classical.detect(image)
        )

        if len(classical_props) == 0:
            return self._empty_result(image)

        # Step 2: Feature extraction
        features = self.extractor.extract_batch(
            preprocessed, classical_labels, classical_props
        )

        # Step 3: ML classification (high precision filter)
        if self.ml.is_trained and features.shape[0] > 0:
            predictions, confidences = self.ml.predict(
                features, threshold=self.config.confidence_threshold
            )

            # Filter to confirmed voids only
            confirmed_props = []
            confirmed_labels = np.zeros_like(classical_labels)
            new_label = 0

            for i, (pred, conf, props) in enumerate(
                zip(predictions, confidences, classical_props)
            ):
                if pred == 1:  # Confirmed void
                    new_label += 1
                    old_label = props["label"]
                    confirmed_labels[classical_labels == old_label] = new_label
                    props["label"] = new_label
                    props["ml_confidence"] = float(conf)
                    props["classification"] = "void"
                    confirmed_props.append(props)
                else:
                    props["ml_confidence"] = float(conf)
                    props["classification"] = "artifact"
        else:
            # No ML model — use classical results directly
            confirmed_labels = classical_labels
            confirmed_props = classical_props
            for p in confirmed_props:
                p["ml_confidence"] = None
                p["classification"] = "void (classical only)"

        # Step 4: Compute porosity metrics
        px_size = self.config.pixel_size_um
        metrics_calc = PorosityMetrics(pixel_size_um=px_size)
        porosity_metrics = metrics_calc.compute(
            confirmed_labels, confirmed_props, image.shape[:2]
        )

        # Add void classification to properties
        for props in confirmed_props:
            props["void_type"] = self._classify_void_type(props)
            # Convert pixel measurements to physical units
            props["area_um2"] = props["area_px"] * (px_size ** 2)
            props["equivalent_diameter_um"] = (
                props["equivalent_diameter_px"] * px_size
            )

        return {
            "labels": confirmed_labels,
            "void_properties": confirmed_props,
            "porosity_metrics": porosity_metrics,
            "features": features,
            "classical_metadata": classical_meta,
            "pipeline_metadata": {
                "method": "hybrid",
                "n_classical_candidates": len(classical_props),
                "n_confirmed_voids": len(confirmed_props),
                "n_rejected_artifacts": (
                    len(classical_props) - len(confirmed_props)
                ),
                "ml_trained": self.ml.is_trained,
                "confidence_threshold": self.config.confidence_threshold,
                "pixel_size_um": px_size,
            },
        }

    def _classify_void_type(self, props: dict) -> str:
        """Classify void formation mechanism based on morphology.

        Domain knowledge encoding:
        - Gas porosity: high circularity (>0.7), moderate size
        - Shrinkage porosity: low circularity (<0.5), irregular shape
        - Delamination/disbond: very high aspect ratio (>3)
        - Micro-void: very small equivalent diameter
        """
        circ = props.get("circularity", 0)
        ar = props.get("aspect_ratio", 1)
        area = props.get("area_px", 0)

        if ar > 3.0:
            return "delamination"
        elif circ > 0.7:
            return "gas_porosity"
        elif circ < 0.4:
            return "shrinkage_porosity"
        elif area < 100:
            return "micro_void"
        else:
            return "unclassified"

    def _empty_result(self, image: np.ndarray) -> dict:
        """Return empty result when no voids detected."""
        return {
            "labels": np.zeros(image.shape[:2], dtype=int),
            "void_properties": [],
            "porosity_metrics": {
                "porosity_area_fraction": 0.0,
                "porosity_percent": 0.0,
                "void_count": 0,
                "total_void_area_px": 0,
                "total_void_area_um2": 0.0,
                "mean_void_area_px": 0.0,
                "void_size_distribution": {},
            },
            "features": np.zeros((0, len(FeatureExtractor.FEATURE_NAMES))),
            "classical_metadata": {"n_voids_detected": 0},
            "pipeline_metadata": {
                "method": "hybrid",
                "n_classical_candidates": 0,
                "n_confirmed_voids": 0,
                "n_rejected_artifacts": 0,
            },
        }
