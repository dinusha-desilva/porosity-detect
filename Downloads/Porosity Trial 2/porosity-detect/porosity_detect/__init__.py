"""
porosity_detect — Hybrid Classical + ML Void/Porosity Detection for Optical Microscopy

A materials informatics toolkit that bridges materials science domain knowledge
with data science methods to detect, segment, and quantify voids and porosity
in optical microscopy cross-section images of aerospace materials.

Author: Dinusha
License: Apache-2.0
"""

__version__ = "0.1.0"
__author__ = "Dinusha"

from porosity_detect.classical import ClassicalDetector
from porosity_detect.features import FeatureExtractor
from porosity_detect.ml_model import MLDetector
from porosity_detect.hybrid import HybridPipeline
from porosity_detect.metrics import PorosityMetrics
from porosity_detect.two_pass import TwoPassDetector, TwoPassParams, PRESETS

__all__ = [
    "ClassicalDetector",
    "FeatureExtractor",
    "MLDetector",
    "HybridPipeline",
    "PorosityMetrics",
    "TwoPassDetector",
    "TwoPassParams",
    "PRESETS",
]
