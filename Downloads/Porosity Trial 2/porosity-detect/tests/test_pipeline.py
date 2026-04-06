"""Unit tests for porosity-detect pipeline."""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_synthetic_generation():
    """Test synthetic image generation produces valid output."""
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig

    config = SyntheticConfig(width=256, height=256, seed=42)
    gen = SyntheticMicrograph(config)
    image, gt, meta = gen.generate()

    assert image.shape == (256, 256), f"Wrong shape: {image.shape}"
    assert image.min() >= 0 and image.max() <= 1, "Image out of range"
    assert gt.shape == (256, 256), f"Wrong GT shape"
    assert gt.dtype == bool, "GT should be boolean"
    assert meta["n_true_voids"] > 0, "Should have voids"
    assert 0 < meta["true_porosity_fraction"] < 1, "Porosity out of range"
    print("✓ Synthetic generation OK")


def test_classical_detection():
    """Test classical detection pipeline."""
    from porosity_detect.classical import ClassicalDetector, ClassicalParams
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig

    config = SyntheticConfig(width=256, height=256, seed=42)
    gen = SyntheticMicrograph(config)
    image, gt, _ = gen.generate()

    params = ClassicalParams(min_void_area_px=10)
    detector = ClassicalDetector(params)
    labels, props, meta = detector.detect(image)

    assert labels.shape == image.shape, "Labels wrong shape"
    assert len(props) > 0, "Should detect some voids"
    assert meta["n_voids_detected"] == len(props)
    assert 0 <= meta["porosity_fraction"] <= 1

    # Check void properties
    for p in props:
        assert "area_px" in p
        assert "circularity" in p
        assert 0 <= p["circularity"] <= 1
        assert p["aspect_ratio"] >= 1
    print(f"✓ Classical detection OK ({len(props)} voids detected)")


def test_feature_extraction():
    """Test feature extraction produces correct-size vectors."""
    from porosity_detect.classical import ClassicalDetector, ClassicalParams
    from porosity_detect.features import FeatureExtractor
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig

    config = SyntheticConfig(width=256, height=256, seed=42)
    gen = SyntheticMicrograph(config)
    image, _, _ = gen.generate()

    detector = ClassicalDetector(ClassicalParams(min_void_area_px=10))
    labels, props, _ = detector.detect(image)
    preprocessed = detector.preprocess(image)

    extractor = FeatureExtractor()
    features = extractor.extract_batch(preprocessed, labels, props)

    n_expected = len(FeatureExtractor.FEATURE_NAMES)
    assert features.shape == (len(props), n_expected), (
        f"Expected ({len(props)}, {n_expected}), got {features.shape}"
    )
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"
    print(f"✓ Feature extraction OK ({features.shape})")


def test_ml_model():
    """Test ML model training and prediction."""
    from porosity_detect.ml_model import MLDetector, MLParams

    params = MLParams(n_estimators=10, max_depth=5)
    detector = MLDetector(params)

    X_syn, y_syn = detector.generate_synthetic_training_data(
        n_voids=50, n_artifacts=50
    )

    assert X_syn.shape == (100, 25), f"Wrong shape: {X_syn.shape}"
    assert len(y_syn) == 100

    metrics = detector.train(X_syn, y_syn)
    assert detector.is_trained
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] > 0.5, "Model should be better than random"

    preds, proba = detector.predict(X_syn[:10])
    assert len(preds) == 10
    assert len(proba) == 10
    assert all(p in [0, 1] for p in preds)
    assert all(0 <= p <= 1 for p in proba)
    print(f"✓ ML model OK (accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f})")


def test_hybrid_pipeline():
    """Test full hybrid pipeline end-to-end."""
    from porosity_detect.hybrid import HybridPipeline, HybridConfig
    from porosity_detect.synthetic import SyntheticMicrograph, SyntheticConfig

    config = SyntheticConfig(width=256, height=256, seed=42)
    gen = SyntheticMicrograph(config)
    image, gt, gt_meta = gen.generate()

    pipeline = HybridPipeline(HybridConfig(auto_train=True))
    results = pipeline.analyze(image)

    assert "labels" in results
    assert "void_properties" in results
    assert "porosity_metrics" in results
    assert "pipeline_metadata" in results

    pm = results["porosity_metrics"]
    assert 0 <= pm["porosity_percent"] <= 100
    assert pm["void_count"] >= 0

    meta = results["pipeline_metadata"]
    assert meta["method"] == "hybrid"
    assert meta["n_confirmed_voids"] <= meta["n_classical_candidates"]

    # Check void properties have expected fields
    for vp in results["void_properties"]:
        assert "void_type" in vp
        assert "ml_confidence" in vp
        assert "area_px" in vp

    print(f"✓ Hybrid pipeline OK")
    print(f"  True porosity:     {gt_meta['true_porosity_percent']:.4f}%")
    print(f"  Detected porosity: {pm['porosity_percent']:.4f}%")
    print(f"  Voids detected:    {pm['void_count']}")


def test_metrics():
    """Test porosity metrics computation."""
    from porosity_detect.metrics import PorosityMetrics

    calc = PorosityMetrics(pixel_size_um=0.5)

    # Create dummy data
    labels = np.zeros((100, 100), dtype=int)
    labels[10:20, 10:20] = 1
    labels[50:55, 50:55] = 2

    props = [
        {
            "label": 1, "area_px": 100, "circularity": 0.8,
            "aspect_ratio": 1.2, "equivalent_diameter_px": 11.3,
            "centroid_y": 15.0, "centroid_x": 15.0,
        },
        {
            "label": 2, "area_px": 25, "circularity": 0.9,
            "aspect_ratio": 1.0, "equivalent_diameter_px": 5.6,
            "centroid_y": 52.5, "centroid_x": 52.5,
        },
    ]

    metrics = calc.compute(labels, props, (100, 100))

    assert metrics["void_count"] == 2
    assert metrics["total_void_area_px"] == 125
    assert abs(metrics["porosity_percent"] - 1.25) < 0.01
    assert metrics["image_area_px"] == 10000

    report = calc.format_report(metrics)
    assert "POROSITY ANALYSIS REPORT" in report
    assert "1.25" in report
    print(f"✓ Metrics OK (porosity={metrics['porosity_percent']:.2f}%)")


def test_empty_image():
    """Test pipeline handles image with no voids."""
    from porosity_detect.hybrid import HybridPipeline

    # Uniform gray image (no voids)
    image = np.full((100, 100), 0.6)

    pipeline = HybridPipeline()
    results = pipeline.analyze(image)

    assert results["porosity_metrics"]["void_count"] == 0
    assert results["porosity_metrics"]["porosity_percent"] == 0.0
    print("✓ Empty image handling OK")


if __name__ == "__main__":
    print("Running porosity-detect tests...\n")
    test_synthetic_generation()
    test_classical_detection()
    test_feature_extraction()
    test_ml_model()
    test_hybrid_pipeline()
    test_metrics()
    test_empty_image()
    print("\n✅ All tests passed!")
