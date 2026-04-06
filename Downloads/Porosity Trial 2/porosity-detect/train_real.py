"""
Train porosity-detect classifier on real annotated data.

Uses annotations created by the annotate.py tool to train
a Random Forest classifier on real micrograph data. The trained
model can then be used for future analyses without re-annotation.

Usage:
    python train_real.py annotations/
    python train_real.py annotations/ --combine-synthetic
    python train_real.py annotations/ --output trained_model.json
"""

import argparse
import json
import sys
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Train void classifier on real annotated data"
    )
    parser.add_argument(
        "annotation_dir",
        help="Directory containing X_train.npy and y_train.npy from annotate.py",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save trained model (default: annotation_dir/trained_model.json)",
    )
    parser.add_argument(
        "--combine-synthetic", action="store_true",
        help="Augment real data with synthetic training data",
    )
    parser.add_argument(
        "--synthetic-ratio", type=float, default=0.3,
        help="Fraction of synthetic data when combining (default: 0.3)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees in Random Forest (default: 100)",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    args = parser.parse_args()

    ann_dir = args.annotation_dir

    # Load training data
    X_path = os.path.join(ann_dir, "X_train.npy")
    y_path = os.path.join(ann_dir, "y_train.npy")

    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        print(f"Error: Training data not found in {ann_dir}")
        print(f"Expected: X_train.npy and y_train.npy")
        print(f"Run annotate.py first and press 'S' to save.")
        sys.exit(1)

    X_real = np.load(X_path)
    y_real = np.load(y_path)

    n_voids = int((y_real == 1).sum())
    n_artifacts = int((y_real == 0).sum())
    print(f"Loaded real training data:")
    print(f"  Samples: {len(y_real)} ({n_voids} voids, {n_artifacts} artifacts)")
    print(f"  Features: {X_real.shape[1]}")

    if n_voids < 3 or n_artifacts < 3:
        print(f"\nWarning: Very few samples. Consider annotating more regions.")
        print(f"Minimum recommended: 10+ voids, 10+ artifacts.")

    # Optionally combine with synthetic data
    if args.combine_synthetic:
        try:
            from porosity_detect.ml_model import MLDetector

            detector = MLDetector()
            n_syn = int(len(y_real) * args.synthetic_ratio / (1 - args.synthetic_ratio))
            n_syn_voids = max(n_syn // 2, 5)
            n_syn_artifacts = max(n_syn // 2, 5)

            X_syn, y_syn = detector.generate_synthetic_training_data(
                n_voids=n_syn_voids, n_artifacts=n_syn_artifacts
            )

            X_combined = np.vstack([X_real, X_syn])
            y_combined = np.concatenate([y_real, y_syn])

            print(f"\nCombined with synthetic data:")
            print(f"  Real: {len(y_real)}, Synthetic: {len(y_syn)}")
            print(f"  Total: {len(y_combined)}")

            X_train = X_combined
            y_train = y_combined
        except ImportError:
            print("Warning: Could not import porosity_detect. Using real data only.")
            X_train = X_real
            y_train = y_real
    else:
        X_train = X_real
        y_train = y_real

    # Train/test split
    n = len(y_train)
    indices = np.random.RandomState(42).permutation(n)
    n_test = max(int(n * args.test_split), 1)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_te, y_te = X_train[test_idx], y_train[test_idx]

    print(f"\nTrain/test split: {len(y_tr)} train, {len(y_te)} test")

    # Train
    try:
        from porosity_detect.ml_model import MLDetector, MLParams

        params = MLParams(n_estimators=args.n_estimators)
        detector = MLDetector(params)
        metrics = detector.train(X_tr, y_tr)

        print(f"\nTraining metrics (on training set):")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall:    {metrics.get('recall', 0):.3f}")
        print(f"  F1 Score:  {metrics.get('f1_score', 0):.3f}")

        # Evaluate on test set
        if len(y_te) > 0:
            preds, proba = detector.predict(X_te)
            tp = ((preds == 1) & (y_te == 1)).sum()
            fp = ((preds == 1) & (y_te == 0)).sum()
            fn = ((preds == 0) & (y_te == 1)).sum()
            tn = ((preds == 0) & (y_te == 0)).sum()

            test_acc = (tp + tn) / max(len(y_te), 1)
            test_prec = tp / max(tp + fp, 1)
            test_rec = tp / max(tp + fn, 1)
            test_f1 = 2 * test_prec * test_rec / max(test_prec + test_rec, 1e-6)

            print(f"\nTest set metrics:")
            print(f"  Accuracy:  {test_acc:.3f}")
            print(f"  Precision: {test_prec:.3f}")
            print(f"  Recall:    {test_rec:.3f}")
            print(f"  F1 Score:  {test_f1:.3f}")
            print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

        # Save model
        model_path = args.output or os.path.join(ann_dir, "trained_model.json")
        detector.save_model(model_path)
        print(f"\nTrained model saved: {model_path}")
        print(f"\nTo use this model for future analyses:")
        print(f"  porosity-detect analyze image.tif --model {model_path}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure porosity-detect is installed (pip install -e .)")
        sys.exit(1)


if __name__ == "__main__":
    main()
