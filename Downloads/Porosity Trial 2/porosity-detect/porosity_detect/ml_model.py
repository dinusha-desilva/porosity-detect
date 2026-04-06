"""
Machine learning classifier for void/artifact discrimination.

Trains a Random Forest classifier on physics-informed features to distinguish
true manufacturing-induced voids from common artifacts in optical microscopy:

- Sample preparation scratches
- Polishing debris/contamination
- Staining artifacts
- Imaging noise
- Inclusions (non-metallic particles that appear dark but aren't voids)

The ML layer adds value over classical methods by learning complex
decision boundaries in feature space that encode expert knowledge about
what constitutes a "real" void vs an artifact.
"""

import numpy as np
import json
import warnings
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class MLParams:
    """Parameters for the Random Forest classifier."""

    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    class_weight: str = "balanced"  # Handle imbalanced void/artifact ratio


class SimpleDecisionTree:
    """Minimal decision tree implementation for void classification."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _build_tree(self, X, y, w, depth):
        n_samples = len(y)
        if w is None:
            w = np.ones(n_samples)

        class_counts = {}
        for c in np.unique(y):
            class_counts[int(c)] = float(w[y == c].sum())

        node = {"class_counts": class_counts, "n_samples": n_samples}

        # Leaf conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node["leaf"] = True
            node["prediction"] = max(class_counts, key=class_counts.get)
            node["probability"] = {
                k: v / sum(class_counts.values())
                for k, v in class_counts.items()
            }
            return node

        # Find best split (random subset of features)
        n_features = X.shape[1]
        feature_subset = self.rng.choice(
            n_features,
            size=max(1, int(np.sqrt(n_features))),
            replace=False,
        )

        best_gain = -1
        best_feat = None
        best_thresh = None

        total_weight = w.sum()
        parent_impurity = self._gini(y, w)

        for feat in feature_subset:
            values = np.unique(X[:, feat])
            if len(values) <= 1:
                continue

            # Try a subset of thresholds
            if len(values) > 20:
                thresholds = np.percentile(
                    X[:, feat], np.linspace(10, 90, 10)
                )
            else:
                thresholds = (values[:-1] + values[1:]) / 2

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                left_w = w[left_mask].sum()
                right_w = w[right_mask].sum()

                gain = parent_impurity - (
                    left_w / total_weight
                    * self._gini(y[left_mask], w[left_mask])
                    + right_w / total_weight
                    * self._gini(y[right_mask], w[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            node["leaf"] = True
            node["prediction"] = max(class_counts, key=class_counts.get)
            node["probability"] = {
                k: v / sum(class_counts.values())
                for k, v in class_counts.items()
            }
            return node

        node["leaf"] = False
        node["feature"] = int(best_feat)
        node["threshold"] = float(best_thresh)

        left_mask = X[:, best_feat] <= best_thresh
        node["left"] = self._build_tree(
            X[left_mask], y[left_mask], w[left_mask], depth + 1
        )
        node["right"] = self._build_tree(
            X[~left_mask], y[~left_mask], w[~left_mask], depth + 1
        )

        return node

    def _gini(self, y, w):
        total = w.sum()
        if total == 0:
            return 0.0
        impurity = 1.0
        for c in np.unique(y):
            p = w[y == c].sum() / total
            impurity -= p ** 2
        return impurity

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            prob = self._traverse(x, self.tree)
            proba[i, 0] = prob.get(0, 0.0)
            proba[i, 1] = prob.get(1, 0.0)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def _traverse(self, x, node):
        if node["leaf"]:
            return node["probability"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])


class SimpleRandomForest:
    """Minimal Random Forest for void classification."""

    def __init__(self, params: MLParams):
        self.params = params
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.params.random_state)
        n_samples = len(y)

        # Compute class weights
        if self.params.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = n_samples
            weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        else:
            weights = {0: 1.0, 1: 1.0}

        sample_weight = np.array([weights[int(yi)] for yi in y])

        self.trees = []
        for i in range(self.params.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            w_boot = sample_weight[indices]

            tree = SimpleDecisionTree(
                max_depth=self.params.max_depth,
                min_samples_split=self.params.min_samples_split,
                min_samples_leaf=self.params.min_samples_leaf,
                random_state=self.params.random_state + i,
            )
            tree.fit(X_boot, y_boot, w_boot)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba = np.array([t.predict_proba(X) for t in self.trees])
        return all_proba.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class MLDetector:
    """ML-based void classifier using Random Forest on physics-informed features.

    Wraps a Random Forest that classifies candidate regions as true voids
    vs artifacts based on the feature vectors from FeatureExtractor.
    """

    def __init__(self, params: Optional[MLParams] = None):
        self.params = params or MLParams()
        self.model = None
        self.is_trained = False
        self.feature_importance = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Train the classifier on labeled void/artifact data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (1 = true void, 0 = artifact).

        Returns:
            Training metrics dict.
        """
        if X.shape[0] < 10:
            warnings.warn("Very small training set. Results may be unreliable.")

        self.model = SimpleRandomForest(self.params)
        self.model.fit(X, y)
        self.is_trained = True

        # Training predictions for metrics
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        metrics = self._compute_metrics(y, y_pred, y_proba)
        metrics["n_training_samples"] = int(len(y))
        metrics["n_voids"] = int((y == 1).sum())
        metrics["n_artifacts"] = int((y == 0).sum())

        return metrics

    def predict(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict void/artifact classification.

        Args:
            X: Feature matrix.
            threshold: Classification threshold (default 0.5).

        Returns:
            Tuple of (predictions, probabilities).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        proba = self.model.predict_proba(X)
        predictions = (proba[:, 1] >= threshold).astype(int)

        return predictions, proba[:, 1]

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        """Compute classification metrics."""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        accuracy = (tp + tn) / max(len(y_true), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-6)
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    def save_model(self, path: str):
        """Save trained model to JSON."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save.")
        # Serialize tree structures
        model_data = {
            "params": self.params.__dict__,
            "n_trees": len(self.model.trees),
            "trees": [self._serialize_tree(t.tree) for t in self.model.trees],
        }
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def _serialize_tree(self, node: dict) -> dict:
        """Recursively serialize a decision tree node."""
        serialized = {
            "leaf": node["leaf"],
            "class_counts": {str(k): v for k, v in node["class_counts"].items()},
            "n_samples": node["n_samples"],
        }
        if node["leaf"]:
            serialized["prediction"] = node["prediction"]
            serialized["probability"] = {str(k): v for k, v in node["probability"].items()}
        else:
            serialized["feature"] = node["feature"]
            serialized["threshold"] = node["threshold"]
            serialized["left"] = self._serialize_tree(node["left"])
            serialized["right"] = self._serialize_tree(node["right"])
        return serialized

    def generate_synthetic_training_data(
        self, n_voids: int = 200, n_artifacts: int = 200, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data based on domain knowledge.

        Creates realistic feature distributions for true voids and
        common artifacts based on materials science expertise.

        This encodes expert knowledge about how different types of
        regions differ in their feature signatures.
        """
        rng = np.random.RandomState(seed)

        # --- True void features ---
        void_features = []
        for _ in range(n_voids):
            # Gas porosity (60% of voids) — round, moderate size
            # Shrinkage porosity (30%) — irregular, varied size
            # Delamination (10%) — elongated
            void_type = rng.choice(
                ["gas", "shrinkage", "delam"], p=[0.6, 0.3, 0.1]
            )

            if void_type == "gas":
                area = rng.lognormal(mean=6.0, sigma=0.8)
                circularity = rng.beta(8, 2) * 0.7 + 0.3
                aspect_ratio = 1.0 + rng.exponential(0.3)
            elif void_type == "shrinkage":
                area = rng.lognormal(mean=6.5, sigma=1.0)
                circularity = rng.beta(3, 5) * 0.6 + 0.1
                aspect_ratio = 1.0 + rng.exponential(0.8)
            else:  # delamination
                area = rng.lognormal(mean=7.0, sigma=0.5)
                circularity = rng.beta(2, 8) * 0.4 + 0.05
                aspect_ratio = 3.0 + rng.exponential(2.0)

            perimeter = np.sqrt(4 * np.pi * area / max(circularity, 0.01))
            eq_diameter = np.sqrt(4 * area / np.pi)
            solidity = rng.beta(6, 2) * 0.5 + 0.5
            extent = rng.beta(5, 3) * 0.5 + 0.3
            compactness = perimeter ** 2 / max(area, 1)

            mean_intensity = rng.beta(2, 8) * 0.3  # Dark
            std_intensity = rng.exponential(0.02) + 0.01
            p10 = mean_intensity * rng.uniform(0.5, 0.8)
            p25 = mean_intensity * rng.uniform(0.7, 0.9)
            p50 = mean_intensity
            p75 = mean_intensity * rng.uniform(1.05, 1.2)
            p90 = mean_intensity * rng.uniform(1.1, 1.4)
            boundary_contrast = rng.uniform(0.2, 0.6)
            int_range = rng.uniform(0.05, 0.2)
            skewness = rng.normal(0.3, 0.5)

            grad_mean = rng.uniform(0.05, 0.2)
            grad_std = rng.uniform(0.02, 0.08)
            homogeneity = rng.uniform(0.6, 0.95)
            edge_sharpness = rng.uniform(0.1, 0.4)

            local_density = rng.exponential(0.5)
            relative_brightness = rng.uniform(0.1, 0.5)
            dist_to_edge = rng.uniform(0.05, 0.95)

            void_features.append([
                area, perimeter, circularity, aspect_ratio, eq_diameter,
                solidity, extent, compactness,
                mean_intensity, std_intensity, p10, p25, p50, p75, p90,
                boundary_contrast, int_range, skewness,
                grad_mean, grad_std, homogeneity, edge_sharpness,
                local_density, relative_brightness, dist_to_edge,
            ])

        # --- Artifact features ---
        artifact_features = []
        for _ in range(n_artifacts):
            artifact_type = rng.choice(
                ["scratch", "debris", "stain", "noise"], p=[0.3, 0.3, 0.2, 0.2]
            )

            if artifact_type == "scratch":
                area = rng.lognormal(mean=5.5, sigma=1.0)
                circularity = rng.beta(2, 10) * 0.2
                aspect_ratio = 5.0 + rng.exponential(3.0)
            elif artifact_type == "debris":
                area = rng.lognormal(mean=4.0, sigma=0.5)
                circularity = rng.beta(5, 3) * 0.8 + 0.1
                aspect_ratio = 1.0 + rng.exponential(0.5)
            elif artifact_type == "stain":
                area = rng.lognormal(mean=7.0, sigma=1.5)
                circularity = rng.beta(3, 3) * 0.6 + 0.2
                aspect_ratio = 1.0 + rng.exponential(1.0)
            else:  # noise
                area = rng.lognormal(mean=3.5, sigma=0.5)
                circularity = rng.beta(4, 4) * 0.8 + 0.1
                aspect_ratio = 1.0 + rng.exponential(0.5)

            perimeter = np.sqrt(4 * np.pi * area / max(circularity, 0.01))
            eq_diameter = np.sqrt(4 * area / np.pi)
            solidity = rng.beta(3, 4) * 0.6 + 0.2
            extent = rng.beta(3, 3) * 0.6 + 0.1
            compactness = perimeter ** 2 / max(area, 1)

            mean_intensity = rng.beta(3, 5) * 0.5 + 0.1  # Brighter than voids
            std_intensity = rng.exponential(0.05) + 0.03
            p10 = mean_intensity * rng.uniform(0.4, 0.7)
            p25 = mean_intensity * rng.uniform(0.6, 0.85)
            p50 = mean_intensity
            p75 = mean_intensity * rng.uniform(1.1, 1.3)
            p90 = mean_intensity * rng.uniform(1.2, 1.6)
            boundary_contrast = rng.uniform(0.02, 0.25)
            int_range = rng.uniform(0.1, 0.5)
            skewness = rng.normal(-0.2, 0.8)

            grad_mean = rng.uniform(0.02, 0.12)
            grad_std = rng.uniform(0.01, 0.06)
            homogeneity = rng.uniform(0.3, 0.7)
            edge_sharpness = rng.uniform(0.02, 0.2)

            local_density = rng.exponential(0.3)
            relative_brightness = rng.uniform(0.3, 0.8)
            dist_to_edge = rng.uniform(0.0, 0.9)

            artifact_features.append([
                area, perimeter, circularity, aspect_ratio, eq_diameter,
                solidity, extent, compactness,
                mean_intensity, std_intensity, p10, p25, p50, p75, p90,
                boundary_contrast, int_range, skewness,
                grad_mean, grad_std, homogeneity, edge_sharpness,
                local_density, relative_brightness, dist_to_edge,
            ])

        X = np.vstack([void_features, artifact_features])
        y = np.concatenate([np.ones(n_voids), np.zeros(n_artifacts)])

        # Shuffle
        perm = rng.permutation(len(y))
        return X[perm], y[perm]
