"""
Unit tests for ML Agent pipeline components.
Run with: ./.venv/bin/python -m unittest tests/test_unit.py
"""
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import unittest
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from src.feature_selection import variance_filter, correlation_filter
from src.class_balancing import class_distribution, compute_class_weights
from src.pca_analysis import run_pca
from src.clustering_kmeans import fit_kmeans, evaluate_k_range
from src.clustering_hierarchical import fit_hierarchical
from src.model_selection import run_model_selection
from src.model_selection_regression import run_model_selection as run_model_selection_reg


class TestFeatureSelection(unittest.TestCase):
    def test_variance_filter_drops_constant_feature(self):
        df = pd.DataFrame({
            "constant_col": [1.0, 1.0, 1.0, 1.0],
            "var_col": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1]
        })
        filtered_df, log = variance_filter(df, target="target", threshold=0.01)
        self.assertNotIn("constant_col", filtered_df.columns)
        self.assertIn("var_col", filtered_df.columns)
        self.assertIn("target", filtered_df.columns)

    def test_correlation_filter_drops_collinear_pair(self):
        x = np.linspace(1, 10, 20)
        df = pd.DataFrame({
            "col_a": x,
            "col_b": x * 2.0,  # perfect correlation
            "col_c": np.random.RandomState(42).randn(20),
            "target": np.random.RandomState(42).randint(0, 2, 20)
        })
        filtered_df, log = correlation_filter(df, target="target", threshold=0.95)
        # Exactly one of col_a or col_b should be dropped
        cols = filtered_df.columns.tolist()
        self.assertTrue(("col_a" in cols) ^ ("col_b" in cols))
        self.assertIn("col_c", cols)


class TestClassBalancing(unittest.TestCase):
    def test_class_distribution_percentages(self):
        y = pd.Series([0, 0, 0, 1])
        dist = class_distribution(y)
        self.assertEqual(dist["0"]["count"], 3)
        self.assertEqual(dist["0"]["pct"], 75.0)
        self.assertEqual(dist["1"]["count"], 1)
        self.assertEqual(dist["1"]["pct"], 25.0)

    def test_compute_class_weights(self):
        y = pd.Series([0, 0, 0, 1])
        weights = compute_class_weights(y)
        # Class 1 should have higher weight than class 0
        self.assertGreater(weights[1], weights[0])


class TestUnsupervisedLearning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50) * 2,
            "f3": np.random.randn(50) + 5,
            "target": np.random.randint(0, 2, 50)
        })

    def test_pca_analysis(self):
        pca, pca_df, loadings, num_cols, y = run_pca(self.df, target="target", n_components=2)
        self.assertEqual(pca_df.shape[1], 3)  # PC1, PC2, target
        self.assertIn("PC1", pca_df.columns)
        self.assertIn("PC2", pca_df.columns)
        self.assertIn("target", pca_df.columns)
        self.assertAlmostEqual(pca.explained_variance_ratio_.sum(), sum(pca.explained_variance_ratio_), places=5)
        self.assertEqual(loadings.shape, (3, 2))  # 3 numeric features x 2 components

    def test_kmeans_clustering(self):
        kmeans, df_clustered, centroids, cluster_sizes, sil, _, _ = fit_kmeans(self.df, target="target", k=3)
        self.assertIn("cluster", df_clustered.columns)
        self.assertEqual(len(centroids), 3)
        self.assertEqual(cluster_sizes.sum(), 50)
        self.assertIsNotNone(sil)
        self.assertGreaterEqual(sil, -1.0)
        self.assertLessEqual(sil, 1.0)

    def test_hierarchical_clustering_metrics(self):
        # Euclidean
        model_euc, link_euc, df_euc, _, sizes_euc, _ = fit_hierarchical(self.df, target="target", n_clusters=2, metric="euclidean", method="ward")
        self.assertEqual(sizes_euc.sum(), 50)

        # Manhattan (cityblock)
        model_man, link_man, df_man, _, sizes_man, _ = fit_hierarchical(self.df, target="target", n_clusters=2, metric="cityblock", method="complete")
        self.assertEqual(sizes_man.sum(), 50)


class TestParamParsing(unittest.TestCase):
    def test_parses_json_and_python_literal(self):
        raw_json = '{"n_estimators": 100, "learning_rate": 0.1}'
        raw_python = "{'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': None}"

        # JSON parse
        p1 = json.loads(raw_json)
        self.assertEqual(p1["n_estimators"], 100)

        # Fallback to ast.literal_eval
        try:
            p2 = json.loads(raw_python)
        except Exception:
            p2 = ast.literal_eval(raw_python)
        self.assertEqual(p2["n_estimators"], 100)
        self.assertIsNone(p2["max_depth"])


class TestModelSelectionValidation(unittest.TestCase):
    def test_missing_target_raises_error(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "target": [0, np.nan, 1]
        })
        with self.assertRaises(ValueError):
            run_model_selection(df, target="target")

    def test_missing_features_raises_error(self):
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0],
            "target": [0, 1, 0]
        })
        with self.assertRaises(ValueError):
            run_model_selection(df, target="target")


if __name__ == "__main__":
    unittest.main()
