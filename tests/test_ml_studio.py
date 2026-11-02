import gzip
import unittest

import numpy as np
import pandas as pd

import app


class MLStudioWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        rows = 120
        signal = rng.normal(size=rows)
        labels = np.where(signal > 0, "positive", "negative")
        cls.classification = pd.DataFrame(
            {
                "signal": signal,
                "category": np.where(signal > 0, "high", "low"),
                "target": labels,
            }
        )
        cls.classification.loc[::13, "signal"] = np.nan
        cls.classification.loc[::17, "category"] = None
        cls.regression = pd.DataFrame(
            {
                "signal": signal,
                "category": np.where(signal > 0, "high", "low"),
                "target": (4.5 * signal + rng.normal(0, 0.3, rows)).astype(str),
            }
        )
        cls.clustering = pd.DataFrame(
            {
                "x": np.r_[rng.normal(-2, 0.3, 60), rng.normal(2, 0.3, 60)],
                "y": np.r_[rng.normal(-2, 0.3, 60), rng.normal(2, 0.3, 60)],
                "known_group": ["left"] * 60 + ["right"] * 60,
            }
        )
        cls.settings = {
            "random_state": 19,
            "max_depth": 4,
            "n_estimators": 30,
            "neighbors": 5,
            "regularization": 1.0,
            "alpha": 1.0,
            "clusters": 2,
            "eps": 0.6,
            "min_samples": 3,
            "components": 2,
            "contamination": 0.05,
        }

    def assert_models_succeed(self, bundles):
        results = [result for bundle in bundles for result in bundle["results"]]
        self.assertTrue(results)
        failures = [f"{result['model']}: {result['error']}" for result in results if result["error"]]
        self.assertEqual(failures, [])
        return results

    def test_classification_catalog_and_seeded_holdout(self):
        model_names = app.CLASSIFICATION_MODELS
        for index in range(0, len(model_names), app.MAX_MODELS_PER_RUN):
            bundles = app.run_models(
                self.classification,
                "Classification",
                "target",
                ["signal", "category"],
                model_names[index : index + app.MAX_MODELS_PER_RUN],
                0.2,
                self.settings,
            )
            results = self.assert_models_succeed(bundles)
            self.assertIn("Test macro F1", results[1]["metrics"])
            self.assertIn("confusion_matrix", results[1]["details"])

        first = app.run_models(self.classification, "Classification", "target", ["signal", "category"], ["Decision Tree"], 0.2, self.settings)
        second = app.run_models(self.classification, "Classification", "target", ["signal", "category"], ["Decision Tree"], 0.2, self.settings)
        self.assertEqual(first[0]["results"][1]["metrics"], second[0]["results"][1]["metrics"])

    def test_regression_catalog_accepts_numeric_text_targets(self):
        model_names = app.REGRESSION_MODELS
        for index in range(0, len(model_names), app.MAX_MODELS_PER_RUN):
            bundles = app.run_models(
                self.regression,
                "Regression",
                "target",
                ["signal", "category"],
                model_names[index : index + app.MAX_MODELS_PER_RUN],
                0.2,
                self.settings,
            )
            results = self.assert_models_succeed(bundles)
            self.assertIn("Test MAE", results[1]["metrics"])
            self.assertIn("Residual", results[1]["details"]["output"].columns)

    def test_cross_validation_and_feature_importance(self):
        bundles = app.run_models(
            self.classification,
            "Classification",
            "target",
            ["signal", "category"],
            ["Logistic Regression"],
            0.2,
            self.settings,
            cv_folds=3,
            explain_features=True,
        )
        result = self.assert_models_succeed(bundles)[1]
        self.assertIn("CV Macro F1 (mean)", result["metrics"])
        self.assertIn("feature_importance", result["details"])

    def test_clustering_pca_and_anomaly_workflows(self):
        clustering = app.run_models(
            self.clustering,
            "Clustering",
            None,
            ["x", "y"],
            app.UNSUPERVISED_MODELS["Clustering"],
            0.2,
            self.settings,
            evaluation_label="known_group",
        )
        cluster_results = self.assert_models_succeed(clustering)
        self.assertIn("Adjusted Rand (known labels)", cluster_results[0]["metrics"])

        pca = app.run_models(self.clustering, "Dimensionality reduction", None, ["x", "y"], ["PCA"], 0.2, self.settings)
        self.assert_models_succeed(pca)
        self.assertIn("Explained variance", pca[0]["results"][0]["metrics"])

        anomaly = app.run_models(
            self.clustering,
            "Anomaly detection",
            None,
            ["x", "y"],
            app.UNSUPERVISED_MODELS["Anomaly detection"],
            0.2,
            self.settings,
        )
        self.assert_models_succeed(anomaly)

    def test_large_row_download_is_bounded_and_readable(self):
        output = pd.DataFrame({"value": ["student-record-" + ("x" * 80)] * 100_000})
        preview, payload, suffix, mime = app.prepare_row_download(output, seed=19)
        self.assertEqual(len(preview), app.MAX_OUTPUT_PREVIEW_ROWS)
        self.assertEqual(suffix, ".csv.gz")
        self.assertEqual(mime, "application/gzip")
        self.assertIn(b"student-record", gzip.decompress(payload))

    def test_report_records_settings_and_metrics(self):
        report = app.build_markdown_report(
            [{
                "name": "lesson_data",
                "config": {
                    "task": "Classification",
                    "target": "target",
                    "features": ["signal"],
                    "models": ["Decision Tree"],
                    "settings": {"random_state": 19, "max_depth": 4},
                    "test_fraction": 0.2,
                    "cv_folds": 3,
                },
                "bundles": [{"results": [{"model": "Decision Tree", "metrics": {"Test macro F1": 0.8}, "error": None}]}],
            }]
        )
        self.assertIn("Random seed", report)
        self.assertIn("max_depth=4", report)
        self.assertIn("Test macro F1", report)

    def test_free_tier_model_limit_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "up to 4 models"):
            app.run_models(
                self.classification,
                "Classification",
                "target",
                ["signal"],
                app.CLASSIFICATION_MODELS[:5],
                0.2,
                self.settings,
            )

    def test_csv_header_validation_and_resource_guard(self):
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            app.parse_csv_bytes(b"feature,feature\n1,2\n")
        with self.assertRaisesRegex(ValueError, "Every CSV column needs a name"):
            app.parse_csv_bytes(b"feature,\n1,2\n")

        large = pd.DataFrame({"x": np.zeros(app.MAX_DENSITY_MODEL_ROWS + 1)})
        bundles = app.run_models(
            large,
            "Clustering",
            None,
            ["x"],
            ["DBSCAN"],
            0.2,
            self.settings,
        )
        self.assertIn("limited to", bundles[0]["results"][0]["error"])


if __name__ == "__main__":
    unittest.main()
