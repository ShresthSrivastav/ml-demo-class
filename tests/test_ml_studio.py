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

    def test_eda_summaries_and_bounded_chart_data(self):
        frame = pd.DataFrame({
            "number": np.arange(250),
            "group": [f"category-{i % 15}" for i in range(250)],
            "empty": [None if i % 3 == 0 else i for i in range(250)],
        })
        missing = app.missingness_summary(frame).set_index("Column")
        self.assertEqual(missing.loc["empty", "Missing rows"], 84)
        self.assertAlmostEqual(missing.loc["empty", "Missing %"], 33.6)

        histogram = app.histogram_data(frame, "number", bins=7, limit=40)
        self.assertEqual(len(histogram), 7)
        self.assertEqual(histogram["rows"].sum(), 40)
        pd.testing.assert_frame_equal(histogram, app.histogram_data(frame, "number", bins=7, limit=40))

        categories = app.category_count_data(frame, "group", limit=10)
        self.assertEqual(len(categories), 11)
        self.assertEqual(categories.loc[categories["category"] == "Other categories", "rows"].iloc[0], 80)

    def test_eda_relationship_limits_and_empty_inputs(self):
        frame = pd.DataFrame({f"n{i}": np.arange(120) + i for i in range(22)})
        correlations = app.correlation_data(frame, list(frame.columns))
        self.assertEqual(correlations["x"].nunique(), app.MAX_CORRELATION_COLUMNS)
        self.assertEqual(correlations["y"].nunique(), app.MAX_CORRELATION_COLUMNS)
        self.assertEqual(len(correlations), app.MAX_CORRELATION_COLUMNS ** 2)

        patterned = frame.assign(group=[f"g{i % 12}" for i in range(len(frame))])
        missing_pattern = app.missingness_pattern_data(patterned, list(patterned.columns))
        self.assertEqual(missing_pattern["row"].nunique(), app.MAX_MISSINGNESS_PLOT_ROWS)
        self.assertEqual(missing_pattern["column"].nunique(), app.MAX_CORRELATION_COLUMNS)
        self.assertEqual(len(missing_pattern), app.MAX_MISSINGNESS_PLOT_ROWS * app.MAX_CORRELATION_COLUMNS)

        pairs = app.pairwise_pairs(["a", "b", "c", "d"])
        self.assertEqual(len(pairs), 6)
        with self.assertRaisesRegex(ValueError, "at most four"):
            app.pairwise_pairs(["a", "b", "c", "d", "e"])

        large = pd.DataFrame({"x": np.arange(3_000), "y": np.arange(3_000) ** 2})
        pair_sample = app.bounded_frame_sample(large, ("x", "y"), app.MAX_PAIRWISE_PLOT_ROWS)
        self.assertEqual(len(pair_sample), app.MAX_PAIRWISE_PLOT_ROWS)
        pd.testing.assert_frame_equal(pair_sample, app.bounded_frame_sample(large, ("x", "y"), app.MAX_PAIRWISE_PLOT_ROWS))

        relation = app.category_relationship_data(patterned, "group", "n0", row_limit=25)
        self.assertLessEqual(int(relation["rows"].sum()), 25)
        self.assertTrue(app.correlation_data(pd.DataFrame({"only": [1, 2]}), ["only"]).empty)


if __name__ == "__main__":
    unittest.main()
