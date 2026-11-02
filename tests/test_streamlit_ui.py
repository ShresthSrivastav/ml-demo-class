import unittest
import hashlib
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitInterfaceTests(unittest.TestCase):
    @staticmethod
    def fresh_app():
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        return AppTest.from_file(str(app_path), default_timeout=30).run()

    def test_all_learning_examples_run_without_ui_errors(self):
        examples = [
            "Classification example",
            "Regression example",
            "Clustering example",
            "PCA example",
            "Anomaly detection example",
        ]
        for example in examples:
            with self.subTest(example=example):
                at = self.fresh_app()
                at.radio(key="single_source").set_value("Try a learning example").run()
                at.selectbox(key="single_demo_name").set_value(example).run()
                at.button(key="single_run").click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.error), 0)
                self.assertTrue(any(element.label.startswith("Download run report") for element in at.get("download_button")))

    def test_two_dataset_comparison_modes_and_session_reset(self):
        first = b"measurement,label\n" + b"".join(
            f"{i / 10},{'yes' if i % 2 else 'no'}\n".encode() for i in range(80)
        )
        second = b"measurement,label\n" + b"".join(
            f"{i / 12},{'yes' if i % 2 else 'no'}\n".encode() for i in range(80)
        )
        at = self.fresh_app()
        at.radio(key="workspace").set_value("Compare datasets").run()
        at.file_uploader(key="multi_upload").set_value(
            [
                ("first.csv", first, "text/csv"),
                ("second.csv", second, "text/csv"),
            ]
        ).run()
        for target in [widget for widget in at.selectbox if widget.label == "Target column"]:
            target.set_value("label")
        at.run()
        at.button(key="compare_run").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)
        self.assertTrue(any(element.label.startswith("Download comparison report") for element in at.get("download_button")))

        at.radio(key="compare_mode").set_value("Same model across datasets").run()
        at.button(key="compare_run").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        at.button(key="reset_session").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.radio(key="workspace").value, "Single dataset")

    def test_explore_eda_for_numeric_categorical_mixed_and_unlabeled_csvs(self):
        examples = {
            "numeric": b"x,y\n" + b"".join(f"{i},{i * 2}\n".encode() for i in range(40)),
            "categorical": b"group\n" + b"".join(f"g{i % 4}\n".encode() for i in range(40)),
            "mixed_missing": b"measure,group,label\n" + b"".join(
                f"{'' if i % 9 == 0 else i / 10},{'a' if i % 2 else 'b'},{'yes' if i % 2 else 'no'}\n".encode()
                for i in range(60)
            ),
            "high_cardinality": b"measure,identifier\n" + b"".join(f"{i % 7},{'id-' + str(i)}\n".encode() for i in range(60)),
        }
        for name, payload in examples.items():
            with self.subTest(dataset=name):
                at = self.fresh_app()
                at.file_uploader(key="single_upload").set_value([(f"{name}.csv", payload, "text/csv")]).run()
                dataset_key = hashlib.sha256(payload).hexdigest()[:16]
                eda_key = f"single_eda_{dataset_key}_view"
                at.radio(key=eda_key).set_value("Explore dataset").run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.error), 0)
                self.assertEqual(at.radio(key=eda_key).value, "Explore dataset")
                if name == "numeric":
                    at.checkbox(key=f"single_eda_{dataset_key}_pair_enabled").set_value(True).run()
                    self.assertEqual(len(at.exception), 0)

    def test_classification_and_regression_target_plots(self):
        classification = b"signal,category,label\n" + b"".join(
            f"{i / 10},{'a' if i % 2 else 'b'},{'yes' if i % 2 else 'no'}\n".encode()
            for i in range(60)
        )
        at = self.fresh_app()
        at.file_uploader(key="single_upload").set_value([("classes.csv", classification, "text/csv")]).run()
        class_key = hashlib.sha256(classification).hexdigest()[:16]
        at.selectbox(key=f"single_target_{class_key}").set_value("label").run()
        self.assertEqual(len(at.exception), 0)

        regression = b"signal,target\n" + b"".join(f"{i},{i * 3.5}\n".encode() for i in range(60))
        at = self.fresh_app()
        at.file_uploader(key="single_upload").set_value([("regression.csv", regression, "text/csv")]).run()
        at.selectbox(key="single_task").set_value("Regression").run()
        regression_key = hashlib.sha256(regression).hexdigest()[:16]
        at.selectbox(key=f"single_target_{regression_key}").set_value("target").run()
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main()
