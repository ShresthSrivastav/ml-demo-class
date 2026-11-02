import unittest
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


if __name__ == "__main__":
    unittest.main()
