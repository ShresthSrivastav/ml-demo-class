from __future__ import annotations

import io
import hashlib
import json
import math
import csv
import gzip
import re
import warnings
from datetime import datetime
from itertools import combinations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
    classification_report,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.inspection import permutation_importance


TASKS = [
    "Classification",
    "Regression",
    "Clustering",
    "Dimensionality reduction",
    "Anomaly detection",
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Gaussian Naive Bayes",
]
REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Support Vector Regression",
]
UNSUPERVISED_MODELS = {
    "Clustering": ["K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"],
    "Dimensionality reduction": ["PCA"],
    "Anomaly detection": ["Isolation Forest", "Local Outlier Factor"],
}

TASK_LESSONS = {
    "Classification": "Predict a category. Accuracy is the fraction correct, but can hide rare-class failures. Macro F1 weights classes equally; weighted F1 weights them by frequency. Check per-class recall and the confusion matrix.",
    "Regression": "Predict a numeric value. MAE is average absolute error in target units; RMSE penalizes large misses more; R² compares with a mean-value baseline. Inspect residuals for patterns the metrics hide.",
    "Clustering": "Group rows without using a target. Cluster IDs are arbitrary names, not ranks. Silhouette measures separation without labels; optional known labels report ARI/NMI for external evaluation only. ARI is chance-adjusted; NMI is not.",
    "Dimensionality reduction": "Compress features into fewer components. PCA preserves as much variance as possible at each step; explained variance describes information retained, not prediction accuracy.",
    "Anomaly detection": "Find rows unlike the majority. A flagged row needs investigation, not automatic deletion. The contamination setting controls the expected fraction flagged; anomaly scores are not probabilities.",
}

TASK_EXERCISES = {
    "Classification": [
        "Change tree depth and compare train accuracy with test macro F1. When does the gap grow?",
        "Compare the baseline with per-class recall. Which class is hardest to predict?",
    ],
    "Regression": [
        "Compare the mean baseline with a model using MAE and RMSE. Are large errors driving the difference?",
        "Inspect residuals. Do they fan out or curve, suggesting a pattern the model missed?",
    ],
    "Clustering": [
        "Change cluster count or DBSCAN density settings. Do the groups remain meaningful?",
        "Compare silhouette with ARI/NMI when known labels are available. Why can these measures disagree?",
    ],
    "Dimensionality reduction": [
        "Increase the component count and track total explained variance. How much compression is useful?",
        "Look at the first two components. Do visible groups explain the full dataset or only a projection?",
    ],
    "Anomaly detection": [
        "Change contamination and compare which rows are flagged.",
        "Inspect flagged measurements. Which are data errors and which could be rare but valid cases?",
    ],
}

MODEL_LESSONS = {
    "Logistic Regression": "A linear classification baseline. Fast and easier to interpret; it may underfit curved decision boundaries.",
    "Linear Regression": "Fits a linear relationship by minimizing squared error. Coefficients are easiest to interpret when features are well prepared and not strongly collinear.",
    "Ridge": "Linear regression with L2 regularization. Shrinks coefficients to reduce sensitivity to correlated or noisy features.",
    "Lasso": "Linear regression with L1 regularization. Can shrink some coefficients to zero, but results can be unstable with strongly correlated features.",
    "Decision Tree": "Learns a sequence of feature-based rules. Easy to visualize conceptually, but deep trees can memorize training data.",
    "Random Forest": "Averages many trees trained with randomness. Often a strong tabular baseline, with less interpretability than one tree.",
    "Extra Trees": "Averages highly randomized trees. Compare it with Random Forest to see how added split randomness affects variance and accuracy.",
    "Gradient Boosting": "Adds small trees sequentially to correct earlier errors. It can model complex patterns but may need tuning and can overfit.",
    "K-Nearest Neighbors": "Predicts from nearby examples. Feature scaling matters, and prediction can slow down as the training set grows.",
    "Support Vector Machine": "Finds a maximum-margin boundary; the default RBF kernel can model nonlinear boundaries. Scaling and dataset size matter.",
    "Support Vector Regression": "Uses a margin around predictions rather than fitting every point exactly. Scaling and kernel choice strongly affect results.",
    "Gaussian Naive Bayes": "A fast probabilistic classifier that assumes features are conditionally independent. The assumption is simple and often imperfect.",
    "K-Means": "Partitions rows around k centroids. It favors compact, roughly round clusters and requires choosing k.",
    "Agglomerative Clustering": "Builds nested groups by repeatedly merging nearby clusters. It is useful for teaching hierarchy but can be slow on large datasets.",
    "DBSCAN": "Groups dense regions and marks sparse rows as noise. It can find irregular shapes, but results depend on distance scale and density settings.",
    "Gaussian Mixture": "Models data as a mixture of probability distributions. It can assign soft membership, but the app displays the most likely component.",
    "PCA": "Creates orthogonal linear components ordered by explained variance. Components are combinations of features, not original columns.",
    "Isolation Forest": "Flags rows that are isolated quickly by random partitions. It works well as a general anomaly baseline.",
    "Local Outlier Factor": "Compares each row's local density with its neighbors. It detects local anomalies but depends on neighborhood size.",
}

DEMO_TASKS = {
    "Classification example": "Classification",
    "Regression example": "Regression",
    "Clustering example": "Clustering",
    "PCA example": "Dimensionality reduction",
    "Anomaly detection example": "Anomaly detection",
}
MAX_DENSE_FEATURE_CELLS = 5_000_000
MAX_AGGLOMERATIVE_ROWS = 5_000
MAX_SLOW_KERNEL_ROWS = 10_000
MAX_DENSITY_MODEL_ROWS = 15_000
MAX_NEIGHBOR_ROWS = 20_000
MAX_MODELS_PER_RUN = 4
MAX_COMPARISON_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_OUTPUT_PREVIEW_ROWS = 5_000
MAX_EDA_PLOT_ROWS = 5_000
MAX_PAIRWISE_PLOT_ROWS = 2_000
MAX_CORRELATION_COLUMNS = 20
MAX_MISSINGNESS_PLOT_ROWS = 100
MAX_MISSINGNESS_CHART_COLUMNS = 50
MAX_CATEGORY_LEVELS = 10


@st.cache_data(scope="session", max_entries=4, show_spinner=False)
def make_demo_dataset(name: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if name == "Classification example":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=600,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            weights=[0.78, 0.22],
            random_state=42,
        )
        frame = pd.DataFrame(X, columns=[f"measurement_{i + 1}" for i in range(X.shape[1])])
        frame["region"] = rng.choice(["north", "south", "east"], size=len(frame))
        frame["label"] = np.where(y == 1, "positive", "negative")
        frame.loc[::29, "measurement_1"] = np.nan
        frame.attrs["mlstudio_cache_key"] = f"demo:{name}"
        return frame
    if name == "Regression example":
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=600, n_features=6, noise=12, random_state=42)
        frame = pd.DataFrame(X, columns=[f"input_{i + 1}" for i in range(X.shape[1])])
        frame["target"] = y
        frame.loc[::31, "input_2"] = np.nan
        frame.attrs["mlstudio_cache_key"] = f"demo:{name}"
        return frame
    if name in ("Clustering example", "PCA example"):
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=600, centers=4, n_features=2, cluster_std=1.1, random_state=42)
        frame = pd.DataFrame(X, columns=["measurement_x", "measurement_y"])
        frame.attrs["mlstudio_cache_key"] = f"demo:{name}"
        return frame
    frame = np.vstack([rng.normal(0, 0.6, size=(580, 2)), rng.uniform(-6, 6, size=(20, 2))])
    result = pd.DataFrame(frame, columns=["measurement_x", "measurement_y"])
    result.attrs["mlstudio_cache_key"] = f"demo:{name}"
    return result


def render_learning_guide(task: str, models: list[str] | None = None) -> None:
    st.info(TASK_LESSONS[task])
    with st.expander("Teacher notes: how to interpret this run"):
        st.markdown("**Workflow:** inspect the data → define the question → choose inputs → fit the model → evaluate on data the model did not train on.")
        st.markdown("**Important:** a high score is not proof of causation or fairness. Check leakage, class balance, data quality, and whether the test data resembles the intended real-world use.")
        st.markdown("**Try this experiment**")
        for exercise in TASK_EXERCISES[task]:
            st.markdown(f"- {exercise}")
        if models:
            st.markdown("**Selected model ideas**")
            for model in models:
                st.markdown(f"- **{model}:** {MODEL_LESSONS.get(model, 'A scikit-learn estimator for the selected task.')}")
def read_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV without writing it to disk."""
    raw = uploaded_file.getvalue()
    cache_key = hashlib.sha256(raw).hexdigest()
    frame = _parse_csv(cache_key, raw)
    frame.attrs["mlstudio_cache_key"] = cache_key
    return frame


def safe_file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return stem[:80] or "dataset"


def prepare_row_download(output: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, bytes, str, str]:
    """Keep a small preview in session state and stream large CSV exports through gzip."""
    preview = output
    if len(output) > MAX_OUTPUT_PREVIEW_ROWS:
        preview = output.sample(MAX_OUTPUT_PREVIEW_ROWS, random_state=seed)
    preview = preview.copy()
    estimated_bytes = int(output.memory_usage(index=True, deep=True).sum())
    if estimated_bytes <= 5 * 1024 * 1024:
        payload = output.to_csv(index=False).encode("utf-8")
        return preview, payload, ".csv", "text/csv"

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_file:
            output.to_csv(text_file, index=False, chunksize=10_000)
    return preview, buffer.getvalue(), ".csv.gz", "application/gzip"


def build_markdown_report(runs: list[dict], title: str = "ML Studio run report") -> str:
    """Create a lightweight, portable record of run settings and results."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"# {title}", "", f"Generated: {timestamp}", ""]
    for run in runs:
        name = str(run.get("name", "Dataset")).replace("|", "\\|")
        config = run.get("config", {})
        lines.extend([f"## {name}", ""])
        if run.get("error"):
            lines.extend([f"Run error: {run['error']}", ""])
            continue

        if config.get("task") in ("Classification", "Regression"):
            split_text = f"{1 - config.get('test_fraction', 0):.0%} / {config.get('test_fraction', 0):.0%}"
        else:
            split_text = "Not used"
        settings = config.get("settings", {})
        setting_text = ", ".join(f"{key}={value}" for key, value in settings.items() if key != "random_state") or "Defaults"
        metadata = [
            ("Task", config.get("task", "Not recorded")),
            ("Target", config.get("target") or "None"),
            ("Known labels for external evaluation", config.get("evaluation_label") or "None"),
            ("Features", ", ".join(map(str, config.get("features", []))) or "None"),
            ("Models", ", ".join(map(str, config.get("models", []))) or "None"),
            ("Train/test", split_text),
            ("Cross-validation folds", config.get("cv_folds", 0) or "Not used"),
            ("Permutation importance", "Enabled" if config.get("explain_features") else "Disabled"),
            ("Random seed", settings.get("random_state", 42)),
            ("Model settings", setting_text),
        ]
        for label, value in metadata:
            escaped_value = str(value).replace("|", "\\|")
            lines.append(f"- **{label}:** {escaped_value}")
        lines.append("")

        for bundle in run.get("bundles", []):
            if bundle.get("note"):
                lines.extend([f"Note: {bundle['note']}", ""])
            for result in bundle.get("results", []):
                lines.extend([f"### {result['model']}", ""])
                if result.get("error"):
                    lines.extend([f"Error: {result['error']}", ""])
                    continue
                metrics = result.get("metrics", {})
                if metrics:
                    lines.extend(["| Metric | Value |", "| --- | ---: |"])
                    for metric, value in metrics.items():
                        if isinstance(value, (float, np.floating)):
                            value = "Not defined" if not np.isfinite(value) else f"{value:.6g}"
                        escaped_metric = str(metric).replace("|", "\\|")
                        escaped_value = str(value).replace("|", "\\|")
                        lines.append(f"| {escaped_metric} | {escaped_value} |")
                    lines.append("")
    lines.extend(["---", "Preprocessing is fit on training folds/splits for supervised runs. Results are educational estimates, not guarantees of real-world performance.", ""])
    return "\n".join(lines)


def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Parse and validate CSV bytes without depending on Streamlit session state."""
    encoding = "utf-8-sig"
    try:
        with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding, newline="") as text_file:
            header = next(csv.reader(text_file), [])
    except UnicodeDecodeError:
        encoding = "latin-1"
        with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding, newline="") as text_file:
            header = next(csv.reader(text_file), [])
    if not header or all(not str(column).strip() for column in header):
        raise ValueError("This CSV needs a header row with at least one named column.")
    if len(header) != len(set(header)):
        raise ValueError("The CSV contains duplicate column names. Rename them and upload again.")
    if any(not str(column).strip() for column in header):
        raise ValueError("Every CSV column needs a name. Rename blank headers and upload again.")
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding=encoding)
    except UnicodeDecodeError:
        if encoding == "latin-1":
            raise
        frame = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("This CSV has no rows or columns.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names. Rename them and upload again.")
    return frame


@st.cache_data(scope="session", ttl=1800, max_entries=8, show_spinner=False)
def _parse_csv(cache_key: str, _raw: bytes) -> pd.DataFrame:
    # The digest keys the cache; the underscore avoids hashing the upload bytes a second time.
    return parse_csv_bytes(_raw)


@st.cache_data(scope="session", max_entries=8, show_spinner=False)
def _eda_profile(cache_key: str, _frame: pd.DataFrame) -> dict:
    sample = _frame if len(_frame) <= 50_000 else _frame.sample(50_000, random_state=42)
    numeric = list(_frame.select_dtypes(include=np.number).columns)
    missing = _frame.isna().mean()
    unique = sample.nunique(dropna=True)
    unique_label = "Unique values" if len(sample) == len(_frame) else "Unique values (sample)"
    schema = pd.DataFrame(
        {
            "Column": _frame.columns,
            "Type": [str(dtype) for dtype in _frame.dtypes],
            "Missing": _frame.isna().sum().to_numpy(),
            "Missing %": (missing.to_numpy() * 100).round(1),
            unique_label: unique.to_numpy(),
        }
    )
    likely_ids = likely_identifier_columns(_frame, sample, unique)
    constants = [column for column in _frame.columns if _frame[column].nunique(dropna=True) <= 1]
    return {
        "schema": schema,
        "summary": sample.describe(include="all").transpose(),
        "likely_ids": likely_ids,
        "constants": constants,
        "sampled": len(sample) < len(_frame),
        "missing_cells": int(_frame.isna().sum().sum()),
        "duplicates": int(_frame.duplicated().sum()),
    }


@st.cache_data(scope="session", max_entries=32, show_spinner=False)
def _eda_sample(cache_key: str, _frame: pd.DataFrame, columns: tuple, limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    return bounded_frame_sample(_frame, columns, limit, seed)


def bounded_frame_sample(frame: pd.DataFrame, columns: tuple | list, limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    selected = list(columns)
    sample = frame.loc[:, selected]
    if len(sample) > limit:
        sample = sample.sample(limit, random_state=seed)
    return sample.replace([np.inf, -np.inf], np.nan)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    return pd.DataFrame({"Column": missing.index.astype(str), "Missing rows": missing.to_numpy(), "Missing %": (missing / max(1, len(frame)) * 100).to_numpy()})


def histogram_data(frame: pd.DataFrame, column: str, bins: int = 20, limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) > limit:
        values = values.sample(limit, random_state=seed)
    if values.empty:
        return pd.DataFrame(columns=["bin_start", "bin_end", "rows"])
    counts, edges = np.histogram(values, bins=max(1, min(int(bins), 50)))
    return pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "rows": counts})


def category_count_data(frame: pd.DataFrame, column: str, limit: int = MAX_CATEGORY_LEVELS, row_limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    values = frame[column]
    if len(values) > row_limit:
        values = values.sample(row_limit, random_state=seed)
    counts = values.fillna("(missing)").astype(str).value_counts()
    top = counts.head(limit)
    if len(counts) > limit:
        top.loc["Other categories"] = int(counts.iloc[limit:].sum())
    return top.rename_axis("category").reset_index(name="rows")


def correlation_data(frame: pd.DataFrame, columns: list[str], limit: int = MAX_CORRELATION_COLUMNS, row_limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    selected = [column for column in columns if column in frame and pd.api.types.is_numeric_dtype(frame[column])][:limit]
    if len(selected) < 2:
        return pd.DataFrame(columns=["x", "y", "correlation"])
    sample = frame[selected]
    if len(sample) > row_limit:
        sample = sample.sample(row_limit, random_state=seed)
    correlations = sample.replace([np.inf, -np.inf], np.nan).corr(numeric_only=True)
    result = pd.DataFrame(
        [(str(x), str(y), correlations.loc[y, x]) for x in correlations.columns for y in correlations.index],
        columns=["x", "y", "correlation"],
    )
    return result.dropna(subset=["correlation"])


def missingness_pattern_data(frame: pd.DataFrame, columns: list[str], limit: int = MAX_MISSINGNESS_PLOT_ROWS, max_columns: int = MAX_CORRELATION_COLUMNS, seed: int = 42) -> pd.DataFrame:
    selected = list(columns)[:max_columns]
    if not selected:
        return pd.DataFrame(columns=["row", "column", "missing"])
    sample = frame[selected]
    if len(sample) > limit:
        sample = sample.sample(limit, random_state=seed)
    pattern = sample.isna().copy()
    pattern.index = np.arange(len(pattern))
    pattern.index.name = "row"
    return pattern.rename_axis(columns="column").reset_index().melt(id_vars="row", var_name="column", value_name="missing")


def category_relationship_data(frame: pd.DataFrame, x: str, y: str, limit: int = MAX_CATEGORY_LEVELS, row_limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    sample = frame[[x, y]]
    if len(sample) > row_limit:
        sample = sample.sample(row_limit, random_state=seed)
    result = sample.copy()
    result.columns = ["x", "y"]
    result = result.fillna("(missing)").astype(str)
    for column in ("x", "y"):
        top = result[column].value_counts().head(limit).index
        result.loc[~result[column].isin(top), column] = "Other categories"
    return result.groupby(["x", "y"], observed=True).size().reset_index(name="rows")


def numeric_category_sample(frame: pd.DataFrame, numeric: str, category: str, row_limit: int = MAX_EDA_PLOT_ROWS, seed: int = 42) -> pd.DataFrame:
    sample = frame[[numeric, category]]
    if len(sample) > row_limit:
        sample = sample.sample(row_limit, random_state=seed)
    sample = sample.rename(columns={numeric: "value", category: "category"}).copy()
    sample["value"] = pd.to_numeric(sample["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sample["category"] = sample["category"].fillna("(missing)").astype(str)
    top = sample["category"].value_counts().head(MAX_CATEGORY_LEVELS).index
    sample.loc[~sample["category"].isin(top), "category"] = "Other categories"
    return sample.dropna(subset=["value"])


def pairwise_pairs(columns: list[str]) -> list[tuple[str, str]]:
    if len(columns) > 4:
        raise ValueError("Choose at most four numeric columns for pairwise exploration.")
    return list(combinations(columns, 2))


def likely_identifier_columns(frame: pd.DataFrame, sample: pd.DataFrame | None = None, unique: pd.Series | None = None) -> list[str]:
    sample = sample if sample is not None else (frame if len(frame) <= 50_000 else frame.sample(50_000, random_state=42))
    unique = unique if unique is not None else sample.nunique(dropna=True)
    return [
        column
        for column in frame.columns
        if unique[column] > 50
        and (
            str(column).lower() in {"id", "key"}
            or str(column).lower().endswith(("_id", " id", "_key", " key"))
            or (
                not pd.api.types.is_numeric_dtype(frame[column])
                and unique[column] / max(1, len(sample)) >= 0.98
            )
        )
    ]


def available_models(task: str) -> list[str]:
    if task == "Classification":
        return CLASSIFICATION_MODELS
    if task == "Regression":
        return REGRESSION_MODELS
    return UNSUPERVISED_MODELS[task]


def feature_columns(frame: pd.DataFrame, target: str | None, excluded: list[str] | None = None) -> list[str]:
    excluded = set(excluded or [])
    if target:
        excluded.add(target)
    return [column for column in frame.columns if column not in excluded]


def choose_features(frame: pd.DataFrame, target: str | None, key: str, excluded: list[str] | None = None) -> list[str]:
    candidates = feature_columns(frame, target, excluded)
    cache_key = frame.attrs.get("mlstudio_cache_key", key)
    likely_ids = [
        column
        for column in _eda_profile(cache_key, frame)["likely_ids"]
        if column in candidates
    ]
    exclude_ids = False
    if likely_ids:
        exclude_ids = st.checkbox(
            f"Exclude likely ID columns ({', '.join(map(str, likely_ids[:5]))}{'…' if len(likely_ids) > 5 else ''})",
            value=True,
            help="Unique row identifiers usually add noise rather than predictive information. Turn this off if the ID itself has meaning.",
            key=f"{key}_exclude_ids",
        )
    if exclude_ids:
        candidates = [column for column in candidates if column not in likely_ids]
    use_all = st.checkbox("Use all remaining columns", value=True, key=f"{key}_all_features")
    if use_all:
        return candidates
    return st.multiselect(
        "Choose input features",
        options=candidates,
        default=candidates,
        key=f"{key}_selected_features",
    )


def make_preprocessor(frame: pd.DataFrame, columns: list[str], dense: bool = False) -> ColumnTransformer:
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numeric]
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=50,
                        sparse_output=not dense,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("Choose at least one input feature.")
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0 if dense else 0.3,
    )


def check_dense_capacity(frame: pd.DataFrame, columns: list[str], rows: int, model_name: str) -> None:
    numeric_count = sum(pd.api.types.is_numeric_dtype(frame[column]) for column in columns)
    categorical_count = len(columns) - numeric_count
    estimated_width = numeric_count + 50 * categorical_count
    estimated_cells = rows * estimated_width
    if estimated_cells > MAX_DENSE_FEATURE_CELLS:
        raise ValueError(
            f"{model_name} needs a dense feature matrix (estimated {estimated_cells:,} cells). "
            "Reduce the row/column count or choose a sparse-compatible model."
        )


def build_supervised_model(task: str, name: str, settings: dict | None = None):
    settings = settings or {}
    random_state = settings.get("random_state", 42)
    max_depth = settings.get("max_depth")
    n_estimators = settings.get("n_estimators", 150)
    neighbors = settings.get("neighbors", 5)
    regularization = settings.get("regularization", 1.0)
    alpha = settings.get("alpha", 1.0)
    if task == "Classification":
        models = {
            "Logistic Regression": (LogisticRegression(C=regularization, max_iter=1000), False),
            "Decision Tree": (DecisionTreeClassifier(max_depth=max_depth, random_state=random_state), False),
            "Random Forest": (RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state), False),
            "Extra Trees": (ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state), False),
            "Gradient Boosting": (GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth or 3, random_state=random_state), True),
            "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=neighbors), False),
            "Support Vector Machine": (SVC(C=regularization), False),
            "Gaussian Naive Bayes": (GaussianNB(), True),
        }
    else:
        models = {
            "Linear Regression": (LinearRegression(), False),
            "Ridge": (Ridge(alpha=alpha), False),
            "Lasso": (Lasso(alpha=alpha, max_iter=5000), False),
            "Decision Tree": (DecisionTreeRegressor(max_depth=max_depth, random_state=random_state), False),
            "Random Forest": (RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state), False),
            "Extra Trees": (ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state), False),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth or 3, random_state=random_state), True),
            "K-Nearest Neighbors": (KNeighborsRegressor(n_neighbors=neighbors), False),
            "Support Vector Regression": (SVR(C=regularization), False),
        }
    return models[name]


def run_supervised(
    frame: pd.DataFrame,
    task: str,
    target: str,
    features: list[str],
    model_names: list[str],
    test_fraction: float,
    cv_folds: int = 0,
    explain_features: bool = False,
    model_settings: dict | None = None,
) -> list[dict]:
    if not features:
        raise ValueError("Choose at least one input feature.")
    seed = (model_settings or {}).get("random_state", 42)
    data = frame.loc[frame[target].notna(), features + [target]].copy()
    if task == "Regression":
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data[np.isfinite(data[target])]
    if len(data) < 4:
        raise ValueError("At least four rows with a non-missing target are needed for a train/test split.")
    X = data[features].replace([np.inf, -np.inf], np.nan)
    y = data[target]
    if task == "Classification" and y.nunique() < 2:
        raise ValueError("Classification needs at least two target classes.")
    if task == "Regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression needs a numeric target column.")

    stratify = None
    stratification_note = None
    if task == "Classification":
        class_counts = y.value_counts()
        test_rows = math.ceil(len(y) * test_fraction)
        train_rows = len(y) - test_rows
        if class_counts.min() >= 2 and test_rows >= len(class_counts) and train_rows >= len(class_counts):
            stratify = y
        else:
            stratification_note = "Some target classes are too small for a stratified split; using a random split."

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"The selected train/test split is not possible: {exc}") from exc

    if task == "Classification" and y_train.nunique() < 2:
        raise ValueError("The training split contains only one class. Add more examples of each class or choose a larger training set.")

    results = []
    baseline = DummyClassifier(strategy="most_frequent") if task == "Classification" else DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_prediction = baseline.predict(np.zeros((len(y_test), 1)))
    if task == "Classification":
        baseline_metrics = {
            "Test accuracy": accuracy_score(y_test, baseline_prediction),
            "Test balanced accuracy": balanced_accuracy_score(y_test, baseline_prediction),
            "Test macro F1": f1_score(y_test, baseline_prediction, average="macro", zero_division=0),
            "Test weighted F1": f1_score(y_test, baseline_prediction, average="weighted", zero_division=0),
        }
    else:
        baseline_metrics = {
            "Test R²": r2_score(y_test, baseline_prediction) if len(y_test) > 1 else np.nan,
            "Test MAE": mean_absolute_error(y_test, baseline_prediction),
            "Test RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
        }
    baseline_output = pd.DataFrame({"Actual": y_test, "Predicted": baseline_prediction}, index=y_test.index)
    if task == "Regression":
        baseline_output["Residual"] = y_test - baseline_prediction
    baseline_output = baseline_output.reset_index(names="Original row")
    preview, download_data, download_suffix, download_mime = prepare_row_download(baseline_output, seed)
    results.append({
        "model": "Baseline (most frequent class)" if task == "Classification" else "Baseline (predict the mean)",
        "metrics": baseline_metrics,
        "details": {
            "output": preview,
            "output_rows": len(baseline_output),
            "download_data": download_data,
            "download_suffix": download_suffix,
            "download_mime": download_mime,
        },
        "error": None,
    })

    for name in model_names:
        if name == "Support Vector Machine" and len(X_train) > MAX_SLOW_KERNEL_ROWS:
            results.append({"model": name, "metrics": {}, "details": {}, "error": f"Limited to {MAX_SLOW_KERNEL_ROWS:,} training rows on the free tier because kernel SVM can become very slow."})
            continue
        if name == "Support Vector Regression" and len(X_train) > MAX_SLOW_KERNEL_ROWS:
            results.append({"model": name, "metrics": {}, "details": {}, "error": f"Limited to {MAX_SLOW_KERNEL_ROWS:,} training rows on the free tier because kernel SVR can become very slow."})
            continue
        if name == "K-Nearest Neighbors" and len(X_train) > MAX_NEIGHBOR_ROWS:
            results.append({"model": name, "metrics": {}, "details": {}, "error": f"Limited to {MAX_NEIGHBOR_ROWS:,} training rows on the free tier to keep neighbor predictions responsive."})
            continue
        estimator, dense = build_supervised_model(task, name, model_settings)
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(frame, features, dense=dense)),
                ("model", estimator),
            ]
        )
        try:
            if dense:
                check_dense_capacity(frame, features, len(X_train), name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cv_folds:
                    if task == "Classification":
                        smallest_class = int(y_train.value_counts().min())
                        actual_folds = min(cv_folds, smallest_class)
                        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed) if actual_folds >= 2 else None
                        scoring = {
                            "accuracy": "accuracy",
                            "balanced_accuracy": "balanced_accuracy",
                            "macro_f1": "f1_macro",
                            "weighted_f1": "f1_weighted",
                        }
                    else:
                        actual_folds = min(cv_folds, len(y_train))
                        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=seed) if actual_folds >= 2 else None
                        scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
                    if cv is not None:
                        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
                    else:
                        cv_result = None
                pipeline.fit(X_train, y_train)
                train_prediction = pipeline.predict(X_train)
                prediction = pipeline.predict(X_test)
            if task == "Classification":
                metrics = {
                    "Train accuracy": accuracy_score(y_train, train_prediction),
                    "Test accuracy": accuracy_score(y_test, prediction),
                    "Test balanced accuracy": balanced_accuracy_score(y_test, prediction),
                    "Train macro F1": f1_score(y_train, train_prediction, average="macro", zero_division=0),
                    "Test macro F1": f1_score(y_test, prediction, average="macro", zero_division=0),
                    "Test weighted F1": f1_score(y_test, prediction, average="weighted", zero_division=0),
                    "Test precision (weighted)": precision_score(y_test, prediction, average="weighted", zero_division=0),
                    "Test recall (weighted)": recall_score(y_test, prediction, average="weighted", zero_division=0),
                }
                labels = list(pd.unique(pd.concat([y_test.reset_index(drop=True), pd.Series(prediction)])))
                matrix = pd.DataFrame(
                    confusion_matrix(y_test, prediction, labels=labels),
                    index=pd.Index(labels, name="Actual"),
                    columns=pd.Index(labels, name="Predicted"),
                )
                report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True, zero_division=0)).transpose()
                details = {"confusion_matrix": matrix, "classification_report": report}
            else:
                metrics = {
                    "Train R²": r2_score(y_train, train_prediction) if len(y_train) > 1 else np.nan,
                    "Test R²": r2_score(y_test, prediction) if len(y_test) > 1 else np.nan,
                    "Test MAE": mean_absolute_error(y_test, prediction),
                    "Test RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                }
                details = {}
            if cv_folds:
                if cv_result is None:
                    details["cv_note"] = "Cross-validation was skipped because the training data has fewer than two members of at least one class."
                else:
                    for metric, scores in scoring.items():
                        values = cv_result[f"test_{metric}"]
                        if metric in ("mae", "rmse"):
                            values = -values
                        friendly = {
                            "accuracy": "Accuracy",
                            "balanced_accuracy": "Balanced accuracy",
                            "macro_f1": "Macro F1",
                            "weighted_f1": "Weighted F1",
                            "r2": "R²",
                            "mae": "MAE",
                            "rmse": "RMSE",
                        }[metric]
                        metrics[f"CV {friendly} (mean)"] = float(np.mean(values))
                        metrics[f"CV {friendly} (std)"] = float(np.std(values))
                    if actual_folds < cv_folds:
                        details["cv_note"] = f"Used {actual_folds}-fold cross-validation because the training set has a class with only {actual_folds} examples."
            output = pd.DataFrame({"Actual": y_test, "Predicted": prediction}, index=y_test.index)
            if task == "Regression":
                output["Residual"] = y_test - prediction
            output = output.reset_index(names="Original row")
            preview, download_data, download_suffix, download_mime = prepare_row_download(output, seed)
            details.update({
                "output": preview,
                "output_rows": len(output),
                "download_data": download_data,
                "download_suffix": download_suffix,
                "download_mime": download_mime,
            })
            if explain_features:
                scoring_name = "f1_weighted" if task == "Classification" else "neg_mean_absolute_error"
                importance_X = X_test
                importance_y = y_test
                importance_sampled = len(importance_X) > 5_000
                if len(importance_X) > 5_000:
                    importance_X = importance_X.sample(5_000, random_state=seed)
                    importance_y = y_test.loc[importance_X.index]
                importance = permutation_importance(
                    pipeline,
                    importance_X,
                    importance_y,
                    scoring=scoring_name,
                    n_repeats=3,
                    random_state=seed,
                    n_jobs=1,
                )
                details["feature_importance"] = pd.DataFrame(
                    {"Feature": features, "Importance": importance.importances_mean}
                ).sort_values("Importance", ascending=False)
                details["importance_sampled"] = importance_sampled
            results.append({"model": name, "metrics": metrics, "details": details, "error": None})
        except Exception as exc:  # report one model's incompatibility without losing other results
            results.append({"model": name, "metrics": {}, "details": {}, "error": str(exc)})
    return [{"note": stratification_note, "results": results}]


def cluster_model(name: str, settings: dict):
    if name == "DBSCAN":
        return DBSCAN(eps=settings["eps"], min_samples=settings["min_samples"]), False
    clusters = settings["clusters"]
    if name == "K-Means":
        return KMeans(n_clusters=clusters, n_init=10, random_state=settings.get("random_state", 42)), False
    if name == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=clusters), True
    return GaussianMixture(n_components=clusters, random_state=settings.get("random_state", 42), n_init=3), True


def run_unsupervised(
    frame: pd.DataFrame,
    task: str,
    features: list[str],
    model_names: list[str],
    settings: dict,
    evaluation_label: str | None = None,
) -> list[dict]:
    if not features:
        raise ValueError("Choose at least one input feature.")
    X = frame[features].replace([np.inf, -np.inf], np.nan)
    if len(X) < 2:
        raise ValueError("At least two rows are needed for unsupervised learning.")
    seed = settings.get("random_state", 42)
    results = []
    for name in model_names:
        try:
            if task == "Clustering":
                estimator, dense = cluster_model(name, settings)
                if name == "Agglomerative Clustering" and len(X) > MAX_AGGLOMERATIVE_ROWS:
                    raise ValueError(f"Agglomerative Clustering is limited to {MAX_AGGLOMERATIVE_ROWS:,} rows because it grows quadratically with dataset size.")
                if name == "DBSCAN" and len(X) > MAX_DENSITY_MODEL_ROWS:
                    raise ValueError(f"DBSCAN is limited to {MAX_DENSITY_MODEL_ROWS:,} rows in this free-tier app to keep memory use predictable.")
                if dense:
                    check_dense_capacity(frame, features, len(X), name)
                transformed = make_preprocessor(frame, features, dense=dense).fit_transform(X)
                if name == "Gaussian Mixture":
                    estimator.fit(transformed)
                    labels = estimator.predict(transformed)
                else:
                    labels = estimator.fit_predict(transformed)
                output = frame[features].copy()
                output["Cluster"] = [str(label) for label in labels]
                unique_labels = set(labels)
                metrics = {"Rows": len(labels), "Clusters (excluding noise)": len(unique_labels - {-1})}
                if evaluation_label and evaluation_label in frame.columns:
                    known = frame[evaluation_label]
                    labeled = known.notna()
                    if labeled.sum() >= 2:
                        predicted_labels = np.asarray(labels)[labeled.to_numpy()]
                        known_labels = known[labeled].to_numpy()
                        metrics["Adjusted Rand (known labels)"] = adjusted_rand_score(known_labels, predicted_labels)
                        metrics["NMI (known labels)"] = normalized_mutual_info_score(known_labels, predicted_labels)
                        output["Known label (evaluation only)"] = known.to_numpy()
                if -1 in unique_labels:
                    metrics["Noise rows"] = int(np.sum(np.asarray(labels) == -1))
                score_mask = np.asarray(labels) != -1 if -1 in unique_labels else np.ones(len(labels), dtype=bool)
                score_labels = np.asarray(labels)[score_mask]
                score_features = transformed[score_mask]
                if len(set(score_labels)) > 1 and len(set(score_labels)) < len(score_labels):
                    try:
                        score = silhouette_score(
                            score_features,
                            score_labels,
                            sample_size=min(5000, len(score_labels)),
                            random_state=seed,
                        )
                        metrics["Silhouette score"] = score
                    except ValueError:
                        metrics["Silhouette score"] = np.nan
                scatter_columns = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])][:2]
                details = {
                    "output": output.reset_index(names="Original row"),
                    "cluster_sizes": pd.Series(labels).astype(str).value_counts().rename_axis("Cluster").reset_index(name="Rows"),
                }
                if len(scatter_columns) == 2:
                    details["scatter_columns"] = scatter_columns
            elif task == "Dimensionality reduction":
                check_dense_capacity(frame, features, len(X), "PCA")
                transformed = make_preprocessor(frame, features, dense=True).fit_transform(X)
                max_components = min(transformed.shape[0], transformed.shape[1])
                components = min(max(1, settings["components"]), max_components)
                estimator = PCA(n_components=components, random_state=seed)
                projection = estimator.fit_transform(transformed)
                component_names = [f"PC{i + 1}" for i in range(projection.shape[1])]
                output = pd.DataFrame(projection, columns=component_names)
                output.insert(0, "Original row", frame.index)
                metrics = {"Rows": len(output), "Components": components, "Explained variance": estimator.explained_variance_ratio_.sum()}
                details = {
                    "output": output,
                    "explained_variance": pd.DataFrame(
                        {
                            "Component": component_names,
                            "Explained variance ratio": estimator.explained_variance_ratio_,
                            "Cumulative variance": np.cumsum(estimator.explained_variance_ratio_),
                        },
                    ),
                }
            else:
                contamination = settings["contamination"]
                estimator = (
                    IsolationForest(contamination=contamination, random_state=seed)
                    if name == "Isolation Forest"
                    else LocalOutlierFactor(contamination=contamination)
                )
                if name == "Local Outlier Factor":
                    if len(X) > MAX_DENSITY_MODEL_ROWS:
                        raise ValueError(f"Local Outlier Factor is limited to {MAX_DENSITY_MODEL_ROWS:,} rows in this free-tier app to keep memory use predictable.")
                    check_dense_capacity(frame, features, len(X), name)
                transformed = make_preprocessor(frame, features, dense=name == "Local Outlier Factor").fit_transform(X)
                labels = estimator.fit_predict(transformed)
                scores = (
                    estimator.decision_function(transformed)
                    if name == "Isolation Forest"
                    else estimator.negative_outlier_factor_
                )
                output = frame[features].copy()
                output["Result"] = np.where(np.asarray(labels) == -1, "Anomaly", "Inlier")
                output["Anomaly score (lower is more unusual)"] = scores
                metrics = {
                    "Rows": len(labels),
                    "Anomalies": int(np.sum(np.asarray(labels) == -1)),
                    "Anomaly rate": float(np.mean(np.asarray(labels) == -1)),
                }
                scatter_columns = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])][:2]
                details = {"output": output.reset_index(names="Original row")}
                if len(scatter_columns) == 2:
                    details["scatter_columns"] = scatter_columns
            output = details["output"]
            output = output.reset_index(names="Original row") if "Original row" not in output.columns else output
            preview, download_data, download_suffix, download_mime = prepare_row_download(output, seed)
            details.update({
                "output": preview,
                "output_rows": len(output),
                "download_data": download_data,
                "download_suffix": download_suffix,
                "download_mime": download_mime,
            })
            results.append({"model": name, "metrics": metrics, "details": details, "error": None})
        except Exception as exc:
            results.append({"model": name, "metrics": {}, "details": {}, "error": str(exc)})
    return [{"note": None, "results": results}]


def run_models(
    frame: pd.DataFrame,
    task: str,
    target: str | None,
    features: list[str],
    models: list[str],
    test_fraction: float,
    settings: dict,
    cv_folds: int = 0,
    explain_features: bool = False,
    evaluation_label: str | None = None,
) -> list[dict]:
    if len(models) > MAX_MODELS_PER_RUN:
        raise ValueError(f"Choose up to {MAX_MODELS_PER_RUN} models per dataset in one run to keep the free-tier app responsive.")
    if task in ("Classification", "Regression"):
        if not target:
            raise ValueError("Choose a target column for supervised learning.")
        return run_supervised(frame, task, target, features, models, test_fraction, cv_folds, explain_features, settings)
    return run_unsupervised(frame, task, features, models, settings, evaluation_label)


def config_signature(frame: pd.DataFrame, config: dict) -> str:
    values = {"dataset": frame.attrs.get("mlstudio_cache_key", ""), **config}
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _histogram_chart(data: pd.DataFrame, title: str, height: int = 280):
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("bin_start:Q", title=title), x2="bin_end:Q", y=alt.Y("rows:Q", title="Rows"),
        tooltip=[alt.Tooltip("bin_start:Q", title="From", format=".3g"), alt.Tooltip("bin_end:Q", title="To", format=".3g"), "rows:Q"],
    ).properties(height=height)


def _scatter_chart(data: pd.DataFrame, x: str, y: str, color: str | None, title: str, height: int = 320):
    encodings = {
        "x": alt.X(f"{x}:Q", title=x), "y": alt.Y(f"{y}:Q", title=y),
        "tooltip": [alt.Tooltip(f"{x}:Q", format=".3g"), alt.Tooltip(f"{y}:Q", format=".3g")],
    }
    if color and color in data:
        encodings["color"] = alt.Color(f"{color}:N", title=color)
        encodings["tooltip"].append(alt.Tooltip(f"{color}:N", title=color))
    return alt.Chart(data).mark_circle(size=45, opacity=0.65).encode(**encodings).properties(title=title, height=height)


def _render_correlation(frame: pd.DataFrame, columns: list[str], key: str, height: int = 420) -> None:
    sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, tuple(columns))
    data = correlation_data(sample, columns)
    if data.empty:
        st.info("Choose at least two numeric columns with usable values to calculate correlations.")
        return
    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X("x:O", title=None), y=alt.Y("y:O", title=None),
        color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"), title="Pearson r"),
        tooltip=["x:O", "y:O", alt.Tooltip("correlation:Q", format=".2f")],
    ).properties(height=height)
    st.altair_chart(chart, width="stretch", key=key)
    st.caption("How to read: values near +1 move together linearly; values near −1 move in opposite directions; values near 0 have little linear association. Correlation does not establish causation.")


def _render_missingness(frame: pd.DataFrame, key: str) -> None:
    summary = missingness_summary(frame).query("`Missing rows` > 0").sort_values("Missing %", ascending=False)
    if summary.empty:
        st.success("No missing values found.")
        return
    if len(summary) > MAX_MISSINGNESS_CHART_COLUMNS:
        summary = summary.head(MAX_MISSINGNESS_CHART_COLUMNS)
        st.caption(f"Showing the {MAX_MISSINGNESS_CHART_COLUMNS} columns with the highest missingness.")
    chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X("Missing %:Q", title="Missing rows (%)"),
        y=alt.Y("Column:N", sort="-x", title=None),
        tooltip=["Column:N", "Missing rows:Q", alt.Tooltip("Missing %:Q", format=".1f")],
        color=alt.Color("Missing %:Q", scale=alt.Scale(scheme="orangered"), legend=None),
    ).properties(height=min(400, 24 * len(summary) + 40))
    st.altair_chart(chart, width="stretch", key=key)
    st.caption("How to read: each bar is the share of rows absent from that column. Consider whether missingness itself may carry information before choosing an imputation strategy.")


def _render_target_charts(frame: pd.DataFrame, target: str | None, task: str | None, numeric: list[str], categorical: list[str], key: str) -> None:
    if not target or target not in frame:
        return
    st.markdown(f"**Target exploration: {target}**")
    if task == "Classification":
        target_frame = frame.dropna(subset=[target])
        target_sample = _eda_sample(target_frame.attrs.get("mlstudio_cache_key", key), target_frame, (target,))
        counts = category_count_data(target_sample, target)
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("category:N", sort="-y", title=target), y=alt.Y("rows:Q", title="Rows"),
            color=alt.Color("category:N", legend=None), tooltip=["category:N", "rows:Q"],
        ).properties(height=280)
        st.altair_chart(chart, width="stretch", key=f"{key}_balance")
        st.caption("How to read: compare class counts before training. Rare classes can be hard to learn; accuracy can hide poor performance on them.")
        features = [column for column in numeric if column != target]
        if features:
            feature = st.selectbox("Numeric feature to compare across classes", features, key=f"{key}_class_feature")
            sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (feature, target))
            data = numeric_category_sample(sample, feature, target)
            if data.empty:
                st.info("No finite feature values are available for this class comparison.")
            else:
                box = alt.Chart(data).mark_boxplot().encode(
                    x=alt.X("category:N", title=target), y=alt.Y("value:Q", title=feature),
                    color=alt.Color("category:N", legend=None), tooltip=["category:N", alt.Tooltip("value:Q", format=".3g")],
                ).properties(height=300)
                st.altair_chart(box, width="stretch", key=f"{key}_class_box")
                st.caption("How to read: compare medians and spread for this one feature across classes. This view is descriptive; it does not show model importance or prove causation.")
        else:
            features = [column for column in categorical if column != target]
            if features:
                feature = st.selectbox("Categorical feature to compare with classes", features, key=f"{key}_class_cat_feature")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (feature, target))
                relation = category_relationship_data(sample, feature, target)
                chart = alt.Chart(relation).mark_rect().encode(
                    x=alt.X("x:O", title=feature), y=alt.Y("y:O", title=target),
                    color=alt.Color("rows:Q", title="Rows"), tooltip=["x:O", "y:O", "rows:Q"],
                ).properties(height=260)
                st.altair_chart(chart, width="stretch", key=f"{key}_class_category")
    elif task == "Regression":
        sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (target,))
        target_hist = histogram_data(sample, target)
        if not target_hist.empty:
            st.altair_chart(_histogram_chart(target_hist, target), width="stretch", key=f"{key}_target_hist")
            st.caption("How to read: inspect skew and extreme values; a long tail can make average-error measures sensitive to large misses.")
        else:
            st.info("The selected target has no finite numeric values to plot.")
        features = [column for column in numeric if column != target]
        if features:
            feature = st.selectbox("Numeric feature to compare with target", features, key=f"{key}_reg_feature")
            sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (feature, target))
            sample = sample.rename(columns={feature: "x", target: "y"}).dropna()
            if len(sample):
                st.altair_chart(_scatter_chart(sample, "x", "y", None, f"{feature} vs {target}"), width="stretch", key=f"{key}_reg_scatter")
                st.caption("How to read: look for a trend, curvature, changing spread, and extreme points. This association is exploratory and does not establish causation.")
        else:
            st.info("No other numeric feature is available for a feature-versus-target plot.")


def render_eda(frame: pd.DataFrame, target: str | None, task: str | None, key: str) -> None:
    st.markdown("#### Exploratory data analysis")
    profile = _eda_profile(frame.attrs.get("mlstudio_cache_key", key), frame)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Columns", f"{len(frame.columns):,}")
    c3.metric("Missing cells", f"{profile['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{profile['duplicates']:,}")
    st.caption("Uploads are kept in session memory only. Distribution and relationship charts use a repeatable sample of up to 5,000 rows.")

    warnings_to_show = []
    if len(frame) > MAX_SLOW_KERNEL_ROWS:
        warnings_to_show.append(f"Free-tier runtime limits: kernel SVM/SVR up to {MAX_SLOW_KERNEL_ROWS:,} training rows, KNN up to {MAX_NEIGHBOR_ROWS:,}, DBSCAN/Local Outlier Factor up to {MAX_DENSITY_MODEL_ROWS:,}, and Agglomerative Clustering up to {MAX_AGGLOMERATIVE_ROWS:,}.")
    missing_columns = profile["schema"].loc[profile["schema"]["Missing %"] >= 30, "Column"].tolist()
    if missing_columns:
        warnings_to_show.append(f"High missingness (30% or more): {', '.join(map(str, missing_columns[:8]))}.")
    if profile["constants"]:
        warnings_to_show.append(f"Constant columns carry no variation: {', '.join(map(str, profile['constants'][:8]))}.")
    if profile["likely_ids"]:
        warnings_to_show.append(f"Possible identifiers: {', '.join(map(str, profile['likely_ids'][:8]))}. Check whether to exclude them from features.")
    numeric_frame = frame.select_dtypes(include=np.number)
    infinite_cells = int(np.isinf(numeric_frame.to_numpy()).sum()) if not numeric_frame.empty else 0
    if infinite_cells:
        warnings_to_show.append(f"Found {infinite_cells:,} infinite numeric values; preprocessing treats them as missing.")
    if target and target in frame:
        missing_target = int(frame[target].isna().sum())
        if missing_target:
            warnings_to_show.append(f"{missing_target:,} rows have no target and will be left out of supervised training.")
        if task == "Classification":
            counts = frame[target].dropna().value_counts()
            if len(counts) > 1 and counts.min() / counts.max() < 0.2:
                warnings_to_show.append("The target is imbalanced. Accuracy alone may be misleading; compare macro and weighted F1 and per-class recall.")
    if warnings_to_show:
        with st.expander("Data quality checks — review before training", expanded=True):
            for note in warnings_to_show:
                st.warning(note)
    if task in ("Classification", "Regression"):
        st.info("Teaching note: the app splits rows before learning imputers, encoders, and scalers. Keep the test set untouched until the final evaluation to avoid data leakage.")

    numeric = list(frame.select_dtypes(include=np.number).columns)
    categorical = [column for column in frame.columns if column not in numeric]
    view = st.radio("EDA view", ["Guided walkthrough", "Explore dataset"], horizontal=True, key=f"{key}_view")
    with st.expander("Preview and column details"):
        st.dataframe(frame.head(100), width="stretch")
        st.dataframe(profile["schema"], hide_index=True, width="stretch")
        st.dataframe(profile["summary"], width="stretch")

    if view == "Guided walkthrough":
        st.markdown("##### 1. Check missing data")
        _render_missingness(frame, f"{key}_missing")
        st.markdown("##### 2. Inspect distributions")
        left, right = st.columns(2)
        with left:
            if numeric:
                column = st.selectbox("Numeric column", numeric, key=f"{key}_guided_num")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (column,))
                bins = histogram_data(sample, column)
                if bins.empty:
                    st.info("This column has no finite numeric values to plot.")
                else:
                    st.altair_chart(_histogram_chart(bins, column), width="stretch", key=f"{key}_guided_hist")
                    st.caption("How to read: shape, skew, gaps, and outliers can reveal data issues or useful transformations. What would you check before modeling?")
            else:
                st.info("No numeric columns are available for a histogram.")
        with right:
            if categorical:
                column = st.selectbox("Categorical column", categorical, key=f"{key}_guided_cat")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (column,))
                counts = category_count_data(sample, column)
                chart = alt.Chart(counts).mark_bar().encode(
                    x=alt.X("category:N", sort="-y", title=column), y=alt.Y("rows:Q", title="Rows"),
                    tooltip=["category:N", "rows:Q"], color=alt.Color("category:N", legend=None),
                ).properties(height=280)
                st.altair_chart(chart, width="stretch", key=f"{key}_guided_counts")
                st.caption("How to read: bars show common categories; rare categories are grouped as Other. Are any groups too small to evaluate reliably?")
            else:
                st.info("No categorical columns are available for category counts.")
        st.markdown("##### 3. Compare numeric relationships")
        if len(numeric) >= 2:
            _render_correlation(frame, numeric[:12], f"{key}_guided_corr", 360)
            st.caption("Discussion: which pair is most strongly associated, and what other explanation could produce that pattern?")
        else:
            st.info("At least two numeric columns are needed for a correlation heatmap.")
        _render_target_charts(frame, target, task, numeric, categorical, f"{key}_guided_target")
        if task in ("Classification", "Regression"):
            st.caption("EDA relationships help form questions; they do not establish causation or reveal how a trained model will perform.")
        with st.expander("Questions for students"):
            st.markdown("- Which columns have missing values, duplicates, or almost no variation?")
            st.markdown("- Could an identifier or target-related field give the model an unfair shortcut?")
            st.markdown("- How might imbalance, outliers, or skew change your choice of model or metric?")
    else:
        st.markdown("##### Explore distributions and missingness")
        left, right = st.columns(2)
        with left:
            if numeric:
                column = st.selectbox("Numeric column", numeric, key=f"{key}_explore_numeric")
                bins = st.slider("Histogram bins", 5, 50, 20, key=f"{key}_explore_bins")
                kind = st.radio("Distribution chart", ["Histogram", "Boxplot"], horizontal=True, key=f"{key}_explore_dist")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (column,))
                if kind == "Histogram":
                    histogram = histogram_data(sample, column, bins)
                    if histogram.empty:
                        st.info("This column has no finite numeric values to plot.")
                    else:
                        st.altair_chart(_histogram_chart(histogram, column), width="stretch", key=f"{key}_explore_hist")
                else:
                    data = sample.rename(columns={column: "value"}).dropna()
                    if data.empty:
                        st.info("This column has no finite numeric values to plot.")
                    else:
                        chart = alt.Chart(data).mark_boxplot().encode(y=alt.Y("value:Q", title=column), tooltip=[alt.Tooltip("value:Q", format=".3g")]).properties(height=280)
                        st.altair_chart(chart, width="stretch", key=f"{key}_explore_box")
            else:
                st.info("No numeric columns are available for distribution charts.")
        with right:
            if categorical:
                column = st.selectbox("Categorical column", categorical, key=f"{key}_explore_category")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (column,))
                counts = category_count_data(sample, column)
                chart = alt.Chart(counts).mark_bar().encode(
                    x=alt.X("category:N", sort="-y", title=column), y=alt.Y("rows:Q", title="Rows"),
                    tooltip=["category:N", "rows:Q"], color=alt.Color("category:N", legend=None),
                ).properties(height=280)
                st.altair_chart(chart, width="stretch", key=f"{key}_explore_counts")
            else:
                st.info("No categorical columns are available for category counts.")
        if frame.isna().any().any():
            missing_columns = st.multiselect("Columns in missingness pattern (up to 20)", list(frame.columns), default=list(frame.columns[:min(12, len(frame.columns))]), max_selections=MAX_CORRELATION_COLUMNS, key=f"{key}_missing_columns")
            sampled_pattern = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, tuple(missing_columns), MAX_MISSINGNESS_PLOT_ROWS)
            pattern = missingness_pattern_data(sampled_pattern, missing_columns)
            if not pattern.empty:
                chart = alt.Chart(pattern).mark_rect().encode(
                    x=alt.X("column:O", title=None), y=alt.Y("row:O", title="Sampled row", axis=None),
                    color=alt.Color("missing:N", scale=alt.Scale(domain=["False", "True"], range=["#dbeafe", "#ef4444"]), title="Missing"),
                    tooltip=["row:O", "column:O", "missing:N"],
                ).properties(height=240)
                st.altair_chart(chart, width="stretch", key=f"{key}_missing_pattern")
                st.caption("How to read: this sampled matrix shows missing cells in red. Patterns can suggest shared collection or recording issues.")
            else:
                st.info("Select one or more columns to show the missingness pattern.")
        else:
            st.success("No missing values found; a missingness pattern chart is not needed.")

        st.markdown("##### Correlation heatmap")
        if numeric:
            default_corr = numeric[:min(12, len(numeric))]
            corr_columns = st.multiselect("Numeric columns (up to 20)", numeric, default=default_corr, max_selections=MAX_CORRELATION_COLUMNS, key=f"{key}_corr_columns")
            _render_correlation(frame, corr_columns, f"{key}_explore_corr")
        else:
            st.info("No numeric columns are available for correlations.")

        st.markdown("##### Relationships")
        relation_types = []
        if len(numeric) >= 2:
            relation_types.append("Numeric vs numeric")
        if numeric and categorical:
            relation_types.append("Numeric by category")
        if len(categorical) >= 2:
            relation_types.append("Category vs category")
        if relation_types:
            relation = st.selectbox("Relationship view", relation_types, key=f"{key}_relationship_type")
            if relation == "Numeric vs numeric":
                axes = st.columns(2)
                x_column = axes[0].selectbox("X numeric", numeric, key=f"{key}_scatter_x")
                y_options = [column for column in numeric if column != x_column] or numeric
                y_column = axes[1].selectbox("Y numeric", y_options, key=f"{key}_scatter_y")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (x_column, y_column)).dropna()
                data = sample.rename(columns={x_column: "x", y_column: "y"})
                st.altair_chart(_scatter_chart(data, "x", "y", None, "Numeric relationship"), width="stretch", key=f"{key}_relation_scatter")
                st.caption("Look for direction, form, spread, and unusual points. A relationship does not establish causation.")
            elif relation == "Numeric by category":
                axes = st.columns(2)
                value_column = axes[0].selectbox("Numeric value", numeric, key=f"{key}_box_numeric")
                category_column = axes[1].selectbox("Category", categorical, key=f"{key}_box_category")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (value_column, category_column))
                data = numeric_category_sample(sample, value_column, category_column)
                if data.empty:
                    st.info("No finite numeric values are available for this comparison.")
                else:
                    chart = alt.Chart(data).mark_boxplot().encode(
                        x=alt.X("category:N", title=category_column), y=alt.Y("value:Q", title=value_column),
                        color=alt.Color("category:N", legend=None), tooltip=["category:N", alt.Tooltip("value:Q", format=".3g")],
                    ).properties(height=320)
                    st.altair_chart(chart, width="stretch", key=f"{key}_relation_box")
                    st.caption("Compare medians, spread, and outliers across the ten most common categories; remaining categories are grouped. These differences are descriptive, not causal.")
            else:
                axes = st.columns(2)
                x_column = axes[0].selectbox("First category", categorical, key=f"{key}_cat_x")
                y_options = [column for column in categorical if column != x_column] or categorical
                y_column = axes[1].selectbox("Second category", y_options, key=f"{key}_cat_y")
                sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, (x_column, y_column))
                data = category_relationship_data(sample, x_column, y_column)
                chart = alt.Chart(data).mark_rect().encode(
                    x=alt.X("x:O", title=x_column), y=alt.Y("y:O", title=y_column),
                    color=alt.Color("rows:Q", title="Rows"), tooltip=["x:O", "y:O", "rows:Q"],
                ).properties(height=320)
                st.altair_chart(chart, width="stretch", key=f"{key}_relation_categories")
                st.caption("Color shows row counts for category combinations; each variable is reduced to its ten most common values plus Other.")
        else:
            st.info("A relationship plot needs two numeric columns, or at least one numeric and one categorical column, or two categorical columns.")

        _render_target_charts(frame, target, task, numeric, categorical, f"{key}_explore_target")
        if len(numeric) >= 2:
            if st.checkbox("Enable pairwise numeric exploration (optional)", value=False, key=f"{key}_pair_enabled"):
                chosen = st.multiselect("Choose up to four numeric columns", numeric, default=numeric[:min(3, len(numeric))], max_selections=4, key=f"{key}_pair_columns")
                pairs = pairwise_pairs(chosen)
                if not pairs:
                    st.info("Choose at least two numeric columns to make pairwise scatterplots.")
                else:
                    st.caption(f"Each plot uses the same repeatable sample of at most {MAX_PAIRWISE_PLOT_ROWS:,} rows. Up to six pair plots are shown.")
                    sample = _eda_sample(frame.attrs.get("mlstudio_cache_key", key), frame, tuple(chosen), MAX_PAIRWISE_PLOT_ROWS)
                    for start in range(0, len(pairs), 2):
                        cols = st.columns(2)
                        for slot, (x_column, y_column) in zip(cols, pairs[start:start + 2]):
                            data = sample[[x_column, y_column]].dropna().rename(columns={x_column: "x", y_column: "y"})
                            if data.empty:
                                slot.info(f"{x_column} and {y_column} have no overlapping finite values.")
                            else:
                                slot.altair_chart(_scatter_chart(data, "x", "y", None, f"{x_column} vs {y_column}", 260), width="stretch", key=f"{key}_pair_{start}_{x_column}_{y_column}")
                    st.caption("Patterns can be nonlinear or driven by another variable. Visual association does not establish causation.")


def render_results(dataset_name: str, bundles: list[dict], key_prefix: str) -> list[dict]:
    st.markdown(f"### Results: {dataset_name}")
    flattened = []
    for bundle in bundles:
        if bundle.get("note"):
            st.warning(bundle["note"])
        for result in bundle["results"]:
            flattened.append({"Dataset": dataset_name, "Model": result["model"], **result["metrics"], "Error": result["error"] or ""})
            with st.expander(result["model"], expanded=len(bundle["results"]) == 1):
                if result["error"]:
                    st.error(result["error"])
                    continue
                st.dataframe(pd.DataFrame([result["metrics"]]), hide_index=True, width="stretch")
                if result["model"].startswith("Baseline"):
                    st.caption("Baseline models ignore the input features. A useful model should beat this simple reference on the same held-out rows.")
                if result["details"].get("cv_note"):
                    st.info(result["details"]["cv_note"])
                if any(column.startswith("CV ") for column in result["metrics"]):
                    st.caption("CV results summarize folds from the training portion. The separate test score is still the final holdout check.")
                if "Adjusted Rand (known labels)" in result["metrics"]:
                    st.caption("Known labels were excluded from the clustering inputs. ARI is chance-adjusted (1 is perfect, near 0 is chance, and it can be negative); NMI is not chance-adjusted.")
                if "confusion_matrix" in result["details"]:
                    st.markdown("**Confusion matrix**")
                    matrix = result["details"]["confusion_matrix"]
                    if len(matrix) <= 30:
                        matrix_data = pd.DataFrame(
                            [(str(actual), str(predicted), int(matrix.loc[actual, predicted])) for actual in matrix.index for predicted in matrix.columns],
                            columns=["Actual", "Predicted", "Rows"],
                        )
                        heat = alt.Chart(matrix_data).mark_rect().encode(
                            x=alt.X("Predicted:O", title="Predicted class"), y=alt.Y("Actual:O", title="Actual class"),
                            color=alt.Color("Rows:Q", title="Rows", scale=alt.Scale(scheme="blues")),
                            tooltip=["Actual:O", "Predicted:O", "Rows:Q"],
                        ).properties(height=min(500, max(260, 28 * len(matrix))))
                        st.altair_chart(heat, width="stretch", key=f"{key_prefix}_cm_{len(flattened)}")
                        st.caption("How to read: the diagonal is correct predictions; off-diagonal cells show which classes are confused. Compare errors by class, not only the total.")
                    else:
                        st.info("There are more than 30 classes, so the matrix is shown as a table to keep labels readable.")
                    st.dataframe(matrix, width="stretch")
                    st.markdown("**Per-class precision, recall, and F1**")
                    report = result["details"]["classification_report"]
                    st.dataframe(report, width="stretch")
                    class_report = report.drop(index=[name for name in ("accuracy", "macro avg", "weighted avg") if name in report.index], errors="ignore")
                    if not class_report.empty:
                        recall_data = class_report.reset_index(names="Class")
                        recall_chart = alt.Chart(recall_data).mark_bar().encode(
                            x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("Class:N", sort="-x", title=None), color=alt.Color("recall:Q", legend=None),
                            tooltip=["Class:N", alt.Tooltip("recall:Q", format=".3f")],
                        ).properties(height=min(420, 28 * len(recall_data) + 30))
                        st.altair_chart(recall_chart, width="stretch", key=f"{key_prefix}_recall_{len(flattened)}")
                        st.caption("How to read: recall is the share of actual examples of a class that were found. Low recall means many examples of that class were missed.")
                if "Train accuracy" in result["metrics"] or "Train R²" in result["metrics"]:
                    st.caption("Train scores describe fit to examples the model saw. Test scores estimate performance on held-out examples. A large gap can indicate overfitting.")
                output = result["details"].get("output")
                if "explained_variance" in result["details"]:
                    variance = result["details"]["explained_variance"]
                    st.markdown("**PCA variance retained**")
                    bars = alt.Chart(variance).mark_bar(color="#4878cf").encode(
                        x=alt.X("Component:O", title="Component"), y=alt.Y("Explained variance ratio:Q", title="Variance ratio"),
                        tooltip=["Component:O", alt.Tooltip("Explained variance ratio:Q", format=".3f"), alt.Tooltip("Cumulative variance:Q", format=".3f")],
                    )
                    line = alt.Chart(variance).mark_line(color="#d94841", point=True).encode(
                        x=alt.X("Component:O"), y=alt.Y("Cumulative variance:Q", title="Cumulative variance", scale=alt.Scale(domain=[0, 1])),
                    )
                    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=300), width="stretch", key=f"{key_prefix}_pca_variance_{len(flattened)}")
                    st.caption("Bars show the variance share in each component; the red line shows cumulative variance. Retained variance measures compression, not predictive accuracy.")
                if output is not None:
                    output_rows = result["details"].get("output_rows", len(output))
                    if output_rows > len(output):
                        st.caption(f"Showing a repeatable sample of {len(output):,} of {output_rows:,} result rows. Download the full file below.")
                    st.dataframe(output.head(100), width="stretch")
                    if {"Actual", "Predicted"}.issubset(output.columns):
                        actual = output[["Actual", "Predicted"]].apply(pd.to_numeric, errors="coerce").dropna()
                        if not actual.empty:
                            low = float(actual.min().min())
                            high = float(actual.max().max())
                            points = alt.Chart(actual).mark_circle(size=45, opacity=0.6).encode(
                                x=alt.X("Actual:Q", title="Actual"), y=alt.Y("Predicted:Q", title="Predicted"),
                                tooltip=[alt.Tooltip("Actual:Q", format=".3g"), alt.Tooltip("Predicted:Q", format=".3g")],
                            )
                            reference = alt.Chart(pd.DataFrame({"Actual": [low, high], "Predicted": [low, high]})).mark_line(color="#d94841", strokeDash=[5, 4]).encode(x="Actual:Q", y="Predicted:Q")
                            st.markdown("**Actual vs predicted**")
                            st.altair_chart(alt.layer(points, reference).properties(height=320), width="stretch", key=f"{key_prefix}_actual_pred_{len(flattened)}")
                            st.caption("The dashed line is perfect prediction. Points farther away have larger errors; this view uses held-out rows only.")
                    if "Residual" in output.columns:
                        st.markdown("**Residual diagnostics (actual − predicted)**")
                        residuals = output[["Predicted", "Residual"]].apply(pd.to_numeric, errors="coerce").dropna()
                        if not residuals.empty:
                            left, right = st.columns(2)
                            residual_data = histogram_data(output, "Residual", bins=25)
                            left.altair_chart(_histogram_chart(residual_data, "Residual", 260), width="stretch", key=f"{key_prefix}_resid_hist_{len(flattened)}")
                            right.altair_chart(_scatter_chart(residuals.rename(columns={"Predicted": "x", "Residual": "y"}), "x", "y", None, "Residual vs predicted", 260), width="stretch", key=f"{key_prefix}_resid_scatter_{len(flattened)}")
                            st.caption("A centered residual distribution is useful; curves or changing spread in residual-vs-predicted plots can reveal a missed pattern or unequal error variance.")
                    if "cluster_sizes" in result["details"]:
                        sizes = result["details"]["cluster_sizes"]
                        size_chart = alt.Chart(sizes).mark_bar().encode(
                            x=alt.X("Rows:Q", title="Rows"), y=alt.Y("Cluster:N", sort="-x", title=None),
                            color=alt.Color("Cluster:N", legend=None), tooltip=["Cluster:N", "Rows:Q"],
                        ).properties(height=min(320, 35 * len(sizes) + 30))
                        st.altair_chart(size_chart, width="stretch", key=f"{key_prefix}_cluster_sizes_{len(flattened)}")
                        st.caption("How to read: cluster IDs are arbitrary. Check cluster sizes and context; a visual group is not automatically meaningful.")
                    color_column = "Cluster" if "Cluster" in output.columns else "Result" if "Result" in output.columns else None
                    if color_column:
                        excluded = {"Original row", "Anomaly score (lower is more unusual)"}
                        plot_columns = [column for column in output.select_dtypes(include=np.number).columns if column not in excluded]
                        if len(plot_columns) >= 2:
                            axes = st.columns(2)
                            x_column = axes[0].selectbox("Plot X feature", plot_columns, key=f"{key_prefix}_result_x_{len(flattened)}")
                            y_options = [column for column in plot_columns if column != x_column] or plot_columns
                            y_column = axes[1].selectbox("Plot Y feature", y_options, key=f"{key_prefix}_result_y_{len(flattened)}")
                            scatter = output[[x_column, y_column, color_column]].dropna().rename(columns={x_column: "x", y_column: "y"})
                            st.altair_chart(_scatter_chart(scatter, "x", "y", color_column, "Two-feature projection"), width="stretch", key=f"{key_prefix}_cluster_scatter_{len(flattened)}")
                            st.caption("Illustrative two-feature view: the model may have used more columns and dimensions. This projection is not a full measure of clustering quality.")
                    if "Anomaly score (lower is more unusual)" in output.columns:
                        st.markdown("**Anomaly score distribution**")
                        scores = output[["Anomaly score (lower is more unusual)", "Result"]].dropna().rename(columns={"Anomaly score (lower is more unusual)": "score"})
                        score_chart = alt.Chart(scores).mark_bar(opacity=0.7).encode(
                            x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Score (lower is more unusual)"),
                            y=alt.Y("count():Q", title="Rows"), color=alt.Color("Result:N", title="Flag"),
                            tooltip=["Result:N", alt.Tooltip("count():Q", title="Rows")],
                        ).properties(height=300)
                        st.altair_chart(score_chart, width="stretch", key=f"{key_prefix}_anomaly_hist_{len(flattened)}")
                        st.caption("The score is a ranking signal, not a probability. Lower scores are more unusual; flagged status depends on the selected contamination setting.")
                    if {"PC1", "PC2"}.issubset(output.columns):
                        pc_data = output[["PC1", "PC2"]].dropna()
                        st.altair_chart(_scatter_chart(pc_data, "PC1", "PC2", None, "First two principal components"), width="stretch", key=f"{key_prefix}_pca_scatter_{len(flattened)}")
                        st.caption("This is a two-component projection. It can hide structure in later components and does not measure predictive performance.")
                    download_data = result["details"].get("download_data")
                    if download_data is None:
                        download_data = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download row-level results",
                        data=download_data,
                        file_name=f"{safe_file_stem(dataset_name)}_{safe_file_stem(result['model'])}_results{result['details'].get('download_suffix', '.csv')}",
                        mime=result["details"].get("download_mime", "text/csv"),
                        key=f"download_{key_prefix}_{len(flattened)}",
                    )
                if "feature_importance" in result["details"]:
                    st.markdown("**Permutation feature importance**")
                    st.caption("This measures how much the held-out score changes when one input column is shuffled. It is model-specific and does not prove causation.")
                    if result["details"].get("importance_sampled"):
                        st.caption("For speed, importance used a repeatable sample of up to 5,000 held-out rows.")
                    st.dataframe(result["details"]["feature_importance"].head(15), hide_index=True, width="stretch")
    if flattened:
        st.markdown("**Model comparison for this dataset**")
        comparison = pd.DataFrame(flattened).drop(columns=["Dataset"])
        st.dataframe(comparison, hide_index=True, width="stretch")
    return flattened


def task_controls(task: str, models: list[str], rows: int, key: str) -> dict:
    settings = {}
    if task == "Clustering" and any(model != "DBSCAN" for model in models):
        maximum = max(2, min(20, rows - 1))
        settings["clusters"] = st.number_input(
            "Number of clusters (K-Means, Agglomerative, Gaussian Mixture)",
            min_value=2,
            max_value=maximum,
            value=min(3, maximum),
            key=f"{key}_clusters",
        )
    if task == "Clustering" and "DBSCAN" in models:
        settings["eps"] = st.number_input("DBSCAN epsilon", min_value=0.01, max_value=10.0, value=0.5, step=0.1, key=f"{key}_eps")
        settings["min_samples"] = st.number_input("DBSCAN minimum samples", min_value=2, max_value=100, value=5, key=f"{key}_min_samples")
    if task == "Dimensionality reduction":
        maximum = max(1, min(rows, 50))
        settings["components"] = st.number_input("PCA components", min_value=1, max_value=maximum, value=min(2, maximum), key=f"{key}_components")
    if task == "Anomaly detection":
        settings["contamination"] = st.slider("Expected anomaly fraction", 0.01, 0.25, 0.05, 0.01, key=f"{key}_contamination")
    return settings


def evaluation_controls(task: str, key: str) -> tuple[int, bool]:
    if task not in ("Classification", "Regression"):
        return 0, False
    cv_folds = st.selectbox(
        "Compare reliability",
        options=[0, 3, 5],
        index=0,
        format_func=lambda value: "Holdout only (fast)" if value == 0 else f"{value}-fold CV on training data + final holdout",
        help="Cross-validation helps compare models using several training folds. The final test set remains untouched.",
        key=f"{key}_cv_folds",
    )
    explain_features = st.checkbox(
        "Calculate permutation feature importance (slower)",
        value=False,
        help="Shuffle each original input column on held-out rows to show how the score changes. Uses at most 5,000 held-out rows, is model-specific, and does not prove causation.",
        key=f"{key}_permutation_importance",
    )
    return cv_folds, explain_features


def reproducibility_control(key: str) -> int:
    return int(st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        help="Use the same seed and settings to repeat the split and randomized models.",
        key=f"{key}_random_seed",
    ))


def supervised_controls(models: list[str], rows: int, key: str) -> dict:
    settings = {}
    tree_models = {"Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting"}
    ensemble_models = {"Random Forest", "Extra Trees", "Gradient Boosting"}
    if tree_models.intersection(models):
        settings["max_depth"] = st.slider(
            "Maximum tree depth",
            1,
            20,
            5,
            help="Shallow trees are easier to explain and may generalize better; deep trees can memorize training rows.",
            key=f"{key}_max_depth",
        )
    if ensemble_models.intersection(models):
        settings["n_estimators"] = st.slider(
            "Number of trees",
            50,
            300,
            150,
            50,
            help="More trees usually take longer; they do not always improve test performance.",
            key=f"{key}_n_estimators",
        )
    if "K-Nearest Neighbors" in models:
        max_neighbors = max(1, min(50, rows - 1))
        settings["neighbors"] = st.number_input(
            "Number of neighbors (k)",
            min_value=1,
            max_value=max_neighbors,
            value=min(5, max_neighbors),
            help="Small k can be sensitive to noise; large k smooths decision boundaries.",
            key=f"{key}_neighbors",
        )
    regularized = {"Logistic Regression", "Support Vector Machine", "Support Vector Regression"}
    if regularized.intersection(models):
        settings["regularization"] = st.select_slider(
            "Regularization strength (C)",
            options=[0.1, 0.5, 1.0, 5.0, 10.0],
            value=1.0,
            help="Higher C penalizes training errors less and can fit more complex boundaries.",
            key=f"{key}_regularization",
        )
    if {"Ridge", "Lasso"}.intersection(models):
        settings["alpha"] = st.select_slider(
            "Linear-model regularization (alpha)",
            options=[0.01, 0.1, 1.0, 10.0],
            value=1.0,
            help="Higher alpha shrinks coefficients more strongly.",
            key=f"{key}_alpha",
        )
    return settings


def model_controls(task: str, models: list[str], rows: int, key: str) -> dict:
    if task in ("Classification", "Regression"):
        return supervised_controls(models, rows, key)
    return task_controls(task, models, rows, key)


def clear_session_data() -> None:
    st.cache_data.clear()
    st.session_state.clear()


def render_single_dataset() -> None:
    source = st.radio("Start with", ["Upload a CSV", "Try a learning example"], horizontal=True, key="single_source")
    dataset_name = ""
    if source == "Try a learning example":
        dataset_name = st.selectbox("Learning example", list(DEMO_TASKS), key="single_demo_name")
        frame = make_demo_dataset(dataset_name)
        task = DEMO_TASKS[dataset_name]
        st.caption("Built-in examples are generated in memory and include small, intentionally varied teaching datasets.")
    else:
        upload = st.file_uploader("Upload a CSV file (up to 25 MB)", type=["csv"], max_upload_size=25, key="single_upload")
        if upload is None:
            st.info("Upload a tabular CSV to begin, or choose a built-in learning example.")
            return
        try:
            frame = read_csv(upload)
            dataset_name = upload.name.rsplit(".", 1)[0]
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            return
        task = st.selectbox("Machine learning task", TASKS, key="single_task")

    dataset_key = frame.attrs.get("mlstudio_cache_key", dataset_name).replace("demo:", "")[:16]
    target = None
    if task in ("Classification", "Regression"):
        expected_target = {"Classification example": "label", "Regression example": "target"}.get(dataset_name)
        target_options = [None, *frame.columns]
        target_index = target_options.index(expected_target) if expected_target in frame.columns else 0
        target = st.selectbox(
            "Target column",
            target_options,
            index=target_index,
            format_func=lambda value: "Choose the value to predict" if value is None else str(value),
            key=f"single_target_{dataset_key}",
        )
    evaluation_label = None
    if task == "Clustering":
        evaluation_label = st.selectbox(
            "Optional known labels (evaluation only; never used to fit clusters)",
            options=[None, *frame.columns],
            format_func=lambda value: "No known labels" if value is None else str(value),
            key=f"single_evaluation_label_{dataset_key}",
        )
    render_eda(frame, target, task, f"single_eda_{dataset_key}")
    features = choose_features(frame, target, f"single_{dataset_key}", [evaluation_label] if evaluation_label else None)
    models = st.multiselect(
        "Models to run",
        options=available_models(task),
        default=available_models(task)[:1],
        key=f"single_models_{task}",
    )
    render_learning_guide(task, models)
    test_fraction = 0.2
    if task in ("Classification", "Regression"):
        test_fraction = st.slider("Test set size", 10, 50, 20, 5, key="single_test_split") / 100
    cv_folds, explain_features = evaluation_controls(task, "single_evaluation")
    random_seed = reproducibility_control("single")
    settings = model_controls(task, models, len(frame), "single_settings") if models else {}
    settings["random_state"] = random_seed
    if len(models) > MAX_MODELS_PER_RUN:
        st.warning(f"Choose at most {MAX_MODELS_PER_RUN} models per run to keep the free-tier app responsive.")
    result_key = "single_saved_results"
    run_config = {
        "task": task,
        "target": target,
        "evaluation_label": evaluation_label,
        "features": features,
        "models": models,
        "test_fraction": test_fraction,
        "settings": settings,
        "cv_folds": cv_folds,
        "explain_features": explain_features,
    }
    signature = config_signature(
        frame,
        {"dataset_name": dataset_name, **run_config},
    )
    if st.button("Run selected model(s)", type="primary", key="single_run", disabled=len(models) > MAX_MODELS_PER_RUN):
        if not models:
            st.error("Select at least one model.")
        else:
            try:
                with st.spinner(f"Training {len(models)} model(s)…"):
                    bundles = run_models(
                        frame,
                        task,
                        target,
                        features,
                        models,
                        test_fraction,
                        settings,
                        cv_folds,
                        explain_features,
                        evaluation_label,
                    )
                st.session_state[result_key] = {"signature": signature, "name": dataset_name, "config": run_config, "bundles": bundles}
            except Exception as exc:
                st.error(str(exc))
    saved = st.session_state.get(result_key)
    if saved and saved["signature"] == signature:
        render_results(saved["name"], saved["bundles"], "single")
        report = build_markdown_report([saved], title="ML Studio run report")
        st.download_button(
            "Download run report (.md)",
            data=report.encode("utf-8"),
            file_name=f"{safe_file_stem(saved['name'])}_ml_report.md",
            mime="text/markdown",
            key="download_single_report",
        )
        if st.button("Clear saved results", key="single_clear_results"):
            del st.session_state[result_key]
            st.rerun()
    elif saved:
        st.info("Settings changed since the last run. Run the model(s) again to refresh these results.")


def render_dataset_configuration(upload, index: int, mode: str, shared_task: str | None = None, shared_model: str | None = None):
    try:
        frame = read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None

    key = f"compare_{index}_{frame.attrs.get('mlstudio_cache_key', '')[:12]}"
    with st.expander(f"{upload.name} — {len(frame):,} rows × {len(frame.columns):,} columns", expanded=index == 0):
        task = shared_task or st.selectbox("Task", TASKS, key=f"{key}_task")
        target = None
        if task in ("Classification", "Regression"):
            target = st.selectbox(
                "Target column",
                [None, *frame.columns],
                format_func=lambda value: "Choose the value to predict" if value is None else str(value),
                key=f"{key}_target",
            )
        evaluation_label = None
        if task == "Clustering":
            evaluation_label = st.selectbox(
                "Optional known labels (evaluation only; never used to fit clusters)",
                options=[None, *frame.columns],
                format_func=lambda value: "No known labels" if value is None else str(value),
                key=f"{key}_evaluation_label",
            )
        render_eda(frame, target, task, f"{key}_eda")
        features = choose_features(frame, target, key, [evaluation_label] if evaluation_label else None)
        if mode == "Within each dataset":
            models = st.multiselect(
                "Models to compare",
                options=available_models(task),
                default=available_models(task)[:1],
                key=f"{key}_models_{task}",
            )
        else:
            models = [shared_model] if shared_model else []
        render_learning_guide(task, models)
        test_fraction = 0.2
        if task in ("Classification", "Regression"):
            test_fraction = st.slider("Test set size", 10, 50, 20, 5, key=f"{key}_split") / 100
        cv_folds, explain_features = evaluation_controls(task, f"{key}_evaluation")
        random_seed = reproducibility_control(key)
        settings = model_controls(task, models, len(frame), f"{key}_settings") if models else {}
        settings["random_state"] = random_seed
        if len(models) > MAX_MODELS_PER_RUN:
            st.warning(f"Choose at most {MAX_MODELS_PER_RUN} models for this dataset in one run.")
    return {
        "name": upload.name.rsplit(".", 1)[0],
        "frame": frame,
        "task": task,
        "target": target,
        "evaluation_label": evaluation_label,
        "features": features,
        "models": models,
        "test_fraction": test_fraction,
        "settings": settings,
        "cv_folds": cv_folds,
        "explain_features": explain_features,
    }


def render_comparison() -> None:
    uploads = st.file_uploader(
        "Upload 2–4 CSV files (up to 25 MB each)",
        type=["csv"],
        accept_multiple_files=True,
        max_upload_size=25,
        key="multi_upload",
    )
    if not uploads:
        st.info("Upload at least two CSV files to compare models across datasets.")
        return
    if len(uploads) > 4:
        st.error("Compare up to four datasets at a time to keep memory and model runtimes predictable.")
        return
    total_upload_bytes = sum(upload.size for upload in uploads)
    if total_upload_bytes > MAX_COMPARISON_UPLOAD_BYTES:
        st.error(f"The combined comparison upload limit is {MAX_COMPARISON_UPLOAD_BYTES // (1024 * 1024)} MB. Remove files or use smaller samples to protect free-tier memory.")
        return
    mode = st.radio(
        "Comparison mode",
        ["Within each dataset", "Same model across datasets"],
        horizontal=True,
        key="compare_mode",
    )
    shared_task = None
    shared_model = None
    if mode == "Same model across datasets":
        shared_task = st.selectbox("Task for all datasets", TASKS, key="shared_task")
        shared_model = st.selectbox("Model to run on every dataset", available_models(shared_task), key=f"shared_model_{shared_task}")

    configs = []
    for index, upload in enumerate(uploads):
        config = render_dataset_configuration(upload, index, mode, shared_task, shared_model)
        if config is not None:
            configs.append(config)

    if len(uploads) < 2:
        st.warning("Upload at least two CSV files for a dataset comparison.")
    signature = hashlib.sha256(
        json.dumps(
            {"mode": mode, "configs": [
                config_signature(
                    config["frame"],
                    {
                        "task": config["task"],
                        "name": config["name"],
                        "target": config["target"],
                        "evaluation_label": config["evaluation_label"],
                        "features": config["features"],
                        "models": config["models"],
                        "test_fraction": config["test_fraction"],
                        "settings": config["settings"],
                        "cv_folds": config["cv_folds"],
                        "explain_features": config["explain_features"],
                    },
                )
                for config in configs
            ]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    result_key = "compare_saved_results"
    too_many_models = any(len(config["models"]) > MAX_MODELS_PER_RUN for config in configs)
    if too_many_models:
        st.warning(f"Choose at most {MAX_MODELS_PER_RUN} models per dataset before running the comparison.")
    if st.button("Run comparison", type="primary", key="compare_run", disabled=too_many_models):
        if len(configs) < 2:
            st.error("At least two readable CSV files are required.")
            return
        saved_runs = []
        for index, config in enumerate(configs):
            if not config["models"]:
                saved_runs.append({"name": config["name"], "error": "Select at least one model.", "bundles": []})
                continue
            try:
                with st.spinner(f"Training models for {config['name']}…"):
                    bundles = run_models(
                        config["frame"],
                        config["task"],
                        config["target"],
                        config["features"],
                        config["models"],
                        config["test_fraction"],
                        config["settings"],
                        config["cv_folds"],
                        config["explain_features"],
                        config["evaluation_label"],
                    )
                run_config = {
                    "task": config["task"],
                    "target": config["target"],
                    "evaluation_label": config["evaluation_label"],
                    "features": config["features"],
                    "models": config["models"],
                    "test_fraction": config["test_fraction"],
                    "settings": config["settings"],
                    "cv_folds": config["cv_folds"],
                    "explain_features": config["explain_features"],
                }
                saved_runs.append({"name": config["name"], "config": run_config, "error": None, "bundles": bundles, "key": f"compare_{index}"})
            except Exception as exc:
                saved_runs.append({"name": config["name"], "error": str(exc), "bundles": []})
        st.session_state[result_key] = {"signature": signature, "runs": saved_runs}

    saved = st.session_state.get(result_key)
    if saved and saved["signature"] == signature:
        all_rows = []
        for run in saved["runs"]:
            st.markdown("---")
            if run["error"]:
                st.error(f"{run['name']}: {run['error']}")
                continue
            all_rows.extend(render_results(run["name"], run["bundles"], run["key"]))
        if all_rows:
            summary = pd.DataFrame(all_rows)
            st.download_button(
                "Download per-dataset metrics (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
                key="download_comparison_summary",
            )
            st.caption("The CSV keeps each dataset in its own row and does not average scores across datasets. On-screen model summaries are shown separately under each dataset.")
        report = build_markdown_report(saved["runs"], title="ML Studio dataset comparison report")
        st.download_button(
            "Download comparison report (.md)",
            data=report.encode("utf-8"),
            file_name="ml_dataset_comparison_report.md",
            mime="text/markdown",
            key="download_comparison_report",
        )
        if st.button("Clear saved comparison", key="compare_clear_results"):
            del st.session_state[result_key]
            st.rerun()
    elif saved:
        st.info("Comparison settings changed since the last run. Run the comparison again to refresh these results.")


def main() -> None:
    st.set_page_config(page_title="ML Studio", page_icon="🧪", layout="wide")
    with st.sidebar:
        st.markdown("## Personal workspace")
        st.caption("Uploads and results stay in this active session; the app does not write them to storage.")
        st.button(
            "Clear uploads, cache, and results",
            key="reset_session",
            on_click=clear_session_data,
            width="stretch",
        )
        st.divider()
        st.markdown("### Access and privacy")
        st.info("For one-person use, set **Who can view this app** to **Only specific people** in Streamlit Cloud settings. A public app link is open to anyone, and a public GitHub repository keeps the source visible.")

    st.title("🧪 ML Studio")
    st.caption("A guided, in-memory lab for exploring data, learning ML ideas, and comparing models.")
    st.markdown("**Learning path:** 1. Inspect data → 2. choose the task and inputs → 3. train → 4. explain the results.")
    workspace = st.radio("Choose a workspace", ["Single dataset", "Compare datasets"], horizontal=True, key="workspace")
    if workspace == "Single dataset":
        render_single_dataset()
    else:
        render_comparison()


if __name__ == "__main__":
    main()
