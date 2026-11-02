from __future__ import annotations

import io
import hashlib
import json
import math
import warnings

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


st.set_page_config(page_title="ML Studio", page_icon="ðŸ§ª", layout="wide")

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
    "Regression": "Predict a numeric value. MAE is average absolute error in target units; RMSE penalizes large misses more; RÂ² compares with a mean-value baseline. Inspect residuals for patterns the metrics hide.",
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
        st.markdown("**Workflow:** inspect the data â†’ define the question â†’ choose inputs â†’ fit the model â†’ evaluate on data the model did not train on.")
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


@st.cache_data(scope="session", ttl=1800, max_entries=8, show_spinner=False)
def _parse_csv(cache_key: str, _raw: bytes) -> pd.DataFrame:
    # The digest keys the cache; the underscore avoids hashing the upload bytes a second time.
    try:
        frame = pd.read_csv(io.BytesIO(_raw))
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(_raw), encoding="latin-1")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("This CSV has no rows or columns.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names. Rename them and upload again.")
    return frame


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
    correlations = sample[numeric[:30]].corr(numeric_only=True).round(2) if numeric else pd.DataFrame()
    return {
        "schema": schema,
        "summary": sample.describe(include="all").transpose(),
        "correlations": correlations,
        "likely_ids": likely_ids,
        "constants": constants,
        "sampled": len(sample) < len(_frame),
        "missing_cells": int(_frame.isna().sum().sum()),
        "duplicates": int(_frame.duplicated().sum()),
    }


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


@st.cache_data(scope="session", max_entries=64, show_spinner=False)
def _numeric_histogram(cache_key: str, _frame: pd.DataFrame, column: str) -> pd.DataFrame:
    values = pd.to_numeric(_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) > 100_000:
        values = values.sample(100_000, random_state=42)
    if values.empty:
        return pd.DataFrame()
    counts, edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
    labels = [f"{edges[i]:.3g}â€“{edges[i + 1]:.3g}" for i in range(len(counts))]
    return pd.DataFrame({"Rows": counts}, index=labels)


@st.cache_data(scope="session", max_entries=64, show_spinner=False)
def _category_counts(cache_key: str, _frame: pd.DataFrame, column: str) -> pd.Series:
    values = _frame[column]
    if len(values) > 100_000:
        values = values.sample(100_000, random_state=42)
    return values.fillna("(missing)").astype(str).value_counts().head(20).rename("Rows")


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
            f"Exclude likely ID columns ({', '.join(map(str, likely_ids[:5]))}{'â€¦' if len(likely_ids) > 5 else ''})",
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
    max_depth = settings.get("max_depth")
    n_estimators = settings.get("n_estimators", 150)
    neighbors = settings.get("neighbors", 5)
    regularization = settings.get("regularization", 1.0)
    alpha = settings.get("alpha", 1.0)
    if task == "Classification":
        models = {
            "Logistic Regression": (LogisticRegression(C=regularization, max_iter=1000), False),
            "Decision Tree": (DecisionTreeClassifier(max_depth=max_depth, random_state=42), False),
            "Random Forest": (RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42), False),
            "Extra Trees": (ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42), False),
            "Gradient Boosting": (GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth or 3, random_state=42), True),
            "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=neighbors), False),
            "Support Vector Machine": (SVC(C=regularization), False),
            "Gaussian Naive Bayes": (GaussianNB(), True),
        }
    else:
        models = {
            "Linear Regression": (LinearRegression(), False),
            "Ridge": (Ridge(alpha=alpha), False),
            "Lasso": (Lasso(alpha=alpha, max_iter=5000), False),
            "Decision Tree": (DecisionTreeRegressor(max_depth=max_depth, random_state=42), False),
            "Random Forest": (RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42), False),
            "Extra Trees": (ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42), False),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth or 3, random_state=42), True),
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
    data = frame.loc[frame[target].notna(), features + [target]].copy()
    if task == "Regression":
        data = data[np.isfinite(pd.to_numeric(data[target], errors="coerce"))]
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
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"The selected train/test split is not possible: {exc}") from exc

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
            "Test RÂ²": r2_score(y_test, baseline_prediction) if len(y_test) > 1 else np.nan,
            "Test MAE": mean_absolute_error(y_test, baseline_prediction),
            "Test RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
        }
    results.append({
        "model": "Baseline (most frequent class)" if task == "Classification" else "Baseline (predict the mean)",
        "metrics": baseline_metrics,
        "details": {"output": pd.DataFrame({"Actual": y_test, "Predicted": baseline_prediction}, index=y_test.index).reset_index(names="Original row")},
        "error": None,
    })

    for name in model_names:
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
                        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42) if actual_folds >= 2 else None
                        scoring = {
                            "accuracy": "accuracy",
                            "balanced_accuracy": "balanced_accuracy",
                            "macro_f1": "f1_macro",
                            "weighted_f1": "f1_weighted",
                        }
                    else:
                        actual_folds = min(cv_folds, len(y_train))
                        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42) if actual_folds >= 2 else None
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
                    "Train RÂ²": r2_score(y_train, train_prediction) if len(y_train) > 1 else np.nan,
                    "Test RÂ²": r2_score(y_test, prediction) if len(y_test) > 1 else np.nan,
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
                            "r2": "RÂ²",
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
            details["output"] = output.reset_index(names="Original row")
            if explain_features:
                scoring_name = "f1_weighted" if task == "Classification" else "neg_mean_absolute_error"
                importance = permutation_importance(
                    pipeline,
                    X_test,
                    y_test,
                    scoring=scoring_name,
                    n_repeats=3,
                    random_state=42,
                    n_jobs=1,
                )
                details["feature_importance"] = pd.DataFrame(
                    {"Feature": features, "Importance": importance.importances_mean}
                ).sort_values("Importance", ascending=False)
            results.append({"model": name, "metrics": metrics, "details": details, "error": None})
        except Exception as exc:  # report one model's incompatibility without losing other results
            results.append({"model": name, "metrics": {}, "details": {}, "error": str(exc)})
    return [{"note": stratification_note, "results": results}]


def cluster_model(name: str, settings: dict):
    if name == "DBSCAN":
        return DBSCAN(eps=settings["eps"], min_samples=settings["min_samples"]), False
    clusters = settings["clusters"]
    if name == "K-Means":
        return KMeans(n_clusters=clusters, n_init=10, random_state=42), False
    if name == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=clusters), True
    return GaussianMixture(n_components=clusters, random_state=42, n_init=3), True


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
    results = []
    for name in model_names:
        try:
            if task == "Clustering":
                estimator, dense = cluster_model(name, settings)
                if name == "Agglomerative Clustering" and len(X) > MAX_AGGLOMERATIVE_ROWS:
                    raise ValueError(f"Agglomerative Clustering is limited to {MAX_AGGLOMERATIVE_ROWS:,} rows because it grows quadratically with dataset size.")
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
                            random_state=42,
                        )
                        metrics["Silhouette score"] = score
                    except ValueError:
                        metrics["Silhouette score"] = np.nan
                scatter_columns = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])][:2]
                details = {"output": output.reset_index(names="Original row")}
                if len(scatter_columns) == 2:
                    details["scatter_columns"] = scatter_columns
            elif task == "Dimensionality reduction":
                check_dense_capacity(frame, features, len(X), "PCA")
                transformed = make_preprocessor(frame, features, dense=True).fit_transform(X)
                max_components = min(transformed.shape[0], transformed.shape[1])
                components = min(max(1, settings["components"]), max_components)
                estimator = PCA(n_components=components, random_state=42)
                projection = estimator.fit_transform(transformed)
                component_names = [f"PC{i + 1}" for i in range(projection.shape[1])]
                output = pd.DataFrame(projection, columns=component_names)
                output.insert(0, "Original row", frame.index)
                metrics = {"Rows": len(output), "Components": components, "Explained variance": estimator.explained_variance_ratio_.sum()}
                details = {
                    "output": output,
                    "explained_variance": pd.DataFrame(
                        {"Explained variance ratio": estimator.explained_variance_ratio_},
                        index=component_names,
                    ),
                }
            else:
                contamination = settings["contamination"]
                estimator = (
                    IsolationForest(contamination=contamination, random_state=42)
                    if name == "Isolation Forest"
                    else LocalOutlierFactor(contamination=contamination)
                )
                if name == "Local Outlier Factor":
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
    if task in ("Classification", "Regression"):
        if not target:
            raise ValueError("Choose a target column for supervised learning.")
        return run_supervised(frame, task, target, features, models, test_fraction, cv_folds, explain_features, settings)
    return run_unsupervised(frame, task, features, models, settings, evaluation_label)


def config_signature(frame: pd.DataFrame, config: dict) -> str:
    values = {"dataset": frame.attrs.get("mlstudio_cache_key", ""), **config}
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_eda(frame: pd.DataFrame, target: str | None, task: str | None, key: str) -> None:
    st.markdown("#### Exploratory data analysis")
    profile = _eda_profile(frame.attrs.get("mlstudio_cache_key", key), frame)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Columns", f"{len(frame.columns):,}")
    c3.metric("Missing cells", f"{profile['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{profile['duplicates']:,}")
    st.caption("Uploads are read in memory and are not saved by this app.")
    if profile["sampled"]:
        st.caption("Summary statistics and correlations use a reproducible sample of up to 50,000 rows; distribution charts use up to 100,000 rows for speed.")

    warnings_to_show = []
    if len(frame) > 25_000:
        warnings_to_show.append("Large dataset: SVM, KNN, DBSCAN, and Agglomerative Clustering can be slow. Start with one model or a smaller teaching sample.")
    missing_columns = profile["schema"].loc[profile["schema"]["Missing %"] >= 30, "Column"].tolist()
    if missing_columns:
        warnings_to_show.append(f"High missingness (30% or more): {', '.join(map(str, missing_columns[:8]))}.")
    if profile["constants"]:
        warnings_to_show.append(f"Constant columns carry no variation: {', '.join(map(str, profile['constants'][:8]))}.")
    if profile["likely_ids"]:
        warnings_to_show.append(f"Possible identifiers: {', '.join(map(str, profile['likely_ids'][:8]))}. Check whether to exclude them from features.")
    numeric = frame.select_dtypes(include=np.number)
    infinite_cells = int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    if infinite_cells:
        warnings_to_show.append(f"Found {infinite_cells:,} infinite numeric values; preprocessing treats them as missing.")
    if target:
        missing_target = int(frame[target].isna().sum())
        if missing_target:
            warnings_to_show.append(f"{missing_target:,} rows have no target and will be left out of supervised training.")
        if task == "Classification":
            counts = frame[target].dropna().value_counts()
            if len(counts) > 1 and counts.min() / counts.max() < 0.2:
                warnings_to_show.append("The target is imbalanced. Accuracy alone may be misleading; compare macro and weighted F1 and per-class recall.")
    if warnings_to_show:
        with st.expander("Data quality checks â€” review before training", expanded=True):
            for note in warnings_to_show:
                st.warning(note)
    if task in ("Classification", "Regression"):
        st.info("Teaching note: the app splits rows before learning imputers, encoders, and scalers. Keep the test set untouched until the final evaluation to avoid data leakage.")
        st.caption("Runs are reproducible: random splits and randomized models use seed 42.")

    with st.expander("Questions for students"):
        st.markdown("- Which columns have missing values, duplicates, or almost no variation?")
        st.markdown("- Could any identifier or target-related field give the model an unfair shortcut?")
        if task == "Classification":
            st.markdown("- Are all classes similarly common? Which class would accuracy hide?")
        elif task == "Regression":
            st.markdown("- Are the feature/target patterns roughly linear? Are there extreme values?")
        elif task == "Clustering":
            st.markdown("- What similarity does the chosen feature scaling imply? Do the groups make sense in context?")
        elif task == "Anomaly detection":
            st.markdown("- Are flagged rows true data problems, rare valid cases, or simply unusual measurements?")

    with st.expander("Preview and column details", expanded=True):
        st.dataframe(frame.head(100), width="stretch")
        st.dataframe(profile["schema"], hide_index=True, width="stretch")
        st.dataframe(profile["summary"], width="stretch")

    numeric_columns = list(frame.select_dtypes(include=np.number).columns)
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    left, right = st.columns(2)
    with left:
        st.markdown("**Numeric distribution**")
        if numeric_columns:
            column = st.selectbox("Numeric column", numeric_columns, key=f"{key}_numeric_chart")
            chart = _numeric_histogram(frame.attrs.get("mlstudio_cache_key", key), frame, column)
            if not chart.empty:
                st.bar_chart(chart)
        else:
            st.info("No numeric columns to plot.")
    with right:
        st.markdown("**Category counts**")
        if categorical_columns:
            column = st.selectbox("Categorical column", categorical_columns, key=f"{key}_category_chart")
            counts = _category_counts(frame.attrs.get("mlstudio_cache_key", key), frame, column)
            st.bar_chart(counts)
        else:
            st.info("No categorical columns to plot.")

    if numeric_columns:
        st.markdown("**Numeric correlations**")
        st.dataframe(profile["correlations"], width="stretch")
    if target and target in frame.columns:
        st.markdown(f"**Target: {target}**")
        if task == "Classification":
            st.bar_chart(_category_counts(frame.attrs.get("mlstudio_cache_key", key), frame, target))
        elif task == "Regression":
            st.bar_chart(_numeric_histogram(frame.attrs.get("mlstudio_cache_key", key), frame, target))


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
                    st.dataframe(result["details"]["confusion_matrix"], width="stretch")
                    st.markdown("**Per-class precision, recall, and F1**")
                    st.dataframe(result["details"]["classification_report"], width="stretch")
                if "Train accuracy" in result["metrics"] or "Train RÂ²" in result["metrics"]:
                    st.caption("Train scores describe fit to examples the model saw. Test scores estimate performance on held-out examples. A large gap can indicate overfitting.")
                output = result["details"].get("output")
                if "explained_variance" in result["details"]:
                    st.markdown("**Variance retained by each PCA component**")
                    st.bar_chart(result["details"]["explained_variance"])
                if output is not None:
                    st.dataframe(output.head(100), width="stretch")
                    scatter_columns = result["details"].get("scatter_columns")
                    color_column = "Cluster" if "Cluster" in output.columns else "Result" if "Result" in output.columns else None
                    if scatter_columns and color_column:
                        st.scatter_chart(output, x=scatter_columns[0], y=scatter_columns[1], color=color_column)
                    if "Residual" in output.columns:
                        st.markdown("**Residuals (actual âˆ’ predicted)**")
                        st.scatter_chart(output, x="Predicted", y="Residual")
                    if {"PC1", "PC2"}.issubset(output.columns):
                        st.scatter_chart(output, x="PC1", y="PC2")
                    st.download_button(
                        "Download row-level results",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dataset_name.replace(' ', '_')}_{result['model'].lower().replace(' ', '_')}_results.csv",
                        mime="text/csv",
                        key=f"download_{key_prefix}_{len(flattened)}",
                    )
                if "feature_importance" in result["details"]:
                    st.markdown("**Permutation feature importance**")
                    st.caption("This measures how much the held-out score changes when one input column is shuffled. It is model-specific and does not prove causation.")
                    st.dataframe(result["details"]["feature_importance"].head(15), hide_index=True, width="stretch")
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
        options=[0, 3, 5, 10],
        index=0,
        format_func=lambda value: "Holdout only (fast)" if value == 0 else f"{value}-fold CV on training data + final holdout",
        help="Cross-validation helps compare models using several training folds. The final test set remains untouched.",
        key=f"{key}_cv_folds",
    )
    explain_features = st.checkbox(
        "Calculate permutation feature importance (slower)",
        value=False,
        help="Shuffle each original input column on held-out rows to show how the score changes. This is model-specific, not causal.",
        key=f"{key}_permutation_importance",
    )
    return cv_folds, explain_features


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
    settings = model_controls(task, models, len(frame), "single_settings") if models else {}
    result_key = "single_saved_results"
    signature = config_signature(
        frame,
        {
            "dataset_name": dataset_name,
            "task": task,
            "target": target,
            "evaluation_label": evaluation_label,
            "features": features,
            "models": models,
            "test_fraction": test_fraction,
            "settings": settings,
            "cv_folds": cv_folds,
            "explain_features": explain_features,
        },
    )
    if st.button("Run selected model(s)", type="primary", key="single_run"):
        if not models:
            st.error("Select at least one model.")
        else:
            try:
                with st.spinner(f"Training {len(models)} model(s)â€¦"):
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
                st.session_state[result_key] = {"signature": signature, "name": dataset_name, "bundles": bundles}
            except Exception as exc:
                st.error(str(exc))
    saved = st.session_state.get(result_key)
    if saved and saved["signature"] == signature:
        render_results(saved["name"], saved["bundles"], "single")
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
    with st.expander(f"{upload.name} â€” {len(frame):,} rows Ã— {len(frame.columns):,} columns", expanded=index == 0):
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
        settings = model_controls(task, models, len(frame), f"{key}_settings") if models else {}
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
    if st.button("Run comparison", type="primary", key="compare_run"):
        if len(configs) < 2:
            st.error("At least two readable CSV files are required.")
            return
        saved_runs = []
        for index, config in enumerate(configs):
            if not config["models"]:
                saved_runs.append({"name": config["name"], "error": "Select at least one model.", "bundles": []})
                continue
            try:
                with st.spinner(f"Training models for {config['name']}â€¦"):
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
                saved_runs.append({"name": config["name"], "error": None, "bundles": bundles, "key": f"compare_{index}"})
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
            st.markdown("### Per-dataset comparison summary")
            summary = pd.DataFrame(all_rows)
            st.dataframe(summary, hide_index=True, width="stretch")
            st.download_button(
                "Download comparison metrics",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
                key="download_comparison_summary",
            )
        if st.button("Clear saved comparison", key="compare_clear_results"):
            del st.session_state[result_key]
            st.rerun()
    elif saved:
        st.info("Comparison settings changed since the last run. Run the comparison again to refresh these results.")


st.title("ðŸ§ª ML Studio")
st.caption("A guided, in-memory lab for exploring data, learning ML ideas, and comparing models.")
st.markdown("**Learning path:** 1. Inspect data â†’ 2. choose the task and inputs â†’ 3. train â†’ 4. explain the results.")
workspace = st.radio("Choose a workspace", ["Single dataset", "Compare datasets"], horizontal=True, key="workspace")
if workspace == "Single dataset":
    render_single_dataset()
else:
    render_comparison()
