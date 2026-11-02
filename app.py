from __future__ import annotations

import io
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
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="ML Studio", page_icon="🧪", layout="wide")

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
def read_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV without writing it to disk."""
    try:
        frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding="latin-1")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("This CSV has no rows or columns.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names. Rename them and upload again.")
    return frame


def available_models(task: str) -> list[str]:
    if task == "Classification":
        return CLASSIFICATION_MODELS
    if task == "Regression":
        return REGRESSION_MODELS
    return UNSUPERVISED_MODELS[task]


def feature_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    return [column for column in frame.columns if column != target]


def choose_features(frame: pd.DataFrame, target: str | None, key: str) -> list[str]:
    candidates = feature_columns(frame, target)
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


def build_supervised_model(task: str, name: str):
    if task == "Classification":
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=1000), False),
            "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
            "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=42), False),
            "Extra Trees": (ExtraTreesClassifier(n_estimators=150, random_state=42), False),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=42), True),
            "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), False),
            "Support Vector Machine": (SVC(), False),
            "Gaussian Naive Bayes": (GaussianNB(), True),
        }
    else:
        models = {
            "Linear Regression": (LinearRegression(), False),
            "Ridge": (Ridge(), False),
            "Lasso": (Lasso(max_iter=5000), False),
            "Decision Tree": (DecisionTreeRegressor(random_state=42), False),
            "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=42), False),
            "Extra Trees": (ExtraTreesRegressor(n_estimators=150, random_state=42), False),
            "Gradient Boosting": (GradientBoostingRegressor(random_state=42), True),
            "K-Nearest Neighbors": (KNeighborsRegressor(n_neighbors=5), False),
            "Support Vector Regression": (SVR(), False),
        }
    return models[name]


def run_supervised(
    frame: pd.DataFrame,
    task: str,
    target: str,
    features: list[str],
    model_names: list[str],
    test_fraction: float,
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
    for name in model_names:
        estimator, dense = build_supervised_model(task, name)
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(frame, features, dense=dense)),
                ("model", estimator),
            ]
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
                prediction = pipeline.predict(X_test)
            if task == "Classification":
                metrics = {
                    "Accuracy": accuracy_score(y_test, prediction),
                    "Precision (weighted)": precision_score(y_test, prediction, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, prediction, average="weighted", zero_division=0),
                    "F1 (weighted)": f1_score(y_test, prediction, average="weighted", zero_division=0),
                }
                labels = list(pd.unique(pd.concat([y_test.reset_index(drop=True), pd.Series(prediction)])))
                matrix = pd.DataFrame(
                    confusion_matrix(y_test, prediction, labels=labels),
                    index=pd.Index(labels, name="Actual"),
                    columns=pd.Index(labels, name="Predicted"),
                )
                details = {"confusion_matrix": matrix}
            else:
                metrics = {
                    "R²": r2_score(y_test, prediction) if len(y_test) > 1 else np.nan,
                    "MAE": mean_absolute_error(y_test, prediction),
                    "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                }
                details = {}
            output = pd.DataFrame({"Actual": y_test, "Predicted": prediction}, index=y_test.index)
            details["output"] = output.reset_index(names="Original row")
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
                transformed = make_preprocessor(frame, features, dense=dense).fit_transform(X)
                if name == "Gaussian Mixture":
                    estimator.fit(transformed)
                    labels = estimator.predict(transformed)
                else:
                    labels = estimator.fit_predict(transformed)
                output = frame[features].copy()
                output["Cluster"] = labels
                unique_labels = set(labels)
                metrics = {"Rows": len(labels), "Clusters (excluding noise)": len(unique_labels - {-1})}
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
                details = {"output": output.reset_index(names="Original row")}
            elif task == "Dimensionality reduction":
                transformed = make_preprocessor(frame, features, dense=True).fit_transform(X)
                max_components = min(transformed.shape[0], transformed.shape[1])
                components = min(max(1, settings["components"]), max_components)
                estimator = PCA(n_components=components, random_state=42)
                projection = estimator.fit_transform(transformed)
                component_names = [f"PC{i + 1}" for i in range(projection.shape[1])]
                output = pd.DataFrame(projection, columns=component_names)
                output.insert(0, "Original row", frame.index)
                metrics = {"Rows": len(output), "Components": components, "Explained variance": estimator.explained_variance_ratio_.sum()}
                details = {"output": output}
            else:
                contamination = settings["contamination"]
                estimator = (
                    IsolationForest(contamination=contamination, random_state=42)
                    if name == "Isolation Forest"
                    else LocalOutlierFactor(contamination=contamination)
                )
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
                details = {"output": output.reset_index(names="Original row")}
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
) -> list[dict]:
    if task in ("Classification", "Regression"):
        if not target:
            raise ValueError("Choose a target column for supervised learning.")
        return run_supervised(frame, task, target, features, models, test_fraction)
    return run_unsupervised(frame, task, features, models, settings)


def render_eda(frame: pd.DataFrame, target: str | None, task: str | None, key: str) -> None:
    st.markdown("#### Exploratory data analysis")
    missing_cells = int(frame.isna().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Columns", f"{len(frame.columns):,}")
    c3.metric("Missing cells", f"{missing_cells:,}")
    c4.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    st.caption("Uploads are read in memory and are not saved by this app.")

    with st.expander("Preview and column details", expanded=True):
        st.dataframe(frame.head(100), use_container_width=True)
        schema = pd.DataFrame(
            {
                "Column": frame.columns,
                "Type": [str(dtype) for dtype in frame.dtypes],
                "Missing": frame.isna().sum().to_numpy(),
                "Missing %": (frame.isna().mean().to_numpy() * 100).round(1),
                "Unique values": frame.nunique(dropna=True).to_numpy(),
            }
        )
        st.dataframe(schema, hide_index=True, use_container_width=True)
        st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)

    numeric_columns = list(frame.select_dtypes(include=np.number).columns)
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    left, right = st.columns(2)
    with left:
        st.markdown("**Numeric distribution**")
        if numeric_columns:
            column = st.selectbox("Numeric column", numeric_columns, key=f"{key}_numeric_chart")
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                counts, edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
                labels = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(counts))]
                st.bar_chart(pd.DataFrame({"Rows": counts}, index=labels))
        else:
            st.info("No numeric columns to plot.")
    with right:
        st.markdown("**Category counts**")
        if categorical_columns:
            column = st.selectbox("Categorical column", categorical_columns, key=f"{key}_category_chart")
            counts = frame[column].fillna("(missing)").astype(str).value_counts().head(20)
            st.bar_chart(counts.rename("Rows"))
        else:
            st.info("No categorical columns to plot.")

    if numeric_columns:
        st.markdown("**Numeric correlations**")
        corr = frame[numeric_columns[:30]].corr(numeric_only=True).round(2)
        st.dataframe(corr, use_container_width=True)
    if target and target in frame.columns:
        st.markdown(f"**Target: {target}**")
        if task == "Classification":
            counts = frame[target].fillna("(missing)").astype(str).value_counts().head(30)
            st.bar_chart(counts.rename("Rows"))
        elif task == "Regression":
            values = pd.to_numeric(frame[target], errors="coerce").dropna()
            if len(values):
                counts, edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
                labels = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(counts))]
                st.bar_chart(pd.DataFrame({"Rows": counts}, index=labels))


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
                st.dataframe(pd.DataFrame([result["metrics"]]), hide_index=True, use_container_width=True)
                if "confusion_matrix" in result["details"]:
                    st.markdown("**Confusion matrix**")
                    st.dataframe(result["details"]["confusion_matrix"], use_container_width=True)
                output = result["details"].get("output")
                if output is not None:
                    st.dataframe(output.head(100), use_container_width=True)
                    st.download_button(
                        "Download row-level results",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dataset_name.replace(' ', '_')}_{result['model'].lower().replace(' ', '_')}_results.csv",
                        mime="text/csv",
                        key=f"download_{key_prefix}_{len(flattened)}",
                    )
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


def render_single_dataset() -> None:
    upload = st.file_uploader("Upload a CSV file", type=["csv"], key="single_upload")
    if upload is None:
        st.info("Upload a tabular CSV to begin.")
        return
    try:
        frame = read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return

    task = st.selectbox("Machine learning task", TASKS, key="single_task")
    target = None
    if task in ("Classification", "Regression"):
        target = st.selectbox("Target column", list(frame.columns), key="single_target")
    features = choose_features(frame, target, "single")
    render_eda(frame, target, task, "single_eda")
    models = st.multiselect(
        "Models to run",
        options=available_models(task),
        default=available_models(task)[:1],
        key=f"single_models_{task}",
    )
    test_fraction = 0.2
    if task in ("Classification", "Regression"):
        test_fraction = st.slider("Test set size", 10, 50, 20, 5, key="single_test_split") / 100
    settings = task_controls(task, models, len(frame), "single_settings") if models else {}
    if st.button("Run selected model(s)", type="primary", key="single_run"):
        if not models:
            st.error("Select at least one model.")
        else:
            try:
                bundles = run_models(frame, task, target, features, models, test_fraction, settings)
                render_results(upload.name.rsplit(".", 1)[0], bundles, "single")
            except Exception as exc:
                st.error(str(exc))


def render_dataset_configuration(upload, index: int, mode: str, shared_task: str | None = None, shared_model: str | None = None):
    key = f"compare_{index}"
    try:
        frame = read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None

    with st.expander(f"{upload.name} — {len(frame):,} rows × {len(frame.columns):,} columns", expanded=index == 0):
        task = shared_task or st.selectbox("Task", TASKS, key=f"{key}_task")
        target = None
        if task in ("Classification", "Regression"):
            target = st.selectbox("Target column", list(frame.columns), key=f"{key}_target")
        features = choose_features(frame, target, key)
        render_eda(frame, target, task, f"{key}_eda")
        if mode == "Within each dataset":
            models = st.multiselect(
                "Models to compare",
                options=available_models(task),
                default=available_models(task)[:1],
                key=f"{key}_models_{task}",
            )
        else:
            models = [shared_model] if shared_model else []
        test_fraction = 0.2
        if task in ("Classification", "Regression"):
            test_fraction = st.slider("Test set size", 10, 50, 20, 5, key=f"{key}_split") / 100
        settings = task_controls(task, models, len(frame), f"{key}_settings") if models else {}
    return {
        "name": upload.name.rsplit(".", 1)[0],
        "frame": frame,
        "task": task,
        "target": target,
        "features": features,
        "models": models,
        "test_fraction": test_fraction,
        "settings": settings,
    }


def render_comparison() -> None:
    uploads = st.file_uploader("Upload two or more CSV files", type=["csv"], accept_multiple_files=True, key="multi_upload")
    if not uploads:
        st.info("Upload at least two CSV files to compare models across datasets.")
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
    if st.button("Run comparison", type="primary", key="compare_run"):
        if len(configs) < 2:
            st.error("At least two readable CSV files are required.")
            return
        all_rows = []
        for index, config in enumerate(configs):
            st.markdown("---")
            if not config["models"]:
                st.error(f"{config['name']}: select at least one model.")
                continue
            try:
                bundles = run_models(
                    config["frame"],
                    config["task"],
                    config["target"],
                    config["features"],
                    config["models"],
                    config["test_fraction"],
                    config["settings"],
                )
                all_rows.extend(render_results(config["name"], bundles, f"compare_{index}"))
            except Exception as exc:
                st.error(f"{config['name']}: {exc}")
        if all_rows:
            st.markdown("### Per-dataset comparison summary")
            summary = pd.DataFrame(all_rows)
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.download_button(
                "Download comparison metrics",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
                key="download_comparison_summary",
            )


st.title("🧪 ML Studio")
st.caption("Explore CSV data, train common machine learning models, and compare results.")
single_tab, comparison_tab = st.tabs(["Single dataset", "Compare datasets"])
with single_tab:
    render_single_dataset()
with comparison_tab:
    render_comparison()
