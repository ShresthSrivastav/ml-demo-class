# ML Studio

A Streamlit app for exploring CSV datasets, training common supervised models, and running unsupervised learning workflows.

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features

- EDA for dataset shape, columns, missing values, duplicates, summary statistics, distributions, correlations, and supervised targets.
- Built-in classification, regression, clustering, PCA, and anomaly examples for students who do not have a dataset yet.
- Task/model explanations, student discussion questions, and repeatable classroom exercises.
- Supervised classification and regression with an adjustable train/test split.
- A simple baseline, train-versus-test scores, per-class reports, residual plots, and optional cross-validation on training data.
- Interactive controls for common hyperparameters such as tree depth, tree count, neighbors, and regularization.
- Unsupervised clustering, PCA dimensionality reduction, and anomaly detection.
- Optional known labels for external clustering evaluation; labels are excluded from model inputs.
- Automatic or manual feature selection, likely-ID suggestions, preprocessing for numeric/categorical data, result tables, and CSV downloads.
- Single-dataset runs and multi-dataset model comparisons.
- Session-local caching and saved results reduce repeated parsing, EDA, and training during Streamlit reruns.
- Uploads are capped at 25 MB each; compare up to four datasets at once. Dense models and Agglomerative Clustering have protective size limits.

The app reads uploads in memory and does not save uploaded data. Do not use the public deployment for data that must remain private.

## Deploy on Streamlit Community Cloud

1. The source is published at [ShresthSrivastav/ml-demo-class](https://github.com/ShresthSrivastav/ml-demo-class).
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub if prompted.
3. Create an app from the repository, using `app.py` as the app file.
4. Deploy. The public app URL can be shared with anyone.
