# ML Studio

A teaching app for exploring tabular data, learning common machine-learning workflows, and comparing models. It is designed for one person at a time on Streamlit Community Cloud's free tier.

## Features

- Upload a CSV and explore its shape, column types, missing values, duplicates, summary statistics, distributions, correlations, and target balance.
- Get data-quality flags for likely identifiers, constant columns, high missingness, class imbalance, and infinite numeric values.
- Choose automatic or manual input features and configure supervised targets.
- Train common classification and regression models, compare them with a baseline, and inspect holdout metrics, confusion matrices, per-class scores, or residuals.
- Run clustering, PCA, and anomaly detection with relevant metrics, plots, and row-level outputs.
- Optionally run up to five-fold cross-validation or permutation importance on a bounded sample.
- Compare model results within each dataset, or use one compatible model across datasets. Scores are displayed per dataset and are never averaged across datasets.
- Use five built-in learning examples with teaching notes and student exercises.
- Reproduce runs with a selected random seed. Download row-level results, comparison metrics, or a Markdown report of settings and results.
- Clear uploaded files, session results, and cached data with one button.

## Free-tier guardrails

- Single uploads are limited to 25 MB. Dataset comparisons allow up to four CSV files with a combined upload limit of 40 MB.
- Choose at most four models per dataset in one run.
- Kernel SVM/SVR, KNN, DBSCAN, Local Outlier Factor, Agglomerative Clustering, and dense transformations have row or memory limits to keep runtimes predictable.
- Large row-level outputs keep a small preview in session state and download as compressed CSV when needed.
- Cross-validation and permutation importance are optional because they require extra computation.

Streamlit Community Cloud has resource limits that can change, and inactive apps may sleep. Keep datasets modest and run a few models at a time. See [Community Cloud resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Run locally

Use Python 3.14.10 to match the current Community Cloud deployment, then install the tested dependencies and start the app:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run checks

```bash
python -m unittest discover -s tests -v
```

## Privacy and access

Uploads and run results are processed in memory for the active session. The app does not write datasets or models to persistent storage. Use the sidebar's clear button when finished. Do not upload sensitive data to an app that is publicly accessible.

For one-person access, open the app's settings in Community Cloud, go to **Sharing**, and select **Only specific people**. Invite only your own email. Streamlit currently allows one private app per workspace. The GitHub repository is currently public, so its source stays public even if app access is private. To hide the source too, make the GitHub repository private and connect it to Community Cloud. See the [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Deploy or update the live app

- Repository: [ShresthSrivastav/ml-demo-class](https://github.com/ShresthSrivastav/ml-demo-class)
- Live app: [ml-demo-class-2s4jjxd8rwzktnpi9mylbc.streamlit.app](https://ml-demo-class-2s4jjxd8rwzktnpi9mylbc.streamlit.app/)
- Entrypoint: `app.py`
- Branch: `main`
- Python version: 3.14.10 (current deployment runtime)

Community Cloud deploys updates from GitHub after changes are pushed to the selected branch. See the [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
