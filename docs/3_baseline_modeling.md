# Baseline Modeling Notebook

## 1. Objective

[`2_baseline_modeling.ipynb`](../notebooks/2_baseline_modeling.ipynb)
builds the first validated model lineup. It establishes a sanity baseline,
checks whether linear modeling is sufficient, and confirms whether boosted tree
models are the right direction.

## 2. Step Flow

| Step | Purpose | Main Output |
| --- | --- | --- |
| Setup and Configuration | Define metrics, model flags, and shared constants. | Controlled fast/full validation behavior. |
| Load Data | Read competition inputs and optionally sample for faster iteration. | Modeling train/test frames. |
| Feature Engineering | Add safe row-level strategy features used by every baseline. | Consistent baseline feature matrix. |
| Preprocessing Helpers | Build reusable numeric and categorical preprocessing pipelines. | Linear and tree preprocessors. |
| Model Lineup | Register models from simple to strong. | Dummy, logistic, sklearn tree, LightGBM, and XGBoost candidates. |
| Cross-Validation | Score every candidate with stratified folds. | Fold and OOF metrics. |
| Compare Baselines | Rank models across ROC AUC, average precision, and log loss. | Baseline leaderboard and metric chart. |
| Train Best Model and Create Submission | Fit the best baseline candidate on available training data. | `submission.csv` candidate. |
| Next Experiments | Translate baseline results into tuning priorities. | Optimization plan for notebook 3. |

## 3. Published Results

| Model | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| LightGBM | 0.94569 | 0.79933 | 0.23285 |
| XGBoost | 0.94503 | 0.79800 | 0.23413 |
| HistGradientBoosting | 0.94358 | 0.79206 | 0.23705 |
| Logistic Regression | 0.86534 | 0.59577 | 0.46610 |
| Dummy Prior | 0.49998 | 0.19898 | 0.49899 |

## 4. Interpretation

The dummy model confirms the class-prior floor. Logistic regression learns
signal, but its gap to tree models shows that pit-stop timing depends on
nonlinear strategy interactions. LightGBM and XGBoost become the main
optimization candidates because they dominate both ranking and probability
metrics.

Runtime and stability notes from the rendered output:

| Model | OOF ROC AUC | Fold ROC AUC Range | Fit Seconds |
| --- | ---: | ---: | ---: |
| LightGBM | 0.94569 | 0.94392-0.94717 | 46.1 |
| XGBoost | 0.94503 | 0.94349-0.94676 | 35.2 |
| HistGradientBoosting | 0.94358 | 0.94190-0.94521 | 46.0 |
| Logistic Regression | 0.86534 | 0.86324-0.86899 | 742.7 |
| Dummy Prior | 0.49998 | 0.50000-0.50000 | 7.2 |

The fold ranges confirm that LightGBM's win is stable enough for a baseline
decision, but the gap to XGBoost is small enough to justify the blend check in
notebook 3. Logistic regression is both weaker and much slower in this setup,
so it should remain a sanity baseline only.
