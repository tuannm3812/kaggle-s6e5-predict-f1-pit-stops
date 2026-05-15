# Kaggle Predict F1 Pit Stops

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E5-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Notebook](https://img.shields.io/badge/Notebook-EDA%20%2B%20Modeling-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Baseline%20Ready-2E7D32?style=flat-square)

![Formula 1 pit stop](https://media.formula1.com/image/upload/t_16by9Centre/c_lfill,w_3392/q_auto/v1740000001/fom-website/2025/Miscellaneous/GettyImages-666401980.webp)

This repository contains an exploratory workflow for the Kaggle competition **Predict F1 Pit Stops**, where the goal is to predict whether a Formula 1 car will pit on the next lap.

**Repository description:** Exploratory data analysis and preprocessing workflow for Kaggle Playground Series S6E5, predicting next-lap Formula 1 pit stops from race, tyre, lap-time, and position features.

Current notebooks:

- [`notebooks/01_eda_and_circuit_context.ipynb`](notebooks/01_eda_and_circuit_context.ipynb)
- [`notebooks/02_baseline_modeling.ipynb`](notebooks/02_baseline_modeling.ipynb)
- [`notebooks/03_model_optimization_and_ensemble.ipynb`](notebooks/03_model_optimization_and_ensemble.ipynb)
- [`notebooks/04_challenger_models_and_feature_importance.ipynb`](notebooks/04_challenger_models_and_feature_importance.ipynb)

The workflow is grouped into four notebooks: EDA and circuit context, baseline modeling, model optimization with feature validation and ensembling, and challenger models with feature importance.

## 1. Competition Overview

The dataset is inspired by Formula 1 race strategy data. Each row represents race-lap context for a driver, with the binary target:

- `PitNextLap = 1`: the driver pits on the next lap
- `PitNextLap = 0`: the driver does not pit on the next lap

The competition data intentionally removes `Normalized_TyreLife`, which would make the task too direct. The original F1 strategy dataset may still be useful for external exploration or augmentation, but the first notebook focuses on understanding the competition-provided train and test files.

Expected Kaggle input path:

```text
/kaggle/input/competitions/playground-series-s6e5
```

Files:

- `train.csv`: training data with `PitNextLap`
- `test.csv`: test data requiring probability predictions
- `sample_submission.csv`: required submission format

## 2. Dataset Snapshot

The training data has roughly 439k rows and 16 columns. Key feature groups include:

- Race context: `Race`, `Year`, `LapNumber`, `RaceProgress`
- Driver and stint context: `Driver`, `Stint`, `Position`, `Position_Change`
- Tyre context: `Compound`, `TyreLife`, `PitStop`
- Lap performance: `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`
- Target: `PitNextLap`

The target is imbalanced. From the dataset profile, about 20% of rows have `PitNextLap = 1`, so validation should preserve class balance with stratified splits.

## 3. Notebook Workflow

The EDA and circuit-context notebook covers:

1. Environment setup and Kaggle/local data path detection
2. Data loading with memory downcasting
3. Schema checks, missing values, duplicate IDs, and cardinality
4. Target distribution and positive-rate review
5. Categorical feature summaries and target-rate plots
6. Numerical distributions and binned target-rate plots
7. Outlier checks for extreme lap-time and degradation values
8. Train/test distribution drift checks, including PSI
9. Starter feature engineering
10. A reusable scikit-learn preprocessing pipeline
11. Prepared train/test feature exports for Kaggle working sessions
12. Race-level pit-window context and optional FastF1 circuit maps

The baseline notebook compares sanity, linear, sklearn tree, LightGBM, and XGBoost models. The optimization notebook tunes LightGBM, validates feature groups, and tests a LightGBM/XGBoost blend. The challenger notebook contains the CNN baseline, CatBoost, optional RealMLP, and feature-importance analysis.

## 4. Initial EDA Insights

These are the main findings from the current notebook run and dataset profile:

- **The data is clean.** The current run found no missing values, no duplicated rows, no duplicated IDs, and no train/test ID overlap.
- **Target imbalance matters.** `PitNextLap` has a positive rate of about `19.90%`, so accuracy alone will be misleading. AUC, log loss, PR AUC, and calibration should be considered for model selection.
- **Compound has a strong raw signal.** In the current run, `HARD` has the highest pit-next-lap rate at about `32.75%`, while `MEDIUM` is much lower at about `10.11%`. This is probably a strategy/timing effect, so it should be analyzed together with `TyreLife`, `Stint`, and `RaceProgress`.
- **Tyre age is likely central.** `TyreLife`, `Stint`, `Compound`, and `RaceProgress` should strongly influence pit probability. Since `Normalized_TyreLife` was removed, ratio-style tyre features should be validated carefully.
- **Race progress adds strategic context.** The probability of pitting is unlikely to be linear across a race. Early laps, pit windows, and late-race behavior may differ substantially.
- **Lap-time features contain extreme values.** `LapTime (s)`, `LapTime_Delta`, and `Cumulative_Degradation` show long tails and unusual event-like outliers. Tree models may handle these naturally, while scaled models may benefit from clipping or robust transforms.
- **Categorical features need careful handling.** `Driver` is high-cardinality with `887` train values and `801` test values. Test has no unseen driver values, but train has `86` drivers absent from test.
- **Numeric train/test drift appears very low.** The largest PSI values in the current run are tiny, led by `TyreLife`, `RaceProgress`, and `LapNumber`. Broad distribution shift does not look like the first-order risk.
- **Sequential `id` should usually be excluded.** The notebook keeps `id` out of the modeling preprocessor because it is unlikely to represent causal race information.

## 5. Deep-Dive Analysis

The notebook now includes additional checks for:

- Compound-by-stint target-rate heatmap
- Tyre-life-by-race-progress target-rate heatmap
- Outlier target-rate impact for lap-time and degradation fields
- Train/test drift interpretation

These analyses are intended to answer whether the strongest raw signals still matter after accounting for race phase and stint context.

## 6. Baseline Modeling Insights

The baseline modeling notebook was run in `RUN_FAST = True` mode on a stratified 180k-row sample. The current model ranking is:

| Model | OOF ROC AUC | OOF Average Precision | OOF Log Loss | Fit Time |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 0.9457 | 0.7993 | 0.2328 | 33.4s |
| XGBoost | 0.9450 | 0.7980 | 0.2341 | 24.4s |
| HistGradientBoosting | 0.9436 | 0.7921 | 0.2371 | 32.9s |
| Logistic Regression | 0.8653 | 0.5958 | 0.4661 | 492.4s |
| Dummy Prior | 0.5000 | 0.1990 | 0.4990 | 5.5s |

Key takeaways:

- Gradient boosting is clearly the right model family for the next phase.
- LightGBM is the current leader across AUC, average precision, and log loss.
- XGBoost is very close and faster in this sampled run, so it remains a good challenger.
- Logistic regression is much weaker and unexpectedly slow with one-hot high-cardinality features.
- The gap between HistGradientBoosting and LightGBM is small enough that feature engineering may matter as much as model choice.

The next modeling step should be LightGBM-focused tuning on full data, with XGBoost kept as a benchmark.

## 7. Feature Validation and Challenger Insights

Feature validation on a 180k-row stratified sample showed:

| Feature Set | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| Safe + ratios, no Driver | 0.9473 | 0.8050 | 0.2293 |
| Safe + ratios | 0.9473 | 0.8048 | 0.2293 |
| Safe + ratios, no PitStop | 0.9466 | 0.8028 | 0.2307 |
| Safe engineered | 0.9462 | 0.8026 | 0.2316 |
| Raw | 0.9453 | 0.7994 | 0.2334 |

The engineered ratio features help. Dropping `Driver` very slightly improves validation, which suggests driver identity is not essential and may add mild noise. Dropping `PitStop` hurts, so it should be kept if it is valid test-time information.

The CNN baseline is not competitive yet. Its OOF ROC AUC is about `0.799`, with much worse log loss than the tree models. For now, neural models should be treated as experimental challengers rather than submission candidates.

## 8. Starter Feature Engineering

The notebook creates a conservative row-level feature set available in both train and test:

- Estimated race length from `LapNumber / RaceProgress`
- Estimated laps remaining
- Tyre-life ratios
- Absolute lap-time delta
- Approximate previous position
- Absolute position change
- Compound indicator flags

These features are intended for experimentation. The tyre-life ratio features may be powerful, but they should be tested with robust cross-validation to avoid overfitting the synthetic data generation process.

## 9. Recommended Next Steps

1. Use `03_model_optimization_and_ensemble.ipynb` with the selected `safe_plus_ratios_no_driver` feature set.
2. Review LightGBM/CatBoost feature importance agreement.
3. Add CatBoost to the ensemble only if its OOF score or prediction diversity justifies it.
4. Inspect slice performance by `Compound`, `Stint`, `RaceProgress`, and `TyreLife` bins.
5. Inspect calibration before final submission because the target is probability-based.
6. Optionally test whether the original F1 strategy dataset improves validation performance.

## 10. Repository Structure

```text
.
|-- README.md
|-- .gitignore
`-- notebooks
    |-- 01_eda_and_circuit_context.ipynb
    |-- 02_baseline_modeling.ipynb
    |-- 03_model_optimization_and_ensemble.ipynb
    `-- 04_challenger_models_and_feature_importance.ipynb
```
