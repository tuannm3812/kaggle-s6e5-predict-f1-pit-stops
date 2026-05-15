# Kaggle Predict F1 Pit Stops

This repository contains an exploratory workflow for the Kaggle competition **Predict F1 Pit Stops**, where the goal is to predict whether a Formula 1 car will pit on the next lap.

The current focus is the first notebook:

- [`notebooks/01_preprocessing_eda.ipynb`](notebooks/01_preprocessing_eda.ipynb)

It prepares the data, runs initial exploratory data analysis, checks train/test drift, and defines reusable preprocessing helpers for later modeling.

## Competition Overview

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

## Dataset Snapshot

The training data has roughly 439k rows and 16 columns. Key feature groups include:

- Race context: `Race`, `Year`, `LapNumber`, `RaceProgress`
- Driver and stint context: `Driver`, `Stint`, `Position`, `Position_Change`
- Tyre context: `Compound`, `TyreLife`, `PitStop`
- Lap performance: `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`
- Target: `PitNextLap`

The target is imbalanced. From the dataset profile, about 20% of rows have `PitNextLap = 1`, so validation should preserve class balance with stratified splits.

## Notebook Summary

The preprocessing and EDA notebook covers:

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
11. Optional export of prepared artifacts to parquet or CSV

## Initial EDA Insights

These are the main early findings suggested by the dataset profile and encoded into the notebook workflow:

- **Target imbalance matters.** `PitNextLap` has a positive rate near 20%, so accuracy alone will be misleading. AUC, log loss, and calibration should be considered for model selection.
- **Tyre age is likely central.** `TyreLife`, `Stint`, `Compound`, and `RaceProgress` should strongly influence pit probability. Since `Normalized_TyreLife` was removed, ratio-style tyre features should be validated carefully.
- **Race progress adds strategic context.** The probability of pitting is unlikely to be linear across a race. Early laps, pit windows, and late-race behavior may differ substantially.
- **Lap-time features contain extreme values.** `LapTime (s)`, `LapTime_Delta`, and `Cumulative_Degradation` show long tails and unusual event-like outliers. Tree models may handle these naturally, while scaled models may benefit from clipping or robust transforms.
- **Categorical features need careful handling.** `Driver` is high-cardinality, while `Compound` and `Race` have fewer values. One-hot encoding is a reasonable baseline, but CatBoost or cross-validated target encoding may be stronger later.
- **Train/test drift should be checked before modeling.** The notebook compares feature distributions to surface fields where synthetic train/test generation may differ.
- **Sequential `id` should usually be excluded.** The notebook keeps `id` out of the modeling preprocessor because it is unlikely to represent causal race information.

## Starter Feature Engineering

The notebook creates a conservative row-level feature set available in both train and test:

- Estimated race length from `LapNumber / RaceProgress`
- Estimated laps remaining
- Tyre-life ratios
- Absolute lap-time delta
- Approximate previous position
- Absolute position change
- Compound indicator flags

These features are intended for experimentation. The tyre-life ratio features may be powerful, but they should be tested with robust cross-validation to avoid overfitting the synthetic data generation process.

## Recommended Next Steps

1. Execute `01_preprocessing_eda.ipynb` on Kaggle and review the generated plots.
2. Create a baseline modeling notebook with stratified cross-validation.
3. Compare Logistic Regression, LightGBM/XGBoost, and CatBoost baselines.
4. Evaluate feature sets with and without engineered tyre-ratio features.
5. Inspect calibration and choose a submission threshold only if the metric requires labels rather than probabilities.
6. Optionally test whether the original F1 strategy dataset improves validation performance.

## Repository Structure

```text
.
├── README.md
└── notebooks
    └── 01_preprocessing_eda.ipynb
```

## Usage

On Kaggle, attach the competition dataset and run:

```text
notebooks/01_preprocessing_eda.ipynb
```

Locally, place the competition files in one of the supported locations:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

The notebook will automatically detect the available path.
