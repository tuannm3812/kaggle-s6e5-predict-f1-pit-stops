# Kaggle Predict F1 Pit Stops

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E5-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-2E7D32?style=flat-square)
![Status](https://img.shields.io/badge/Status-Optimization%20Ready-2E7D32?style=flat-square)

![Formula 1 pit stop](https://media.formula1.com/image/upload/t_16by9Centre/c_lfill,w_3392/q_auto/v1740000001/fom-website/2025/Miscellaneous/GettyImages-666401980.webp)

This repository contains an end-to-end notebook workflow for Kaggle Playground Series S6E5, **Predict F1 Pit Stops**. The task is to predict the probability that a Formula 1 driver will pit on the next lap from race, stint, tyre, lap-time, and position context.

**Repository description:** EDA, feature validation, model tuning, calibration diagnostics, and submission workflow for next-lap Formula 1 pit-stop prediction.

## 1. Project Overview

The competition dataset is inspired by Formula 1 strategy data. Each row describes one driver-lap state, and the target is binary:

- `PitNextLap = 1`: the driver pits on the next lap
- `PitNextLap = 0`: the driver does not pit on the next lap

The competition intentionally removes `Normalized_TyreLife`, a highly predictive feature from the original strategy dataset. This workflow therefore focuses on validating robust tyre-life, race-progress, and stint features without assuming that reconstructed features will automatically improve leaderboard performance.

Expected Kaggle input path:

```text
/kaggle/input/competitions/playground-series-s6e5
```

Input files:

- `train.csv`: training data with `PitNextLap`
- `test.csv`: test rows for probability prediction
- `sample_submission.csv`: required submission format

## 2. Repository Structure

```text
.
|-- README.md
|-- .gitignore
|-- dist/
|   `-- kaggle-predict-f1-pit-stops-source.zip
`-- notebooks/
    |-- 01_eda_and_circuit_context.ipynb
    |-- 02_baseline_modeling.ipynb
    |-- 03_model_optimization_and_ensemble.ipynb
    `-- 04_challenger_models_and_feature_importance.ipynb
```

## 3. Notebook Workflow

| Notebook | Purpose |
| --- | --- |
| `01_eda_and_circuit_context.ipynb` | Data quality, target behavior, categorical and numerical signal, train/test drift, race-level context. |
| `02_baseline_modeling.ipynb` | Sanity baseline, linear model, sklearn tree baseline, LightGBM, and XGBoost comparison. |
| `03_model_optimization_and_ensemble.ipynb` | LightGBM tuning, feature-set validation, XGBoost blend check, calibration/error analysis, final submission. |
| `04_challenger_models_and_feature_importance.ipynb` | CatBoost challenger, prediction-diversity check, non-tree model decision, feature importance. |

## 4. Data and EDA Insights

The current EDA run shows a clean and stable dataset:

- Training data has `439,140` rows and `16` columns.
- Test data has `188,165` rows and `15` columns.
- `PitNextLap` positive rate is about `19.90%`.
- No missing values were found in the train/test files.
- No duplicated rows, duplicated IDs, or train/test ID overlap were found.
- Numeric train/test drift is very low. The largest PSI values are still tiny, led by `TyreLife`, `RaceProgress`, and `LapNumber`.

Important raw signals:

- `Compound` is highly informative. `HARD` has a much higher pit-next-lap rate than `MEDIUM`, while `WET` is rare and low-rate.
- `TyreLife`, `Stint`, `RaceProgress`, and race identity carry the strongest strategy context.
- Lap-time features contain extreme values, especially `LapTime_Delta`, `LapTime (s)`, and `Cumulative_Degradation`. Tree models are preferred because they handle this kind of nonlinearity naturally.
- `Driver` is high-cardinality. Validation shows that dropping it slightly improves the best feature set, so it is not part of the current champion model.

## 5. Modeling Results

Baseline modeling on a stratified 180k-row sample confirmed that gradient boosting is the right model family.

| Model | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| LightGBM | 0.94569 | 0.79933 | 0.23285 |
| XGBoost | 0.94503 | 0.79800 | 0.23413 |
| HistGradientBoosting | 0.94358 | 0.79206 | 0.23705 |
| Logistic Regression | 0.86534 | 0.59577 | 0.46610 |
| Dummy Prior | 0.49998 | 0.19898 | 0.49899 |

The tuned LightGBM workflow is the current champion:

| Experiment | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| Tuned LightGBM, `safe_plus_ratios_no_driver` | 0.94721 | 0.80489 | 0.22949 |
| Tuned LightGBM + XGBoost blend | 0.94721 | 0.80489 | 0.22949 |

The blend search selected `LightGBM = 1.0` and `XGBoost = 0.0`, so XGBoost does not currently improve the tuned model.

## 6. Feature Validation

The selected feature set is `safe_plus_ratios_no_driver`.

| Feature Set | Features | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `safe_plus_ratios_no_driver` | 27 | 0.94721 | 0.80489 | 0.22949 |
| `safe_plus_ratios` | 28 | 0.94718 | 0.80499 | 0.22959 |
| `safe_plus_ratios_lifecycle` | 56 | 0.94660 | 0.80225 | 0.23077 |
| `safe_plus_ratios_lifecycle_no_driver` | 55 | 0.94654 | 0.80233 | 0.23082 |
| `safe_plus_ratios_no_pitstop` | 27 | 0.94648 | 0.80265 | 0.23101 |
| `safe_engineered` | 26 | 0.94616 | 0.80277 | 0.23195 |
| `safe_plus_ratios_lifecycle_no_pitstop` | 55 | 0.94576 | 0.79961 | 0.23239 |
| `raw` | 15 | 0.94548 | 0.79997 | 0.23344 |

Key conclusion: the broad tyre-lifecycle reconstruction inspired by external baselines did **not** improve this LightGBM setup. It added many redundant/noisy features. The simpler ratio-based feature set remains stronger.

## 7. Calibration and Diagnostics

The tuned LightGBM model is generally well calibrated, with small but visible underprediction in some higher-risk regions:

- Highest prediction decile: predicted `0.849`, actual `0.860`.
- Mid-risk band around `0.05-0.20`: predicted `0.115`, actual `0.122`.

Top-ranked predictions are strong:

| Top Prediction Slice | Precision | Recall |
| --- | ---: | ---: |
| Top 0.5% | 0.96889 | 0.02435 |
| Top 1% | 0.95222 | 0.04785 |
| Top 2% | 0.94083 | 0.09456 |
| Top 5% | 0.90756 | 0.22805 |
| Top 10% | 0.85983 | 0.43211 |
| Top 20% | 0.73942 | 0.74320 |

Slice diagnostics show the remaining errors are more race-specific than model-family-specific. Larger calibration gaps appear in races such as Chinese, Belgian, Japanese, Hungarian, Monaco, Saudi Arabian, and Las Vegas Grand Prix.

## 8. Challenger Models

CatBoost is useful as a challenger and diversity check, but it is not the primary model.

| Model / Blend | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| LightGBM challenger run | 0.94482 | 0.79530 | 0.23428 |
| CatBoost challenger run | 0.94391 | 0.79393 | 0.23596 |
| LightGBM/CatBoost blend, smaller challenger run | 0.94567 | 0.79906 | 0.23246 |

CatBoost predictions are highly correlated with LightGBM predictions (`0.981`), so the ensemble benefit is limited. CNN and RealMLP were deprioritized because earlier results/runtime tradeoffs did not justify more time compared with tuned tree models.

## 9. Recommended Next Steps

1. Keep `safe_plus_ratios_no_driver` as the champion feature set.
2. Build a tuned LightGBM seed ensemble using the current champion features.
3. Test only a small lifecycle subset if needed: `TyreLife_Normalized`, `Laps_Until_Due`, and `Overdue`.
4. Explore CV-safe race-level calibration or race interaction features because the largest remaining gaps are race-specific.
5. Optionally test original dataset augmentation with an `is_original` flag, but validate it separately from competition-only CV.
6. Submit the tuned LightGBM output from notebook 03 as the primary candidate.
