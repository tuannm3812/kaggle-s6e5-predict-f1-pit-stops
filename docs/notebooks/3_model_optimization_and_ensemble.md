# Model Optimization and Ensemble Notebook

## 1. Objective

[`3_model_optimization_and_ensemble.ipynb`](../../notebooks/3_model_optimization_and_ensemble.ipynb)
turns the best baseline direction into the primary submission workflow. It tunes
LightGBM, validates feature sets, tests whether XGBoost adds blend value, checks
calibration, and writes the final submission candidate.

## 2. Step Flow

| Step | Purpose | Main Output |
| --- | --- | --- |
| Setup and Configuration | Define model flags, metrics, and selected feature set. | Reusable optimization settings. |
| Load Data | Read competition inputs and create the validation sample. | Train/test frames for tuning. |
| Feature Sets | Define raw, safe, ratio, and lifecycle feature variants. | Candidate feature-set registry. |
| LightGBM Tuning | Search a focused LightGBM parameter space. | Best LightGBM parameters by OOF ROC AUC. |
| Feature Validation | Compare feature families with the tuned model. | Feature-set leaderboard. |
| LightGBM + XGBoost Blend | Test whether XGBoost improves the tuned LightGBM OOF predictions. | Blend-weight table. |
| Calibration and Error Analysis | Inspect decile calibration, top-risk slices, and race slices. | Calibration and slice diagnostics. |
| Train Final Submission | Fit the selected model on full training data and predict test rows. | `submission.csv`. |

## 3. Champion Result

| Experiment | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| Tuned LightGBM, `safe_plus_ratios_no_driver` | 0.94721 | 0.80489 | 0.22949 |
| Tuned LightGBM + XGBoost blend | 0.94721 | 0.80489 | 0.22949 |

The blend selected `LightGBM = 1.0` and `XGBoost = 0.0`, so the single tuned
LightGBM model remains the champion.

## 4. Feature Decision

`safe_plus_ratios_no_driver` is the selected feature set. It keeps compact,
row-safe tyre-life and race-progress ratios while dropping the high-cardinality
`Driver` field. The broader lifecycle reconstruction did not improve validation
metrics and appears to add redundant or noisy predictors.

## 5. Error Pattern

The tuned model is generally well calibrated, with small underprediction in
some higher-risk regions. The largest remaining gaps are race-specific, which
makes race interaction features or CV-safe race calibration the most promising
next line of work.
