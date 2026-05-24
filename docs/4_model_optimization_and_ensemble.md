# Model Optimization and Ensemble Notebook

## 1. Objective

[`3_model_optimization_and_ensemble.ipynb`](../notebooks/3_model_optimization_and_ensemble.ipynb)
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

The best sampled LightGBM parameter set is:

| Parameter | Value |
| --- | ---: |
| `learning_rate` | 0.015 |
| `num_leaves` | 127 |
| `min_child_samples` | 150 |
| `subsample` | 0.95 |
| `colsample_bytree` | 0.80 |
| `reg_alpha` | 0.03 |
| `reg_lambda` | 10.0 |
| `max_bin` | 255 |

The top tuning candidates are close, but the winner is a conservative
regularized configuration rather than a very aggressive tree setup.

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

Rendered calibration details:

| Prediction Slice | Predicted Rate | Actual Rate | Interpretation |
| --- | ---: | ---: | --- |
| `0.0536-0.202` band | 0.11497 | 0.12233 | Underpredicts mid-risk rows. |
| Highest decile | 0.849 | 0.860 | Slight underprediction at the top. |

Top-risk retrieval is strong:

| Top Prediction Slice | Precision | Recall |
| --- | ---: | ---: |
| Top 1% | 0.95222 | 0.04785 |
| Top 5% | 0.90756 | 0.22805 |
| Top 10% | 0.85983 | 0.43211 |
| Top 20% | 0.73942 | 0.74320 |

The final submission path should remain pure tuned LightGBM unless a future
experiment beats both the OOF score and the calibration profile.
