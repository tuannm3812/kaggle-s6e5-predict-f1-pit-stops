# Competition Instructions and Project Approach

## 1. Objective

Kaggle Playground Series S6E5, **Predict F1 Pit Stops**, asks competitors to
predict the probability that a Formula 1 driver will pit on the next lap.

Each row describes a driver-lap state. The target is binary:

- `PitNextLap = 1`: the driver pits on the next lap.
- `PitNextLap = 0`: the driver does not pit on the next lap.

The submission file must contain one probability per test row, aligned to the
sample submission format.

## 2. Input Files

Expected Kaggle path:

```text
/kaggle/input/competitions/playground-series-s6e5
```

Files:

- `train.csv`: training rows with `PitNextLap`.
- `test.csv`: test rows without `PitNextLap`.
- `sample_submission.csv`: required submission schema.

## 3. Evaluation Mindset

The notebooks optimize probability quality, not hard class labels. The core
validation metrics are:

- ROC AUC for ranking quality.
- Average precision for positive-class retrieval.
- Log loss for probability calibration.

Because pit-stop events are relatively rare, average precision and calibration
diagnostics are important complements to ROC AUC.

## 4. Problem Framing

The strongest signal is expected to come from strategy context:

- tyre compound;
- tyre age and stint progress;
- lap number and race progress;
- prior pit-stop count;
- lap-time deterioration;
- race identity and circuit-specific behavior;
- track position and time gaps where available.

The competition removes `Normalized_TyreLife`, a highly predictive feature from
the original strategy dataset. This project therefore validates reconstructed
tyre-life and ratio features instead of assuming they will improve leaderboard
performance.

## 5. Project Solution

The current solution is a notebook-first gradient-boosting workflow:

1. Run EDA and train/test drift checks in
   [`1_eda_and_circuit_context.ipynb`](../notebooks/1_eda_and_circuit_context.ipynb).
2. Establish baselines in
   [`2_baseline_modeling.ipynb`](../notebooks/2_baseline_modeling.ipynb).
3. Tune LightGBM, validate feature sets, check XGBoost blending, and write the
   primary submission in
   [`3_model_optimization_and_ensemble.ipynb`](../notebooks/3_model_optimization_and_ensemble.ipynb).
4. Compare CatBoost and feature importance in
   [`4_challenger_models_and_feature_importance.ipynb`](../notebooks/4_challenger_models_and_feature_importance.ipynb).

The current champion is tuned LightGBM with the
`safe_plus_ratios_no_driver` feature set:

| Model | Feature Set | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | --- | ---: | ---: | ---: |
| Tuned LightGBM | `safe_plus_ratios_no_driver` | 0.94721 | 0.80489 | 0.22949 |

## 6. Current Decision

Use the tuned LightGBM notebook output as the primary submission candidate.
XGBoost and CatBoost remain useful challengers, but their standalone and blend
results do not currently justify replacing the champion.

Recommended next experiments:

1. Build a small tuned LightGBM seed ensemble.
2. Test a narrow lifecycle subset only: `TyreLife_Normalized`,
   `Laps_Until_Due`, and `Overdue`.
3. Explore CV-safe race-level calibration or race interaction features.
4. Optionally test original-dataset augmentation with an `is_original` flag.
