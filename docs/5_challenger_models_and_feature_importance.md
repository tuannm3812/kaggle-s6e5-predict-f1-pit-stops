# Challenger Models and Feature Importance Notebook

## 1. Objective

[`4_challenger_models_and_feature_importance.ipynb`](../notebooks/4_challenger_models_and_feature_importance.ipynb)
checks whether CatBoost provides useful model diversity and records feature
importance from the tree-model challengers.

## 2. Step Flow

| Step | Purpose | Main Output |
| --- | --- | --- |
| Setup and Configuration | Define challenger settings and selected feature set. | Shared challenger configuration. |
| Load Data | Read competition inputs and create the challenger validation sample. | Train/test frames for challenger runs. |
| Shared Feature Set | Recreate the selected safe ratio feature set. | LightGBM/CatBoost-ready features. |
| CatBoost and LightGBM Comparison | Score both models on the same folds. | Challenger OOF metrics. |
| Prediction Diversity Check | Measure correlation and blend behavior. | Diversity and blend tables. |
| Non-Tree Model Decision | Record why CNN and RealMLP are not prioritized. | Model-scope decision table. |
| Feature Importance | Compare top tree-model drivers. | Importance tables and charts. |
| Challenger Summary | Decide whether the challenger should affect the champion. | Model-family recommendation. |

## 3. Published Results

| Model / Blend | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| --- | ---: | ---: | ---: |
| LightGBM challenger run | 0.94482 | 0.79530 | 0.23428 |
| CatBoost challenger run | 0.94391 | 0.79393 | 0.23596 |
| LightGBM/CatBoost blend, smaller challenger run | 0.94567 | 0.79906 | 0.23246 |

CatBoost predictions are highly correlated with LightGBM predictions
(`0.981`), so the blend opportunity is limited.

Detailed diversity output:

| Metric | Value |
| --- | ---: |
| Prediction correlation | 0.98112 |
| Mean absolute prediction delta | 0.03074 |

The best LightGBM/CatBoost blend on the smaller challenger run is:

| LightGBM Weight | CatBoost Weight | OOF ROC AUC | OOF Average Precision | OOF Log Loss |
| ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.40 | 0.94567 | 0.79906 | 0.23246 |

## 4. Interpretation

CatBoost is a credible challenger, but current evidence still favors tuned
LightGBM. Feature importance should be used to sanity-check strategy signals and
guide small interaction experiments rather than to expand the feature set
indiscriminately.

Feature importance agrees across the challenger models. The strongest signals
include:

- `Stint`;
- `AbsLapTime_Delta`;
- `TyreLife_to_LapNumber`;
- `EstimatedRaceLaps`;
- `TyreLife`;
- `Year`;
- `Race`.

This supports the current feature direction: compact race, stint, tyre, and
lap-time context is more useful than broad feature expansion.
