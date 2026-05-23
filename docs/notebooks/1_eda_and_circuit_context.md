# EDA and Circuit Context Notebook

## 1. Objective

[`1_eda_and_circuit_context.ipynb`](../../notebooks/1_eda_and_circuit_context.ipynb)
establishes the data profile and modeling hypotheses for predicting
`PitNextLap`. The notebook answers three questions:

1. Is the competition data clean enough for direct modeling?
2. Which raw fields explain pit-next-lap behavior?
3. Does train/test drift look large enough to require special validation?

## 2. Step Flow

| Step | Purpose | Main Output |
| --- | --- | --- |
| Setup and Configuration | Define constants, plotting defaults, and shared runtime values. | Reproducible notebook configuration. |
| Load Data | Read `train.csv`, `test.csv`, and `sample_submission.csv`. | Memory-reduced train/test frames. |
| Data Quality | Check missingness, duplicated rows, duplicated IDs, and schema shape. | Dataset health tables. |
| Target and Categorical Signal | Measure target balance and categorical pit-rate differences. | Target-rate and category-rate summaries. |
| Numerical Signal and Outliers | Inspect numeric distributions and relationship with the target. | Numeric summary and outlier context. |
| Strategy Interactions | Study compound, stint, tyre-life, and race-progress interactions. | Heatmaps that motivate engineered features. |
| Train/Test Drift | Compare train and test distributions. | PSI and category coverage diagnostics. |
| Circuit Context and Strategy Maps | Summarize race-level pit behavior and visualize race-progress pit windows. | Race slices and stylized circuit dashboards. |
| EDA Summary | Convert findings into modeling directions. | Feature and validation recommendations. |

## 3. Key Findings

The dataset is clean: no missing values, duplicated rows, duplicated IDs, or
train/test ID overlap were found in the recorded EDA run. The target positive
rate is about `19.90%`, so probability quality and average precision matter.

Compound, tyre age, stint, race progress, and race identity carry the core
strategy signal. Lap-time and degradation fields include extreme values, which
supports tree-based models as the primary family.

Train/test numeric drift is low. The largest remaining modeling risk is
race-specific calibration rather than broad distribution shift.

The circuit-style maps add a race-progress deep dive. They use deterministic
stylized circuit avatars rather than real track geometry, then color each path
by pit-risk or median tyre-life windows. This makes it easier to spot races
where pit timing is concentrated in specific portions of race progress.

## 4. Downstream Decisions

The EDA motivates safe row-level feature engineering around tyre-life ratios,
estimated race distance, stint progress, and lap-time deterioration. It also
sets up the later decision to evaluate race-level calibration slices after
tuning the main model.
