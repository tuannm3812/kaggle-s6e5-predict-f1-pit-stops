# EDA Insights

## 1. Dataset Health

The EDA notebook shows a clean, stable dataset:

| Check | Result |
| --- | ---: |
| Training rows | 439,140 |
| Training columns | 16 |
| Test rows | 188,165 |
| Test columns | 15 |
| Target positive rate | 19.90% |
| Missing values | 0 |
| Duplicated rows | 0 |
| Duplicated IDs | 0 |
| Train/test ID overlap | 0 |

Implication: modeling effort can focus on feature quality and validation rather
than heavy cleaning or imputation strategy.

## 2. Target Behavior

`PitNextLap` is imbalanced but not extremely rare. About one in five training
rows is positive. This supports probability modeling with stratified
cross-validation and makes average precision a useful secondary metric.

![Target balance](assets/target_balance.svg)

## 3. Categorical Signal

`Compound` is one of the strongest raw signals. The README-level EDA summary
records that `HARD` has a much higher pit-next-lap rate than `MEDIUM`, while
`WET` is rare and low-rate.

Implication: categorical handling must preserve compound identity. Tree models
with ordinal-encoded categoricals work well here because the strongest
interactions are nonlinear and strategy-dependent.

## 4. Numerical Signal

Important raw signals include:

- `TyreLife`;
- `Stint`;
- `RaceProgress`;
- `LapNumber`;
- lap-time and degradation features;
- race identity and circuit context.

Lap-time features contain extreme values, especially `LapTime_Delta`,
`LapTime (s)`, and `Cumulative_Degradation`. This favors tree-based models over
purely linear methods.

## 5. Train/Test Drift

Numeric train/test drift is very low. The largest PSI values are still tiny,
led by `TyreLife`, `RaceProgress`, and `LapNumber`.

Implication: the public test distribution appears close enough to train that
standard stratified CV is a useful first validation strategy. The remaining risk
is less about global drift and more about race-specific calibration.

## 6. Feature Engineering Implications

The best-performing engineered set is `safe_plus_ratios_no_driver`.

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

![Feature set validation](assets/feature_set_validation.svg)

The broad lifecycle reconstruction did not improve this LightGBM setup. It
added many redundant or noisy features. The simpler ratio-based set is stronger.

## 7. Calibration and Error Pattern

The tuned LightGBM model is generally well calibrated, but underpredicts some
higher-risk regions:

| Slice | Predicted Rate | Actual Rate |
| --- | ---: | ---: |
| Highest prediction decile | 0.849 | 0.860 |
| Mid-risk band around 0.05-0.20 | 0.115 | 0.122 |

Top-ranked predictions are strong:

| Top Prediction Slice | Precision | Recall |
| --- | ---: | ---: |
| Top 0.5% | 0.96889 | 0.02435 |
| Top 1% | 0.95222 | 0.04785 |
| Top 2% | 0.94083 | 0.09456 |
| Top 5% | 0.90756 | 0.22805 |
| Top 10% | 0.85983 | 0.43211 |
| Top 20% | 0.73942 | 0.74320 |

The largest remaining gaps are race-specific, including Chinese, Belgian,
Japanese, Hungarian, Monaco, Saudi Arabian, and Las Vegas Grand Prix slices.

## 8. Circuit-Style Deep Dive

The EDA notebook now includes stylized circuit avatars for race-progress
analysis. These are deterministic visual maps generated from race names, not
real circuit geometry. They are still useful because the competition data
contains `RaceProgress` but does not contain GPS track coordinates.

The circuit maps support three diagnostics:

- pit-risk windows by race, using empirical `PitNextLap` rate across
  race-progress bins;
- median tyre life around the selected race-progress path;
- compound-specific pit-rate curves for a selected race.

Implication: the model should not only learn global tyre-life behavior. It
should also be checked for race-specific timing windows, especially when
calibration gaps appear in race slices.
