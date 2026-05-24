# Coding Standards

## 1. Repository Scope

This repository is intentionally notebook-first. Kaggle notebooks are the
executable source of truth for the Playground Series S6E5 workflow, while
`docs/` captures competition instructions, EDA findings, model results, and
project decisions.

Keep the root small:

- `notebooks/` for Kaggle notebooks.
- `docs/` for standards, instructions, EDA notes, notebook writeups, and small
  supporting figures.
- `README.md` for the high-level project overview and current champion result.

Avoid adding local-only folders such as `data/`, `models/`, `outputs/`,
`configs/`, or `scripts/` unless the project direction changes back to local
training. Kaggle data should be read from the competition input mount, not
committed to the repo.

## 2. Notebook Naming

Use numbered, stable notebook names that match the project workflow:

1. `1_eda_and_circuit_context.ipynb`
2. `2_baseline_modeling.ipynb`
3. `3_model_optimization_and_ensemble.ipynb`
4. `4_challenger_models_and_feature_importance.ipynb`

Notebook names should describe the actual Kaggle workflow. Do not split
training and submission into separate notebooks when a notebook is intended to
run end-to-end on Kaggle.

## 3. Code Style

Follow PEP 8 for Python code:

- Use 4 spaces for indentation.
- Keep lines to 79 characters or fewer where practical.
- Prefer f-strings, list comprehensions, and small utility functions when they
  improve readability.
- Add type hints for reusable functions when the type is clear.
- Group imports in this order:
  1. Standard library
  2. Third-party libraries
  3. Local modules, if the project later adds them
- Separate import groups with a blank line.

Use Google-style docstrings for reusable functions that carry project logic:

```python
def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate probability metrics for pit-next-lap predictions.

    Args:
        y_true: Ground-truth binary target.
        y_pred: Predicted probability for `PitNextLap = 1`.

    Returns:
        A dictionary with ROC AUC, average precision, and log loss.
    """
```

Add short inline comments only when they explain why a decision was made. Avoid
comments that restate what the code already says.

## 4. Notebook Style

Each notebook should include:

- a short purpose statement at the top;
- a clear configuration section near the top for tunable values;
- explicit mode flags such as `RUN_FAST`, `FAST_SAMPLE_SIZE`, `N_SPLITS`, and
  `SELECTED_FEATURE_SET`;
- the fixed Kaggle competition path
  `/kaggle/input/competitions/playground-series-s6e5`;
- Markdown insight cells after important plots or metrics;
- artifact-writing cells for reusable outputs such as `submission.csv`,
  histories, or figures.

Prefer readable, self-contained notebook code over imports from local project
modules. Kaggle should be able to run each notebook with the competition dataset
available at the fixed input path.

When notebook code changes, clear outputs before committing and rerun the
notebook on Kaggle to regenerate trusted outputs. Keep committed notebooks
lightweight; Kaggle is the execution record.

Competition notebooks should not depend on internet access during final reruns.
For runtime packages that differ from the Kaggle image, attach a Kaggle input
dataset containing wheels or pretrained assets and load them explicitly. If
exploratory installation is ever allowed, gate it behind an explicit config flag
and keep the default offline-safe.

Submission paths must be optimized for Kaggle scoring limits. Do not run EDA in
submission mode. Keep final scoring focused on loading inputs, fitting or
loading the selected model, inference, and writing `submission.csv`.

## 5. Feature Engineering Standards

Feature engineering must use only fields available in both `train.csv` and
`test.csv` unless the target is explicitly being attached for validation. The
current champion feature set is `safe_plus_ratios_no_driver`.

Project-safe feature themes:

- tyre-life ratios such as life relative to stint or compound context;
- race-progress and estimated race-lap context;
- stint and pit-stop count context;
- lap-time deltas and degradation signals;
- categorical race and compound identifiers.

Avoid target leakage:

- Do not use future laps from the same driver-race row sequence.
- Do not use `PitNextLap` in feature construction.
- Treat race-level calibration or target encoding as cross-validation-only
  unless the transform is recomputed safely inside each fold.

## 6. Plot Style

Use the Viridis palette as the default visual language across notebooks:

- Use `sns.color_palette("viridis", ...)` for categorical or sequential
  accents.
- Use `"viridis"` as the default colormap for heatmaps.
- Change color palettes only when a chart needs clearer contrast, semantic
  coloring, or accessibility improvement.
- Keep chart titles short and analytical; avoid decorative styling.

## 7. Documentation Style

Documentation should be written for a competition reviewer or teammate who wants
the reasoning quickly:

- Use numbered sections.
- Lead with findings and implications.
- Include exact metrics when available.
- Link notebooks and docs with relative paths.
- Keep model results grouped under the notebook page that produced them.
- Keep broad narrative in the root `README.md`; keep detailed evidence in the
  numbered docs in `docs/`.

## 8. Git Hygiene

Do not commit:

- raw Kaggle competition data;
- local model checkpoints;
- Kaggle working directories;
- large cached feature tables;
- Python caches or notebook checkpoints;
- ad hoc experiment dumps.

Commit lightweight artifacts only when they directly support written analysis,
such as small figures used by EDA markdown and notebook result pages.
