# iv-woe-filter

Scikit-learn compatible IV/WOE binning, feature selection, and PSI auditing for credit risk modeling.

[![PyPI version](https://img.shields.io/pypi/v/iv-woe-filter?style=flat-square&cacheSeconds=60)](https://pypi.org/project/iv-woe-filter/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/dilshodazamjonov/iv-woe-filter/ci.yaml?style=flat-square&label=CI)](https://github.com/dilshodazamjonov/iv-woe-filter/actions)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange?style=flat-square)](https://scikit-learn.org/)

---

## Motivation

In credit risk modeling, WOE encoding and IV-based feature selection are standard practice. In most teams, they are also a source of recurring problems: hand-rolled binning scripts that differ between analysts, WOE maps that are never persisted, transformations applied inconsistently between training and scoring, and no formal record of how features were selected or why.

The consequences are familiar - model validation findings, scorecard drift, leakage that surfaces only in production, cycles that require reconstructing preprocessing decisions from memory or scattered notebooks.

`iv-woe-filter` addresses this directly. It wraps the full IV/WOE preprocessing pipeline - binning, merging, special-code isolation, WOE encoding, IV-based feature selection, Gini-based rank-ordering validation, leakage risk flagging, monotonicity auditing, and population stability measurement - into a single, sklearn-compatible transformer. Every fit is reproducible, every decision is logged, and every artifact needed for internal governance review is written automatically.

---

## Quick Start

### Install

**pip (any platform)**

```bash
pip install iv-woe-filter
```

**pip inside a virtual environment (recommended)**

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install iv-woe-filter

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
pip install iv-woe-filter

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install iv-woe-filter
```

**uv**

```bash
uv add iv-woe-filter
```

**conda**

```bash
conda create -n credit-risk python=3.11
conda activate credit-risk
pip install iv-woe-filter   # iv-woe-filter is not on conda-forge; install via pip into the conda env
```

### Usage Example

```python
import pandas as pd
from sklearn.datasets import make_classification
from iv_woe_filter import IVWOEFilter

# 1. Generate synthetic data
X_arr, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    random_state=42
)
X = pd.DataFrame(X_arr, columns=[
    "bureau_score", "months_employed", "loan_to_value", "num_delinquencies",
    "credit_utilisation", "income_band", "months_since_last_default",
    "debt_to_income", "num_inquiries", "age_at_app",
])

# 2. Fit the filter (train phase)
filter_ = IVWOEFilter(
    binning_method="tree",
    n_bins=6,
    min_iv=0.02,
    min_gini=0.05,
    tree_max_depth=3,
    tree_min_samples_leaf=0.05,
    random_state=42,
    drop_unselected=True,
    output_dir="audit/",
)
X_woe = filter_.fit_transform(X, y)

print(filter_.iv_table_)          # IV and Gini scores for all features
print(filter_.selected_features_)  # Features that passed both IV and Gini thresholds

# 3. Simulate a population shift (out-of-time phase)
X_test = X.copy()
X_test["age_at_app"] += 2.0

# 4. Audit feature stability (PSI)
psi_report = filter_.calculate_psi(X_test, save=True)
print(psi_report)                 # PSI score and shift status

# 5. Save per-feature audit plots
filter_.save_feature_plot(output_dir="audit/plots/", feature="all")
```

For the default unsupervised path, omit `binning_method` and the tree-specific parameters. `IVWOEFilter()` uses quantile binning for numeric features unless you explicitly switch to a supervised alternative such as `binning_method="chi_merge"` or `binning_method="tree"`.

### Pipeline Usage

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from iv_woe_filter import IVWOEFilter

pipe = Pipeline([
    ("woe",   IVWOEFilter(min_iv=0.02, min_gini=0.05, n_jobs=-1)),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

```python
pipe = Pipeline([
    (
        "woe",
        IVWOEFilter(
            binning_method="tree",
            n_bins=6,
            min_iv=0.02,
            min_gini=0.05,
            tree_max_depth=3,
            tree_min_samples_leaf=0.05,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

---

## What It Does

`iv-woe-filter` is a single-class library. `IVWOEFilter` handles every step of the standard IV/WOE preprocessing workflow in one `fit()` call.

| Capability | Description |
|---|---|
| IV-based feature filtering | Features that fall below a configurable IV threshold are excluded at transform time. The threshold and all IV scores are recorded. |
| Gini-based rank-ordering validation | Gini is derived from ROC-AUC using `2 * max(AUC, 1 - AUC) - 1` and is computed on the fitted WOE representation for every feature. Features below the `min_gini` threshold are excluded, ensuring the retained encoded features demonstrate meaningful predictive separation. |
| WOE encoding | Raw feature values are replaced with Weight of Evidence scores derived from the training distribution. Encoding is applied consistently via the stored WOE map at transform time. |
| Configurable numeric binning | Numeric features use equal-frequency quantile bins by default, with supervised ChiMerge and tree-based bins available through `binning_method`. |
| Categorical support | String and low-cardinality integer columns are handled natively; no manual encoding is required before fitting. |
| Small bin merging | Bins below a configurable population share are merged with their nearest neighbour, eliminating zero-event and statistically unstable bins before WOE computation. |
| Special codes isolation | Sentinel values (e.g. `-99`, `9999`) are removed from the continuous distribution before binning and assigned dedicated WOE bins, preserving their signal without distorting numeric boundaries. |
| Target proxy signal flagging | Features with IV above a configurable upper threshold are flagged as potential target proxy signals and recorded in the audit output for modeler review. |
| Monotonicity auditing | The WOE trend direction is checked across ordered bins for every numeric feature. Non-monotonic features are flagged in the audit output. |
| Population Stability Index (PSI) | `calculate_psi()` compares training-time bin distributions against a new dataset to detect population drift, producing a per-feature stability report. |
| Parallel processing | All per-feature operations run in parallel via `joblib`, with backend selection controlled by `parallel_backend`. |
| Audit artifact generation | Structured CSV files are written at the end of every `fit()` and `calculate_psi()` call, covering IV scores, Gini coefficients, bin-level statistics, per-feature governance diagnostics, and population stability. |

---

## How It Works

```text
raw features -> special code isolation -> numeric binning -> small bin merging
             -> WOE computation -> IV + Gini calculation -> feature selection -> audit logs
```

### 1. Binning

When `fit()` is called, each feature is binned independently. Numeric features are assigned to equal-frequency quantile bins by default, ensuring each bin has a comparable share of the training population. If `binning_method="chi_merge"` is selected, adjacent ordered value groups are merged using a binary-target chi-square test until the requested bin budget is reached. If `binning_method="tree"` is selected, numeric boundaries are learned from the binary target using a decision tree constrained by `n_bins` and the optional tree parameters, then passed through the same WOE/IV workflow. Categorical features are binned by value. Before binning, any values listed in `special_codes` for that feature are extracted and assigned their own dedicated bin.

Once initial bins are created, any bin whose population share falls below `min_bin_pct` is merged with its nearest neighbour. This step prevents unstable WOE estimates and zero-event bins, both of which introduce noise into IV calculations and can cause calibration issues downstream.

### 2. WOE Computation

Within each bin, the ratio of non-events (target = 0) to events (target = 1) is computed relative to the total non-event and event counts in the training set. WOE is the natural log of this ratio. Computation is vectorized across bins for performance.

```text
WOE_i = ln(%Goods_i / %Bads_i)
```

In plain terms: a bin where good accounts are overrepresented relative to the overall population gets a positive WOE; a bin where bad accounts dominate gets a negative WOE. A WOE of zero means the bin's bad rate equals the overall bad rate.

The resulting WOE map - a dictionary from bin ID to WOE value - is stored on the fitted object and applied at `transform()` time.

### 3. IV and Gini Calculation

Information Value aggregates the separation power of a feature across all its bins. It is the sum over bins of the difference in non-event and event distributions, weighted by the WOE of each bin.

```text
IV = sum_i ((%Goods_i - %Bads_i) * WOE_i)
```

In plain terms: IV measures how well a feature's bins separate defaults from non-defaults. A higher IV means stronger separation. After fitting, all IV scores are stored in `iv_table_` sorted in descending order.

Gini measures how well the fitted feature representation ranks good versus bad observations. In this package, that representation is the stored WOE mapping applied to the learned bins for both numeric and categorical features. This keeps `min_gini` comparable across feature types and expresses separation in the same transformed space the model will actually consume. It is derived from the feature-level ROC-AUC and expressed on a 0-1 scale.

```text
Gini = 2 * max(AUC, 1 - AUC) - 1
```

Gini and IV are complementary. IV reflects the aggregate separation signal captured by the chosen bin structure, while Gini evaluates the discriminatory power of the fitted WOE representation. Both are recorded in `iv_table_` and `feature_audit.csv`.

### 4. Feature Selection

Features must satisfy both the IV and Gini thresholds to be retained. Features with IV below `min_iv` or Gini below `min_gini` are excluded from the output of `transform()` when `drop_unselected=True`. Features with IV above `max_iv_for_leakage` are retained but flagged in `leakage_flags_` and in the audit output. The selection logic is deterministic and fully inspectable via fitted attributes.

### 5. Stability Auditing (PSI)

After fitting, `calculate_psi(X_test, save=True)` compares the bin distribution observed during training against the distribution in a new dataset, typically an out-of-time holdout or a recent production sample. If the population has shifted since the model was trained, the proportion of observations falling into each bin will differ from what was seen at fit time. PSI quantifies this divergence per feature: a low PSI means the distribution is stable; a high PSI signals that the feature's population has drifted and model performance may be affected.

Formally, PSI is computed as the sum over bins of the log-weighted difference between the new dataset distribution (*Actual*) and the training distribution (*Expected*):

```text
PSI = sum_i ((Actual_i - Expected_i) * ln(Actual_i / Expected_i))
```

PSI is interpreted against configurable thresholds set via `psi_thresholds`:

- Below the lower threshold: `Stable`, no action required.
- Between the two thresholds: `Minor Shift`, worth monitoring.
- Above the upper threshold: `Significant Shift`, model performance should be re-evaluated.

The method returns a DataFrame with one row per feature and, if `save=True`, writes `stability_report.csv` to `output_dir`. PSI auditing does not require refitting and can be run at any point after `fit()`.

### 6. Audit Generation

If `output_dir` is set, CSV files are written automatically at the end of `fit()`. These files record every binning decision, WOE value, IV score, Gini coefficient, and governance flag produced during the fit - see the Audit and Governance Outputs section for details.

---

## API Reference

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_bins` | `int` | `10` | Maximum number of bins for numeric features |
| `binning_method` | `str` | `"quantile"` | Numeric binning strategy. Supported values are `"quantile"`, `"chi_merge"`, and `"tree"` |
| `min_iv` | `float` | `0.02` | Minimum IV to retain a feature at transform time |
| `min_gini` | `float` | `0.05` | Minimum Gini coefficient to retain a feature at transform time, computed on the fitted WOE representation |
| `max_iv_for_leakage` | `float` | `0.8` | Features with IV above this value are flagged as potential target proxy signals |
| `min_bin_pct` | `float or None` | `0.05` | Minimum bin population share (0-1); bins below this are merged |
| `special_codes` | `dict or None` | `{}` | Per-feature lists of sentinel values to isolate before binning |
| `encode` | `bool` | `True` | If `True`, `transform()` returns float WOE values. If `False`, `transform()` returns the original processed columns without WOE encoding. |
| `drop_unselected` | `bool` | `True` | If `True`, features that do not meet both the `min_iv` and `min_gini` thresholds are excluded from `transform()` output |
| `psi_thresholds` | `tuple[float, float]` | `(0.1, 0.2)` | Lower and upper PSI thresholds used to classify each feature as `Stable`, `Minor Shift`, or `Significant Shift` in the stability report |
| `random_state` | `int or None` | `42` | Random state passed to tree-based binning |
| `tree_criterion` | `str` | `"gini"` | Split criterion used when `binning_method="tree"` |
| `tree_max_depth` | `int or None` | `None` | Maximum depth used when `binning_method="tree"` |
| `tree_min_samples_leaf` | `int, float, or None` | `None` | Minimum samples per tree leaf. If `None`, `min_bin_pct` is used when available |
| `tree_min_samples_split` | `int or float` | `2` | Minimum samples required to split an internal tree node |
| `n_jobs` | `int` | `-1` | Number of parallel workers passed to `joblib` (`-1` = all cores) |
| `parallel_backend` | `str` | `"auto"` | Joblib backend preference. `"auto"` uses threads for supervised numeric binning and processes otherwise; `"threads"` and `"processes"` force a specific backend |
| `output_dir` | `str or None` | `None` | Directory for audit CSV artifacts; no files are written if `None` |
| `verbose` | `bool` | `True` | If `True`, progress is logged via the standard `logging` module |

### Fitted Attributes

These attributes are available after calling `fit()`.

| Attribute | Type | Description |
|---|---|---|
| `iv_table_` | `pd.DataFrame` | IV and Gini scores for all processed features, sorted descending by IV |
| `selected_features_` | `list[str]` | Features that met both the `min_iv` and `min_gini` thresholds |
| `woe_maps_` | `dict[str, dict]` | Bin-to-WOE mapping per feature, applied at transform time |
| `binning_` | `dict[str, NumericBinConfig \| CategoricalBinConfig]` | Fitted bin configuration per feature, including the learned `binning_method`, numeric bin edges or categorical mappings, and any persisted post-fit merge map |
| `monotonicity_report_` | `dict[str, MonotonicityResult]` | Monotonicity check results per feature, including `is_monotonic` and `direction` |
| `leakage_flags_` | `dict[str, bool]` | `True` for each feature whose IV exceeded `max_iv_for_leakage` |
| `feature_types_` | `dict[str, str]` | `"numeric"` or `"categorical"` per feature |
| `reference_distributions_` | `dict[str, pd.Series]` | Training-time bin distributions stored per feature; used as the expected baseline when `calculate_psi()` is called |

---

## Audit and Governance Outputs

When `output_dir` is specified, the following files are written to disk at the end of every `fit()` call. Their primary purpose is to support model validation reviews and internal governance documentation. `stability_report.csv` is written when `calculate_psi(save=True)` is called.

```
audit_outputs/
    iv_summary.csv
    bin_stats.csv
    feature_audit.csv
    stability_report.csv
    plots/
        bureau_score.png
        months_employed.png
        ...
```

### `iv_summary.csv`

One row per feature. Contains the IV score and Gini coefficient for every feature processed during fit, sorted by descending IV. This file is the starting point for a model validation team reviewing feature selection decisions - it records what was considered, what was kept, and the predictive signal behind each decision.

### `bin_stats.csv`

One row per bin per feature. Contains bin boundaries (or category labels), event count, non-event count, event rate, population share, and WOE value for every bin of every feature. This is the primary artifact for auditing binning decisions. Any reviewer questioning why a feature has a particular WOE value can trace it directly to this file.

### `feature_audit.csv`

One row per feature. Contains feature type, binning method, IV, Gini, monotonicity direction, and leakage risk flag. This file is designed for the governance layer - it answers the four questions most commonly asked during internal model risk reviews: how predictive is this feature, does it exhibit adequate rank-ordering separation, does its WOE trend make directional sense, and is there a risk that its predictive signal is too close to the target.

### `stability_report.csv`

One row per feature. Contains the PSI score computed against the dataset passed to `calculate_psi()`, along with a shift status derived from `psi_thresholds` (`Stable`, `Minor Shift`, or `Significant Shift`). This file is intended for periodic out-of-time monitoring and provides a documented record of population drift assessments across model deployment cycles.

### Feature Plots

`IVWOEFilter` also provides built-in plotting methods through `plot_feature_audit()` and `save_feature_plot()`. Each feature plot uses bin labels on the x-axis, bad-rate bars on the left y-axis, and a WOE line on the right y-axis. Numeric bin ranges are rounded for readability, and each bar is annotated with total count, good count, and bad count so the visual can be used directly in audit discussions.

Use `save_feature_plot(feature="all")` to save every fitted feature, or pass a specific feature name to save just one plot.

---

## Advanced Features

### Tree-Based Numeric Binning

For numeric features, `binning_method="tree"` learns split points from the binary target using `DecisionTreeClassifier`. This mode is numeric-only. The target is used only to discover thresholds; after that, the package uses the same bin IDs, WOE maps, IV table, monotonicity report, PSI calculation, and audit artifacts as the default quantile path.

```python
IVWOEFilter(
    binning_method="tree",
    n_bins=6,
    tree_max_depth=3,
    tree_min_samples_leaf=0.05,
    random_state=42,
)
```

`n_bins` is passed as the maximum number of tree leaves. If `tree_min_samples_leaf` is not set, `min_bin_pct` is used as the leaf-size guardrail, so very small leaves are discouraged before WOE is calculated.

In practice, `tree` is useful when you want the binning stage to pick target-aware cut points instead of equal-frequency intervals. The tradeoff is that tree-based thresholds are more sample-dependent than quantile bins, so they should be reviewed carefully when stability matters.

### ChiMerge Numeric Binning

For numeric features, `binning_method="chi_merge"` learns ordered bins by repeatedly merging the adjacent pair with the smallest chi-square separation in the binary target. This mode is numeric-only and supervised: it uses the target during fit to find risk-homogeneous intervals, then reuses the learned cut points in the same WOE/IV workflow as the quantile and tree paths.

```python
IVWOEFilter(
    binning_method="chi_merge",
    n_bins=6,
)
```

`n_bins` is treated as the maximum number of final numeric intervals. On moderate-cardinality features, ChiMerge starts from one bin per distinct value. On very high-cardinality features, the implementation first compresses the distribution into ordered quantile seed bins before merging, which keeps runtime and memory usage practical on larger datasets. The tradeoff is that its learned thresholds are still sample-dependent and should be reviewed with the same stability discipline as any other supervised binning method.

### Special Codes Handling

Credit bureau and application data frequently contain sentinel values that encode system states rather than measured quantities. Common examples include `-99` for "bureau record not found" and `9999` for "self-employed, income not verifiable". Feeding these values into numeric binning silently distorts bin boundaries, compresses the distribution of real values, and produces misleading WOE estimates.

`iv-woe-filter` extracts declared special codes from each feature before binning and assigns each a dedicated WOE bin. The predictive signal of the sentinel value is preserved; the underlying distribution is not affected.

```python
IVWOEFilter(
    special_codes={
        "bureau_score":    [-99, -1],     # bureau not available / query refused
        "months_employed": [9999],        # self-employed
        "loan_amount":     [-1],          # missing at application
    }
)
```

Each special code appears as a labelled row in `bin_stats.csv`.

### Monotonicity Checks

Many internal model risk frameworks and central bank guidelines require that WOE values exhibit a monotonic relationship with risk across ordered bins - either consistently increasing or consistently decreasing as the feature value increases. A non-monotonic WOE curve may indicate overfitting to the training distribution, an unstable binning, or a feature that requires further investigation before inclusion in a scorecard.

After WOE computation, `iv-woe-filter` checks the trend direction for every numeric feature and stores the result in `monotonicity_report_` and `feature_audit.csv`. Features that fail the check are not dropped - the decision is left to the modeler - but they are clearly identified.

### Parallel Processing

Per-feature operations - binning, WOE and IV computation, Gini calculation, monotonicity checking, and leakage flagging - are dispatched in parallel using `joblib.Parallel`. On datasets with a large number of features, this materially reduces the wall time of the fit step.

```python
IVWOEFilter(n_jobs=-1)   # all logical cores
IVWOEFilter(n_jobs=4)    # fixed worker count
IVWOEFilter(n_jobs=1)    # single-threaded (useful for debugging)
```

By default, `parallel_backend="auto"` uses thread-based workers for supervised numeric binning (`chi_merge` and `tree`) and process-based workers otherwise. This keeps the heavier supervised paths more stable on Windows while preserving the default process backend for the simpler quantile path. If you need to force one strategy, pass `parallel_backend="threads"` or `parallel_backend="processes"`.

### Sklearn Compatibility

`IVWOEFilter` extends `BaseEstimator` and `TransformerMixin` and is fully compatible with the sklearn ecosystem. It can be used as a stage in `Pipeline`, tuned via `GridSearchCV`, and evaluated within `cross_val_score`. `get_feature_names_out()` is implemented for pipeline introspection.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("woe",   IVWOEFilter(min_iv=0.02, min_gini=0.05, n_jobs=-1)),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

---

## Limitations

The following are known constraints of the current implementation. They are documented here to support informed usage and to avoid misapplication in production.

**IV is sensitive to binning strategy.** Information Value is not a fixed property of a feature - it depends on how that feature is binned. Different values of `binning_method`, `n_bins`, or `min_bin_pct` will produce different IV scores for the same feature on the same dataset. IV scores should not be compared across fits that used different binning configurations.

**Supervised thresholds are sample-dependent.** When `binning_method="tree"` or `binning_method="chi_merge"` is used, learned split points come from the observed relationship between a feature and the target in the fit sample. On smaller or noisier datasets, those thresholds can be less stable than quantile bins and should be reviewed with PSI, monotonicity, and holdout performance in mind.

**The target proxy signal flag is a heuristic, not a detection mechanism.** `max_iv_for_leakage` identifies features with unusually high IV relative to a configurable threshold. A high IV score is a signal worth investigating, not a confirmation of leakage. Legitimate features can have high IV; some forms of leakage produce moderate IV. The flag is an input to human judgment, not a substitute for it.

**Categorical high-cardinality features may produce unstable bins.** Features with many distinct categories can produce bins with very low population share even after merging. For high-cardinality categoricals, consider grouping low-frequency categories before fitting or applying a separate encoding strategy outside this transformer.

**Special codes must be declared manually.** The library does not attempt to infer sentinel values from the data distribution. Any value that should be isolated must be declared explicitly in `special_codes`. Undeclared sentinel values will be treated as standard observations and will influence bin boundaries and WOE estimates accordingly.

**Binary targets only.** The current implementation is designed for binary classification tasks (0/1 target). Multi-class targets are not supported.

---

## When Not To Use This

`iv-woe-filter` is purpose-built for binary credit risk scorecards using linear or logistic models. It is not the right tool in the following situations.

**Tree-based models.** Gradient boosted trees and random forests do not require WOE encoding. These models handle non-linear relationships and mixed feature types natively. Applying WOE transformation before a tree-based model collapses information that the model would otherwise learn from the raw distribution.

**Deep learning pipelines.** Neural networks learn their own representations from raw inputs. WOE encoding reduces features to a single scalar per bin, discarding distributional detail that deep models depend on. Standard normalization or embedding approaches are more appropriate.

**Unsupervised learning.** WOE and IV are defined relative to a binary target. They have no meaningful interpretation in clustering, dimensionality reduction, or anomaly detection contexts.

**High-cardinality free-text or embedding features.** Features that are inherently continuous at high resolution - raw embeddings, image features, unstructured text representations - are not suited to IV/WOE binning. Apply appropriate feature extraction before using this transformer.

---

## Development and Testing

### Set Up

```bash
git clone https://github.com/dilshodazamjonov/iv-woe-filter.git
cd iv-woe-filter
uv sync --all-extras
```

### Run Tests

```bash
# Full test suite
uv run pytest

# With coverage report
uv run pytest --cov=src/iv_woe_filter --cov-report=term-missing

# Verbose output for a single module
uv run pytest tests/test_tree_binning.py -v
```

### CI

Pull requests are validated automatically via GitHub Actions. Configuration is at `.github/workflows/ci.yaml`.

### Project Layout

```
iv-woe-filter/
    src/
        iv_woe_filter/
            __init__.py
            iv_woe_filter.py     # IVWOEFilter transformer
            binning.py           # Quantile, ChiMerge, tree, and categorical binning logic
            woe.py               # Vectorized WOE / IV computation and monotonicity checks
            metrics.py           # Gini and PSI calculations
    tests/
        conftest.py
        test_artifacts_and_psi.py
        test_binning_behavior.py
        test_fit_transform.py
        test_single_class_target.py
        test_tree_binning.py
        test_validation_and_selection.py
    pyproject.toml
    uv.lock
    .python-version
    LICENSE
    .github/
        workflows/
            ci.yaml
```

---

## License

Released under the [MIT License](LICENSE).
