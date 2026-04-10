# iv-woe-filter

Financial-grade IV filtering and WOE transformation for credit risk feature preprocessing.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/dilshodazamjonov/iv-woe-filter/ci.yaml?style=flat-square&label=CI)](https://github.com/dilshodazamjonov/iv-woe-filter/actions)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange?style=flat-square)](https://scikit-learn.org/)
[![Managed by uv](https://img.shields.io/badge/managed%20by-uv-purple?style=flat-square)](https://github.com/astral-sh/uv)

---

## Motivation

In credit risk modeling, WOE encoding and IV-based feature selection are standard practice. In most teams, they are also a source of recurring problems: hand-rolled binning scripts that differ between analysts, WOE maps that are never persisted, transformations applied inconsistently between training and scoring, and no formal record of how features were selected or why.

The consequences are familiar — model validation findings, scorecard drift, leakage that surfaces only in production, and review cycles that require reconstructing preprocessing decisions from memory or scattered notebooks.

`iv-woe-filter` addresses this directly. It wraps the full IV/WOE preprocessing pipeline — binning, merging, special-code isolation, WOE encoding, feature selection, leakage risk flagging, and monotonicity auditing — into a single, sklearn-compatible transformer. Every fit is reproducible, every decision is logged, and every artifact needed for internal governance review is written automatically.

---

## Quick Start

### Install

```bash
pip install iv-woe-filter
# or
uv add iv-woe-filter
```

### Usage Example

```python
import pandas as pd
from sklearn.datasets import make_classification
from iv_woe_filter import IVWOEFilter

X_arr, y = make_classification(n_samples=5000, n_features=10, n_informative=6, random_state=42)
X = pd.DataFrame(X_arr, columns=[
    "bureau_score", "months_employed", "loan_to_value", "num_delinquencies",
    "credit_utilisation", "income_band", "months_since_last_default",
    "debt_to_income", "num_inquiries", "age_at_app",
])

filter_ = IVWOEFilter(min_iv=0.02, min_bin_pct=0.05, n_jobs=-1, output_dir="audit/")
X_woe = filter_.fit_transform(X, y)

print(filter_.iv_table_)          # IV scores for all features
print(filter_.selected_features_) # features that passed the threshold
```

### Pipeline Usage

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("woe",   IVWOEFilter(min_iv=0.02, n_jobs=-1)),
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
| WOE encoding | Raw feature values are replaced with Weight of Evidence scores derived from the training distribution. Encoding is applied consistently via the stored WOE map at transform time. |
| Quantile binning | Numeric features are binned into equal-frequency intervals, ensuring adequate event exposure per bin. |
| Categorical support | String and low-cardinality integer columns are handled natively; no manual encoding is required before fitting. |
| Small bin merging | Bins below a configurable population share are merged with their nearest neighbour, eliminating zero-event and statistically unstable bins before WOE computation. |
| Special codes isolation | Sentinel values (e.g. `-99`, `9999`) are removed from the continuous distribution before binning and assigned dedicated WOE bins, preserving their signal without distorting quantile boundaries. |
| Target proxy signal flagging | Features with IV above a configurable upper threshold are flagged as potential target proxy signals and recorded in the audit output for modeler review. |
| Monotonicity auditing | The WOE trend direction is checked across ordered bins for every numeric feature. Non-monotonic features are flagged in the audit output. |
| Parallel processing | All per-feature operations run in parallel via `joblib`. |
| Audit artifact generation | Three structured CSV files are written at the end of every `fit()`, covering IV scores, bin-level statistics, and per-feature governance diagnostics. |

---

## How It Works

```
raw features  →  special code isolation  →  quantile binning  →  small bin merging
      →  WOE computation  →  IV calculation  →  feature selection  →  audit logs
```

### 1. Binning

When `fit()` is called, each feature is binned independently. Numeric features are assigned to equal-frequency quantile bins, ensuring each bin has a comparable share of the training population. Categorical features are binned by value. Before binning, any values listed in `special_codes` for that feature are extracted and assigned their own dedicated bin.

Once initial bins are created, any bin whose population share falls below `min_bin_pct` is merged with its nearest neighbour. This step prevents unstable WOE estimates and zero-event bins, both of which introduce noise into IV calculations and can cause calibration issues downstream.

### 2. WOE Computation

Within each bin, the ratio of events (target = 1) to non-events (target = 0) is computed relative to the total event and non-event counts in the training set. WOE is the natural log of this ratio. Computation is vectorized across bins for performance.

$$WOE_i = \ln\left(\frac{\%Bads_i}{\%Goods_i}\right)$$

In plain terms: a bin where bad accounts are overrepresented relative to the overall population gets a positive WOE; a bin where good accounts dominate gets a negative WOE. A WOE of zero means the bin's bad rate equals the overall bad rate.

The resulting WOE map — a dictionary from bin ID to WOE value — is stored on the fitted object and applied at `transform()` time.

### 3. IV Calculation

Information Value aggregates the separation power of a feature across all its bins. It is the sum over bins of the difference in event and non-event distributions, weighted by the WOE of each bin.

$$IV = \sum_{i=1}^{n} \left(\%Bads_i - \%Goods_i\right) \times WOE_i$$

In plain terms: IV measures how well a feature's bins separate defaults from non-defaults. A higher IV means stronger separation. After fitting, all IV scores are stored in `iv_table_` sorted in descending order.

### 4. Feature Selection

Features with IV below `min_iv` are excluded from the output of `transform()` when `drop_low_iv=True`. Features with IV above `max_iv_for_leakage` are retained but flagged in `leakage_flags_` and in the audit output. The selection logic is deterministic and fully inspectable via fitted attributes.

### 5. Audit Generation

If `output_dir` is set, three CSV files are written automatically at the end of `fit()`. These files record every binning decision, WOE value, IV score, and governance flag produced during the fit — see the Audit and Governance Outputs section for details.

---

## API Reference

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_bins` | `int` | `10` | Target number of quantile bins for numeric features |
| `min_iv` | `float` | `0.02` | Minimum IV to retain a feature at transform time |
| `max_iv_for_leakage` | `float` | `0.8` | Features with IV above this value are flagged as potential target proxy signals |
| `min_bin_pct` | `float or None` | `0.05` | Minimum bin population share (0–1); bins below this are merged |
| `special_codes` | `dict or None` | `{}` | Per-feature lists of sentinel values to isolate before binning |
| `encode` | `bool` | `True` | If `True`, `transform()` returns float WOE values. If `False`, returns integer bin IDs (zero-indexed, assigned during fit); useful for debugging binning decisions before encoding. |
| `drop_low_iv` | `bool` | `True` | If `True`, features below `min_iv` are excluded from `transform()` output |
| `n_jobs` | `int` | `-1` | Number of parallel workers passed to `joblib` (`-1` = all cores) |
| `output_dir` | `str or None` | `None` | Directory for audit CSV artifacts; no files are written if `None` |
| `verbose` | `bool` | `True` | If `True`, progress is logged via the standard `logging` module |

### Fitted Attributes

These attributes are available after calling `fit()`.

| Attribute | Type | Description |
|---|---|---|
| `iv_table_` | `pd.DataFrame` | IV scores for all processed features, sorted descending |
| `selected_features_` | `list[str]` | Features that met the `min_iv` threshold |
| `woe_maps_` | `dict[str, dict]` | Bin-to-WOE mapping per feature, applied at transform time |
| `binning_` | `dict[str, Any]` | Fitted bin configuration per feature |
| `monotonicity_report_` | `dict[str, Any]` | Monotonicity check results per feature, including direction |
| `leakage_flags_` | `dict[str, bool]` | `True` for each feature whose IV exceeded `max_iv_for_leakage` |
| `feature_types_` | `dict[str, str]` | `"numeric"` or `"categorical"` per feature |

---

## Audit and Governance Outputs

When `output_dir` is specified, the following files are written to disk at the end of every `fit()` call. Their primary purpose is to support model validation reviews and internal governance documentation.

```
audit_outputs/
    iv_summary.csv
    bin_stats.csv
    feature_audit.csv
```

### `iv_summary.csv`

One row per feature. Contains the IV score for every feature processed during fit, sorted by descending IV. This file is the starting point for a model validation team reviewing feature selection decisions — it records what was considered, what was kept, and the predictive signal behind each decision.

### `bin_stats.csv`

One row per bin per feature. Contains bin boundaries (or category labels), event count, non-event count, event rate, population share, and WOE value for every bin of every feature. This is the primary artifact for auditing binning decisions. Any reviewer questioning why a feature has a particular WOE value can trace it directly to this file.

### `feature_audit.csv`

One row per feature. Contains feature type, IV, monotonicity direction, and leakage risk flag. This file is designed for the governance layer — it answers the three questions most commonly asked during internal model risk reviews: how predictive is this feature, does its WOE trend make directional sense, and is there a risk that its predictive signal is too close to the target.

---

## Advanced Features

### Special Codes Handling

Credit bureau and application data frequently contain sentinel values that encode system states rather than measured quantities. Common examples include `-99` for "bureau record not found" and `9999` for "self-employed, income not verifiable". Feeding these values into quantile binning silently distorts bin boundaries, compresses the distribution of real values, and produces misleading WOE estimates.

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

Many internal model risk frameworks and central bank guidelines require that WOE values exhibit a monotonic relationship with risk across ordered bins — either consistently increasing or consistently decreasing as the feature value increases. A non-monotonic WOE curve may indicate overfitting to the training distribution, an unstable binning, or a feature that requires further investigation before inclusion in a scorecard.

After WOE computation, `iv-woe-filter` checks the trend direction for every numeric feature and stores the result in `monotonicity_report_` and `feature_audit.csv`. Features that fail the check are not dropped — the decision is left to the modeler — but they are clearly identified.

### Parallel Processing

Per-feature operations — binning, WOE and IV computation, monotonicity checking, and leakage flagging — are dispatched in parallel using `joblib.Parallel`. On datasets with a large number of features, this materially reduces the wall time of the fit step.

```python
IVWOEFilter(n_jobs=-1)   # all logical cores
IVWOEFilter(n_jobs=4)    # fixed worker count
IVWOEFilter(n_jobs=1)    # single-threaded (useful for debugging)
```

### Sklearn Compatibility

`IVWOEFilter` extends `BaseEstimator` and `TransformerMixin` and is fully compatible with the sklearn ecosystem. It can be used as a stage in `Pipeline`, tuned via `GridSearchCV`, and evaluated within `cross_val_score`. `get_feature_names_out()` is implemented for pipeline introspection.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("woe",   IVWOEFilter(min_iv=0.02, n_jobs=-1)),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

---

## Limitations

The following are known constraints of the current implementation. They are documented here to support informed usage and to avoid misapplication in production.

**IV is sensitive to binning strategy.** Information Value is not a fixed property of a feature — it depends on how that feature is binned. Different values of `n_bins` or `min_bin_pct` will produce different IV scores for the same feature on the same dataset. IV scores should not be compared across fits that used different binning configurations.

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

**High-cardinality free-text or embedding features.** Features that are inherently continuous at high resolution — raw embeddings, image features, unstructured text representations — are not suited to quantile binning. Apply appropriate feature extraction before using this transformer.

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
uv run pytest tests/test_iv_woe_filter.py -v
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
            binning.py           # Quantile and categorical binning logic
            woe.py               # Vectorized WOE / IV computation and monotonicity checks 
    tests/
        test_iv_woe_filter.py
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