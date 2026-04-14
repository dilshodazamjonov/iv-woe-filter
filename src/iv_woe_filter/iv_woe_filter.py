"""
Binning, Information Value (IV), and Weight of Evidence (WOE) Transformer.

This module provides a robust framework for feature selection 
specifically tailored for Binary Classification in highly regulated environments 
such as Credit Risk Scoring (PD models).

Why use IV/WOE?
---------------
1. Non-linear Relationships: WOE linearizes the relationship between a feature 
   and the log-odds of the target, allowing Logistic Regression to capture 
   non-linear patterns.
2. Handling Outliers & Missings: By grouping values into bins, extreme 
   outliers and missing values are treated as distinct categories, reducing 
   the need for manual imputation.
3. Feature Power: Information Value (IV) provides a single, model-agnostic 
   metric to rank the predictive power of features.
4. Interpretability: The transformation creates a "Risk Profile" for each 
   variable, making the model highly explainable for regulatory audits.

Theoretical Definitions
------------------------
* WOE_i = ln( % Non-Events_i / % Events_i )
* IV = Σ ( (% Non-Events_i - % Events_i) * WOE_i )

References
----------


Interpretation of IV:
- < 0.02: Useless for prediction.
- 0.02 to 0.1: Weak predictor.
- 0.1 to 0.3: Medium predictor.
- 0.3 to 0.5: Strong predictor.
- > 0.5: Suspiciously high (Potential Data Leakage).

Implementation Features:
------------------------
- Automatic Feature Typing: Separate handling for numeric and categorical data.
- Monotonicity Checking: Reports whether the risk profile of a feature is 
  consistently increasing or decreasing across bins.
- Regulatory Audit Logs: Generates CSV artifacts including bin-level statistics, 
  IV summaries, and leakage flags for model documentation.
- High Performance: Parallelized fitting using joblib and optimized bin 
  mapping for high-volume datasets.
- Special Value Handling: Isolates "System Codes" (e.g., 9999, -1) into 
  distinct bins to prevent skewing the general population distribution.

Author: Dilshod A'zamjonov
"""


from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .binning import (
    apply_bins,
    fit_categorical_bins,
    fit_numeric_bins,
    merge_non_significant_bins,
    get_bin_labels
)
from .woe import calculate_woe_iv, check_monotonicity, compute_aggregate_stats

logger = logging.getLogger(__name__)

class IVWOEFilter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for Information Value (IV) selection 
    and Weight of Evidence (WOE) encoding.

    This estimator automates the "Binning -> IV calculation -> Feature selection 
    -> WOE transformation" pipeline. It is designed to produce stable, 
    linearized features for Logistic Regression while generating the necessary 
    documentation for regulatory model validation.

    Parameters
    ----------
    n_bins : int, default=10
        Maximum number of bins for numeric variables. The actual number of 
        bins may be lower after merging non-significant or non-monotonic bins.
    min_iv : float, default=0.02
        Minimum Information Value required to retain a feature. Common industry 
        thresholds are 0.02 (weak) or 0.05 (medium).
    max_iv_for_leakage : float, default=0.8
        IV threshold used to flag potential data leakage. Features above 
        this value are flagged in the audit reports.
    min_bin_pct : float or None, default=0.05
        Minimum population fraction required per bin (e.g., 0.05 for 5%). 
        Bins smaller than this are merged to ensure statistical significance.
    special_codes : dict[str, list[Any]] or None, default=None
        Dictionary mapping column names to lists of "special values" 
        (e.g., -99, 9999) to be isolated in their own dedicated bins.
    encode : bool, default=True
        If True, replaces raw values with Weight of Evidence (WOE) scores.
    drop_low_iv : bool, default=True
        If True, the transform method drops columns with IV < `min_iv`.
    n_jobs : int, default=-1
        Number of parallel CPU workers for feature processing.
    output_dir : str or None, default=None
        Path to save CSV audit artifacts (IV summary, bin stats, feature audit).
    verbose : bool, default=True
        If True, logs progress and selection summaries to the console.

    Attributes
    ----------
    selected_features_ : list[str]
        List of features that met the `min_iv` criteria and were not flagged 
        for excessive leakage.
    iv_table_ : pd.DataFrame
        Ranked summary of all input features and their calculated Information Values.
    binning_ : dict[str, Any]
        Fitted bin boundaries (for numeric) or category maps (for categorical).
    woe_maps_ : dict[str, dict[int, float]]
        Mapping of bin IDs to their respective WOE values for each feature.
    monotonicity_report_ : dict[str, dict]
        Audit of whether the WOE trend is monotonic (increasing/decreasing) 
        across bins for each feature.
    feature_types_ : dict[str, str]
        Internal classification of features as 'numeric' or 'categorical'.
    leakage_flags_ : dict[str, bool]
        Boolean flags indicating if a feature exceeded `max_iv_for_leakage`.

    Methods
    -------
    fit(X, y)
        Calculates optimal bins, IV values, and WOE mappings. Identifies the subset 
        of features falling within the predictive range [`min_iv`, `max_iv_for_leakage`].
        
        Returns
        -------
        self : IVWOEFilter
            The fitted estimator instance, now containing the `selected_features_` 
            subset and internal WOE dictionaries.
            
        Side Effects
        ------------
        If `output_dir` is specified, the following CSV artifacts are exported:
        - `iv_summary.csv`: Ranked Information Value for all input features.
        - `bin_stats.csv`: Detailed statistics per bin (Counts, Event Rates, WOE).
        - `feature_audit.csv`: Regulatory report including monotonicity checks, 
          feature types, and leakage flags.

    transform(X)
        Applies the fitted binning logic and maps the resulting bins to their 
        respective Weight of Evidence (WOE) values.
        
        Returns
        -------
        X_out : pd.DataFrame
            The transformed dataset. If `encode=True`, raw values are replaced with 
            WOE scores. If `drop_low_iv=True`, the DataFrame is filtered to include 
            only the features that met the IV and leakage criteria.
    
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_iv: float = 0.02,
        max_iv_for_leakage: float = 0.8,
        min_bin_pct: float | None = 0.05,
        special_codes: dict[str, list[Any]] | None = None,
        encode: bool = True,
        drop_low_iv: bool = True,
        n_jobs: int = -1,
        output_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_iv_for_leakage = max_iv_for_leakage
        self.min_bin_pct = min_bin_pct
        self.special_codes = special_codes or {}
        self.encode = encode
        self.drop_low_iv = drop_low_iv
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.verbose = verbose

    def __repr__(self) -> str:
        return (f"IVWOEFilter(n_bins={self.n_bins}, min_iv={self.min_iv}, "
                f"encode={self.encode}, drop_low_iv={self.drop_low_iv})")

    @staticmethod
    def _process_column(
        col_name: str,
        series: pd.Series,
        y: np.ndarray,
        n_bins: int,
        min_bin_pct: float | None,
        specials: list[Any],
    ) -> dict[str, Any]:
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            bin_config = fit_numeric_bins(series, n_bins, specials)
        else:
            bin_config = fit_categorical_bins(series, specials)

        bin_ids = apply_bins(series, bin_config)

        if min_bin_pct:
            bin_ids = merge_non_significant_bins(bin_ids, min_bin_pct)

        stats = compute_aggregate_stats(bin_ids, y)
        
        labels_map = get_bin_labels(bin_config, series, bin_ids)
        stats.insert(0, "bin_range", stats.index.map(labels_map))

        woe_series, iv_bin_series, iv_value = calculate_woe_iv(stats)

        stats = stats.assign(woe=woe_series, iv_bin=iv_bin_series)
        mono_info = check_monotonicity(woe_series)

        return {
            "column": col_name,
            "iv": iv_value,
            "bin_config": bin_config,
            "stats": stats,
            "woe_map": woe_series.to_dict(),
            "monotonic": mono_info,
            "feature_type": "numeric" if is_numeric else "categorical",
        }


    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> IVWOEFilter:
        """Fit binning and calculate WOE/IV for all columns.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series or np.ndarray
            Binary target variable (0/1).

        Returns
        -------
        IVWOEFilter
            The fitted estimator.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        y_arr = np.asarray(y)
        columns = X.columns.tolist()

        if self.verbose:
            logger.info(f"Fitting {len(columns)} features...")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_column)(
                col,
                X[col],
                y_arr,
                self.n_bins,
                self.min_bin_pct,
                self.special_codes.get(col, []),
            )
            for col in columns
        )

        # Initialize fit attributes
        self.binning_: dict[str, Any] = {}
        self.woe_maps_: dict[str, Any] = {}
        self.iv_table_data_: dict[str, float] = {}
        self._per_feature_stats: dict[str, pd.DataFrame] = {}
        self.monotonicity_report_: dict[str, Any] = {}
        self.feature_types_: dict[str, str] = {}
        self.leakage_flags_: dict[str, bool] = {}

        for res in results:
            col = res["column"]
            iv_val = res["iv"]
            self.binning_[col] = res["bin_config"]
            self.woe_maps_[col] = res["woe_map"]
            self.iv_table_data_[col] = iv_val
            self._per_feature_stats[col] = res["stats"]
            self.monotonicity_report_[col] = res["monotonic"]
            self.feature_types_[col] = res["feature_type"]
            self.leakage_flags_[col] = iv_val > self.max_iv_for_leakage

        # Selection
        self.iv_table_ = pd.DataFrame.from_dict(
            self.iv_table_data_, orient="index", columns=["IV"]
        ).sort_values("IV", ascending=False)

        self.selected_features_ = self.iv_table_[
            self.iv_table_["IV"] >= self.min_iv
        ].index.tolist()

        if self.output_dir:
            self._save_artifacts()

        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted binning and WOE transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

        Returns
        -------
        pd.DataFrame
            Transformed features. If encode=True, values are Weight of Evidence.
        """
        check_is_fitted(self, ["selected_features_", "binning_", "woe_maps_"])

        cols_to_process = (
            self.selected_features_ if self.drop_low_iv else X.columns.tolist()
        )
        X_out = X[cols_to_process].copy()

        if not self.encode:
            return X_out

        for col in cols_to_process:
            bin_ids = apply_bins(X_out[col], self.binning_[col])
            # High performance dictionary mapping with index preservation
            w_map = self.woe_maps_[col]
            X_out[col] = pd.Series(bin_ids, index=X_out.index).map(w_map).fillna(0.0)

        return X_out

    def _save_artifacts(self) -> None:
        """Internal helper to save CSV audit files."""
        self.iv_table_.to_csv(os.path.join(self.output_dir, "iv_summary.csv"))

        bin_stats = pd.concat(
            [df.assign(feature=feat) for feat, df in self._per_feature_stats.items()]
        ).reset_index()
        bin_stats.to_csv(os.path.join(self.output_dir, "bin_stats.csv"), index=False)

        audit_rows = []
        for col in self.iv_table_.index:
            audit_rows.append(
                {
                    "feature": col,
                    "type": self.feature_types_[col],
                    "IV": self.iv_table_data_[col],
                    "is_monotonic": self.monotonicity_report_[col]["is_monotonic"],
                    "direction": self.monotonicity_report_[col]["direction"],
                    "leakage_flag": self.leakage_flags_[col],
                }
            )
        pd.DataFrame(audit_rows).to_csv(
            os.path.join(self.output_dir, "feature_audit.csv"), index=False
        )

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return list of selected feature names for Scikit-Learn pipeline compatibility."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_