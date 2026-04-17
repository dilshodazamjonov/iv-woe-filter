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
from .metrics import calculate_feature_gini, calculate_psi_from_counts

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
        Maximum number of bins for numeric variables.
    min_iv : float, default=0.02
        Minimum Information Value required to retain a feature.
    min_gini : float, default=0.05
        Minimum Gini coefficient required to retain a feature.
    max_iv_for_leakage : float, default=0.8
        IV threshold used to flag potential data leakage.
    min_bin_pct : float or None, default=0.05
        Minimum population fraction required per bin (e.g., 0.05 for 5%).
    special_codes : dict[str, list[Any]] or None, default=None
        Dictionary mapping column names to lists of "special values" to be isolated.
    encode : bool, default=True
        If True, replaces raw values with Weight of Evidence (WOE) scores.
    drop_low_iv : bool, default=True
        If True, the transform method drops columns failing IV/Gini thresholds.
    n_jobs : int, default=-1
        Number of parallel CPU workers for feature processing.
    output_dir : str or None, default=None
        Path to save CSV audit artifacts (IV summary, bin stats, feature audit).
    verbose : bool, default=True
        If True, logs progress and selection summaries to the console.

    Attributes
    ----------
    selected_features_ : list[str]
        List of features that met the IV, Gini, and leakage criteria.
    iv_table_ : pd.DataFrame
        Ranked summary of all input features and their calculated IV and Gini.
    binning_ : dict[str, Any]
        Fitted bin boundaries (for numeric) or category maps (for categorical).
    woe_maps_ : dict[str, dict[int, float]]
        Mapping of bin IDs to their respective WOE values for each feature.
    reference_distributions_ : dict[str, pd.Series]
        Bin counts from the training data, used for later PSI calculation.
    monotonicity_report_ : dict[str, dict]
        Audit of whether the WOE trend is monotonic across bins for each feature.
    feature_types_ : dict[str, str]
        Internal classification of features as 'numeric' or 'categorical'.
    leakage_flags_ : dict[str, bool]
        Boolean flags indicating if a feature exceeded `max_iv_for_leakage`.

    Methods
    -------
    fit(X, y)
        Calculates optimal bins, IV values, Gini, and WOE mappings.
    transform(X)
        Applies the fitted binning logic and maps bins to WOE values.
    calculate_psi(X, save=True)
        Calculates the Population Stability Index (PSI) against a new dataset and exports report.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_iv: float = 0.02,
        min_gini: float = 0.05,
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
        self.min_gini = min_gini
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
                f"min_gini={self.min_gini}, encode={self.encode})")

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
        gini_value = calculate_feature_gini(bin_ids, woe_series.to_dict(), y)

        stats = stats.assign(woe=woe_series, iv_bin=iv_bin_series)
        mono_info = check_monotonicity(woe_series)

        return {
            "column": col_name,
            "iv": iv_value,
            "gini": gini_value,
            "bin_config": bin_config,
            "stats": stats,
            "woe_map": woe_series.to_dict(),
            "monotonic": mono_info,
            "feature_type": "numeric" if is_numeric else "categorical",
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> IVWOEFilter:
        """Fit binning and calculate WOE/IV for all columns."""
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

        self.binning_: dict[str, Any] = {}
        self.woe_maps_: dict[str, Any] = {}
        self.iv_table_data_: dict[str, float] = {}
        self.gini_table_data_: dict[str, float] = {}
        self._per_feature_stats: dict[str, pd.DataFrame] = {}
        self.reference_distributions_: dict[str, pd.Series] = {}
        self.monotonicity_report_: dict[str, Any] = {}
        self.feature_types_: dict[str, str] = {}
        self.leakage_flags_: dict[str, bool] = {}

        for res in results:
            col = res["column"]
            iv_val = res["iv"]
            gini_val = res["gini"]
            self.binning_[col] = res["bin_config"]
            self.woe_maps_[col] = res["woe_map"]
            self.iv_table_data_[col] = iv_val
            self.gini_table_data_[col] = gini_val
            self._per_feature_stats[col] = res["stats"]
            self.reference_distributions_[col] = res["stats"]["count"]
            self.monotonicity_report_[col] = res["monotonic"]
            self.feature_types_[col] = res["feature_type"]
            self.leakage_flags_[col] = iv_val > self.max_iv_for_leakage

        self.iv_table_ = pd.DataFrame(
            {
                "IV": self.iv_table_data_,
                "Gini": self.gini_table_data_,
            }
        ).sort_values("IV", ascending=False)

        self.selected_features_ = self.iv_table_[
            (self.iv_table_["IV"] >= self.min_iv) & 
            (self.iv_table_["Gini"] >= self.min_gini)
        ].index.tolist()

        if self.output_dir:
            self._save_artifacts()

        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted binning and WOE transformation."""
        check_is_fitted(self, ["selected_features_", "binning_", "woe_maps_"])

        cols_to_process = (
            self.selected_features_ if self.drop_low_iv else list(self.binning_.keys())
        )
        X_out = X[cols_to_process].copy()

        if not self.encode:
            return X_out

        for col in cols_to_process:
            bin_ids = apply_bins(X_out[col], self.binning_[col])
            w_map = self.woe_maps_[col]
            X_out[col] = pd.Series(bin_ids, index=X_out.index).map(w_map).fillna(0.0)

        return X_out

    def calculate_psi(self, X: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Calculate Population Stability Index (PSI) on a new dataset.

        Parameters
        ----------
        X : pd.DataFrame
            New dataset (e.g., test or validation set) to compare against training.
        save : bool, default=True
            If True and output_dir is configured, saves results to 'stability_report.csv'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the PSI score and shift status for each selected feature.
        """
        check_is_fitted(self, ["binning_", "reference_distributions_", "selected_features_"])
        
        psi_records = []
        cols_to_process = (
            self.selected_features_ if self.drop_low_iv else list(self.binning_.keys())
        )
        cols_to_process = [c for c in cols_to_process if c in X.columns]
        
        for col in cols_to_process:
            bin_ids = apply_bins(X[col], self.binning_[col])
            actual_counts = pd.Series(bin_ids).value_counts()
            expected_counts = self.reference_distributions_[col]
            
            psi_total, _ = calculate_psi_from_counts(expected_counts, actual_counts)
            
            status = "Stable"
            if psi_total >= 0.2:
                status = "Significant Shift"
            elif psi_total >= 0.1:
                status = "Minor Shift"
                
            psi_records.append({
                "feature": col,
                "PSI": psi_total,
                "status": status
            })
            
        df_psi = pd.DataFrame(psi_records)
        if not df_psi.empty:
            df_psi = df_psi.sort_values("PSI", ascending=False).reset_index(drop=True)
            
        if save and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            df_psi.to_csv(os.path.join(self.output_dir, "stability_report.csv"), index=False)
            
        return df_psi

    def _save_artifacts(self) -> None:
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
                    "Gini": self.gini_table_data_[col],
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
        return self.selected_features_ if self.drop_low_iv else list(self.binning_.keys())