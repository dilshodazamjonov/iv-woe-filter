"""Orchestration module for Information Value (IV) and Weight of Evidence (WOE) filtering."""

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
)
from .woe import calculate_woe_iv, check_monotonicity, compute_aggregate_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main Filter Class
# ---------------------------------------------------------------------------

class IVWOEFilter(BaseEstimator, TransformerMixin):
    """Advanced IV/WOE Filter and Transformer for Credit Scoring.

    This class performs optimal binning, calculates Information Value (IV),
    filters features based on predictive power, and transforms raw data 
    into Weight of Evidence (WOE) values.

    Parameters
    ----------
    n_bins : int, default=10
        Target number of bins for numeric quantile splitting.
    min_iv : float, default=0.02
        Minimum IV threshold required to keep a feature.
    max_iv_for_leakage : float, default=0.8
        IV threshold above which a feature is flagged for potential leakage.
    min_bin_pct : float or None, default=0.05
        Minimum population percentage required per bin.
    special_codes : dict[str, list[Any]] or None
        Dictionary mapping column names to lists of special values.
    encode : bool, default=True
        If True, transform returns WOE values.
    drop_low_iv : bool, default=True
        If True, transform drops columns that fell below min_iv.
    n_jobs : int, default=-1
        Number of parallel jobs for processing features.
    output_dir : str or None
        Directory to save audit artifacts.
    verbose : bool, default=True
        Enable logging progress.
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
        """Internal worker to process a single column: binning -> stats -> enrichment."""
        # 1. Binning Logic
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            bin_config = fit_numeric_bins(series, n_bins, specials)
        else:
            bin_config = fit_categorical_bins(series, specials)

        # Generate Bin IDs (formerly codes)
        bin_ids = apply_bins(series, bin_config)

        # Optional: Merge bins based on population threshold
        if min_bin_pct:
            bin_ids = merge_non_significant_bins(bin_ids, min_bin_pct)

        # 2. Stats Logic
        stats = compute_aggregate_stats(bin_ids, y)
        woe_series, iv_bin_series, iv_value = calculate_woe_iv(stats)

        # 3. Enrichment Logic
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