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
- Configurable Numeric Binning: Quantile or supervised tree-based thresholds.
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
    SUPPORTED_BINNING_METHODS,
    apply_bins,
    fit_categorical_bins,
    fit_numeric_bins,
    get_bin_labels,
    merge_non_significant_bins,
)
from .woe import (
    calculate_woe_iv,
    check_monotonicity,
    check_numeric_monotonicity,
    compute_aggregate_stats,
)
from .metrics import calculate_feature_gini, calculate_gini, calculate_psi_from_counts

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
    binning_method : {"quantile", "tree"}, default="quantile"
        Strategy used to learn numeric bin boundaries.
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
    drop_unselected : bool, default=True
        If True, the transform method keeps only features that passed selection criteria.
        If False, no columns are dropped but encoding still applies.
    psi_thresholds : tuple[float, float], default=(0.1, 0.2)
        Thresholds for Minor and Significant PSI shifts.
    random_state : int or None, default=42
        Random state passed to tree-based binning.
    tree_criterion : str, default="gini"
        Split criterion used by tree-based binning.
    tree_max_depth : int or None, default=None
        Maximum depth used by tree-based binning.
    tree_min_samples_leaf : int, float, or None, default=None
        Minimum samples per tree leaf. If None, `min_bin_pct` is used when available.
    tree_min_samples_split : int or float, default=2
        Minimum samples required to split an internal tree node.
    n_jobs : int, default=-1
        Number of parallel CPU workers for feature processing.
    output_dir : str or None, default=None
        Path to save CSV audit artifacts (IV summary, bin stats, feature audit).
    verbose : bool, default=True
        If True, logs progress and selection summaries to the console.

    Attributes
    ----------
    selected_features_ : list[str]
        List of features that met the IV and Gini selection criteria.
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
        binning_method: str = "quantile",
        min_iv: float = 0.02,
        min_gini: float = 0.05,
        max_iv_for_leakage: float = 0.8,
        min_bin_pct: float | None = 0.05,
        special_codes: dict[str, list[Any]] | None = None,
        encode: bool = True,
        drop_unselected: bool = True,
        psi_thresholds: tuple[float, float] = (0.1, 0.2),
        random_state: int | None = 42,
        tree_criterion: str = "gini",
        tree_max_depth: int | None = None,
        tree_min_samples_leaf: int | float | None = None,
        tree_min_samples_split: int | float = 2,
        n_jobs: int = -1,
        output_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.min_iv = min_iv
        self.min_gini = min_gini
        self.max_iv_for_leakage = max_iv_for_leakage
        self.min_bin_pct = min_bin_pct
        self.special_codes = special_codes or {}
        self.encode = encode
        self.drop_unselected = drop_unselected
        self.psi_thresholds = psi_thresholds
        self.random_state = random_state
        self.tree_criterion = tree_criterion
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.tree_min_samples_split = tree_min_samples_split
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.verbose = verbose

    def __repr__(self) -> str:
        return (f"IVWOEFilter(n_bins={self.n_bins}, binning_method={self.binning_method!r}, "
                f"min_iv={self.min_iv}, min_gini={self.min_gini}, "
                f"encode={self.encode}, drop_unselected={self.drop_unselected})")
    
    def _validate_target(self, y: np.ndarray) -> None:
        """Validate that target y is binary (0/1) and contains no NaN."""
        if np.isnan(y).any():
            raise ValueError("Target contains NaN values")

        unique_vals = np.unique(y)
        if len(unique_vals) > 2:
            raise ValueError(f"Target must be binary (0/1), got {len(unique_vals)} unique values: {unique_vals}")
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"Target values must be 0 or 1, got: {unique_vals}")
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError("Target must contain both classes 0 and 1.")

    def _validate_parameters(self) -> None:
        """Validate estimator parameters before fitting."""
        if self.binning_method not in SUPPORTED_BINNING_METHODS:
            raise ValueError(
                f"binning_method must be one of {sorted(SUPPORTED_BINNING_METHODS)}, "
                f"got {self.binning_method!r}."
            )
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ValueError(f"n_bins must be an integer >= 1, got {self.n_bins!r}.")
        if self.min_iv < 0:
            raise ValueError(f"min_iv must be >= 0, got {self.min_iv!r}.")
        if not 0 <= self.min_gini <= 1:
            raise ValueError(f"min_gini must be between 0 and 1, got {self.min_gini!r}.")
        if self.max_iv_for_leakage < 0:
            raise ValueError(
                f"max_iv_for_leakage must be >= 0, got {self.max_iv_for_leakage!r}."
            )
        if self.min_bin_pct is not None and not 0 < self.min_bin_pct < 1:
            raise ValueError(
                f"min_bin_pct must be None or a float between 0 and 1, got {self.min_bin_pct!r}."
            )
        if not isinstance(self.psi_thresholds, tuple) or len(self.psi_thresholds) != 2:
            raise ValueError(
                f"psi_thresholds must be a tuple of two floats, got {self.psi_thresholds!r}."
            )
        low, high = self.psi_thresholds
        if not 0 <= low <= high:
            raise ValueError(
                f"psi_thresholds must satisfy 0 <= low <= high, got {self.psi_thresholds!r}."
            )
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError(f"n_jobs must be a non-zero integer, got {self.n_jobs!r}.")
        if not isinstance(self.special_codes, dict):
            raise ValueError(f"special_codes must be a dict or None, got {type(self.special_codes).__name__}.")
        if self.tree_criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError(
                "tree_criterion must be one of ['entropy', 'gini', 'log_loss'], "
                f"got {self.tree_criterion!r}."
            )
        if self.tree_max_depth is not None and (
            not isinstance(self.tree_max_depth, int) or self.tree_max_depth < 1
        ):
            raise ValueError(
                f"tree_max_depth must be None or an integer >= 1, got {self.tree_max_depth!r}."
            )
        if self.tree_min_samples_leaf is not None:
            if isinstance(self.tree_min_samples_leaf, int):
                if self.tree_min_samples_leaf < 1:
                    raise ValueError(
                        "tree_min_samples_leaf must be >= 1 when passed as an integer, "
                        f"got {self.tree_min_samples_leaf!r}."
                    )
            elif isinstance(self.tree_min_samples_leaf, float):
                if not 0 < self.tree_min_samples_leaf < 1:
                    raise ValueError(
                        "tree_min_samples_leaf must be between 0 and 1 when passed as a float, "
                        f"got {self.tree_min_samples_leaf!r}."
                    )
            else:
                raise ValueError(
                    "tree_min_samples_leaf must be None, int, or float, "
                    f"got {type(self.tree_min_samples_leaf).__name__}."
                )
        if isinstance(self.tree_min_samples_split, int):
            if self.tree_min_samples_split < 2:
                raise ValueError(
                    "tree_min_samples_split must be >= 2 when passed as an integer, "
                    f"got {self.tree_min_samples_split!r}."
                )
        elif isinstance(self.tree_min_samples_split, float):
            if not 0 < self.tree_min_samples_split <= 1:
                raise ValueError(
                    "tree_min_samples_split must be in (0, 1] when passed as a float, "
                    f"got {self.tree_min_samples_split!r}."
                )
        else:
            raise ValueError(
                "tree_min_samples_split must be an int or float, "
                f"got {type(self.tree_min_samples_split).__name__}."
            )

    @staticmethod
    def _process_column(
        col_name: str,
        series: pd.Series,
        y: np.ndarray,
        n_bins: int,
        binning_method: str,
        min_bin_pct: float | None,
        specials: list[Any],
        random_state: int | None,
        tree_criterion: str,
        tree_max_depth: int | None,
        tree_min_samples_leaf: int | float | None,
        tree_min_samples_split: int | float,
    ) -> dict[str, Any]:

        if specials:
            missing = [s for s in specials if s not in series.values]
            if missing:
                logger.warning(
                    "Special codes %s not found in feature '%s'. "
                    "These will create empty bins.",
                    missing, col_name
                )
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        if is_numeric:
            bin_config = fit_numeric_bins(
                series,
                n_bins,
                specials,
                binning_method=binning_method,
                y=y,
                min_bin_pct=min_bin_pct,
                random_state=random_state,
                tree_criterion=tree_criterion,
                tree_max_depth=tree_max_depth,
                tree_min_samples_leaf=tree_min_samples_leaf,
                tree_min_samples_split=tree_min_samples_split,
            )
        else:
            bin_config = fit_categorical_bins(series, specials)

        bin_ids = apply_bins(series, bin_config)

        if min_bin_pct:
            bin_ids = merge_non_significant_bins(bin_ids, min_bin_pct)

        stats = compute_aggregate_stats(bin_ids, y)
        
        labels_map = get_bin_labels(bin_config, series, bin_ids)
        stats.insert(0, "bin_range", stats.index.map(labels_map))

        woe_series, iv_bin_series, iv_value = calculate_woe_iv(stats)
        if is_numeric:
            numeric_values = pd.to_numeric(series, errors="coerce").to_numpy()
            gini_value = calculate_gini(y, numeric_values)
        else:
            gini_value = calculate_feature_gini(bin_ids, woe_series.to_dict(), y)

        stats = stats.assign(woe=woe_series, iv_bin=iv_bin_series)
        mono_info = check_numeric_monotonicity(woe_series) if is_numeric else check_monotonicity(woe_series)

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

        self._validate_target(y_arr)
        self._validate_parameters()

        columns = X.columns.tolist()

        if self.verbose:
            logger.info("Fitting %d features...", len(columns))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_column)(
                col,
                X[col],
                y_arr,
                self.n_bins,
                self.binning_method,
                self.min_bin_pct,
                self.special_codes.get(col, []),
                self.random_state,
                self.tree_criterion,
                self.tree_max_depth,
                self.tree_min_samples_leaf,
                self.tree_min_samples_split,
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
            logger.info("Selected %d features.", len(self.selected_features_))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted binning and WOE transformation."""
        check_is_fitted(self, ["selected_features_", "binning_", "woe_maps_"])

        cols_to_process = (
            self.selected_features_ if self.drop_unselected else list(self.binning_.keys())
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
            DataFrame containing the PSI score and shift status for each feature.
        """
        check_is_fitted(self, ["binning_", "reference_distributions_", "selected_features_"])
        
        psi_records = []
        cols_to_process = (
            self.selected_features_ if self.drop_unselected else list(self.binning_.keys())
        )
        missing_columns = [c for c in cols_to_process if c not in X.columns]
        if missing_columns:
            logger.warning(
                "The following fitted features are missing from PSI input and will be marked as missing: %s",
                missing_columns,
            )
        available_columns = [c for c in cols_to_process if c in X.columns]
        low, high = self.psi_thresholds
        
        for col in available_columns:
            bin_ids = apply_bins(X[col], self.binning_[col])
            actual_counts = pd.Series(bin_ids).value_counts()
            expected_counts = self.reference_distributions_[col]
            
            psi_total, _ = calculate_psi_from_counts(expected_counts, actual_counts)
            
            status = "Stable"
            if psi_total >= high:
                status = "Significant Shift"
            elif psi_total >= low:
                status = "Minor Shift"
                
            psi_records.append({
                "feature": col,
                "PSI": psi_total,
                "status": status
            })

        for col in missing_columns:
            psi_records.append({
                "feature": col,
                "PSI": np.nan,
                "status": "Missing in Input"
            })
            
        df_psi = pd.DataFrame(psi_records)
        if not df_psi.empty:
            df_psi = df_psi.sort_values("PSI", ascending=False, na_position="last").reset_index(drop=True)
            
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
                    "binning_method": self.binning_[col].get("binning_method"),
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
        return self.selected_features_ if self.drop_unselected else list(self.binning_.keys())

        
