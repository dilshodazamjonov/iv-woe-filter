import os
import logging
from typing import Optional, Dict, List, Any, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .binning import fit_numeric_bins, fit_categorical_bins, apply_bins, merge_non_significant_bins
from .woe import compute_aggregate_stats, calculate_woe_iv, check_monotonicity

logger = logging.getLogger("iv_woe_package")
logging.basicConfig(level=logging.INFO)

class IVWOEFilter(BaseEstimator, TransformerMixin):
    """
    Advanced IV/WOE Filter and Transformer for Credit Scoring.
    
    This class performs optimal binning, calculates Information Value (IV),
    filters features based on predictive power, and transforms raw data 
    into Weight of Evidence (WOE) values.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_iv: float = 0.02,
        max_iv_for_leakage: float = 0.8,
        min_bin_pct: Optional[float] = 0.05,
        special_codes: Optional[Dict[str, List[Any]]] = None,
        encode: bool = True,
        n_jobs: int = -1,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_iv_for_leakage = max_iv_for_leakage
        self.min_bin_pct = min_bin_pct
        self.special_codes = special_codes or {}
        self.encode = encode
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.verbose = verbose

    @staticmethod
    def _process_column(
        col_name: str, 
        series: pd.Series, 
        y: np.ndarray, 
        n_bins: int, 
        min_bin_pct: Optional[float],
        specials: List[Any]
    ) -> Dict[str, Any]:
        """Worker function for parallel processing of a single feature."""
        # 1. Determine Type and Fit Binning
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            bin_config = fit_numeric_bins(series, n_bins, specials)
        else:
            bin_config = fit_categorical_bins(series, specials)

        # 2. Apply Binning to generate codes
        codes = apply_bins(series, bin_config)

        # 3. Optional: Merge small bins
        if min_bin_pct:
            codes = merge_non_significant_bins(codes, y, min_bin_pct)

        # 4. Compute Statistics
        stats = compute_aggregate_stats(codes, y)
        woe_series, iv_bin_series, iv_value = calculate_woe_iv(stats)
        
        # 5. Enrichment
        stats = stats.assign(woe=woe_series, iv_bin=iv_bin_series)
        mono_info = check_monotonicity(woe_series)

        return {
            "column": col_name,
            "iv": iv_value,
            "bin_config": bin_config,
            "stats": stats,
            "woe_map": woe_series.to_dict(),
            "monotonic": mono_info
        }

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "IVWOEFilter":
        """
        Fits binning and calculates WOE/IV for all columns in X.
        
        Args:
            X: Input features.
            y: Binary target (0/1).
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        y_arr = np.asarray(y)
        columns = X.columns.tolist()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_column)(
                col, X[col], y_arr, self.n_bins, self.min_bin_pct, self.special_codes.get(col, [])
            ) for col in columns
        )

        # Initialize fitted attributes
        self.binning_: Dict[str, Any] = {}
        self.woe_maps_: Dict[str, Any] = {}
        self.iv_table_data_: Dict[str, float] = {}
        self._per_feature_stats: Dict[str, pd.DataFrame] = {}
        self.monotonicity_report_: Dict[str, Any] = {}

        for res in results:
            col = res["column"]
            self.binning_[col] = res["bin_config"]
            self.woe_maps_[col] = res["woe_map"]
            self.iv_table_data_[col] = res["iv"]
            self._per_feature_stats[col] = res["stats"]
            self.monotonicity_report_[col] = res["monotonic"]

        # Finalize Selection
        self.iv_table_ = pd.DataFrame.from_dict(
            self.iv_table_data_, orient="index", columns=["IV"]
        ).sort_values("IV", ascending=False)
        
        self.selected_features_ = self.iv_table_[
            self.iv_table_["IV"] >= self.min_iv
        ].index.tolist()

        if self.output_dir:
            self._save_artifacts()

        if self.verbose:
            logger.info(f"Fit complete. Selected {len(self.selected_features_)} features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms features using fitted binning and WOE maps.
        
        Args:
            X: Input features.
        Returns:
            DataFrame with selected features (optionally WOE encoded).
        """
        check_is_fitted(self, ["selected_features_", "binning_", "woe_maps_"])
        
        X_out = X[self.selected_features_].copy()

        if not self.encode:
            return X_out

        for col in self.selected_features_:
            codes = apply_bins(X_out[col], self.binning_[col])
            # Map codes to WOE, defaulting to 0.0 for unknown/unseen bins
            w_map = self.woe_maps_[col]
            X_out[col] = np.vectorize(lambda c: w_map.get(c, 0.0))(codes)

        return X_out

    def _save_artifacts(self) -> None:
        """Serializes fit statistics to the output directory."""
        self.iv_table_.to_csv(os.path.join(self.output_dir, "iv_summary.csv"))
        
        # Bin Level Detailed Stats
        bin_stats = pd.concat([
            df.assign(feature=feat) for feat, df in self._per_feature_stats.items()
        ]).reset_index()
        bin_stats.to_csv(os.path.join(self.output_dir, "bin_stats.csv"), index=False)

        # Monotonicity and Leakage Flags
        audit_rows = []
        for col in self.iv_table_.index:
            audit_rows.append({
                "feature": col,
                "IV": self.iv_table_data_[col],
                "is_monotonic": self.monotonicity_report_[col]["is_monotonic"],
                "direction": self.monotonicity_report_[col]["direction"],
                "leakage_flag": self.iv_table_data_[col] > self.max_iv_for_leakage
            })
        pd.DataFrame(audit_rows).to_csv(os.path.join(self.output_dir, "feature_audit.csv"), index=False)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Compatibility with Sklearn Pipeline."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_