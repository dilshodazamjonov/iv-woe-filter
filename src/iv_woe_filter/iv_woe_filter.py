"""Scikit-learn compatible IV/WOE feature selection and encoding."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .binning import (
    BinConfig,
    NumericBinConfig,
    apply_bins,
    fit_categorical_bins,
    fit_numeric_bins,
    get_categorical_bin_labels,
    get_numeric_bin_labels,
    merge_non_significant_bins,
    remap_bin_ids,
)
from .metrics import calculate_feature_gini, calculate_psi_from_counts
from .plots import prepare_plot_frame, render_feature_plot, save_rendered_feature_plot
from .validation import ParallelBackend, prepare_binary_target, validate_estimator_parameters
from .woe import (
    MonotonicityResult,
    calculate_woe_iv,
    check_monotonicity,
    check_numeric_monotonicity,
    compute_aggregate_stats,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)

FeatureType = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureComputationResult:
    """Internal representation of a fully processed feature."""

    column: str
    iv: float
    gini: float
    bin_config: BinConfig
    stats: pd.DataFrame
    woe_map: dict[int, float]
    monotonicity: MonotonicityResult
    feature_type: FeatureType


class IVWOEFilter(BaseEstimator, TransformerMixin):
    """
    Estimate bins, WOE values, IV scores, and stability diagnostics per feature.

    Parameters
    ----------
    n_bins : int, default=10
        Maximum number of bins for numeric variables.
    binning_method : {"quantile", "chi_merge", "tree"}, default="quantile"
        Strategy used to learn numeric bin boundaries.
    min_iv : float, default=0.02
        Minimum Information Value required to retain a feature.
    min_gini : float, default=0.05
        Minimum Gini coefficient required to retain a feature.
        Gini is computed on the fitted WOE representation for every feature.
    max_iv_for_leakage : float, default=0.8
        IV threshold used to flag potential data leakage.
    min_bin_pct : float or None, default=0.05
        Minimum population fraction required per bin.
    special_codes : dict[str, list[Any]] or None, default=None
        Dictionary mapping feature names to values that must remain isolated.
    encode : bool, default=True
        If True, transform features to WOE values.
    drop_unselected : bool, default=True
        If True, keep only features that satisfy both IV and Gini thresholds.
    psi_thresholds : tuple[float, float], default=(0.1, 0.2)
        Thresholds used to classify PSI drift as minor or significant.
    random_state : int or None, default=42
        Random state passed to tree-based binning.
    tree_criterion : str, default="gini"
        Split criterion used by tree-based binning.
    tree_max_depth : int or None, default=None
        Maximum depth used by tree-based binning.
    tree_min_samples_leaf : int, float, or None, default=None
        Minimum samples per tree leaf. If None, min_bin_pct is reused when available.
    tree_min_samples_split : int or float, default=2
        Minimum samples required to split an internal tree node.
    n_jobs : int, default=-1
        Number of parallel workers used for feature processing.
    parallel_backend : {"auto", "threads", "processes"}, default="auto"
        Joblib backend preference for per-feature processing. "auto" uses threads
        for supervised numeric binning and processes otherwise.
    output_dir : str or None, default=None
        Directory used to write CSV audit artifacts.
    verbose : bool, default=True
        If True, log progress and selection summaries.
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
        parallel_backend: ParallelBackend = "auto",
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
        self.parallel_backend = parallel_backend
        self.output_dir = output_dir
        self.verbose = verbose

    def __repr__(self) -> str:
        return (
            "IVWOEFilter("
            f"n_bins={self.n_bins}, "
            f"binning_method={self.binning_method!r}, "
            f"min_iv={self.min_iv}, "
            f"min_gini={self.min_gini}, "
            f"parallel_backend={self.parallel_backend!r}, "
            f"encode={self.encode}, "
            f"drop_unselected={self.drop_unselected})"
        )

    @staticmethod
    def _warn_missing_special_codes(
        col_name: str,
        series: pd.Series,
        specials: list[Any],
    ) -> None:
        """Log declared special codes that are not present in the current feature."""
        if not specials:
            return

        missing = [special for special in specials if special not in series.values]
        if missing:
            logger.warning(
                "Special codes %s not found in feature '%s'. These will create empty bins.",
                missing,
                col_name,
            )

    @staticmethod
    def _resolve_feature_type(series: pd.Series) -> FeatureType:
        """Classify a feature as numeric or categorical from its dtype."""
        return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"

    @staticmethod
    def _fit_feature_binning(
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
    ) -> tuple[FeatureType, BinConfig]:
        """Fit the appropriate binning configuration for a single feature."""
        feature_type = IVWOEFilter._resolve_feature_type(series)
        if feature_type == "numeric":
            return feature_type, fit_numeric_bins(
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
        return feature_type, fit_categorical_bins(series, specials)

    @staticmethod
    def _apply_min_bin_pct_merge(
        bin_ids: np.ndarray,
        bin_config: BinConfig,
        min_bin_pct: float | None,
    ) -> np.ndarray:
        """Merge undersized bins and persist the merge map on the config."""
        if not min_bin_pct:
            return bin_ids

        merged_bin_ids, bin_id_map = merge_non_significant_bins(
            bin_ids,
            min_bin_pct,
            return_mapping=True,
        )
        if any(original_bin != final_bin for original_bin, final_bin in bin_id_map.items()):
            bin_config["bin_id_map"] = bin_id_map
        return merged_bin_ids

    @staticmethod
    def _build_feature_labels(
        feature_type: FeatureType,
        bin_config: BinConfig,
        series: pd.Series,
        bin_ids: np.ndarray,
    ) -> dict[int, str]:
        """Build human-readable bin labels for a feature."""
        if feature_type == "numeric":
            return get_numeric_bin_labels(cast(NumericBinConfig, bin_config))
        return get_categorical_bin_labels(series, bin_ids)

    @staticmethod
    def _build_feature_stats(
        feature_type: FeatureType,
        series: pd.Series,
        y: np.ndarray,
        bin_config: BinConfig,
        bin_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Aggregate counts and attach display labels for one feature."""
        stats = compute_aggregate_stats(bin_ids, y)
        labels_map = IVWOEFilter._build_feature_labels(feature_type, bin_config, series, bin_ids)
        stats.insert(0, "bin_range", stats.index.map(labels_map))
        return stats

    @staticmethod
    def _calculate_feature_gini(
        y: np.ndarray,
        bin_ids: np.ndarray,
        woe_map: dict[int, float],
    ) -> float:
        """Calculate Gini from the fitted WOE representation of a feature."""
        return calculate_feature_gini(bin_ids, woe_map, y)

    @staticmethod
    def _calculate_monotonicity(
        feature_type: FeatureType,
        woe_series: pd.Series,
    ) -> MonotonicityResult:
        """Run the appropriate monotonicity check for the fitted feature type."""
        if feature_type == "numeric":
            return check_numeric_monotonicity(woe_series)
        return check_monotonicity(woe_series)

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
    ) -> FeatureComputationResult:
        """Fit one feature end-to-end and return its computed artifacts."""
        IVWOEFilter._warn_missing_special_codes(col_name, series, specials)
        feature_type, bin_config = IVWOEFilter._fit_feature_binning(
            series,
            y,
            n_bins,
            binning_method,
            min_bin_pct,
            specials,
            random_state,
            tree_criterion,
            tree_max_depth,
            tree_min_samples_leaf,
            tree_min_samples_split,
        )

        bin_ids = apply_bins(series, bin_config)
        bin_ids = IVWOEFilter._apply_min_bin_pct_merge(bin_ids, bin_config, min_bin_pct)

        stats = IVWOEFilter._build_feature_stats(feature_type, series, y, bin_config, bin_ids)
        woe_series, iv_bin_series, iv_value = calculate_woe_iv(stats)
        stats = stats.assign(woe=woe_series, iv_bin=iv_bin_series)

        woe_map = {int(bin_id): float(woe_value) for bin_id, woe_value in woe_series.items()}
        gini_value = IVWOEFilter._calculate_feature_gini(y, bin_ids, woe_map)
        monotonicity = IVWOEFilter._calculate_monotonicity(feature_type, woe_series)

        return FeatureComputationResult(
            column=col_name,
            iv=iv_value,
            gini=gini_value,
            bin_config=bin_config,
            stats=stats,
            woe_map=woe_map,
            monotonicity=monotonicity,
            feature_type=feature_type,
        )

    def _resolve_parallel_prefer(self) -> str | None:
        """Resolve the joblib backend preference from the estimator settings."""
        if self.parallel_backend == "threads":
            return "threads"
        if self.parallel_backend == "processes":
            return None
        return "threads" if self.binning_method in {"chi_merge", "tree"} else None

    def _log_parallel_strategy(self, parallel_prefer: str | None) -> None:
        """Log the selected parallel execution strategy."""
        if not self.verbose or self.n_jobs == 1:
            return

        backend_label = "thread-based" if parallel_prefer == "threads" else "process-based"
        if self.parallel_backend == "auto":
            logger.info(
                "Using %s parallelism for %s binning.",
                backend_label,
                self.binning_method,
            )
            return

        logger.info(
            "Using %s parallelism as requested by parallel_backend=%r.",
            backend_label,
            self.parallel_backend,
        )

    def _log_chi_merge_completion(self, feature: str, bin_config: BinConfig) -> None:
        """Log ChiMerge completion details for one fitted feature."""
        if not self.verbose or bin_config.get("binning_method") != "chi_merge":
            return

        seed_bins = bin_config.get("chi_merge_seed_bin_count", 0)
        final_bins = bin_config.get("chi_merge_final_bin_count", 0)
        unique_values = bin_config.get("chi_merge_unique_value_count", 0)
        used_prebinning = bin_config.get("chi_merge_used_prebinning", False)
        logger.info(
            "ChiMerge complete for feature '%s': %d final bins from %d seed bins "
            "(unique values=%d%s).",
            feature,
            final_bins,
            seed_bins,
            unique_values,
            ", quantile pre-binning applied" if used_prebinning else "",
        )

    @staticmethod
    def _validate_input_columns(X: pd.DataFrame, columns: list[str], context: str) -> None:
        """Raise a clear error when required columns are missing from the input frame."""
        missing_columns = [column for column in columns if column not in X.columns]
        if missing_columns:
            raise KeyError(f"Missing columns for {context}: {missing_columns}")

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> IVWOEFilter:
        """Fit binning, WOE, IV, Gini, and audit outputs for all input columns."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        y_arr = prepare_binary_target(y)
        validate_estimator_parameters(
            binning_method=self.binning_method,
            n_bins=self.n_bins,
            min_iv=self.min_iv,
            min_gini=self.min_gini,
            max_iv_for_leakage=self.max_iv_for_leakage,
            min_bin_pct=self.min_bin_pct,
            psi_thresholds=self.psi_thresholds,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            special_codes=self.special_codes,
            tree_criterion=self.tree_criterion,
            tree_max_depth=self.tree_max_depth,
            tree_min_samples_leaf=self.tree_min_samples_leaf,
            tree_min_samples_split=self.tree_min_samples_split,
        )

        columns = X.columns.tolist()
        parallel_prefer = self._resolve_parallel_prefer()

        if self.verbose:
            logger.info("Fitting %d features...", len(columns))
        self._log_parallel_strategy(parallel_prefer)

        results = Parallel(n_jobs=self.n_jobs, prefer=parallel_prefer)(
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

        self.binning_: dict[str, BinConfig] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.iv_table_data_: dict[str, float] = {}
        self.gini_table_data_: dict[str, float] = {}
        self._per_feature_stats: dict[str, pd.DataFrame] = {}
        self.reference_distributions_: dict[str, pd.Series] = {}
        self.monotonicity_report_: dict[str, MonotonicityResult] = {}
        self.feature_types_: dict[str, FeatureType] = {}
        self.leakage_flags_: dict[str, bool] = {}

        for result in results:
            self.binning_[result.column] = result.bin_config
            self.woe_maps_[result.column] = result.woe_map
            self.iv_table_data_[result.column] = result.iv
            self.gini_table_data_[result.column] = result.gini
            self._per_feature_stats[result.column] = result.stats
            self.reference_distributions_[result.column] = result.stats["count"]
            self.monotonicity_report_[result.column] = result.monotonicity
            self.feature_types_[result.column] = result.feature_type
            self.leakage_flags_[result.column] = result.iv > self.max_iv_for_leakage
            self._log_chi_merge_completion(result.column, result.bin_config)

        self.iv_table_ = pd.DataFrame(
            {
                "IV": self.iv_table_data_,
                "Gini": self.gini_table_data_,
            }
        ).sort_values("IV", ascending=False)

        self.selected_features_ = self.iv_table_[
            (self.iv_table_["IV"] >= self.min_iv)
            & (self.iv_table_["Gini"] >= self.min_gini)
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
        self._validate_input_columns(X, cols_to_process, "transform")
        X_out = X.loc[:, cols_to_process].copy()

        if not self.encode:
            return X_out

        for col in cols_to_process:
            bin_ids = apply_bins(X_out[col], self.binning_[col])
            bin_ids = remap_bin_ids(bin_ids, self.binning_[col])
            X_out[col] = (
                pd.Series(bin_ids, index=X_out.index).map(self.woe_maps_[col]).fillna(0.0)
            )

        return X_out

    def calculate_psi(self, X: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Calculate PSI against a new dataset."""
        check_is_fitted(self, ["binning_", "reference_distributions_", "selected_features_"])

        cols_to_process = (
            self.selected_features_ if self.drop_unselected else list(self.binning_.keys())
        )
        missing_columns = [column for column in cols_to_process if column not in X.columns]
        available_columns = [column for column in cols_to_process if column in X.columns]
        low, high = self.psi_thresholds
        psi_records: list[dict[str, float | str]] = []

        if missing_columns:
            logger.warning(
                "The following fitted features are missing from PSI input and will be marked as missing: %s",
                missing_columns,
            )

        for col in available_columns:
            bin_ids = apply_bins(X[col], self.binning_[col])
            bin_ids = remap_bin_ids(bin_ids, self.binning_[col])
            actual_counts = pd.Series(bin_ids).value_counts()
            expected_counts = self.reference_distributions_[col]
            psi_total, _ = calculate_psi_from_counts(expected_counts, actual_counts)

            status = "Stable"
            if psi_total >= high:
                status = "Significant Shift"
            elif psi_total >= low:
                status = "Minor Shift"

            psi_records.append({"feature": col, "PSI": psi_total, "status": status})

        for col in missing_columns:
            psi_records.append({"feature": col, "PSI": np.nan, "status": "Missing in Input"})

        df_psi = pd.DataFrame(psi_records)
        if not df_psi.empty:
            df_psi = df_psi.sort_values("PSI", ascending=False, na_position="last").reset_index(
                drop=True
            )

        if save and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            df_psi.to_csv(os.path.join(self.output_dir, "stability_report.csv"), index=False)

        return df_psi

    def _save_artifacts(self) -> None:
        """Persist fit-time audit artifacts to disk."""
        self.iv_table_.to_csv(os.path.join(self.output_dir, "iv_summary.csv"))

        bin_stats = pd.concat(
            [df.assign(feature=feature) for feature, df in self._per_feature_stats.items()]
        ).reset_index()
        bin_stats.to_csv(os.path.join(self.output_dir, "bin_stats.csv"), index=False)

        audit_rows = [
            {
                "feature": col,
                "type": self.feature_types_[col],
                "binning_method": self.binning_[col]["binning_method"],
                "IV": self.iv_table_data_[col],
                "Gini": self.gini_table_data_[col],
                "is_monotonic": self.monotonicity_report_[col]["is_monotonic"],
                "direction": self.monotonicity_report_[col]["direction"],
                "leakage_flag": self.leakage_flags_[col],
            }
            for col in self.iv_table_.index
        ]
        pd.DataFrame(audit_rows).to_csv(
            os.path.join(self.output_dir, "feature_audit.csv"),
            index=False,
        )

    def plot_feature_audit(
        self,
        feature: str,
        *,
        round_digits: int = 2,
        figsize: tuple[float, float] = (12, 6),
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Render the audit plot for one fitted feature."""
        check_is_fitted(self, ["_per_feature_stats", "iv_table_"])
        if feature not in self._per_feature_stats:
            raise KeyError(f"Unknown feature {feature!r}.")

        plot_df = prepare_plot_frame(self._per_feature_stats[feature].copy(), round_digits)
        iv_value = float(self.iv_table_.loc[feature, "IV"])
        gini_value = float(self.iv_table_.loc[feature, "Gini"])
        return render_feature_plot(
            plot_df,
            feature,
            iv_value,
            gini_value,
            figsize=figsize,
        )

    @staticmethod
    def _sanitize_plot_filename(feature: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in feature).strip("_")
        return safe or "feature"

    def save_feature_plot(
        self,
        output_dir: str | None = None,
        *,
        feature: str = "all",
        round_digits: int = 2,
        figsize: tuple[float, float] = (12, 6),
    ) -> str | list[str]:
        """Save one fitted feature plot or all fitted feature plots."""
        check_is_fitted(self, ["_per_feature_stats", "iv_table_"])

        if output_dir is None:
            if not self.output_dir:
                raise ValueError(
                    "output_dir must be provided when the transformer has no output_dir."
                )
            output_dir = os.path.join(self.output_dir, "plots")

        os.makedirs(output_dir, exist_ok=True)

        features_to_plot = list(self.iv_table_.index) if feature == "all" else [feature]
        saved_paths: list[str] = []
        for feature_name in features_to_plot:
            if feature_name not in self._per_feature_stats:
                raise KeyError(f"Unknown feature {feature_name!r}.")

            plot_df = prepare_plot_frame(
                self._per_feature_stats[feature_name].copy(),
                round_digits,
            )
            iv_value = float(self.iv_table_.loc[feature_name, "IV"])
            gini_value = float(self.iv_table_.loc[feature_name, "Gini"])
            output_path = os.path.join(
                output_dir,
                f"{self._sanitize_plot_filename(feature_name)}.png",
            )
            saved_paths.append(
                save_rendered_feature_plot(
                    plot_df,
                    feature_name,
                    iv_value,
                    gini_value,
                    output_path,
                    figsize=figsize,
                )
            )

        return saved_paths if feature == "all" else saved_paths[0]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the fitted output feature names for sklearn compatibility."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_ if self.drop_unselected else list(self.binning_.keys())
