# iv_filter.py
import os
import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Setup module logger
logger = logging.getLogger("iv_filter")
logger.setLevel(logging.INFO)


class IVFilter:
    def __init__(
        self,
        target_col: Optional[str] = None,
        n_bins: int = 10,
        min_iv: float = 0.01,
        max_iv_for_leakage: float = 0.5,
        min_bin_pct: Optional[float] = None,
        verbose: bool = True,
        save_bin_level_stats: bool = True,
        output_dir: Optional[str] = None,
        n_jobs: int = -1 # Parallel jobs for processing features
    ):
        """
        Params:
            target_col: optional name of the target
            n_bins: quantile bins for numeric variables
            min_iv: threshold to keep a feature
            max_iv_for_leakage: flag features with IV >= this value
            min_bin_pct: optional minimum bin fraction check
            verbose: print progress
            save_bin_level_stats: save per-bin stats
            output_dir: directory where outputs will be saved
        """
        self.target_col = target_col
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_iv_for_leakage = max_iv_for_leakage
        self.min_bin_pct = min_bin_pct
        self.verbose = verbose
        self.save_bin_level_stats = save_bin_level_stats
        self.output_dir = output_dir
        self.n_jobs = n_jobs

        self.iv_table_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[list] = None
        self._per_feature_stats: Dict[str, pd.DataFrame] = {}

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _process_column_worker(col_name, series, y_arr, n_bins, min_bin_pct):
        """
        This is the function that runs on each CPU core.
        It is STATIC so it doesn't have to carry the whole 'self' object.
        """
        # Determine if numeric or categorical
        if pd.api.types.is_numeric_dtype(series):
            # Numeric Logic
            try:
                bins = pd.qcut(series, q=n_bins, duplicates="drop")
            except Exception:
                bins = pd.cut(series, bins=n_bins)
            cat = pd.Categorical(bins)
            codes = cat.codes
        else:
            # Categorical Logic
            s = series.fillna("missing").astype(object)
            cat = pd.Categorical(s)
            codes = cat.codes

        # Handle NaNs in codes
        if (codes == -1).any():
            nan_mask = codes == -1
            codes = codes.copy()
            codes[nan_mask] = codes.max() + 1

        stats = IVFilter._compute_group_stats_from_codes(codes, y_arr)

        if min_bin_pct is not None:
            total = stats["count"].sum()
            stats["_small_bin_flag"] = ((stats["count"] / total) < min_bin_pct).astype(int)

        iv_val, woe, iv_bin = IVFilter._iv_from_stats(stats)
        stats = stats.assign(woe=woe, iv_bin=iv_bin)

        if not pd.api.types.is_numeric_dtype(series):
            stats["_categories"] = list(cat.categories)

        return col_name, float(iv_val), stats
    
    @staticmethod
    def _safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

    @staticmethod
    def _compute_group_stats_from_codes(codes, y_arr):
        codes = np.asarray(codes)
        y_arr = np.asarray(y_arr).astype(int)

        uniq, inv = np.unique(codes, return_inverse=True)
        counts = np.bincount(inv)
        bads = np.bincount(inv, weights=y_arr)
        goods = counts - bads

        total_bad = bads.sum()
        total_good = goods.sum()

        bad_pct = IVFilter._safe_div(bads, total_bad)
        good_pct = IVFilter._safe_div(goods, total_good)

        df = pd.DataFrame(
            {
                "group": uniq,
                "count": counts,
                "bad": bads,
                "good": goods,
                "bad_pct": bad_pct,
                "good_pct": good_pct,
            }
        ).set_index("group")

        return df

    @staticmethod
    def _woe_from_stats(stats_df):
        eps = 1e-12
        return np.log((stats_df["good_pct"] + eps) / (stats_df["bad_pct"] + eps))

    @staticmethod
    def _iv_from_stats(stats_df):
        woe = IVFilter._woe_from_stats(stats_df)
        iv_bin = (stats_df["good_pct"] - stats_df["bad_pct"]) * woe
        return iv_bin.sum(), woe, iv_bin

    # -------------------------
    # Main API
    # -------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):

        if self.output_dir is None:
            raise ValueError("output_dir must be provided externally.")

        os.makedirs(self.output_dir, exist_ok=True)

        ivs = {}
        stats_store = {}
        y_arr = np.asarray(y)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_column_worker)(
                col, X[col], y_arr, self.n_bins, self.min_bin_pct
            ) for col in X.columns
        )

        for col_name, iv_val, stats in results:
            ivs[col_name] = iv_val
            stats_store[col_name] = stats
            
        iv_df = pd.DataFrame.from_dict(ivs, orient="index", columns=["IV"]).sort_values("IV", ascending=False)
        self.iv_table_ = iv_df
        self.selected_features_ = iv_df[iv_df["IV"] >= self.min_iv].index.tolist()
        self._per_feature_stats = stats_store

        # -------------------------
        # Save outputs
        # -------------------------
        iv_df.to_csv(os.path.join(self.output_dir, "iv_table.csv"))

        pd.Series(self.selected_features_, name="selected_feature") \
            .to_frame().to_csv(os.path.join(self.output_dir, "iv_selected_features.csv"), index=False)

        iv_df[iv_df["IV"] < self.min_iv] \
            .to_csv(os.path.join(self.output_dir, "iv_dropped_features.csv"))

        # Feature summary
        summaries = []
        for col, stats in stats_store.items():
            summaries.append({
                "feature": col,
                "IV": ivs[col],
                "num_bins": len(stats),
                "max_woe": stats["woe"].max(),
                "min_woe": stats["woe"].min(),
                "mean_woe": stats["woe"].mean(),
                "max_bad_pct": stats["bad_pct"].max(),
                "mean_bad_pct": stats["bad_pct"].mean(),
                "max_iv_bin": stats["iv_bin"].max(),
                "perfect_bin_sep": int((stats["bad_pct"] == 0).any() or (stats["good_pct"] == 0).any()),
                "small_bin_exists": int("_small_bin_flag" in stats and stats["_small_bin_flag"].any())
            })

        pd.DataFrame(summaries) \
            .sort_values("IV", ascending=False) \
            .to_csv(os.path.join(self.output_dir, "iv_feature_summaries.csv"), index=False)

        # Bin-level stats
        if self.save_bin_level_stats:
            long_rows = []
            for col, stats in stats_store.items():
                df = stats.reset_index().rename(columns={"group": "bin_id"})
                df["feature"] = col
                long_rows.append(df[["feature", "bin_id", "count", "bad", "good", "bad_pct", "good_pct", "woe", "iv_bin"]])

            pd.concat(long_rows, ignore_index=True) \
                .to_csv(os.path.join(self.output_dir, "iv_bin_level_stats.csv"), index=False)

        # Leakage flags
        leakage = {}
        for col, iv_val in ivs.items():
            flags = []
            if iv_val >= self.max_iv_for_leakage:
                flags.append("high_iv")
            stats = stats_store[col]
            if (stats["bad_pct"] == 0).any() or (stats["good_pct"] == 0).any():
                flags.append("perfect_sep")
            leakage[col] = ";".join(flags)

        pd.Series(leakage, name="leakage_flags") \
            .to_frame().to_csv(os.path.join(self.output_dir, "iv_leakage_flags.csv"))

        if self.verbose:
            logger.info(f"Saved IV outputs to {self.output_dir}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("IVFilter not fitted.")

        sel = [c for c in self.iv_table_.index if c in self.selected_features_]
        return X.reindex(columns=sel, fill_value=0)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)