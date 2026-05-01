"""
Binning logic for numeric and categorical feature transformation.

Binning Methods
---------------

1. Quantile Binning
--------------------
Divides the observed numeric distribution into bins with roughly equal
population counts.

Key features:
    - Robust against outliers.
    - Useful as a simple unsupervised baseline.

2. Tree-Based Binning
---------------------
A supervised binning technique that trains a
`sklearn.tree.DecisionTreeClassifier` on a single feature and reuses its
learned split points as numeric bin edges.

Key features:
    - Learns target-aware thresholds.
    - Supports tree depth and minimum leaf-size constraints.

3. ChiMerge Binning
---------------------
Starts from ordered seed bins and repeatedly merges the adjacent pair with
the smallest chi-square separation in the binary target until at most
`n_bins` intervals remain. For very high-cardinality numeric features, the
implementation first compresses the distribution into quantile-based seed
bins to keep runtime and memory usage under control.

Algorithm:
    - Each distinct value starts in its own bin when cardinality is moderate.
    - Very high-cardinality features are first compressed into ordered seed bins.
    - Calculate the chi-square score for each adjacent bin pair.
    - Merge the most statistically similar pair.
    - Repeat until at most `n_bins` intervals remain.
    - Return the numeric edges that define the final bins.

"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


SUPPORTED_BINNING_METHODS = {"chi_merge", "quantile", "tree"}
CHI_MERGE_MIN_PREBINS = 100
CHI_MERGE_PREBIN_MULTIPLIER = 20


def _finalize_numeric_bins(bins: list[float] | np.ndarray) -> np.ndarray:
    """Return sorted bin edges with open-ended outer boundaries."""
    bins_array = np.asarray(bins, dtype=float)
    bins_array = bins_array[~np.isnan(bins_array)]
    bins_array = np.unique(bins_array)

    if len(bins_array) < 2:
        return np.array([-np.inf, np.inf])

    bins_array[0], bins_array[-1] = -np.inf, np.inf
    return bins_array


def _fit_quantile_bins(clean_series: pd.Series, n_bins: int) -> np.ndarray:
    """Fit unsupervised numeric bins using equal-frequency quantiles."""
    try:
        _, bins = pd.qcut(clean_series, q=n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        # Fallback to standard cut if qcut fails due to distribution issues.
        _, bins = pd.cut(
            clean_series, bins=min(n_bins, clean_series.nunique()), retbins=True
        )

    return _finalize_numeric_bins(bins)


def _build_chi_merge_seed_bins(
    x: np.ndarray,
    y_clean: np.ndarray,
    n_bins: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Build ordered ChiMerge seed bins, pre-binning when cardinality is very high."""
    unique_value_count = int(np.unique(x).size)
    max_prebins = max(CHI_MERGE_MIN_PREBINS, n_bins * CHI_MERGE_PREBIN_MULTIPLIER)

    metadata = {
        "chi_merge_unique_value_count": unique_value_count,
        "chi_merge_max_prebins": max_prebins,
        "chi_merge_used_prebinning": unique_value_count > max_prebins,
    }

    if not metadata["chi_merge_used_prebinning"]:
        grouped = (
            pd.DataFrame({"value": x, "target": y_clean})
            .groupby("value", sort=True)["target"]
            .agg(count="size", bad="sum")
            .reset_index()
        )
        grouped["lower"] = grouped["value"]
        grouped["upper"] = grouped["value"]
    else:
        seed_edges = _fit_quantile_bins(pd.Series(x), max_prebins)
        seed_ids = pd.cut(x, bins=seed_edges, labels=False, include_lowest=True)
        grouped = (
            pd.DataFrame({"value": x, "target": y_clean, "seed_id": seed_ids})
            .groupby("seed_id", sort=True)
            .agg(lower=("value", "min"), upper=("value", "max"), count=("target", "size"), bad=("target", "sum"))
            .reset_index(drop=True)
        )

    grouped["good"] = grouped["count"] - grouped["bad"]
    metadata["chi_merge_seed_bin_count"] = int(len(grouped))

    intervals = [
        {
            "lower": float(row.lower),
            "upper": float(row.upper),
            "good": float(row.good),
            "bad": float(row.bad),
        }
        for row in grouped.itertuples(index=False)
    ]
    return intervals, metadata


def _calculate_adjacent_chi_square(
    left_good: float,
    left_bad: float,
    right_good: float,
    right_bad: float,
) -> float:
    """Return the chi-square statistic for two adjacent binary-target bins."""

    # saving into observed table of 2x2 
    observed = np.array(
        [
            [left_good, left_bad], 
            [right_good, right_bad]
        ],
        dtype=float,
    )
    # summing all the observations in left, right and total
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    if total == 0:
        return 0.0

    # using matrix multiplication for better performance    
    expected = row_totals @ col_totals / total
    chi_square = np.divide(
        (observed - expected) ** 2,
        expected,
        out=np.zeros_like(observed, dtype=float),
        where=(expected > 0),
    )
    return float(chi_square.sum())


def _fit_chi_merge_binning(
    clean_series: pd.Series,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit supervised numeric bins by merging adjacent low-separation intervals."""
    valid_mask = clean_series.notna().to_numpy()
    x = clean_series[valid_mask].to_numpy(dtype=float)
    y_clean = np.asarray(y)[valid_mask]

    # checking if the binning if possible if not return [-inf, inf]
    if len(x) == 0 or len(np.unique(x)) <= 1 or len(np.unique(y_clean)) < 2 or n_bins <= 1:
        return np.array([-np.inf, np.inf]), {
            "chi_merge_unique_value_count": int(np.unique(x).size),
            "chi_merge_max_prebins": max(CHI_MERGE_MIN_PREBINS, n_bins * CHI_MERGE_PREBIN_MULTIPLIER),
            "chi_merge_used_prebinning": False,
            "chi_merge_seed_bin_count": int(np.unique(x).size),
            "chi_merge_final_bin_count": 1,
        }

    intervals, metadata = _build_chi_merge_seed_bins(x, y_clean, n_bins)

    while len(intervals) > n_bins:
        chi_values = [
            _calculate_adjacent_chi_square(
                intervals[idx]["good"],
                intervals[idx]["bad"],
                intervals[idx + 1]["good"],
                intervals[idx + 1]["bad"],
            )
            for idx in range(len(intervals) - 1)
        ]
        merge_idx = int(np.argmin(chi_values))
        intervals[merge_idx : merge_idx + 2] = [
            {
                "lower": intervals[merge_idx]["lower"],
                "upper": intervals[merge_idx + 1]["upper"],
                "good": intervals[merge_idx]["good"] + intervals[merge_idx + 1]["good"],
                "bad": intervals[merge_idx]["bad"] + intervals[merge_idx + 1]["bad"],
            }
        ]

    metadata["chi_merge_final_bin_count"] = len(intervals)

    boundaries = [-np.inf]
    for left_interval, right_interval in zip(intervals, intervals[1:]):
        boundaries.append((left_interval["upper"] + right_interval["lower"]) / 2.0)
    boundaries.append(np.inf)
    return np.asarray(boundaries, dtype=float), metadata


def _fit_tree_bins(
    clean_series: pd.Series,
    y: np.ndarray,
    n_bins: int,
    min_bin_pct: float | None,
    random_state: int | None,
    tree_criterion: str,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int | float | None,
    tree_min_samples_split: int | float,
) -> np.ndarray:
    """Fit supervised numeric bins from decision-tree split thresholds."""
    valid_mask = clean_series.notna().to_numpy()
    x = clean_series[valid_mask].to_numpy(dtype=float).reshape(-1, 1)
    y_clean = np.asarray(y)[valid_mask]

    if len(x) == 0 or len(np.unique(x)) <= 1 or len(np.unique(y_clean)) < 2 or n_bins <= 1:
        return np.array([-np.inf, np.inf])

    min_samples_leaf = tree_min_samples_leaf
    if min_samples_leaf is None:
        min_samples_leaf = min_bin_pct if min_bin_pct else 1

    tree = DecisionTreeClassifier(
        criterion=tree_criterion,
        max_depth=tree_max_depth,
        max_leaf_nodes=n_bins,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=tree_min_samples_split,
        random_state=random_state,
    )
    tree.fit(x, y_clean)

    split_mask = tree.tree_.children_left != tree.tree_.children_right
    thresholds = tree.tree_.threshold[split_mask]

    if len(thresholds) == 0:
        return np.array([-np.inf, np.inf])

    return np.concatenate(([-np.inf], np.sort(np.unique(thresholds)), [np.inf]))


def remap_bin_ids(bin_ids: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Apply a fitted post-binning merge map when one exists."""
    bin_id_map = config.get("bin_id_map")
    if not bin_id_map:
        return bin_ids

    return np.fromiter(
        (int(bin_id_map.get(int(bin_id), int(bin_id))) for bin_id in bin_ids),
        dtype=int,
        count=len(bin_ids),
    )


def fit_numeric_bins(
    series: pd.Series,
    n_bins: int,
    special_codes: list[int | float] | None = None,
    *,
    binning_method: str = "quantile",
    y: np.ndarray | None = None,
    min_bin_pct: float | None = None,
    random_state: int | None = 42,
    tree_criterion: str = "gini",
    tree_max_depth: int | None = None,
    tree_min_samples_leaf: int | float | None = None,
    tree_min_samples_split: int | float = 2,
) -> dict[str, Any]:
    """Fit numeric bins while isolating special codes.

    Parameters
    ----------
    series : pd.Series
        Numeric input data to be binned.
    n_bins : int
        Target number of bins for the numeric split.
    special_codes : list of (int or float) or None
        Values to be treated as independent bins (e.g., error codes, missing flags).
    binning_method : {"quantile", "chi_merge", "tree"}, default="quantile"
        Numeric binning strategy.
    y : np.ndarray or None
        Binary target array. Required when `binning_method` is `"tree"` or `"chi_merge"`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'is_numeric', 'special_codes', and the 'bins' boundaries.
    """
    binning_data = {
        "is_numeric": True,
        "special_codes": special_codes or [],
        "binning_method": binning_method,
        "bins": None,
    }

    special_mask = series.isin(special_codes or [])
    clean_series = pd.to_numeric(series[~special_mask], errors="coerce")

    if clean_series.dropna().empty:
        binning_data["bins"] = np.array([-np.inf, np.inf])
        return binning_data

    if binning_method == "quantile":
        bins = _fit_quantile_bins(clean_series.dropna(), n_bins)
    elif binning_method == "chi_merge":
        if y is None:
            raise ValueError("Target y is required when binning_method='chi_merge'.")
        clean_y = np.asarray(y)[(~special_mask).to_numpy()]
        bins, chi_merge_metadata = _fit_chi_merge_binning(clean_series, clean_y, n_bins)
        binning_data.update(chi_merge_metadata)
    elif binning_method == "tree":
        if y is None:
            raise ValueError("Target y is required when binning_method='tree'.")
        clean_y = np.asarray(y)[(~special_mask).to_numpy()]
        bins = _fit_tree_bins(
            clean_series,
            clean_y,
            n_bins,
            min_bin_pct,
            random_state,
            tree_criterion,
            tree_max_depth,
            tree_min_samples_leaf,
            tree_min_samples_split,
        )
    else:
        raise ValueError(
            f"binning_method must be one of {sorted(SUPPORTED_BINNING_METHODS)}, "
            f"got {binning_method!r}."
        )

    binning_data["bins"] = bins

    return binning_data


def fit_categorical_bins(
    series: pd.Series,
    special_codes: list[Any] | None = None,
) -> dict[str, Any]:
    """Fit categorical mapping by identifying unique labels and isolating special codes.

    Parameters
    ----------
    series : pd.Series
        Categorical or object-type input data.
    special_codes : list of Any or None
        Values to be treated as independent bins.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'is_numeric', 'special_codes', and 'categories' list.
    """
    s = series.fillna("missing").astype(str)

    binning_data = {
        "is_numeric": False,
        "special_codes": special_codes or [],
        "binning_method": "categorical",
        "categories": s.unique().tolist(),
    }
    return binning_data


def apply_bins(
    series: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply fitted binning rules to a new series to generate bin IDs.

    Parameters
    ----------
    series : pd.Series
        Input data to transform.
    config : dict[str, Any]
        Configuration dictionary generated during the fit phase.

    Returns
    -------
    np.ndarray
        Array of integer bin IDs. Special codes are mapped to negative IDs.
    """
    s = series.copy()
    output_bins = np.full(s.shape, -1, dtype=int)

    # Handle Special Codes
    special_mask = s.isin(config["special_codes"])
    if special_mask.any():
        mapping = {val: -(i + 2) for i, val in enumerate(config["special_codes"])}
        output_bins[special_mask] = s[special_mask].map(mapping)

    remaining_mask = ~special_mask
    if not remaining_mask.any():
        return output_bins

    # Handle Normal Data
    if config["is_numeric"]:
        numeric_s = pd.to_numeric(s[remaining_mask], errors="coerce")
        # pd.cut with labels=False returns float array with NaNs for out-of-bounds/NaNs
        bins = pd.cut(numeric_s, bins=config["bins"], labels=False, include_lowest=True)
        
        # Safely convert to int after replacing NaNs with -1
        output_bins[remaining_mask] = np.nan_to_num(bins, nan=-1).astype(int)
    else:
        cat_s = s[remaining_mask].fillna("missing").astype(str)
        lookup = {cat: i for i, cat in enumerate(config["categories"])}
        output_bins[remaining_mask] = np.array([lookup.get(x, -1) for x in cat_s])

    return output_bins


def merge_non_significant_bins(
    bin_ids: np.ndarray,
    min_pct: float,
    *,
    protect_special_bins: bool = True,
    return_mapping: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[int, int]]:
    """Merge bins that do not meet the minimum population threshold.

    Parameters
    ----------
    bin_ids : np.ndarray
        Array of bin IDs.
    min_pct : float
        Minimum percentage of total population required per bin.

    Returns
    -------
    np.ndarray
        Array of bin IDs with small bins merged into their nearest neighbor.
    """
    original_bin_ids = bin_ids.copy()
    unique_bins = np.unique(original_bin_ids)
    if len(unique_bins) <= 1:
        if return_mapping:
            return original_bin_ids, {int(bin_id): int(bin_id) for bin_id in unique_bins}
        return original_bin_ids

    total_count = len(original_bin_ids)
    new_bin_ids = original_bin_ids.copy()

    while True:
        active_bins = np.unique(new_bin_ids)
        candidate_bins = active_bins[active_bins >= 0] if protect_special_bins else active_bins
        if len(candidate_bins) <= 1:
            break

        counts = {int(bin_id): int((new_bin_ids == bin_id).sum()) for bin_id in candidate_bins}
        small_bins = [
            int(bin_id)
            for bin_id in candidate_bins
            if counts[int(bin_id)] / total_count < min_pct
        ]
        if not small_bins:
            break

        bin_id = min(small_bins, key=lambda value: (counts[value], value))
        neighbor_candidates = candidate_bins[candidate_bins != bin_id]
        if len(neighbor_candidates) == 0:
            break

        dist = np.abs(neighbor_candidates.astype(float) - float(bin_id))
        nearest_neighbor = int(neighbor_candidates[np.argmin(dist)])
        new_bin_ids[new_bin_ids == bin_id] = nearest_neighbor

    if not return_mapping:
        return new_bin_ids

    bin_id_map: dict[int, int] = {}
    for original_bin in unique_bins:
        final_bin = int(new_bin_ids[np.flatnonzero(original_bin_ids == original_bin)[0]])
        bin_id_map[int(original_bin)] = final_bin

    return new_bin_ids, bin_id_map

def get_bin_labels(config: dict[str, Any], series: pd.Series = None, bin_ids: np.ndarray = None) -> dict[int, str]:    
    """Converts bin configurations into human-readable range strings."""
    labels = {}
    if config["is_numeric"]:
        for i, val in enumerate(config["special_codes"]):
            labels[-(i + 2)] = f"Special: {val}"
        bins = config["bins"]
        bin_id_map = config.get("bin_id_map")
        if bin_id_map:
            merged_numeric_bins: dict[int, list[int]] = {}
            for original_bin, final_bin in bin_id_map.items():
                if original_bin < 0 or final_bin < 0:
                    continue
                merged_numeric_bins.setdefault(int(final_bin), []).append(int(original_bin))

            for final_bin, original_bins in merged_numeric_bins.items():
                left_idx = min(original_bins)
                right_idx = max(original_bins) + 1
                labels[final_bin] = f"[{bins[left_idx]}, {bins[right_idx]})"
        else:
            for i in range(len(bins) - 1):
                labels[i] = f"[{bins[i]}, {bins[i+1]})"
        labels[-1] = "Missing/Other"
    else:
        s_str = series.fillna("missing").astype(str)
        temp_df = pd.DataFrame({"label": s_str, "bid": bin_ids})
        labels = (
            temp_df.groupby("bid")["label"]
            .unique()
            .apply(lambda x: ", ".join(sorted(x)))
            .to_dict()
        )
    return labels
