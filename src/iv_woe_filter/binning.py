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

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_BINNING_METHODS = frozenset({"chi_merge", "quantile", "tree"})
CHI_MERGE_MIN_PREBINS = 100
CHI_MERGE_PREBIN_MULTIPLIER = 20


class NumericBinConfig(TypedDict, total=False):
    """Typed numeric binning configuration stored after fitting."""

    is_numeric: Literal[True]
    special_codes: list[Any]
    binning_method: Literal["quantile", "chi_merge", "tree"]
    bins: np.ndarray
    bin_id_map: dict[int, int]
    chi_merge_unique_value_count: int
    chi_merge_max_prebins: int
    chi_merge_used_prebinning: bool
    chi_merge_seed_bin_count: int
    chi_merge_final_bin_count: int


class CategoricalBinConfig(TypedDict, total=False):
    """Typed categorical binning configuration stored after fitting."""

    is_numeric: Literal[False]
    special_codes: list[Any]
    binning_method: Literal["categorical"]
    categories: list[str]
    bin_id_map: dict[int, int]


BinConfig = NumericBinConfig | CategoricalBinConfig


@dataclass(frozen=True)
class ChiMergeInterval:
    """Ordered interval used internally by the ChiMerge algorithm."""

    lower: float
    upper: float
    good: float
    bad: float


def _finalize_numeric_bins(bins: list[float] | np.ndarray) -> np.ndarray:
    """Return sorted numeric bin edges with open outer boundaries."""
    bins_array = np.asarray(bins, dtype=float)
    bins_array = bins_array[~np.isnan(bins_array)]
    bins_array = np.unique(bins_array)

    if len(bins_array) < 2:
        return np.array([-np.inf, np.inf])

    bins_array[0], bins_array[-1] = -np.inf, np.inf
    return bins_array


def _fit_quantile_bins(clean_series: pd.Series, n_bins: int) -> np.ndarray:
    """Fit equal-frequency numeric bins."""
    try:
        _, bins = pd.qcut(clean_series, q=n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        _, bins = pd.cut(
            clean_series,
            bins=min(n_bins, clean_series.nunique()),
            retbins=True,
        )

    return _finalize_numeric_bins(bins)


def _build_chi_merge_seed_bins(
    x: np.ndarray,
    y_clean: np.ndarray,
    n_bins: int,
) -> tuple[list[ChiMergeInterval], dict[str, int | bool]]:
    """Build ordered ChiMerge seed bins and pre-bin when cardinality is high."""
    unique_value_count = int(np.unique(x).size)
    max_prebins = max(CHI_MERGE_MIN_PREBINS, n_bins * CHI_MERGE_PREBIN_MULTIPLIER)
    used_prebinning = unique_value_count > max_prebins

    metadata: dict[str, int | bool] = {
        "chi_merge_unique_value_count": unique_value_count,
        "chi_merge_max_prebins": max_prebins,
        "chi_merge_used_prebinning": used_prebinning,
    }

    if not used_prebinning:
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
            .agg(
                lower=("value", "min"),
                upper=("value", "max"),
                count=("target", "size"),
                bad=("target", "sum"),
            )
            .reset_index(drop=True)
        )

    grouped["good"] = grouped["count"] - grouped["bad"]
    metadata["chi_merge_seed_bin_count"] = int(len(grouped))

    intervals = [
        ChiMergeInterval(
            lower=float(row.lower),
            upper=float(row.upper),
            good=float(row.good),
            bad=float(row.bad),
        )
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
    observed = np.array(
        [[left_good, left_bad], [right_good, right_bad]],
        dtype=float,
    )
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    if total == 0:
        return 0.0

    expected = row_totals @ col_totals / total
    chi_square = np.divide(
        (observed - expected) ** 2,
        expected,
        out=np.zeros_like(observed, dtype=float),
        where=expected > 0,
    )
    return float(chi_square.sum())


def _fit_chi_merge_binning(
    clean_series: pd.Series,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Fit supervised numeric bins with ChiMerge."""
    valid_mask = clean_series.notna().to_numpy()
    x = clean_series[valid_mask].to_numpy(dtype=float)
    y_clean = np.asarray(y)[valid_mask]

    if len(x) == 0 or len(np.unique(x)) <= 1 or len(np.unique(y_clean)) < 2 or n_bins <= 1:
        unique_value_count = int(np.unique(x).size)
        return np.array([-np.inf, np.inf]), {
            "chi_merge_unique_value_count": unique_value_count,
            "chi_merge_max_prebins": max(
                CHI_MERGE_MIN_PREBINS,
                n_bins * CHI_MERGE_PREBIN_MULTIPLIER,
            ),
            "chi_merge_used_prebinning": False,
            "chi_merge_seed_bin_count": unique_value_count,
            "chi_merge_final_bin_count": 1,
        }

    intervals, metadata = _build_chi_merge_seed_bins(x, y_clean, n_bins)

    while len(intervals) > n_bins:
        chi_values = [
            _calculate_adjacent_chi_square(
                intervals[index].good,
                intervals[index].bad,
                intervals[index + 1].good,
                intervals[index + 1].bad,
            )
            for index in range(len(intervals) - 1)
        ]
        merge_index = int(np.argmin(chi_values))
        merged_interval = ChiMergeInterval(
            lower=intervals[merge_index].lower,
            upper=intervals[merge_index + 1].upper,
            good=intervals[merge_index].good + intervals[merge_index + 1].good,
            bad=intervals[merge_index].bad + intervals[merge_index + 1].bad,
        )
        intervals[merge_index : merge_index + 2] = [merged_interval]

    metadata["chi_merge_final_bin_count"] = len(intervals)

    boundaries = [-np.inf]
    for left_interval, right_interval in zip(intervals, intervals[1:]):
        boundaries.append((left_interval.upper + right_interval.lower) / 2.0)
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


def remap_bin_ids(bin_ids: np.ndarray, config: BinConfig) -> np.ndarray:
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
) -> NumericBinConfig:
    """Fit numeric bins while isolating declared special codes."""
    binning_data: NumericBinConfig = {
        "is_numeric": True,
        "special_codes": list(special_codes or []),
        "binning_method": cast(Literal["quantile", "chi_merge", "tree"], binning_method),
        "bins": np.array([-np.inf, np.inf]),
    }

    special_mask = series.isin(special_codes or [])
    clean_series = pd.to_numeric(series[~special_mask], errors="coerce")

    if clean_series.dropna().empty:
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
) -> CategoricalBinConfig:
    """Fit categorical bins by recording the observed category labels."""
    categories = series.fillna("missing").astype(str).unique().tolist()
    return {
        "is_numeric": False,
        "special_codes": list(special_codes or []),
        "binning_method": "categorical",
        "categories": categories,
    }


def apply_bins(
    series: pd.Series,
    config: BinConfig,
) -> np.ndarray:
    """Apply a fitted binning configuration and return integer bin ids."""
    output_bins = np.full(series.shape, -1, dtype=int)

    special_mask = series.isin(config["special_codes"])
    if special_mask.any():
        special_mapping = {
            value: -(index + 2)
            for index, value in enumerate(config["special_codes"])
        }
        output_bins[special_mask] = series[special_mask].map(special_mapping)

    remaining_mask = ~special_mask
    if not remaining_mask.any():
        return output_bins

    if config["is_numeric"]:
        numeric_series = pd.to_numeric(series[remaining_mask], errors="coerce")
        bin_ids = pd.cut(
            numeric_series,
            bins=config["bins"],
            labels=False,
            include_lowest=True,
        )
        output_bins[remaining_mask] = bin_ids.fillna(-1).to_numpy(dtype=int)
        return output_bins

    categorical_series = series[remaining_mask].fillna("missing").astype(str)
    category_lookup = {
        category: index
        for index, category in enumerate(config["categories"])
    }
    mapped = categorical_series.map(category_lookup).fillna(-1).to_numpy(dtype=int)
    output_bins[remaining_mask] = mapped
    return output_bins


def merge_non_significant_bins(
    bin_ids: np.ndarray,
    min_pct: float,
    *,
    protect_special_bins: bool = True,
    return_mapping: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[int, int]]:
    """Merge bins smaller than the requested population share."""
    original_bin_ids = bin_ids.copy()
    unique_bins = np.unique(original_bin_ids)
    if len(unique_bins) <= 1:
        if return_mapping:
            return original_bin_ids, {int(bin_id): int(bin_id) for bin_id in unique_bins}
        return original_bin_ids

    total_count = len(original_bin_ids)
    merged_bin_ids = original_bin_ids.copy()

    while True:
        active_bins = np.unique(merged_bin_ids)
        candidate_bins = active_bins[active_bins >= 0] if protect_special_bins else active_bins
        if len(candidate_bins) <= 1:
            break

        counts = {int(bin_id): int((merged_bin_ids == bin_id).sum()) for bin_id in candidate_bins}
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

        distance = np.abs(neighbor_candidates.astype(float) - float(bin_id))
        nearest_neighbor = int(neighbor_candidates[np.argmin(distance)])
        merged_bin_ids[merged_bin_ids == bin_id] = nearest_neighbor

    if not return_mapping:
        return merged_bin_ids

    bin_id_map: dict[int, int] = {}
    for original_bin in unique_bins:
        original_positions = np.flatnonzero(original_bin_ids == original_bin)
        final_bin = int(merged_bin_ids[original_positions[0]])
        bin_id_map[int(original_bin)] = final_bin

    return merged_bin_ids, bin_id_map


def get_numeric_bin_labels(config: NumericBinConfig) -> dict[int, str]:
    """Return numeric bin labels for a fitted numeric configuration."""
    labels: dict[int, str] = {
        -(index + 2): f"Special: {value}"
        for index, value in enumerate(config["special_codes"])
    }
    bins = config["bins"]
    bin_id_map = config.get("bin_id_map")

    if bin_id_map:
        merged_numeric_bins: dict[int, list[int]] = {}
        for original_bin, final_bin in bin_id_map.items():
            if original_bin < 0 or final_bin < 0:
                continue
            merged_numeric_bins.setdefault(int(final_bin), []).append(int(original_bin))

        for final_bin, original_bins in merged_numeric_bins.items():
            left_index = min(original_bins)
            right_index = max(original_bins) + 1
            labels[final_bin] = f"[{bins[left_index]}, {bins[right_index]})"
    else:
        for index in range(len(bins) - 1):
            labels[index] = f"[{bins[index]}, {bins[index + 1]})"

    labels[-1] = "Missing/Other"
    return labels


def get_categorical_bin_labels(series: pd.Series, bin_ids: np.ndarray) -> dict[int, str]:
    """Return categorical bin labels derived from the observed grouped values."""
    label_frame = pd.DataFrame(
        {
            "label": series.fillna("missing").astype(str),
            "bin_id": bin_ids,
        }
    )
    return (
        label_frame.groupby("bin_id")["label"]
        .unique()
        .apply(lambda labels: ", ".join(sorted(labels)))
        .to_dict()
    )

