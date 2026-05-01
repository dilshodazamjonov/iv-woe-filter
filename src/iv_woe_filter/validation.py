"""Validation helpers for estimator configuration and binary targets."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .binning import SUPPORTED_BINNING_METHODS


ParallelBackend = Literal["auto", "threads", "processes"]
SUPPORTED_PARALLEL_BACKENDS = frozenset({"auto", "threads", "processes"})
SUPPORTED_TREE_CRITERIA = frozenset({"gini", "entropy", "log_loss"})


def prepare_binary_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """Validate and normalize a binary target to a dense integer array."""
    target = pd.Series(np.asarray(y).ravel())
    if target.isna().any():
        raise ValueError("Target contains NaN values")

    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        raise ValueError("Target values must be numeric 0 or 1.")

    unique_values = np.unique(numeric_target.to_numpy())
    if len(unique_values) > 2:
        raise ValueError(
            f"Target must be binary (0/1), got {len(unique_values)} unique values: {unique_values}"
        )
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"Target values must be 0 or 1, got: {unique_values}")
    if len(unique_values) < 2:
        raise ValueError("Target must contain both classes 0 and 1.")

    return numeric_target.to_numpy(dtype=int)


def validate_estimator_parameters(
    *,
    binning_method: str,
    n_bins: int,
    min_iv: float,
    min_gini: float,
    max_iv_for_leakage: float,
    min_bin_pct: float | None,
    psi_thresholds: tuple[float, float],
    n_jobs: int,
    parallel_backend: ParallelBackend,
    special_codes: dict[str, list[Any]],
    tree_criterion: str,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int | float | None,
    tree_min_samples_split: int | float,
) -> None:
    """Validate estimator parameters before fitting."""
    _validate_general_parameters(
        binning_method=binning_method,
        n_bins=n_bins,
        min_iv=min_iv,
        min_gini=min_gini,
        max_iv_for_leakage=max_iv_for_leakage,
        min_bin_pct=min_bin_pct,
        psi_thresholds=psi_thresholds,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        special_codes=special_codes,
    )
    _validate_tree_parameters(
        tree_criterion=tree_criterion,
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=tree_min_samples_leaf,
        tree_min_samples_split=tree_min_samples_split,
    )


def _validate_general_parameters(
    *,
    binning_method: str,
    n_bins: int,
    min_iv: float,
    min_gini: float,
    max_iv_for_leakage: float,
    min_bin_pct: float | None,
    psi_thresholds: tuple[float, float],
    n_jobs: int,
    parallel_backend: ParallelBackend,
    special_codes: dict[str, list[Any]],
) -> None:
    """Validate core estimator settings shared across all binning strategies."""
    if binning_method not in SUPPORTED_BINNING_METHODS:
        raise ValueError(
            f"binning_method must be one of {sorted(SUPPORTED_BINNING_METHODS)}, "
            f"got {binning_method!r}."
        )
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}.")
    if min_iv < 0:
        raise ValueError(f"min_iv must be >= 0, got {min_iv!r}.")
    if not 0 <= min_gini <= 1:
        raise ValueError(f"min_gini must be between 0 and 1, got {min_gini!r}.")
    if max_iv_for_leakage < 0:
        raise ValueError(
            f"max_iv_for_leakage must be >= 0, got {max_iv_for_leakage!r}."
        )
    if min_bin_pct is not None and not 0 < min_bin_pct < 1:
        raise ValueError(
            f"min_bin_pct must be None or a float between 0 and 1, got {min_bin_pct!r}."
        )
    if not isinstance(psi_thresholds, tuple) or len(psi_thresholds) != 2:
        raise ValueError(
            f"psi_thresholds must be a tuple of two floats, got {psi_thresholds!r}."
        )
    low, high = psi_thresholds
    if not 0 <= low <= high:
        raise ValueError(
            f"psi_thresholds must satisfy 0 <= low <= high, got {psi_thresholds!r}."
        )
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError(f"n_jobs must be a non-zero integer, got {n_jobs!r}.")
    if parallel_backend not in SUPPORTED_PARALLEL_BACKENDS:
        raise ValueError(
            "parallel_backend must be one of ['auto', 'processes', 'threads'], "
            f"got {parallel_backend!r}."
        )
    if not isinstance(special_codes, dict):
        raise ValueError(
            f"special_codes must be a dict or None, got {type(special_codes).__name__}."
        )


def _validate_tree_parameters(
    *,
    tree_criterion: str,
    tree_max_depth: int | None,
    tree_min_samples_leaf: int | float | None,
    tree_min_samples_split: int | float,
) -> None:
    """Validate parameters specific to tree-based binning."""
    if tree_criterion not in SUPPORTED_TREE_CRITERIA:
        raise ValueError(
            "tree_criterion must be one of ['entropy', 'gini', 'log_loss'], "
            f"got {tree_criterion!r}."
        )
    if tree_max_depth is not None and (
        not isinstance(tree_max_depth, int) or tree_max_depth < 1
    ):
        raise ValueError(
            f"tree_max_depth must be None or an integer >= 1, got {tree_max_depth!r}."
        )
    if tree_min_samples_leaf is not None:
        _validate_tree_min_samples_leaf(tree_min_samples_leaf)
    _validate_tree_min_samples_split(tree_min_samples_split)


def _validate_tree_min_samples_leaf(tree_min_samples_leaf: int | float) -> None:
    """Validate the minimum leaf size parameter for tree binning."""
    if isinstance(tree_min_samples_leaf, int):
        if tree_min_samples_leaf < 1:
            raise ValueError(
                "tree_min_samples_leaf must be >= 1 when passed as an integer, "
                f"got {tree_min_samples_leaf!r}."
            )
        return
    if isinstance(tree_min_samples_leaf, float):
        if not 0 < tree_min_samples_leaf < 1:
            raise ValueError(
                "tree_min_samples_leaf must be between 0 and 1 when passed as a float, "
                f"got {tree_min_samples_leaf!r}."
            )
        return
    raise ValueError(
        "tree_min_samples_leaf must be None, int, or float, "
        f"got {type(tree_min_samples_leaf).__name__}."
    )


def _validate_tree_min_samples_split(tree_min_samples_split: int | float) -> None:
    """Validate the minimum split size parameter for tree binning."""
    if isinstance(tree_min_samples_split, int):
        if tree_min_samples_split < 2:
            raise ValueError(
                "tree_min_samples_split must be >= 2 when passed as an integer, "
                f"got {tree_min_samples_split!r}."
            )
        return
    if isinstance(tree_min_samples_split, float):
        if not 0 < tree_min_samples_split <= 1:
            raise ValueError(
                "tree_min_samples_split must be in (0, 1] when passed as a float, "
                f"got {tree_min_samples_split!r}."
            )
        return
    raise ValueError(
        "tree_min_samples_split must be an int or float, "
        f"got {type(tree_min_samples_split).__name__}."
    )
