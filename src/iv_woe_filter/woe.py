"""Weight of Evidence and Information Value utilities."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd


class MonotonicityResult(TypedDict):
    """Monotonicity audit result for an ordered WOE sequence."""

    is_monotonic: bool
    direction: Literal["increasing", "decreasing", "none"]


def safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safely divide two arrays, returning zero where the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def compute_aggregate_stats(bin_ids: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Calculate count, good, bad, and distribution statistics per bin."""
    unique_bins, inverse = np.unique(bin_ids, return_inverse=True)
    counts = np.bincount(inverse)
    bads = np.bincount(inverse, weights=y)
    goods = counts - bads

    total_bads = bads.sum()
    total_goods = goods.sum()

    return pd.DataFrame(
        {
            "bin": unique_bins,
            "count": counts,
            "bad": bads,
            "good": goods,
            "bad_pct": safe_div(bads, total_bads),
            "good_pct": safe_div(goods, total_goods),
        }
    ).set_index("bin")


def calculate_woe_iv(
    stats: pd.DataFrame,
    eps: float = 1e-12,
) -> tuple[pd.Series, pd.Series, float]:
    """Compute WOE and IV values from aggregate bin statistics."""
    woe = np.log((stats["good_pct"] + eps) / (stats["bad_pct"] + eps))
    iv_bin = (stats["good_pct"] - stats["bad_pct"]) * woe
    return woe, iv_bin, float(iv_bin.sum())


def check_monotonicity(woe: pd.Series) -> MonotonicityResult:
    """Check whether an ordered WOE sequence is monotonic."""
    if len(woe) < 2:
        return {"is_monotonic": True, "direction": "none"}

    differences = np.diff(woe.to_numpy())
    is_increasing = bool(np.all(differences >= 0))
    is_decreasing = bool(np.all(differences <= 0))

    if is_increasing:
        return {"is_monotonic": True, "direction": "increasing"}
    if is_decreasing:
        return {"is_monotonic": True, "direction": "decreasing"}
    return {"is_monotonic": False, "direction": "none"}


def check_numeric_monotonicity(woe: pd.Series) -> MonotonicityResult:
    """Check monotonicity only across non-special numeric bins."""
    numeric_bins = woe[woe.index >= 0]
    return check_monotonicity(numeric_bins)
