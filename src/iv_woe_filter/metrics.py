"""Evaluation metrics for Credit Risk models including Gini and PSI."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def calculate_gini(y_true: Any, y_score: Any) -> float:
    """Calculate the Gini coefficient using absolute discriminatory power.

    Parameters
    ----------
    y_true : array-like
        Binary target labels (0, 1).
    y_score : array-like
        Predicted scores or Weight of Evidence values.

    Returns
    -------
    float
        The Gini coefficient (2 * AUC - 1), forced to be positive.
    """
    try:
        y_t = np.asarray(y_true)
        y_s = np.asarray(y_score)
        
        mask = ~np.isnan(y_s)
        y_t, y_s = y_t[mask], y_s[mask]

        if len(np.unique(y_t)) < 2:
            return 0.0

        auc = roc_auc_score(y_t, y_s)
        return float(2 * max(auc, 1 - auc) - 1)
    except Exception as e:
        logger.error(f"Failed to calculate Gini: {e}")
        return 0.0


def calculate_feature_gini(
    bin_ids: np.ndarray, 
    woe_map: dict[int, float], 
    y: np.ndarray
) -> float:
    """Calculate Gini for a single feature based on its WOE transformation.

    Parameters
    ----------
    bin_ids : np.ndarray
        Array of bin indices for the feature.
    woe_map : dict[int, float]
        Dictionary mapping bin indices to WOE values.
    y : np.ndarray
        Binary target array.

    Returns
    -------
    float
        Feature-level Gini coefficient.
    """
    y_score = pd.Series(bin_ids).map(woe_map).fillna(0.0).values
    return calculate_gini(y, y_score)


def calculate_psi(
    expected_pct: np.ndarray | pd.Series, 
    actual_pct: np.ndarray | pd.Series, 
    eps: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    Parameters
    ----------
    expected_pct : array-like
        Distribution of the reference population (e.g., Train).
    actual_pct : array-like
        Distribution of the current population (e.g., Test).
    eps : float, default=1e-4
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Total PSI value.
    """
    exp = np.clip(np.asarray(expected_pct, dtype=float), eps, None)
    act = np.clip(np.asarray(actual_pct, dtype=float), eps, None)

    exp /= exp.sum()
    act /= act.sum()

    return float(np.sum((act - exp) * np.log(act / exp)))


def calculate_psi_from_counts(
    expected_counts: pd.Series, 
    actual_counts: pd.Series,
    eps: float = 1e-4
) -> tuple[float, pd.Series]:
    """Calculate PSI directly from raw bin counts with index alignment.

    Parameters
    ----------
    expected_counts : pd.Series
        Bin counts from the reference population.
    actual_counts : pd.Series
        Bin counts from the actual population.
    eps : float, default=1e-4
        Small constant for zero-count bins.

    Returns
    -------
    tuple[float, pd.Series]
        A tuple of (Total PSI, Series of PSI per bin).
    """
    df = pd.DataFrame({"exp": expected_counts, "act": actual_counts}).fillna(0)

    exp_pct = np.clip(df["exp"] / df["exp"].sum(), eps, None)
    act_pct = np.clip(df["act"] / df["act"].sum(), eps, None)

    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()

    psi_per_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(psi_per_bin.sum()), psi_per_bin