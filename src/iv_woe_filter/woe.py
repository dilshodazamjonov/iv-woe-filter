import numpy as np
import pandas as pd
from typing import Tuple, Dict

def safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator, dtype=float), 
        where=(denominator != 0)
    )

def compute_aggregate_stats(codes: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    counts = np.bincount(inverse)
    bads = np.bincount(inverse, weights=y)
    goods = counts - bads
    
    total_bads = bads.sum()
    total_goods = goods.sum()
    
    bad_pct = safe_div(bads, total_bads)
    good_pct = safe_div(goods, total_goods)
    
    return pd.DataFrame({
        "bin": unique_codes,
        "count": counts,
        "bad": bads,
        "good": goods,
        "bad_pct": bad_pct,
        "good_pct": good_pct
    }).set_index("bin")

def calculate_woe_iv(stats: pd.DataFrame, eps: float = 1e-12) -> Tuple[pd.Series, pd.Series, float]:
    woe = np.log((stats["good_pct"] + eps) / (stats["bad_pct"] + eps))
    iv_bin = (stats["good_pct"] - stats["bad_pct"]) * woe
    return woe, iv_bin, float(iv_bin.sum())

def check_monotonicity(woe: pd.Series) -> Dict[str, bool]:
    if len(woe) < 2:
        return {"is_monotonic": True, "direction": "none"}
    
    diff = np.diff(woe.values)
    is_inc = np.all(diff >= 0)
    is_dec = np.all(diff <= 0)
    
    return {
        "is_monotonic": bool(is_inc or is_dec),
        "direction": "increasing" if is_inc else "decreasing" if is_dec else "none"
    }