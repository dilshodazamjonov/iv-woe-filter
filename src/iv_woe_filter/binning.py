import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union

def fit_numeric_bins(
    series: pd.Series, 
    n_bins: int, 
    special_codes: Optional[List[Union[int, float]]] = None
) -> Dict[str, Any]:
    binning_data = {
        "is_numeric": True,
        "special_codes": special_codes or [],
        "bins": None
    }
    
    clean_series = series
    if special_codes:
        clean_series = series[~series.isin(special_codes)]
        
    if clean_series.empty:
        binning_data["bins"] = np.array([-np.inf, np.inf])
        return binning_data

    try:
        _, bins = pd.qcut(clean_series, q=n_bins, duplicates="drop", retbins=True)
    except (ValueError, IndexError):
        _, bins = pd.cut(clean_series, bins=min(n_bins, clean_series.nunique()), retbins=True)
        
    bins = bins.tolist()
    bins[0], bins[-1] = -np.inf, np.inf
    binning_data["bins"] = np.array(bins)
    
    return binning_data

def fit_categorical_bins(
    series: pd.Series, 
    special_codes: Optional[List[Any]] = None
) -> Dict[str, Any]:
    s = series.fillna("missing").astype(str)
    
    binning_data = {
        "is_numeric": False,
        "special_codes": special_codes or [],
        "categories": s.unique().tolist()
    }
    return binning_data

def apply_bins(
    series: pd.Series, 
    config: Dict[str, Any]
) -> np.ndarray:
    s = series.copy()
    output_codes = np.full(s.shape, -1, dtype=int)
    
    special_mask = s.isin(config["special_codes"])
    if special_mask.any():
        mapping = {val: -(i + 2) for i, val in enumerate(config["special_codes"])}
        output_codes[special_mask] = s[special_mask].map(mapping)

    remaining_mask = ~special_mask
    if not remaining_mask.any():
        return output_codes

    if config["is_numeric"]:
        numeric_s = pd.to_numeric(s[remaining_mask], errors='coerce')
        codes = pd.cut(numeric_s, bins=config["bins"], labels=False, include_lowest=True)
        output_codes[remaining_mask] = np.where(np.isnan(codes), -1, codes.astype(int))
    else:
        cat_s = s[remaining_mask].fillna("missing").astype(str)
        lookup = {cat: i for i, cat in enumerate(config["categories"])}
        output_codes[remaining_mask] = np.array([lookup.get(x, -1) for x in cat_s])

    return output_codes

def merge_non_significant_bins(
    codes: np.ndarray, 
    y: np.ndarray, 
    min_pct: float
) -> np.ndarray:
    unique_codes = np.unique(codes)
    if len(unique_codes) <= 1:
        return codes
        
    total_count = len(codes)
    new_codes = codes.copy()
    
    for code in unique_codes:
        mask = (new_codes == code)
        if (mask.sum() / total_count) < min_pct:
            dist = np.abs(unique_codes - code)
            dist[unique_codes == code] = np.inf
            nearest_neighbor = unique_codes[np.argmin(dist)]
            new_codes[mask] = nearest_neighbor
            
    return new_codes