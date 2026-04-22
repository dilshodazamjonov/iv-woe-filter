from __future__ import annotations

import numpy as np
import pandas as pd

from iv_woe_filter import IVWOEFilter


def test_categorical_small_bins_are_merged_into_combined_labels():
    X = pd.DataFrame({"cat_merge": ["A"] * 80 + ["B"] * 10 + ["C"] * 10})
    y = np.array([0] * 50 + [1] * 50)

    transformer = IVWOEFilter(min_bin_pct=0.15, min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    stats = transformer._per_feature_stats["cat_merge"]
    ranges = stats["bin_range"].tolist()

    assert any("," in label for label in ranges)


def test_unknown_category_maps_to_default_woe_value(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    X_new = X.copy()
    X_new.loc[X_new.index[:5], "cat_feat"] = "UNSEEN_CATEGORY"
    X_out = transformer.transform(X_new)

    assert (X_out.loc[X_new.index[:5], "cat_feat"] == 0.0).all()


def test_missing_numeric_values_transform_without_failure(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    X_new = X.copy()
    X_new.loc[X_new.index[:3], "num_feat"] = np.nan
    X_out = transformer.transform(X_new)

    assert len(X_out) == len(X_new)
    assert X_out["num_feat"].iloc[:3].notna().all()


def test_special_bins_are_not_merged_into_regular_bins():
    X = pd.DataFrame({"feature": [-99] * 3 + list(range(30))})
    y = np.array(([1, 0, 1] * 11)[:33])

    transformer = IVWOEFilter(
        special_codes={"feature": [-99]},
        min_bin_pct=0.20,
        min_iv=0.0,
        min_gini=0.0,
        n_bins=4,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    stats = transformer._per_feature_stats["feature"]
    assert -2 in stats.index
    assert stats.loc[-2, "bin_range"] == "Special: -99"
