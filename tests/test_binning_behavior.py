from __future__ import annotations

import numpy as np
import pandas as pd

from iv_woe_filter import IVWOEFilter
from iv_woe_filter.binning import apply_bins, remap_bin_ids
from iv_woe_filter.metrics import calculate_feature_gini


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


def test_transform_reuses_merged_bin_mapping_for_categorical_values():
    X = pd.DataFrame({"cat_merge": ["A"] * 60 + ["B"] * 25 + ["C"] * 10 + ["D"] * 5})
    y = np.array([0] * 60 + [1] * 40)

    transformer = IVWOEFilter(
        min_bin_pct=0.15,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    transformed = transformer.transform(pd.DataFrame({"cat_merge": ["D"]}))
    assert transformed.iloc[0, 0] != 0.0


def test_calculate_psi_uses_merged_bin_mapping():
    X = pd.DataFrame({"cat_merge": ["A"] * 60 + ["B"] * 25 + ["C"] * 10 + ["D"] * 5})
    y = np.array([0] * 60 + [1] * 40)

    transformer = IVWOEFilter(
        min_bin_pct=0.15,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    psi_report = transformer.calculate_psi(X, save=False)
    psi_value = psi_report.loc[psi_report["feature"] == "cat_merge", "PSI"].iat[0]
    assert np.isclose(psi_value, 0.0)


def test_numeric_merged_bin_label_reflects_combined_range():
    X = pd.DataFrame({"score": [0.0] * 60 + [1.0] * 25 + [2.0] * 10 + [3.0] * 5})
    y = np.array([0] * 60 + [1] * 40)

    transformer = IVWOEFilter(
        n_bins=10,
        min_bin_pct=0.15,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    config = transformer.binning_["score"]
    merged_bin = config["bin_id_map"][3]
    bins = config["bins"]
    expected_label = f"[{bins[2]}, {bins[4]})"

    stats = transformer._per_feature_stats["score"]
    assert stats.loc[merged_bin, "bin_range"] == expected_label


def test_numeric_gini_uses_fitted_woe_representation():
    x = np.linspace(0, 100, 120)
    y = (x >= 50).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        n_bins=4,
        min_bin_pct=None,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    config = transformer.binning_["score"]
    bin_ids = apply_bins(X["score"], config)
    bin_ids = remap_bin_ids(bin_ids, config)
    expected_gini = calculate_feature_gini(bin_ids, transformer.woe_maps_["score"], y)

    assert np.isclose(transformer.iv_table_.loc["score", "Gini"], expected_gini)
