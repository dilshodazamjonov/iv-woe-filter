from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from iv_woe_filter import IVWOEFilter


def test_tree_binning_learns_supervised_numeric_threshold():
    x = np.linspace(0, 100, 120)
    y = (x >= 50).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="tree",
        n_bins=4,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    X_out = transformer.fit_transform(X, y)
    bins = transformer.binning_["score"]["bins"]

    assert transformer.binning_["score"]["binning_method"] == "tree"
    assert len(bins) > 2
    assert any(45 <= threshold <= 55 for threshold in bins[1:-1])
    assert X_out["score"].dtype == float


def test_chi_merge_binning_learns_supervised_numeric_threshold():
    x = np.linspace(0, 100, 120)
    y = (x >= 50).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="chi_merge",
        n_bins=4,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    X_out = transformer.fit_transform(X, y)
    bins = transformer.binning_["score"]["bins"]

    assert transformer.binning_["score"]["binning_method"] == "chi_merge"
    assert len(bins) <= 5
    assert any(45 <= threshold <= 55 for threshold in bins[1:-1])
    assert X_out["score"].dtype == float


def test_invalid_binning_method_raises_clear_error(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(binning_method="kmeans", verbose=False)

    with pytest.raises(ValueError, match="binning_method"):
        transformer.fit(X, y)


def test_tree_binning_respects_max_depth():
    x = np.arange(200, dtype=float)
    y = (((x > 35) & (x < 80)) | (x > 130)).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="tree",
        n_bins=8,
        tree_max_depth=1,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    assert len(transformer.binning_["score"]["bins"]) <= 3


def test_tree_binning_is_deterministic_with_fixed_random_state():
    x = np.tile(np.arange(50, dtype=float), 4)
    y = (x >= 25).astype(int)
    X = pd.DataFrame({"score": x})

    first = IVWOEFilter(
        binning_method="tree",
        n_bins=5,
        random_state=7,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    ).fit(X, y)
    second = IVWOEFilter(
        binning_method="tree",
        n_bins=5,
        random_state=7,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    ).fit(X, y)

    np.testing.assert_array_equal(first.binning_["score"]["bins"], second.binning_["score"]["bins"])


def test_tree_binning_preserves_special_code_bin():
    x = np.array([-99] * 10 + list(range(90)), dtype=float)
    y = np.array(([0, 1] * 5) + [int(v >= 45) for v in range(90)])
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="tree",
        special_codes={"score": [-99]},
        n_bins=4,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    assert -2 in transformer.woe_maps_["score"]
    assert transformer._per_feature_stats["score"].loc[-2, "bin_range"] == "Special: -99"
    assert all(threshold > -99 for threshold in transformer.binning_["score"]["bins"][1:-1])


def test_chi_merge_preserves_special_code_bin():
    x = np.array([-99] * 10 + list(range(90)), dtype=float)
    y = np.array(([0, 1] * 5) + [int(v >= 45) for v in range(90)])
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="chi_merge",
        special_codes={"score": [-99]},
        n_bins=4,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    assert -2 in transformer.woe_maps_["score"]
    assert transformer._per_feature_stats["score"].loc[-2, "bin_range"] == "Special: -99"
    assert all(threshold > -99 for threshold in transformer.binning_["score"]["bins"][1:-1])


def test_chi_merge_prebins_high_cardinality_feature_for_scalability():
    x = np.arange(500, dtype=float)
    y = (x >= 250).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="chi_merge",
        n_bins=5,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    config = transformer.binning_["score"]
    assert config["chi_merge_used_prebinning"] is True
    assert config["chi_merge_seed_bin_count"] <= config["chi_merge_max_prebins"]
    assert config["chi_merge_final_bin_count"] <= 5
    assert len(config["bins"]) <= 6


def test_chi_merge_logs_feature_completion(caplog):
    x = np.linspace(0, 100, 120)
    y = (x >= 50).astype(int)
    X = pd.DataFrame({"score": x})

    transformer = IVWOEFilter(
        binning_method="chi_merge",
        n_bins=4,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=True,
    )

    with caplog.at_level(logging.INFO):
        transformer.fit(X, y)

    assert "ChiMerge complete for feature 'score'" in caplog.text


def test_tree_parameters_are_sklearn_visible():
    transformer = IVWOEFilter(
        binning_method="tree",
        random_state=13,
        tree_criterion="entropy",
        tree_max_depth=2,
        tree_min_samples_leaf=0.1,
        tree_min_samples_split=0.2,
    )

    params = transformer.get_params()

    assert params["binning_method"] == "tree"
    assert params["random_state"] == 13
    assert params["tree_criterion"] == "entropy"
    assert params["tree_max_depth"] == 2
    assert params["tree_min_samples_leaf"] == 0.1
    assert params["tree_min_samples_split"] == 0.2
