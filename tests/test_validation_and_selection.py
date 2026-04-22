from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from iv_woe_filter import IVWOEFilter


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.array([0, 1, 2, 0]), "Target must be binary"),
        (np.array([0.0, 1.0, np.nan, 0.0]), "Target"),
        (np.array([0, 2, 0, 2]), "Target values must be 0 or 1"),
    ],
)
def test_target_validation_rejects_invalid_targets(y, message):
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    transformer = IVWOEFilter(verbose=False)

    with pytest.raises(ValueError, match=message):
        transformer.fit(X, y)

def test_special_codes_create_dedicated_negative_bin(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(
        special_codes={"special_feat": [-99]},
        min_iv=0.0,
        min_gini=0.0,
        verbose=False,
    )

    transformer.fit(X, y)

    assert -2 in transformer.woe_maps_["special_feat"]

    stats = transformer._per_feature_stats["special_feat"]
    special_row = stats.loc[-2]
    assert special_row["bin_range"] == "Special: -99"


def test_missing_special_code_logs_warning(sample_data, caplog):
    X, y = sample_data
    transformer = IVWOEFilter(
        special_codes={"special_feat": [-12345]},
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )

    with caplog.at_level(logging.WARNING):
        transformer.fit(X, y)

    assert "Special codes" in caplog.text
    assert "special_feat" in caplog.text


def test_monotonicity_report_has_expected_shape(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)

    transformer.fit(X, y)
    report = transformer.monotonicity_report_["num_feat"]

    assert set(report) == {"is_monotonic", "direction"}
    assert isinstance(report["is_monotonic"], bool)
    assert report["direction"] in {"increasing", "decreasing", "none"}


def test_high_iv_feature_is_flagged_for_leakage():
    X = pd.DataFrame(
        {
            "proxy_feature": [0] * 60 + [1] * 60,
            "noise_feature": np.tile([0, 1], 60),
        }
    )
    y = np.array([0] * 60 + [1] * 60)
    transformer = IVWOEFilter(
        min_iv=0.0,
        min_gini=0.0,
        max_iv_for_leakage=0.1,
        n_jobs=1,
        verbose=False,
    )

    transformer.fit(X, y)

    assert transformer.leakage_flags_["proxy_feature"] is True


def test_gini_column_is_present_and_bounded(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)

    transformer.fit(X, y)

    assert "Gini" in transformer.iv_table_.columns
    assert ((transformer.iv_table_["Gini"] >= 0.0) & (transformer.iv_table_["Gini"] <= 1.0)).all()
