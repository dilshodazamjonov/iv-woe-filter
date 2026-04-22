from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from iv_woe_filter import IVWOEFilter


def test_fit_transform_basic_returns_encoded_dataframe(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, n_bins=5, verbose=False)

    X_transformed = transformer.fit_transform(X, y)

    assert isinstance(X_transformed, pd.DataFrame)
    assert not X_transformed.empty
    assert "num_feat" in X_transformed.columns
    assert X_transformed["num_feat"].dtype == float


def test_fit_transform_matches_fit_plus_transform(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)

    X_fit_transform = transformer.fit_transform(X, y)
    X_transform = transformer.transform(X)

    pd.testing.assert_frame_equal(X_fit_transform, X_transform)


def test_transform_before_fit_raises(sample_data):
    X, _ = sample_data
    transformer = IVWOEFilter()

    with pytest.raises(Exception):
        check_is_fitted(transformer, ["selected_features_", "binning_", "woe_maps_"])

    with pytest.raises(Exception):
        transformer.transform(X)


def test_encode_false_returns_raw_values_for_processed_columns(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, encode=False, verbose=False)

    transformer.fit(X, y)
    X_out = transformer.transform(X)

    pd.testing.assert_series_equal(X["num_feat"], X_out["num_feat"], check_names=True)
    pd.testing.assert_series_equal(X["cat_feat"], X_out["cat_feat"], check_names=True)


def test_drop_unselected_true_removes_low_signal_features(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.5, min_gini=0.0, drop_unselected=True, verbose=False)

    transformer.fit(X, y)
    X_out = transformer.transform(X)

    assert "constant_feat" not in X_out.columns
    assert len(X_out.columns) < len(X.columns)


def test_drop_unselected_false_keeps_all_fitted_columns(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(
        min_iv=0.5,
        min_gini=0.5,
        drop_unselected=False,
        verbose=False,
    )

    transformer.fit(X, y)
    X_out = transformer.transform(X)

    assert list(X_out.columns) == list(X.columns)


def test_get_feature_names_out_matches_transform_columns(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)

    transformer.fit(X, y)
    feature_names = transformer.get_feature_names_out()
    X_out = transformer.transform(X)

    assert feature_names == list(X_out.columns)


def test_sklearn_pipeline_compatibility(sample_data):
    X, y = sample_data
    pipeline = Pipeline(
        [
            ("iv_filter", IVWOEFilter(min_iv=0.05, min_gini=0.0, verbose=False)),
        ]
    )

    pipeline.fit(X, y)
    X_out = pipeline.transform(X)

    assert isinstance(X_out, pd.DataFrame)
