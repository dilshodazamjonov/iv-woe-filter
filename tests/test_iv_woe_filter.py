"""Unit and integration tests for the IVWOEFilter package."""

from __future__ import annotations

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from iv_woe_filter import IVWOEFilter

@pytest.fixture
def sample_data():
    """Create a synthetic dataset with mixed numeric and categorical features, including intentional constants and special codes, alongside a noisy binary target for robust testing."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        "num_feat": np.random.uniform(0, 100, n),
        "cat_feat": np.random.choice(["A", "B", "C"], n),
        "constant_feat": [1] * n,
        "special_feat": [10, 20, 30, -99] * 25
    })
    
    y = (df["num_feat"] > 50).astype(int)
    noise = np.random.choice([0, 1], n, p=[0.9, 0.1])
    y = np.bitwise_xor(y, noise)
    
    return df, y

@pytest.fixture
def output_dir():
    """Provide and manage a temporary directory path for storing and subsequently cleaning up generated audit artifacts during tests."""
    path = "./test_artifacts"
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

def test_fit_transform_basic(sample_data):
    """Validate the standard fit and transform workflow, ensuring the output is a valid pandas DataFrame containing the expected features encoded as Weight of Evidence (float) values."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, n_bins=5)
    
    transformer.fit(X, y)
    X_transformed = transformer.transform(X)
    
    assert isinstance(X_transformed, pd.DataFrame)
    assert not X_transformed.empty
    assert "num_feat" in X_transformed.columns
    assert X_transformed["num_feat"].dtype == float

def test_artifact_generation(sample_data, output_dir):
    """Verify that setting the output directory successfully generates the required regulatory audit CSV artifacts ('iv_summary.csv', 'feature_audit.csv', 'bin_stats.csv') and that the generated 'bin_stats.csv' contains the dynamically calculated 'bin_range' column."""
    X, y = sample_data
    transformer = IVWOEFilter(output_dir=output_dir)
    
    transformer.fit(X, y)
    
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "iv_summary.csv"))
    assert os.path.exists(os.path.join(output_dir, "feature_audit.csv"))
    
    bin_stats_path = os.path.join(output_dir, "bin_stats.csv")
    assert os.path.exists(bin_stats_path)
    
    stats_df = pd.read_csv(bin_stats_path)
    assert "bin_range" in stats_df.columns

    audit_df = pd.read_csv(os.path.join(output_dir, "feature_audit.csv"))
    assert "Gini" in audit_df.columns

def test_special_codes_handling(sample_data):
    """Ensure that user-defined special codes (e.g., -99) are successfully isolated from standard numeric distributions and correctly mapped to dedicated negative bin IDs (e.g., -2) within the internal Weight of Evidence dictionaries."""
    X, y = sample_data
    transformer = IVWOEFilter(special_codes={"special_feat": [-99]})
    
    transformer.fit(X, y)
    
    assert -2 in transformer.woe_maps_["special_feat"]

def test_sklearn_pipeline_compatibility(sample_data):
    """Confirm that the IVWOEFilter inherits correctly from Scikit-Learn base classes and functions seamlessly within a standard sklearn Pipeline without throwing structural or type errors."""
    X, y = sample_data
    pipeline = Pipeline([
        ("iv_filter", IVWOEFilter(min_iv=0.05)),
    ])
    
    pipeline.fit(X, y)
    X_out = pipeline.transform(X)
    
    assert isinstance(X_out, pd.DataFrame)

def test_drop_low_iv(sample_data):
    """Validate that features with an Information Value (IV) strictly lower than the specified 'min_iv' threshold (such as zero-variance constant features) are automatically dropped from both the final DataFrame and the 'selected_features_' attribute."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.1, drop_low_iv=True)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    assert "constant_feat" not in X_out.columns
    assert "constant_feat" not in transformer.selected_features_

def test_monotonicity_reporting(sample_data):
    """Check the internal audit mechanism to ensure that the monotonic trend (increasing, decreasing, or none) of a feature's Weight of Evidence across its bins is properly calculated and recorded in the 'monotonicity_report_' dictionary."""
    X, y = sample_data
    transformer = IVWOEFilter()
    transformer.fit(X, y)
    
    report = transformer.monotonicity_report_["num_feat"]
    assert "is_monotonic" in report
    assert isinstance(report["is_monotonic"], bool)

def test_fit_transform_consistency(sample_data):
    """Verify that invoking 'fit_transform' yields identically calculated and formatted results as invoking 'fit' followed sequentially by 'transform' on the same dataset."""
    X, y = sample_data
    f = IVWOEFilter(min_iv=0.0)

    X1 = f.fit_transform(X, y)
    X2 = f.transform(X)

    pd.testing.assert_frame_equal(X1, X2)

def test_categorical_bin_merging():
    """Ensure that categorical variables falling below the 'min_bin_pct' population threshold are successfully merged with neighboring bins, and that the resulting 'bin_range' label correctly concatenates the original constituent category names."""
    X = pd.DataFrame({
        "cat_merge": ["A"] * 80 + ["B"] * 10 + ["C"] * 10
    })
    y = np.random.randint(0, 2, 100)
    
    transformer = IVWOEFilter(min_bin_pct=0.15, min_iv=0.0)
    transformer.fit(X, y)
    
    stats = transformer._per_feature_stats["cat_merge"]
    ranges = stats["bin_range"].tolist()
    
    has_merged_label = any("," in r for r in ranges)
    assert has_merged_label, f"Expected merged label with comma, got: {ranges}"

def test_encode_false(sample_data):
    """Verify that initializing the filter with 'encode=False' returns the original raw data values for the selected features, bypassing the Weight of Evidence numerical substitution while still applying the Information Value filtering."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, encode=False)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    pd.testing.assert_series_equal(X["num_feat"], X_out["num_feat"])
    pd.testing.assert_series_equal(X["cat_feat"], X_out["cat_feat"])

def test_gini_calculation(sample_data):
    """Ensure that the Gini coefficient is correctly computed for each feature during fitting, bounded between -1.0 and 1.0, and properly surfaced in the iv_table_ attribute."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0)
    
    transformer.fit(X, y)
    
    assert "Gini" in transformer.iv_table_.columns
    
    gini_values = transformer.iv_table_["Gini"].values
    assert all(g >= -1.0 and g <= 1.0 for g in gini_values)