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
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, n_bins=5)
    
    transformer.fit(X, y)
    X_transformed = transformer.transform(X)
    
    assert isinstance(X_transformed, pd.DataFrame)
    assert not X_transformed.empty
    assert "num_feat" in X_transformed.columns
    assert X_transformed["num_feat"].dtype == float

def test_artifact_generation(sample_data, output_dir):
    """Verify that setting the output directory successfully generates the required regulatory audit CSV artifacts."""
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
    """Ensure that user-defined special codes are successfully isolated and mapped to dedicated bin IDs."""
    X, y = sample_data
    transformer = IVWOEFilter(special_codes={"special_feat": [-99]})
    
    transformer.fit(X, y)
    
    assert -2 in transformer.woe_maps_["special_feat"]

def test_sklearn_pipeline_compatibility(sample_data):
    """Confirm that the IVWOEFilter functions seamlessly within a standard sklearn Pipeline."""
    X, y = sample_data
    pipeline = Pipeline([
        ("iv_filter", IVWOEFilter(min_iv=0.05, min_gini=0.0)),
    ])
    
    pipeline.fit(X, y)
    X_out = pipeline.transform(X)
    
    assert isinstance(X_out, pd.DataFrame)

def test_drop_unselected_iv(sample_data):
    """Validate that features with low IV are dropped when drop_unselected=True."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.5, min_gini=0.0, drop_unselected=True)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    assert "constant_feat" not in X_out.columns
    assert len(X_out.columns) < len(X.columns)

def test_drop_unselected_false(sample_data):
    """Validate that no features are dropped when drop_unselected=False."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.5, min_gini=0.5, drop_unselected=False)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    assert len(X_out.columns) == len(X.columns)

def test_target_validation():
    """Ensure that the transformer raises a ValueError if the target is not binary (0/1)."""
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = np.array([0, 1, 2, 0])
    transformer = IVWOEFilter()
    
    with pytest.raises(ValueError, match="Target y must be binary"):
        transformer.fit(X, y)

def test_monotonicity_reporting(sample_data):
    """Check the internal audit mechanism for monotonicity reporting."""
    X, y = sample_data
    transformer = IVWOEFilter()
    transformer.fit(X, y)
    
    report = transformer.monotonicity_report_["num_feat"]
    assert "is_monotonic" in report
    assert isinstance(report["is_monotonic"], bool)

def test_fit_transform_consistency(sample_data):
    """Verify that fit_transform yields same results as fit followed by transform."""
    X, y = sample_data
    f = IVWOEFilter(min_iv=0.0, min_gini=0.0)

    X1 = f.fit_transform(X, y)
    X2 = f.transform(X)

    pd.testing.assert_frame_equal(X1, X2)

def test_categorical_bin_merging():
    """Ensure that categorical variables are merged when below the population threshold."""
    X = pd.DataFrame({
        "cat_merge": ["A"] * 80 + ["B"] * 10 + ["C"] * 10
    })
    y = np.random.randint(0, 2, 100)
    
    transformer = IVWOEFilter(min_bin_pct=0.15, min_iv=0.0, min_gini=0.0)
    transformer.fit(X, y)
    
    stats = transformer._per_feature_stats["cat_merge"]
    ranges = stats["bin_range"].tolist()
    
    has_merged_label = any("," in r for r in ranges)
    assert has_merged_label

def test_encode_false(sample_data):
    """Verify that encode=False returns raw data values for selected features."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, encode=False)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    pd.testing.assert_series_equal(X["num_feat"], X_out["num_feat"])

def test_gini_calculation(sample_data):
    """Ensure that the Gini coefficient is correctly computed and bounded."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0)
    
    transformer.fit(X, y)
    
    assert "Gini" in transformer.iv_table_.columns
    gini_values = transformer.iv_table_["Gini"].values
    assert all(g >= 0.0 and g <= 1.0 for g in gini_values)

def test_calculate_psi(sample_data):
    """Test PSI calculation against a shifted dataset."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0)
    transformer.fit(X, y)
    
    X_shifted = X.copy()
    X_shifted["num_feat"] = X_shifted["num_feat"] + 50.0
    
    psi_report = transformer.calculate_psi(X_shifted, save=False)
    
    assert isinstance(psi_report, pd.DataFrame)
    assert "PSI" in psi_report.columns
    assert "status" in psi_report.columns
    
    num_feat_psi = psi_report[psi_report["feature"] == "num_feat"]["PSI"].values[0]
    assert num_feat_psi > 0

def test_psi_artifact_generation(sample_data, output_dir):
    """Verify that calculate_psi generates a stability_report.csv artifact."""
    X, y = sample_data
    transformer = IVWOEFilter(output_dir=output_dir)
    transformer.fit(X, y)
    
    transformer.calculate_psi(X, save=True)
    
    assert os.path.exists(os.path.join(output_dir, "stability_report.csv"))