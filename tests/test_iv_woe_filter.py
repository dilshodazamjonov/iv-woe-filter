"""Unit and integration tests for the IVWOEFilter package."""

from __future__ import annotations

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from iv_woe_filter import IVWOEFilter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create a synthetic dataset with numeric and categorical features."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        "num_feat": np.random.uniform(0, 100, n),
        "cat_feat": np.random.choice(["A", "B", "C"], n),
        "constant_feat": [1] * n,
        "special_feat": [10, 20, 30, -99] * 25
    })
    
    # Target correlated with num_feat
    y = (df["num_feat"] > 50).astype(int)
    # Add some noise
    noise = np.random.choice([0, 1], n, p=[0.9, 0.1])
    y = np.bitwise_xor(y, noise)
    
    return df, y

@pytest.fixture
def output_dir():
    """Temporary directory for audit artifacts."""
    path = "./test_artifacts"
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_transform_basic(sample_data):
    """Test standard fit and transform workflow."""
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, n_bins=5)
    
    transformer.fit(X, y)
    X_transformed = transformer.transform(X)
    
    assert isinstance(X_transformed, pd.DataFrame)
    assert not X_transformed.empty
    assert "num_feat" in X_transformed.columns
    # Check if WOE encoding happened (should be floats)
    assert X_transformed["num_feat"].dtype == float

def test_artifact_generation(sample_data, output_dir):
    """Verify that audit CSVs are correctly saved."""
    X, y = sample_data
    transformer = IVWOEFilter(output_dir=output_dir)
    
    transformer.fit(X, y)
    
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "iv_summary.csv"))
    assert os.path.exists(os.path.join(output_dir, "bin_stats.csv"))
    assert os.path.exists(os.path.join(output_dir, "feature_audit.csv"))

def test_special_codes_handling(sample_data):
    """Ensure special codes are treated as independent bins."""
    X, y = sample_data
    # -99 is a special code
    transformer = IVWOEFilter(special_codes={"special_feat": [-99]})
    
    transformer.fit(X, y)
    
    # Check if -99 exists in the woe map for that feature
    # Our binning logic maps the first special code to -2
    assert -2 in transformer.woe_maps_["special_feat"]

def test_sklearn_pipeline_compatibility(sample_data):
    """Test if the filter can be used inside an sklearn Pipeline."""
    X, y = sample_data
    pipeline = Pipeline([
        ("iv_filter", IVWOEFilter(min_iv=0.05)),
    ])
    
    # Should work without raising errors
    pipeline.fit(X, y)
    X_out = pipeline.transform(X)
    
    assert isinstance(X_out, pd.DataFrame)

def test_drop_low_iv(sample_data):
    """Verify that low IV features are actually dropped."""
    X, y = sample_data
    # constant_feat should have IV ~ 0
    transformer = IVWOEFilter(min_iv=0.1, drop_low_iv=True)
    
    transformer.fit(X, y)
    X_out = transformer.transform(X)
    
    assert "constant_feat" not in X_out.columns
    assert "constant_feat" not in transformer.selected_features_

def test_monotonicity_reporting(sample_data):
    """Check if monotonicity is detected."""
    X, y = sample_data
    transformer = IVWOEFilter()
    transformer.fit(X, y)
    
    # num_feat was generated to be correlated with target, should likely be monotonic
    report = transformer.monotonicity_report_["num_feat"]
    assert "is_monotonic" in report
    assert isinstance(report["is_monotonic"], bool)


def test_fit_transform_consistency(sample_data):
    X, y = sample_data
    f = IVWOEFilter(min_iv=0.0)

    X1 = f.fit_transform(X, y)
    X2 = f.transform(X)

    assert X1.equals(X2)