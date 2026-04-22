from __future__ import annotations

import os

import pandas as pd

from iv_woe_filter import IVWOEFilter


def test_fit_writes_expected_artifact_files(sample_data, output_dir):
    X, y = sample_data
    transformer = IVWOEFilter(
        output_dir=output_dir,
        min_iv=0.0,
        min_gini=0.0,
        verbose=False,
    )

    transformer.fit(X, y)

    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "iv_summary.csv"))
    assert os.path.exists(os.path.join(output_dir, "bin_stats.csv"))
    assert os.path.exists(os.path.join(output_dir, "feature_audit.csv"))


def test_artifact_content_includes_expected_columns(sample_data, output_dir):
    X, y = sample_data
    transformer = IVWOEFilter(
        output_dir=output_dir,
        min_iv=0.0,
        min_gini=0.0,
        verbose=False,
    )

    transformer.fit(X, y)

    stats_df = pd.read_csv(os.path.join(output_dir, "bin_stats.csv"))
    audit_df = pd.read_csv(os.path.join(output_dir, "feature_audit.csv"))
    iv_df = pd.read_csv(os.path.join(output_dir, "iv_summary.csv"))

    assert {"feature", "bin_range", "woe", "iv_bin"}.issubset(stats_df.columns)
    assert {"feature", "type", "IV", "Gini", "leakage_flag"}.issubset(audit_df.columns)
    assert {"IV", "Gini"}.issubset(iv_df.columns)


def test_calculate_psi_returns_ranked_report(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    X_shifted = X.copy()
    X_shifted["num_feat"] = X_shifted["num_feat"] + 50.0

    psi_report = transformer.calculate_psi(X_shifted, save=False)

    assert isinstance(psi_report, pd.DataFrame)
    assert {"feature", "PSI", "status"}.issubset(psi_report.columns)
    assert psi_report["PSI"].is_monotonic_decreasing
    assert psi_report.loc[psi_report["feature"] == "num_feat", "PSI"].iat[0] > 0


def test_calculate_psi_skips_missing_columns_in_new_data(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    psi_report = transformer.calculate_psi(X[["num_feat"]], save=False)

    assert list(psi_report["feature"]) == ["num_feat"]


def test_calculate_psi_writes_stability_report(sample_data, output_dir):
    X, y = sample_data
    transformer = IVWOEFilter(
        output_dir=output_dir,
        min_iv=0.0,
        min_gini=0.0,
        verbose=False,
    )
    transformer.fit(X, y)

    transformer.calculate_psi(X, save=True)

    report_path = os.path.join(output_dir, "stability_report.csv")
    assert os.path.exists(report_path)
    stability_df = pd.read_csv(report_path)
    assert {"feature", "PSI", "status"}.issubset(stability_df.columns)
