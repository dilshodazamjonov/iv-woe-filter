from __future__ import annotations

import os
import logging

import matplotlib
import pandas as pd

from iv_woe_filter import IVWOEFilter

matplotlib.use("Agg")


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
    assert {"feature", "type", "binning_method", "IV", "Gini", "leakage_flag"}.issubset(audit_df.columns)
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


def test_calculate_psi_marks_missing_columns_in_new_data(sample_data, caplog):
    X, y = sample_data
    transformer = IVWOEFilter(min_iv=0.0, min_gini=0.0, verbose=False)
    transformer.fit(X, y)

    with caplog.at_level(logging.WARNING):
        psi_report = transformer.calculate_psi(X[["num_feat"]], save=False)

    assert "missing from PSI input" in caplog.text
    assert "num_feat" in psi_report["feature"].values
    missing_rows = psi_report[psi_report["status"] == "Missing in Input"]
    assert not missing_rows.empty
    assert missing_rows["PSI"].isna().all()


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


def test_save_feature_plot_writes_png_files_for_all_features(sample_data, output_dir):
    X, y = sample_data
    transformer = IVWOEFilter(
        output_dir=output_dir,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    plot_dir = os.path.join(output_dir, "plots")
    saved_paths = transformer.save_feature_plot(plot_dir, feature="all")

    assert len(saved_paths) == len(transformer.iv_table_.index)
    assert all(path.endswith(".png") for path in saved_paths)
    assert all(os.path.exists(path) for path in saved_paths)


def test_plot_feature_audit_rounds_numeric_bin_labels(sample_data):
    X, y = sample_data
    transformer = IVWOEFilter(
        min_iv=0.0,
        min_gini=0.0,
        n_bins=5,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    fig, (ax_left, _) = transformer.plot_feature_audit("num_feat", round_digits=1)
    labels = [tick.get_text() for tick in ax_left.get_xticklabels()]
    fig.clear()

    assert labels
    assert any(label.startswith("[") for label in labels)
    assert all(".00" not in label for label in labels if label.startswith("["))


def test_save_feature_plot_writes_single_requested_feature(sample_data, output_dir):
    X, y = sample_data
    transformer = IVWOEFilter(
        output_dir=output_dir,
        min_iv=0.0,
        min_gini=0.0,
        n_jobs=1,
        verbose=False,
    )
    transformer.fit(X, y)

    plot_dir = os.path.join(output_dir, "single-plot")
    saved_path = transformer.save_feature_plot(plot_dir, feature="num_feat")

    assert saved_path.endswith("num_feat.png")
    assert os.path.exists(saved_path)
