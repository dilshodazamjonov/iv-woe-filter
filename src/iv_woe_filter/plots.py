"""Internal plotting helpers for IV/WOE feature audits."""

from __future__ import annotations

import os
import re

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)


def _format_edge(value: float, round_digits: int) -> str:
    """Format a numeric bin edge for display."""
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.{round_digits}f}"


def _round_bin_label(label: str, round_digits: int) -> str:
    """Round numeric ranges inside a bin label while leaving text labels unchanged."""
    if not label or "[" not in label:
        return label

    def replace(match: re.Match[str]) -> str:
        return _format_edge(float(match.group(0)), round_digits)

    return _NUMERIC_PATTERN.sub(replace, label)


def _plot_bin_sort_key(bin_id: int) -> tuple[int, int]:
    """Place regular bins first, then special bins, with missing/other last."""
    if bin_id >= 0:
        return (0, bin_id)
    if bin_id == -1:
        return (2, 0)
    return (1, abs(bin_id))


def prepare_plot_frame(
    stats: pd.DataFrame,
    round_digits: int,
) -> pd.DataFrame:
    """Return bin statistics ready for plotting."""
    plot_df = stats.reset_index().rename(columns={"index": "bin_id"})
    if "bin_id" not in plot_df.columns:
        plot_df = plot_df.rename(columns={plot_df.columns[0]: "bin_id"})
    plot_df = plot_df.sort_values(
        by="bin_id",
        key=lambda series: series.map(_plot_bin_sort_key),
        kind="stable",
    ).reset_index(drop=True)
    plot_df["bad_rate"] = plot_df["bad"] / plot_df["count"]
    plot_df["bin_label"] = plot_df["bin_range"].map(lambda value: _round_bin_label(str(value), round_digits))
    return plot_df


def render_feature_plot(
    plot_df: pd.DataFrame,
    feature: str,
    iv_value: float,
    gini_value: float,
    *,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot bad rate and WOE per bin for a fitted feature."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(111)
    x_positions = range(len(plot_df))

    bars = ax_left.bar(
        x_positions,
        plot_df["bad_rate"],
        color="#7a9e9f",
        edgecolor="#2f4f4f",
        alpha=0.9,
    )
    ax_left.set_xlabel("Bins")
    ax_left.set_ylabel("Bad Rate")
    ax_left.set_xticks(list(x_positions))
    ax_left.set_xticklabels(plot_df["bin_label"], rotation=25, ha="right")
    ax_left.set_ylim(0, max(0.05, float(plot_df["bad_rate"].max()) * 1.35))

    ax_right = ax_left.twinx()
    ax_right.plot(
        list(x_positions),
        plot_df["woe"],
        color="#bc4749",
        marker="o",
        linewidth=2,
    )
    ax_right.set_ylabel("WOE")

    for bar, row in zip(bars, plot_df.itertuples(index=False)):
        annotation = f"n={int(row.count)}\ng={int(row.good)}\nb={int(row.bad)}"
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ax_left.get_ylim()[1] * 0.015, 0.01),
            annotation,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f2933",
        )

    ax_left.set_title(f"{feature} | IV={iv_value:.3f} | Gini={gini_value:.3f}")
    fig.tight_layout()
    return fig, (ax_left, ax_right)


def save_rendered_feature_plot(
    plot_df: pd.DataFrame,
    feature: str,
    iv_value: float,
    gini_value: float,
    output_path: str,
    *,
    figsize: tuple[float, float] = (12, 6),
) -> str:
    """Render and save a single feature audit plot."""
    fig, _ = render_feature_plot(
        plot_df,
        feature,
        iv_value,
        gini_value,
        figsize=figsize,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path
