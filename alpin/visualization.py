"""Visualization utilities for ALPIN.

This module provides publication-quality plotting functions for signal analysis,
changepoint detection results, and metric comparisons. It supports both
static (matplotlib) and interactive (plotly) visualizations.
"""

import warnings
from typing import Dict, List, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "signal": "#2C3E50",
    "truth": "#27AE60",
    "pred": "#E74C3C",
    "grid": "#BDC3C7",
    "band": "#3498DB",
}


def plot_signal(
    signal: np.ndarray,
    true_changepoints: Optional[List[int]] = None,
    pred_changepoints: Optional[List[int]] = None,
    title: str = "Signal with Changepoints",
    figsize: tuple[int, int] = (12, 6),
    show: bool = True,
) -> Optional[matplotlib.figure.Figure]:
    """
    Creates a publication-quality static plot showing the signal with optional ground truth and predicted changepoints as vertical lines.
    Returns figure object if show=False for further customization, otherwise displays and returns None.

    Input: signal (np.ndarray) - 1D signal array, true_changepoints (list of int or None) - ground truth indices, pred_changepoints (list of int or None) - predicted indices, title (str) - plot title, figsize (tuple) - (width, height), show (bool) - display immediately
    Output: matplotlib.figure.Figure or None - figure object if show=False, else None after display
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    ax.plot(signal, color=COLORS["signal"], linewidth=1.5, label="Signal", alpha=0.9)

    if true_changepoints:
        for i, cp in enumerate(true_changepoints):
            label = "Ground Truth" if i == 0 else None
            ax.axvline(
                x=cp,
                color=COLORS["truth"],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=label,
            )

    if pred_changepoints:
        for i, cp in enumerate(pred_changepoints):
            label = "Predicted" if i == 0 else None
            ax.axvline(
                x=cp,
                color=COLORS["pred"],
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                label=label,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel("Signal Amplitude", fontsize=12)
    ax.legend(frameon=True, framealpha=0.9, loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Method Comparison",
    metric_keys: Optional[List[str]] = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Optional[matplotlib.figure.Figure]:
    """
    Creates a grouped bar chart comparing metrics across different methods with labeled values.
    Groups methods on x-axis and metrics as bars, automatically selecting all common keys if not specified.

    Input: metrics_dict (dict) - keys are method names, values are metric dicts, title (str) - plot title, metric_keys (list of str or None) - specific metrics to plot, figsize (tuple) - (width, height), show (bool) - display
    Output: matplotlib.figure.Figure or None - figure object if show=False, else None after display
    """
    if not metrics_dict:
        warnings.warn("No metrics provided to plot.")
        return None

    methods = list(metrics_dict.keys())

    if metric_keys is None:
        first_method = methods[0]
        metric_keys = list(metrics_dict[first_method].keys())

    x = np.arange(len(metric_keys))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(methods))]

    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metric_keys]
        offset = (i - len(methods) / 2) * width + width / 2
        rects = ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=colors[i],
            alpha=0.9,
            edgecolor="white",
        )
        ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=11, rotation=45, ha="right")
    ax.legend(loc="best", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_signal_interactive(
    signal: np.ndarray,
    true_changepoints: Optional[List[int]] = None,
    pred_changepoints: Optional[List[int]] = None,
    title: str = "Interactive Signal View",
) -> go.Figure:
    """
    Creates an interactive Plotly figure for signal exploration with hover details and zoom capability.
    Displays ground truth and predicted changepoints as vertical lines with legend and annotations.

    Input: signal (np.ndarray) - 1D signal array, true_changepoints (list of int or None) - ground truth indices, pred_changepoints (list of int or None) - predicted indices, title (str) - plot title
    Output: go.Figure - Plotly figure object with interactive controls
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=signal,
            mode="lines",
            name="Signal",
            line=dict(color=COLORS["signal"], width=1.5),
            opacity=0.8,
        )
    )

    if true_changepoints:
        for i, cp in enumerate(true_changepoints):
            show_legend = i == 0
            fig.add_vline(
                x=cp,
                line_width=2,
                line_dash="dash",
                line_color=COLORS["truth"],
                annotation_text="GT" if show_legend else None,
                annotation_position="top right",
            )
            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=COLORS["truth"], width=2, dash="dash"),
                        name="Ground Truth",
                    )
                )

    if pred_changepoints:
        for i, cp in enumerate(pred_changepoints):
            show_legend = i == 0
            fig.add_vline(
                x=cp,
                line_width=2,
                line_dash="solid",
                line_color=COLORS["pred"],
                annotation_text="Pred" if show_legend else None,
                annotation_position="bottom right",
            )
            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=COLORS["pred"], width=2, dash="solid"),
                        name="Predicted",
                    )
                )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Time Index",
        yaxis_title="Amplitude",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_sweep_results(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    title: str = "Parameter Sweep Results",
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Optional[matplotlib.figure.Figure]:
    """
    Plots parameter sweep results with error bands showing confidence intervals when multiple runs per point exist.
    Groups results optionally by category column, computing mean and std for each group.

    Input: results_df (pd.DataFrame) - sweep results, x_col (str) - x-axis column, y_col (str) - y-axis metric, group_col (str or None) - grouping column, title (str) - plot title, figsize (tuple) - (width, height), show (bool) - display
    Output: matplotlib.figure.Figure or None - figure object if show=False, else None after display
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    if group_col:
        groups = results_df.groupby(group_col)
    else:
        groups = [("Results", results_df)]

    cmap = plt.get_cmap("tab10")

    for i, (name, group) in enumerate(groups):
        stats = group.groupby(x_col)[y_col].agg(["mean", "std", "count"]).reset_index()

        stats = stats.sort_values(x_col)

        color = cmap(i % 10)

        ax.plot(
            stats[x_col],
            stats["mean"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=name,
            color=color,
        )

        if stats["count"].max() > 1:
            ci = 1.96 * stats["std"] / np.sqrt(stats["count"])
            ax.fill_between(
                stats[x_col],
                stats["mean"] - ci,
                stats["mean"] + ci,
                color=color,
                alpha=0.2,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    if group_col or len(groups) > 1:
        ax.legend(frameon=True, loc="best")

    ax.grid(True, linestyle=":", alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig
