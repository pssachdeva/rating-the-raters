from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rating_raters.labels import provider_display_name
from rating_raters.plotting import PROVIDER_COLORS, format_plot_text


VARIANT_SUFFIXES = ("_minimal", "_low", "_medium", "_high", "_xhigh", "_none")
X_COLUMN = "aa_intelligence_index"
Y_COLUMN = "measure"
OTHER_PROVIDER_SLUGS = {"qwen", "xiaomi", "zai"}
AXIS_BACKGROUND_COLOR = "#F7F7F7"


def load_medium_effort_aa_severities(data_path: Path) -> pd.DataFrame:
    """Load AA severity rows and keep medium effort rows when effort variants repeat."""
    data = pd.read_csv(data_path)
    required_columns = {
        "judge_id",
        "provider",
        "provider_label",
        "measure",
        "aa_intelligence_index",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=[X_COLUMN, Y_COLUMN]).copy()
    data["base_model_id"] = data["judge_id"].map(strip_effort_suffix)
    data["scatter_provider"] = data.apply(infer_scatter_provider, axis=1)
    data["scatter_provider_label"] = data["scatter_provider"].map(provider_display_name)

    selected_indices = []
    for _, group in data.groupby("base_model_id", sort=False):
        medium_rows = group[group["judge_id"].str.endswith("_medium")]
        if len(group) > 1 and not medium_rows.empty:
            selected_indices.extend(medium_rows.index.tolist())
        else:
            selected_indices.extend(group.index.tolist())

    return data.loc[selected_indices].reset_index(drop=True)


def infer_scatter_provider(row: pd.Series) -> str:
    """Return the provider grouping used by the AA scatter legend."""
    judge_id = str(row["judge_id"])
    if judge_id.startswith("together_openai_"):
        return "openai"
    if judge_id.startswith("together_meta-llama_"):
        return "meta"

    provider = str(row["provider"])
    if provider in OTHER_PROVIDER_SLUGS:
        return "other"
    return provider


def strip_effort_suffix(judge_id: str) -> str:
    """Return a model-family id without the local reasoning-effort suffix."""
    for suffix in VARIANT_SUFFIXES:
        if judge_id.endswith(suffix):
            return judge_id[: -len(suffix)]
    return judge_id


def plot_aa_severity_scatter(
    axis: plt.Axes,
    data: pd.DataFrame,
    marker_size: float = 46,
    legend_font_size: float = 6.5,
    axis_label_size: float = 9,
    tick_label_size: float = 7.5,
    annotation_size: float = 7.5,
) -> None:
    """Plot AA Intelligence Index against FACETS severity on an existing axis."""
    axis.set_facecolor(AXIS_BACKGROUND_COLOR)
    for provider, provider_data in data.groupby("scatter_provider", sort=True):
        axis.scatter(
            provider_data[X_COLUMN],
            provider_data[Y_COLUMN],
            s=marker_size,
            alpha=0.82,
            color=PROVIDER_COLORS.get(provider, PROVIDER_COLORS["unknown"]),
            edgecolor="#222222",
            linewidth=0.45,
            label=provider_data["scatter_provider_label"].iloc[0],
        )

    add_trend_line(axis, data)
    add_correlation_annotation(axis, data, annotation_size)
    axis.axhline(0, color="#777777", linewidth=0.8, alpha=0.55, zorder=0)
    axis.set_xlabel(format_plot_text("AA Intelligence Index"), fontsize=axis_label_size)
    axis.set_ylabel(format_plot_text(r"Severity ($\alpha_j$)"), fontsize=axis_label_size)
    axis.set_box_aspect(1)
    axis.tick_params(labelsize=tick_label_size)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.legend(
        frameon=False,
        fontsize=legend_font_size,
        loc="lower right",
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def add_trend_line(axis: plt.Axes, data: pd.DataFrame) -> None:
    """Add an ordinary least-squares trend line for the selected scatter points."""
    x_values = data[X_COLUMN].to_numpy(dtype=float)
    y_values = data[Y_COLUMN].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    axis.plot(line_x, slope * line_x + intercept, color="#222222", linewidth=1.4)


def add_correlation_annotation(axis: plt.Axes, data: pd.DataFrame, annotation_size: float) -> None:
    """Annotate a scatter axis with the Pearson correlation and sample size."""
    correlation = data[X_COLUMN].corr(data[Y_COLUMN])
    text = f"$r = {correlation:.2f}$"
    axis.text(
        0.03,
        0.96,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_size,
    )
