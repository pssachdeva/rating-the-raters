"""Reusable helpers for severity decomposition heatmap figures."""

from pathlib import Path

from matplotlib.collections import QuadMesh
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
import numpy as np
import pandas as pd

from mhs_llms.facets.postprocess import parse_facets_score_file
from mhs_llms.labels import model_id_to_plot_label


def load_human_item_difficulty_order(
    score_path: Path,
    item_ids: list[str],
) -> list[str]:
    """Load item labels ordered from easier to harder in the human baseline."""

    score_frame = parse_facets_score_file(score_path)
    _require_columns(score_frame, {"facet_label", "measure"}, "Human item score file")

    item_scores = score_frame.loc[
        score_frame["facet_label"].isin(item_ids),
        ["facet_label", "measure"],
    ].copy()
    missing_items = sorted(set(item_ids).difference(item_scores["facet_label"]))
    if missing_items:
        missing_text = ", ".join(missing_items)
        raise ValueError(f"Human item score file is missing items: {missing_text}")

    item_scores = item_scores.sort_values(["measure", "facet_label"], kind="stable")
    return item_scores["facet_label"].astype(str).tolist()


def load_model_severity_order(score_path: Path, model_ids: list[str]) -> list[str]:
    """Load model labels ordered from lower to higher severity."""

    score_frame = parse_facets_score_file(score_path)
    _require_columns(score_frame, {"facet_label", "measure"}, "Model score file")

    model_scores = score_frame.loc[
        score_frame["facet_label"].isin(model_ids),
        ["facet_label", "measure"],
    ].copy()
    missing_models = sorted(set(model_ids).difference(model_scores["facet_label"]))
    if missing_models:
        missing_text = ", ".join(missing_models)
        raise ValueError(f"Model score file is missing models: {missing_text}")

    model_scores = model_scores.sort_values(["measure", "facet_label"], kind="stable")
    return model_scores["facet_label"].astype(str).tolist()


def load_bias_terms(
    data_path: Path,
    model_order: list[str],
    item_order: list[str],
) -> pd.DataFrame:
    """Load item-dependent severity adjustments for selected models and items."""

    bias_terms = pd.read_csv(data_path)
    selected = bias_terms.loc[
        bias_terms["judge_label"].isin(model_order)
        & bias_terms["item_label"].isin(item_order)
    ].copy()

    missing_pairs = [
        (model_id, item_id)
        for model_id in model_order
        for item_id in item_order
        if selected.loc[
            (selected["judge_label"] == model_id) & (selected["item_label"] == item_id)
        ].empty
    ]
    if missing_pairs:
        missing_text = ", ".join(f"{model}:{item}" for model, item in missing_pairs[:10])
        raise ValueError(f"Bias-term dataset is missing model-item pairs: {missing_text}")
    return selected


def build_x_layout(model_order: list[str], model_labels: dict[str, str]) -> tuple[pd.DataFrame, list[float]]:
    """Return plotted x positions and labels for model columns."""

    rows = []
    column_widths = []
    x_left = 0.0
    for model_id in model_order:
        matrix_column = len(column_widths)
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_labels.get(model_id, model_id_to_plot_label(model_id)),
                "x_position": x_left + 0.5,
                "matrix_column": matrix_column,
            }
        )
        column_widths.append(1.0)
        x_left += 1.0
    return pd.DataFrame(rows), column_widths


def build_heatmap_matrices(
    bias_terms: pd.DataFrame,
    x_layout: pd.DataFrame,
    item_order: list[str],
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build adjustment and significance matrices in item-row by model-column order."""

    value_matrix = np.full((len(item_order), column_count), np.nan)
    significance_matrix = np.full((len(item_order), column_count), "", dtype=object)
    x_lookup = {row.model_id: int(row.matrix_column) for row in x_layout.itertuples(index=False)}
    item_lookup = {item_id: index for index, item_id in enumerate(item_order)}

    for row in bias_terms.itertuples(index=False):
        row_index = item_lookup[row.item_label]
        column_index = x_lookup[row.judge_label]
        value_matrix[row_index, column_index] = float(row.bias_size)
        significance_matrix[row_index, column_index] = significance_marker(float(row.p_value))
    return value_matrix, significance_matrix


def significance_marker(p_value: float) -> str:
    """Return star notation for the configured p-value thresholds."""

    if p_value < 0.001:
        return r"$\ast\ast\ast$"
    if p_value < 0.05:
        return r"$\ast\ast$"
    if p_value < 0.1:
        return r"$\ast$"
    return ""


def draw_severity_heatmap(
    axis: plt.Axes,
    value_matrix: np.ndarray,
    column_widths: list[float],
    heatmap_colors: list[str],
    bad_color: str,
    cell_edge_color: str,
    cell_edge_width: float,
) -> QuadMesh:
    """Draw the severity-adjustment heatmap with a neutral center at zero."""

    finite_values = value_matrix[np.isfinite(value_matrix)]
    distance = max(abs(float(np.nanmin(finite_values))), abs(float(np.nanmax(finite_values))))
    norm = mcolors.TwoSlopeNorm(vmin=-distance, vcenter=0.0, vmax=distance)
    color_map = mcolors.LinearSegmentedColormap.from_list(
        "severity_decomposition_red_white_black",
        heatmap_colors,
    )
    color_map.set_bad(bad_color)
    x_edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    y_edges = np.arange(-0.5, value_matrix.shape[0] + 0.5, 1.0)
    return axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(value_matrix),
        cmap=color_map,
        norm=norm,
        edgecolors=cell_edge_color,
        linewidth=cell_edge_width,
    )


def style_severity_heatmap_axis(
    axis: plt.Axes,
    x_layout: pd.DataFrame,
    column_widths: list[float],
    item_order: list[str],
    item_labels: dict[str, str],
    x_label_rotation: float,
    x_label_pad: float,
    x_tick_label_size: float,
    y_tick_label_size: float,
) -> None:
    """Apply item and model labels to a severity decomposition heatmap axis."""

    axis.set_yticks(list(range(len(item_order))))
    y_labels = fix_labels_for_tex_style([item_labels[item_id] for item_id in item_order])
    axis.set_yticklabels([bold_text(label) for label in y_labels], fontsize=y_tick_label_size)

    tick_positions = x_layout["x_position"].astype(float).tolist()
    x_labels = fix_labels_for_tex_style(x_layout["model_label"].astype(str).tolist())
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [bold_text(label) for label in x_labels],
        rotation=x_label_rotation,
        ha="right",
        fontsize=x_tick_label_size,
    )
    axis.tick_params(axis="x", length=0, pad=x_label_pad)
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(0.0, float(sum(column_widths)))
    axis.set_ylim(len(item_order) - 0.5, -0.5)
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_significance_markers(
    axis: plt.Axes,
    significance_matrix: np.ndarray,
    column_widths: list[float],
    marker_size: float,
    marker_color: str,
    stroke_color: str,
    stroke_width: float,
) -> None:
    """Overlay star markers on cells where adjustments differ significantly from zero."""

    x_edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    row_indices, column_indices = np.where(significance_matrix != "")
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        marker_text = axis.text(
            x_centers[column_index],
            row_index,
            significance_matrix[row_index, column_index],
            ha="center",
            va="center",
            fontsize=marker_size,
            color=marker_color,
            zorder=4,
        )
        marker_text.set_path_effects(
            [
                path_effects.Stroke(linewidth=stroke_width, foreground=stroke_color),
                path_effects.Normal(),
            ]
        )


def add_severity_heatmap_colorbar(
    figure: matplotlib.figure.Figure,
    axis: plt.Axes,
    heatmap: QuadMesh,
    fraction: float,
    pad: float,
    label: str,
    label_size: float,
    label_pad: float,
    tick_size: float,
    top_label: str,
    bottom_label: str,
    endpoint_label_size: float,
    top_label_y: float,
    bottom_label_y: float,
) -> None:
    """Add a labeled colorbar to a severity decomposition heatmap."""

    colorbar = figure.colorbar(heatmap, ax=axis, fraction=fraction, pad=pad)
    colorbar.set_label(
        bold_text(label),
        fontsize=label_size,
        rotation=270,
        labelpad=label_pad,
    )
    colorbar.ax.tick_params(labelsize=tick_size)
    colorbar.ax.text(
        0.5,
        top_label_y,
        bold_text(top_label),
        transform=colorbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=endpoint_label_size,
    )
    colorbar.ax.text(
        0.5,
        bottom_label_y,
        bold_text(bottom_label),
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=endpoint_label_size,
    )


def _require_columns(dataframe: pd.DataFrame, columns: set[str], source_name: str) -> None:
    """Raise a clear error if a parsed dataframe lacks required columns."""

    missing_columns = columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source_name} is missing columns: {missing_text}")
