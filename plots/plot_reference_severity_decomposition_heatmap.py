"""Plot model item-dependent severity adjustments as an item-by-model heatmap."""

from cycler import cycler
from matplotlib.collections import QuadMesh
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import numpy as np
import pandas as pd
from pathlib import Path

from mhs_llms.labels import model_id_to_plot_label
from mhs_llms.facets.postprocess import parse_facets_score_file
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR, FACETS_DIR


MODEL_ORDER = [
    "openai_gpt-4o",
    "openai_gpt-5.4_medium",
    "google_gemini-2.5-pro",
    "google_gemini-3.1-pro-preview_medium",
    "anthropic_claude-sonnet-4-6_medium",
    "anthropic_claude-opus-4-6_medium",
    "xai_grok-3",
    "xai_grok-4-fast-reasoning",
    "openrouter_moonshotai_kimi-k2.5",
    "openrouter_deepseek_deepseek-v3.2",
]
MODEL_LABELS = {
    "openai_gpt-4o": "GPT-4o",
    "openai_gpt-5.4_medium": "GPT-5.4",
    "google_gemini-2.5-pro": "Gemini 2.5 Pro",
    "google_gemini-3.1-pro-preview_medium": "Gemini 3.1 Pro",
    "anthropic_claude-sonnet-4-6_medium": "Claude Sonnet 4.6",
    "anthropic_claude-opus-4-6_medium": "Claude Opus 4.6",
    "xai_grok-3": "Grok 3",
    "xai_grok-4-fast-reasoning": "Grok 4",
    "openrouter_moonshotai_kimi-k2.5": "Kimi K2.5",
    "openrouter_deepseek_deepseek-v3.2": "DeepSeek V3.2",
}

DEFAULT_ITEM_ORDER = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hate_speech",
]
ITEM_ORDER = DEFAULT_ITEM_ORDER.copy()
ITEM_LABELS = {
    "sentiment": "Sentiment",
    "respect": "Respect",
    "insult": "Insult",
    "humiliate": "Humiliate",
    "status": "Status",
    "dehumanize": "Dehumanize",
    "violence": "Violence",
    "genocide": "Genocide",
    "attack_defend": "Attack/Defend",
    "hate_speech": "Hate Speech",
}

DATA_PATH = DATA_DIR / "reference_set_all_severity_decomposition_bias_terms.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_severity_decomposition_heatmap.png"
HUMAN_ITEM_SCORE_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.3.txt"
MODEL_SCORE_PATH: Path | None = None

FIGSIZE = (9.1, 4.9)
DPI = 300
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
]
FIGURE_LEFT_MARGIN = 0.15
FIGURE_RIGHT_MARGIN = 0.90
FIGURE_TOP_MARGIN = 0.96
FIGURE_BOTTOM_MARGIN = 0.34
COLORBAR_PAD = 0.02
COLORBAR_FRACTION = 0.038
CELL_EDGE_COLOR = "white"
CELL_EDGE_WIDTH = 0.7
XTICK_LABEL_SIZE = 7.0
YTICK_LABEL_SIZE = 8.0
COLORBAR_LABEL_SIZE = 8.0
COLORBAR_TICK_SIZE = 7.2
COLORBAR_ENDPOINT_LABEL_SIZE = 7.0
COLORBAR_TOP_LABEL = "More\nsevere\nthan expected"
COLORBAR_BOTTOM_LABEL = "Less\nsevere\nthan expected"
SIGNIFICANCE_MARKER_SIZE = 8.0
SIGNIFICANCE_MARKER_COLOR = "#FFFFFF"
SIGNIFICANCE_STROKE_COLOR = "#111111"
SIGNIFICANCE_STROKE_WIDTH = 0.65
HEATMAP_COLORS = ["#C43C32", "#FFFFFF", "#000000"]
BAD_COLOR = "#FFFFFF"
X_LABEL_ROTATION = 32
X_LABEL_PAD = 0
COLORBAR_LABEL = "Item-dependent severity adjustment"
SAVE_PAD_INCHES = 0.08


def main() -> None:
    """Build and save the item-dependent severity adjustment heatmap."""

    global ITEM_ORDER, MODEL_ORDER
    ITEM_ORDER = load_human_item_difficulty_order(HUMAN_ITEM_SCORE_PATH)
    if MODEL_SCORE_PATH is not None:
        MODEL_ORDER = load_model_severity_order(MODEL_SCORE_PATH, MODEL_ORDER)

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    bias_terms = load_bias_terms(DATA_PATH, MODEL_ORDER)
    x_layout, column_widths = build_x_layout()
    value_matrix, significance_matrix = build_heatmap_matrices(
        bias_terms,
        x_layout,
        len(column_widths),
    )

    figure, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    figure.subplots_adjust(
        left=FIGURE_LEFT_MARGIN,
        right=FIGURE_RIGHT_MARGIN,
        top=FIGURE_TOP_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
    )
    heatmap = draw_heatmap(axis, value_matrix, column_widths)
    style_axis(axis, x_layout, column_widths)
    add_significance_markers(axis, significance_matrix, column_widths)
    add_colorbar(figure, axis, heatmap)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"output={OUTPUT_PATH.resolve()}")


def load_human_item_difficulty_order(score_path: str | Path) -> list[str]:
    """Load item labels ordered from easiest to hardest in the human baseline."""

    score_frame = parse_facets_score_file(Path(score_path))
    required_columns = {"facet_label", "measure"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Human item score file is missing columns: {missing_text}")

    item_scores = score_frame.loc[
        score_frame["facet_label"].isin(DEFAULT_ITEM_ORDER),
        ["facet_label", "measure"],
    ].copy()
    missing_items = sorted(set(DEFAULT_ITEM_ORDER).difference(item_scores["facet_label"]))
    if missing_items:
        missing_text = ", ".join(missing_items)
        raise ValueError(f"Human item score file is missing items: {missing_text}")

    item_scores = item_scores.sort_values(["measure", "facet_label"], kind="stable")
    return item_scores["facet_label"].astype(str).tolist()


def load_model_severity_order(score_path: str | Path, model_ids: list[str]) -> list[str]:
    """Load model labels ordered from lower to higher severity."""

    score_frame = parse_facets_score_file(Path(score_path))
    required_columns = {"facet_label", "measure"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Model score file is missing columns: {missing_text}")

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


def load_bias_terms(data_path: str, model_order: list[str]) -> pd.DataFrame:
    """Load item-dependent severity adjustments for selected models and items."""

    bias_terms = pd.read_csv(data_path)
    selected = bias_terms.loc[
        bias_terms["judge_label"].isin(model_order)
        & bias_terms["item_label"].isin(ITEM_ORDER)
    ].copy()

    missing_pairs = [
        (model_id, item_id)
        for model_id in model_order
        for item_id in ITEM_ORDER
        if selected.loc[
            (selected["judge_label"] == model_id) & (selected["item_label"] == item_id)
        ].empty
    ]
    if missing_pairs:
        missing_text = ", ".join(f"{model}:{item}" for model, item in missing_pairs[:10])
        raise ValueError(f"Bias-term dataset is missing model-item pairs: {missing_text}")
    return selected


def build_x_layout() -> tuple[pd.DataFrame, list[float]]:
    """Return plotted x positions and labels for model columns."""

    rows = []
    column_widths = []
    x_left = 0.0
    for model_id in MODEL_ORDER:
        matrix_column = len(column_widths)
        rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
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
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build adjustment and significance matrices in item-row by model-column order."""

    value_matrix = np.full((len(ITEM_ORDER), column_count), np.nan)
    significance_matrix = np.full((len(ITEM_ORDER), column_count), "", dtype=object)
    x_lookup = {row.model_id: int(row.matrix_column) for row in x_layout.itertuples(index=False)}
    item_lookup = {item_id: index for index, item_id in enumerate(ITEM_ORDER)}

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


def draw_heatmap(
    axis: plt.Axes,
    value_matrix: np.ndarray,
    column_widths: list[float],
) -> QuadMesh:
    """Draw the severity-adjustment heatmap with a neutral center at zero."""

    finite_values = value_matrix[np.isfinite(value_matrix)]
    distance = max(abs(float(np.nanmin(finite_values))), abs(float(np.nanmax(finite_values))))
    norm = mcolors.TwoSlopeNorm(vmin=-distance, vcenter=0.0, vmax=distance)
    color_map = mcolors.LinearSegmentedColormap.from_list(
        "red_white_black",
        HEATMAP_COLORS,
    )
    color_map.set_bad(BAD_COLOR)
    x_edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    y_edges = np.arange(-0.5, len(ITEM_ORDER) + 0.5, 1.0)
    return axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(value_matrix),
        cmap=color_map,
        norm=norm,
        edgecolors=CELL_EDGE_COLOR,
        linewidth=CELL_EDGE_WIDTH,
    )


def style_axis(axis: plt.Axes, x_layout: pd.DataFrame, column_widths: list[float]) -> None:
    """Apply item and model labels to the heatmap axis."""

    axis.set_yticks(list(range(len(ITEM_ORDER))))
    item_labels = fix_labels_for_tex_style([ITEM_LABELS[item_id] for item_id in ITEM_ORDER])
    axis.set_yticklabels([bold_text(label) for label in item_labels], fontsize=YTICK_LABEL_SIZE)

    tick_positions = x_layout["x_position"].astype(float).tolist()
    model_labels = [
        MODEL_LABELS.get(model_id, model_id_to_plot_label(model_id))
        for model_id in x_layout["model_id"].tolist()
    ]
    tick_labels = fix_labels_for_tex_style(model_labels)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [bold_text(label) for label in tick_labels],
        rotation=X_LABEL_ROTATION,
        ha="right",
        fontsize=XTICK_LABEL_SIZE,
    )
    axis.tick_params(axis="x", length=0, pad=X_LABEL_PAD)
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(0.0, float(sum(column_widths)))
    axis.set_ylim(len(ITEM_ORDER) - 0.5, -0.5)
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_significance_markers(
    axis: plt.Axes,
    significance_matrix: np.ndarray,
    column_widths: list[float],
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
            fontsize=SIGNIFICANCE_MARKER_SIZE,
            color=SIGNIFICANCE_MARKER_COLOR,
            zorder=4,
        )
        marker_text.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=SIGNIFICANCE_STROKE_WIDTH,
                    foreground=SIGNIFICANCE_STROKE_COLOR,
                ),
                path_effects.Normal(),
            ]
        )


def add_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    heatmap: QuadMesh,
) -> None:
    """Add the severity-adjustment colorbar."""

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
    )
    colorbar.set_label(
        bold_text(COLORBAR_LABEL),
        fontsize=COLORBAR_LABEL_SIZE,
        rotation=270,
        labelpad=12,
    )
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    colorbar.ax.text(
        0.5,
        1.03,
        bold_text(COLORBAR_TOP_LABEL),
        transform=colorbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=COLORBAR_ENDPOINT_LABEL_SIZE,
    )
    colorbar.ax.text(
        0.5,
        -0.03,
        bold_text(COLORBAR_BOTTOM_LABEL),
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=COLORBAR_ENDPOINT_LABEL_SIZE,
    )


if __name__ == "__main__":
    main()
