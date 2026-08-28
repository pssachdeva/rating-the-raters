"""Build Figure 2: item-dependent severity decomposition heatmap."""

from cycler import cycler
import matplotlib.pyplot as plt
from mpl_lego.style import use_latex_style

from rating_raters.facets.severity_decomposition_plot import (
    add_severity_heatmap_colorbar,
    add_significance_markers,
    build_heatmap_matrices,
    build_x_layout,
    draw_severity_heatmap,
    load_bias_terms,
    load_human_item_difficulty_order,
    load_model_severity_order,
    style_severity_heatmap_axis,
)
from rating_raters.labels import model_id_to_plot_label
from rating_raters.paths import ARTIFACTS_DIR, DATA_DIR, FACETS_DIR


MODEL_IDS = [
    "openai_gpt-5.4_medium",
    "anthropic_claude-opus-4-6_medium",
    "google_gemini-3.1-pro-preview_medium",
    "xai_grok-4-1-fast-reasoning",
    "moonshot_kimi-k2.5",
    "deepseek_deepseek-v4-pro",
    "openrouter_minimax_minimax-m2.5",
    "together_openai_gpt-oss-120b",
    "together_meta-llama_llama-3.3-70b-instruct-turbo",
]
ITEM_IDS = [
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

DATA_PATH = DATA_DIR / "full_run_models_reference_set_severity_decomposition_bias_terms.csv"
HUMAN_ITEM_SCORE_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.3.txt"
MODEL_SCORE_PATH = (
    FACETS_DIR / "full_run_models_reference_set" / "full_run_models_reference_set_scores.2.txt"
)
OUTPUT_PATH = ARTIFACTS_DIR / "figure2_severity_decomposition_heatmap.pdf"

FIGSIZE = (9.1, 4.8)
DPI = 300
SAVE_PAD_INCHES = 0.08
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
COLORBAR_LABEL_PAD = 12.0
COLORBAR_TOP_LABEL_Y = 1.03
COLORBAR_BOTTOM_LABEL_Y = -0.03
CELL_EDGE_COLOR = "white"
CELL_EDGE_WIDTH = 0.7
XTICK_LABEL_SIZE = 10.0
YTICK_LABEL_SIZE = 10.0
COLORBAR_LABEL_SIZE = 8.0
COLORBAR_TICK_SIZE = 7.2
COLORBAR_ENDPOINT_LABEL_SIZE = 9.0
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


def main() -> None:
    """Build and save the Figure 2 severity decomposition heatmap."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    item_order = load_human_item_difficulty_order(HUMAN_ITEM_SCORE_PATH, ITEM_IDS)
    model_order = load_model_severity_order(MODEL_SCORE_PATH, MODEL_IDS)
    model_labels = {model_id: model_id_to_plot_label(model_id) for model_id in MODEL_IDS}
    bias_terms = load_bias_terms(DATA_PATH, model_order, item_order)
    x_layout, column_widths = build_x_layout(model_order, model_labels)
    value_matrix, significance_matrix = build_heatmap_matrices(
        bias_terms=bias_terms,
        x_layout=x_layout,
        item_order=item_order,
        column_count=len(column_widths),
    )

    figure, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    figure.subplots_adjust(
        left=FIGURE_LEFT_MARGIN,
        right=FIGURE_RIGHT_MARGIN,
        top=FIGURE_TOP_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
    )
    heatmap = draw_severity_heatmap(
        axis=axis,
        value_matrix=value_matrix,
        column_widths=column_widths,
        heatmap_colors=HEATMAP_COLORS,
        bad_color=BAD_COLOR,
        cell_edge_color=CELL_EDGE_COLOR,
        cell_edge_width=CELL_EDGE_WIDTH,
    )
    style_severity_heatmap_axis(
        axis=axis,
        x_layout=x_layout,
        column_widths=column_widths,
        item_order=item_order,
        item_labels=ITEM_LABELS,
        x_label_rotation=X_LABEL_ROTATION,
        x_label_pad=X_LABEL_PAD,
        x_tick_label_size=XTICK_LABEL_SIZE,
        y_tick_label_size=YTICK_LABEL_SIZE,
    )
    add_significance_markers(
        axis=axis,
        significance_matrix=significance_matrix,
        column_widths=column_widths,
        marker_size=SIGNIFICANCE_MARKER_SIZE,
        marker_color=SIGNIFICANCE_MARKER_COLOR,
        stroke_color=SIGNIFICANCE_STROKE_COLOR,
        stroke_width=SIGNIFICANCE_STROKE_WIDTH,
    )
    add_severity_heatmap_colorbar(
        figure=figure,
        axis=axis,
        heatmap=heatmap,
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        label=COLORBAR_LABEL,
        label_size=COLORBAR_LABEL_SIZE,
        label_pad=COLORBAR_LABEL_PAD,
        tick_size=COLORBAR_TICK_SIZE,
        top_label=COLORBAR_TOP_LABEL,
        bottom_label=COLORBAR_BOTTOM_LABEL,
        endpoint_label_size=COLORBAR_ENDPOINT_LABEL_SIZE,
        top_label_y=COLORBAR_TOP_LABEL_Y,
        bottom_label_y=COLORBAR_BOTTOM_LABEL_Y,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"output={OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
