"""Plot target-identity DRF odds multipliers as a model-by-identity heatmap."""

from pathlib import Path
import math

from cycler import cycler
from matplotlib.collections import QuadMesh
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import numpy as np
import pandas as pd

from rating_raters.paths import ARTIFACTS_DIR, DATA_DIR


POOLED_TARGET_TERMS_PATH = DATA_DIR / "full_set_all_models_target_drf_free_judges_target_terms.csv"
MODEL_TERM_PATHS = {
    "openai_gpt-5.4_medium": POOLED_TARGET_TERMS_PATH,
    "anthropic_claude-opus-4-6_medium": POOLED_TARGET_TERMS_PATH,
    "google_gemini-3.1-pro-preview_medium": POOLED_TARGET_TERMS_PATH,
    "xai_grok-4-1-fast-reasoning": POOLED_TARGET_TERMS_PATH,
    "deepseek_deepseek-v4-pro": POOLED_TARGET_TERMS_PATH,
    "moonshot_kimi-k2.5": POOLED_TARGET_TERMS_PATH,
    "openrouter_minimax_minimax-m2.5": POOLED_TARGET_TERMS_PATH,
    "together_openai_gpt-oss-120b": POOLED_TARGET_TERMS_PATH,
    "together_meta-llama_llama-3.3-70b-instruct-turbo": POOLED_TARGET_TERMS_PATH,
}
MODEL_ORDER = [
    "anthropic_claude-opus-4-6_medium",
    "google_gemini-3.1-pro-preview_medium",
    "openai_gpt-5.4_medium",
    "xai_grok-4-1-fast-reasoning",
    "deepseek_deepseek-v4-pro",
    "moonshot_kimi-k2.5",
    "openrouter_minimax_minimax-m2.5",
    "together_openai_gpt-oss-120b",
    "together_meta-llama_llama-3.3-70b-instruct-turbo",
]
MODEL_LABELS = {
    "openai_gpt-5.4_medium": "GPT-5.4",
    "anthropic_claude-opus-4-6_medium": "Claude Opus 4.6",
    "google_gemini-3.1-pro-preview_medium": "Gemini 3.1 Pro",
    "xai_grok-4-1-fast-reasoning": "Grok 4.1",
    "deepseek_deepseek-v4-pro": "DeepSeek V4 Pro",
    "moonshot_kimi-k2.5": "Kimi K2.5",
    "openrouter_minimax_minimax-m2.5": "MiniMax M2.5",
    "together_openai_gpt-oss-120b": "GPT-OSS 120B",
    "together_meta-llama_llama-3.3-70b-instruct-turbo": "Llama 3.3 70B",
}
IDENTITY_GROUPS = {
    "Gender": [
        ("target_gender_men", "Men"),
        ("target_gender_women", "Women"),
        ("target_gender_transgender", "Transgender"),
    ],
    "Race/Ethnicity": [
        ("target_race_asian", "Asian"),
        ("target_race_black", "Black"),
        ("target_race_latinx", "Latinx"),
        ("target_race_middle_eastern", "Middle Eastern"),
        ("target_race_white", "White"),
    ],
    "Religion": [
        ("target_religion_christian", "Christian"),
        ("target_religion_jewish", "Jewish"),
        ("target_religion_muslim", "Muslim"),
    ],
    "Sexuality": [
        ("target_sexuality_bisexual", "Bisexual"),
        ("target_sexuality_gay", "Gay"),
        ("target_sexuality_lesbian", "Lesbian"),
    ],
}
TARGET_LABELS = {
    target_id: target_label
    for targets in IDENTITY_GROUPS.values()
    for target_id, target_label in targets
}
TARGET_ORDER = [target_id for targets in IDENTITY_GROUPS.values() for target_id, _ in targets]

OUTPUT_PATH = ARTIFACTS_DIR / "figure4_target_drf_heatmap.pdf"
FIGSIZE = (8.8, 4.2)
DPI = 300
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
]
FIGURE_LEFT_MARGIN = 0.16
FIGURE_RIGHT_MARGIN = 0.90
FIGURE_TOP_MARGIN = 0.90
FIGURE_BOTTOM_MARGIN = 0.33
COLORBAR_PAD = 0.02
COLORBAR_FRACTION = 0.038
CELL_EDGE_COLOR = "white"
CELL_EDGE_WIDTH = 0.7
GAP_WIDTH = 0.35
GROUP_LABEL_Y = 1.04
GROUP_LABEL_SIZE = 11.0
XTICK_LABEL_SIZE = 10.0
YTICK_LABEL_SIZE = 10.0
Y_TOP_ROW_INDEX = 0
Y_BOTTOM_ROW_INDEX = None
COLORBAR_LABEL_SIZE = 10.0
COLORBAR_LABEL_PAD = 16.0
COLORBAR_TICK_SIZE = 9.0
COLORBAR_ENDPOINT_LABEL_SIZE = 8.0
COLORBAR_TOP_LABEL = "Higher\nHate\nThreshold"
COLORBAR_BOTTOM_LABEL = "Lower\nHate\nThreshold"
COLORBAR_SCALE = "log"
COLORBAR_ODDS_MIN = 0.5
COLORBAR_ODDS_MAX = 2.0
COLORBAR_ODDS_TICKS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SIGNIFICANCE_MARKER_SIZE = 8.0
SIGNIFICANCE_MARKER_COLOR = "#FFFFFF"
SIGNIFICANCE_STROKE_COLOR = "#111111"
SIGNIFICANCE_STROKE_WIDTH = 0.65
HEATMAP_COLORS = ["#C43C32", "#FFFFFF", "#000000"]
BAD_COLOR = "#FFFFFF"
X_LABEL_ROTATION = 30
X_LABEL_PAD = 0
COLORBAR_LABEL = r"Odds multiplier $\exp(\beta_{jm})$"


def main() -> None:
    """Build and save the target-identity DRF odds heatmap."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    beta_terms = load_target_drf_terms(MODEL_TERM_PATHS, MODEL_ORDER)
    x_layout, column_widths = build_x_layout()
    odds_matrix, significance_matrix = build_heatmap_matrices(
        beta_terms,
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
    heatmap = draw_heatmap(axis, odds_matrix, column_widths)
    style_axis(axis, x_layout, column_widths)
    add_significance_markers(axis, significance_matrix, column_widths)
    add_group_labels(axis, x_layout)
    add_colorbar(figure, axis, heatmap)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print(f"output={OUTPUT_PATH.resolve()}")


def load_target_drf_terms(data_paths: dict[str, Path], model_order: list[str]) -> pd.DataFrame:
    """Load and combine target-DRF beta terms for selected models."""

    frames = []
    for model_id in model_order:
        data_path = data_paths[model_id]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing target-DRF terms file: {data_path}")
        frame = pd.read_csv(data_path)
        if "judge_label" in frame.columns:
            frame = frame.loc[frame["judge_label"].eq(model_id)].copy()
            if frame.empty:
                raise ValueError(f"Missing pooled target-DRF rows for model: {model_id}")
        frame["model_id"] = model_id
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined["target_identity"].isin(TARGET_ORDER)].copy()
    missing_pairs = [
        (model_id, target_id)
        for model_id in model_order
        for target_id in TARGET_ORDER
        if combined.loc[
            (combined["model_id"] == model_id) & (combined["target_identity"] == target_id)
        ].empty
    ]
    if missing_pairs:
        missing_text = ", ".join(f"{model}:{target}" for model, target in missing_pairs[:10])
        raise ValueError(f"Target-DRF terms are missing model-target pairs: {missing_text}")
    return combined


def build_x_layout() -> tuple[pd.DataFrame, list[float]]:
    """Return plotted x positions, labels, and group spans for target identities."""

    rows = []
    column_widths = []
    x_left = 0.0
    group_items = list(IDENTITY_GROUPS.items())
    for group_index, (group_name, targets) in enumerate(group_items):
        for target_id, target_label in targets:
            matrix_column = len(column_widths)
            rows.append(
                {
                    "target_identity": target_id,
                    "target_label": target_label,
                    "group": group_name,
                    "x_position": x_left + 0.5,
                    "matrix_column": matrix_column,
                }
            )
            column_widths.append(1.0)
            x_left += 1.0
        if group_index < len(group_items) - 1:
            column_widths.append(GAP_WIDTH)
            x_left += GAP_WIDTH
    return pd.DataFrame(rows), column_widths


def build_heatmap_matrices(
    beta_terms: pd.DataFrame,
    x_layout: pd.DataFrame,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build odds and significance matrices in model-row by identity-column order."""

    odds_matrix = np.full((len(MODEL_ORDER), column_count), np.nan)
    significance_matrix = np.full((len(MODEL_ORDER), column_count), "", dtype=object)
    x_lookup = {
        row.target_identity: int(row.matrix_column) for row in x_layout.itertuples(index=False)
    }
    model_lookup = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}

    for row in beta_terms.itertuples(index=False):
        row_index = model_lookup[row.model_id]
        column_index = x_lookup[row.target_identity]
        odds_matrix[row_index, column_index] = math.exp(float(row.beta_jm))
        significance_matrix[row_index, column_index] = significance_marker(float(row.p_value))
    return odds_matrix, significance_matrix


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
    odds_matrix: np.ndarray,
    column_widths: list[float],
) -> QuadMesh:
    """Draw the odds-ratio heatmap with a neutral center at one."""

    finite_values = odds_matrix[np.isfinite(odds_matrix)]
    plot_values = transform_odds_for_color_scale(odds_matrix)
    if COLORBAR_SCALE == "log":
        log_values = np.log(finite_values)
        if COLORBAR_ODDS_MIN is None or COLORBAR_ODDS_MAX is None:
            scale_distance = float(np.nanmax(np.abs(log_values)))
        else:
            scale_distance = max(abs(math.log(COLORBAR_ODDS_MIN)), abs(math.log(COLORBAR_ODDS_MAX)))
        norm = mcolors.TwoSlopeNorm(vmin=-scale_distance, vcenter=0.0, vmax=scale_distance)
    elif COLORBAR_SCALE == "linear":
        scale_min = float(np.nanmin(finite_values)) if COLORBAR_ODDS_MIN is None else COLORBAR_ODDS_MIN
        scale_max = float(np.nanmax(finite_values)) if COLORBAR_ODDS_MAX is None else COLORBAR_ODDS_MAX
        norm = mcolors.TwoSlopeNorm(vmin=scale_min, vcenter=1.0, vmax=scale_max)
    else:
        raise ValueError(f"Unsupported COLORBAR_SCALE: {COLORBAR_SCALE}")
    color_map = mcolors.LinearSegmentedColormap.from_list(
        "red_white_black",
        HEATMAP_COLORS,
    )
    color_map.set_bad(BAD_COLOR)
    x_edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    y_edges = np.arange(-0.5, len(MODEL_ORDER) + 0.5, 1.0)
    return axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(plot_values),
        cmap=color_map,
        norm=norm,
        edgecolors=CELL_EDGE_COLOR,
        linewidth=CELL_EDGE_WIDTH,
    )


def transform_odds_for_color_scale(odds_matrix: np.ndarray) -> np.ndarray:
    """Return heatmap values in the configured odds color scale."""

    if COLORBAR_SCALE == "log":
        return np.log(odds_matrix)
    if COLORBAR_SCALE == "linear":
        return odds_matrix
    raise ValueError(f"Unsupported COLORBAR_SCALE: {COLORBAR_SCALE}")


def style_axis(axis: plt.Axes, x_layout: pd.DataFrame, column_widths: list[float]) -> None:
    """Apply model and target labels to the heatmap axis."""

    axis.set_yticks(list(range(len(MODEL_ORDER))))
    model_labels = fix_labels_for_tex_style([MODEL_LABELS[model_id] for model_id in MODEL_ORDER])
    axis.set_yticklabels([bold_text(label) for label in model_labels], fontsize=YTICK_LABEL_SIZE)

    tick_positions = x_layout["x_position"].astype(float).tolist()
    tick_labels = fix_labels_for_tex_style(x_layout["target_label"].tolist())
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
    bottom_row = len(MODEL_ORDER) - 1 if Y_BOTTOM_ROW_INDEX is None else Y_BOTTOM_ROW_INDEX
    axis.set_ylim(bottom_row + 0.5, Y_TOP_ROW_INDEX - 0.5)
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_significance_markers(
    axis: plt.Axes,
    significance_matrix: np.ndarray,
    column_widths: list[float],
) -> None:
    """Overlay star markers on cells where beta differs significantly from zero."""

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


def add_group_labels(axis: plt.Axes, x_layout: pd.DataFrame) -> None:
    """Add broader identity group labels above the heatmap."""

    for group_name, group_frame in x_layout.groupby("group", sort=False):
        group_midpoint = float(group_frame["x_position"].mean())
        axis.text(
            group_midpoint,
            GROUP_LABEL_Y,
            bold_text(group_name),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=GROUP_LABEL_SIZE,
            color="#333333",
            clip_on=False,
        )


def add_colorbar(
    figure: plt.Figure,
    axis: plt.Axes,
    heatmap: QuadMesh,
) -> None:
    """Add the odds-multiplier colorbar."""

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
        labelpad=COLORBAR_LABEL_PAD,
    )
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    odds_ticks = [
        tick
        for tick in COLORBAR_ODDS_TICKS
        if heatmap.norm.vmin <= transform_odds_tick_for_color_scale(tick) <= heatmap.norm.vmax
    ]
    colorbar.set_ticks([transform_odds_tick_for_color_scale(tick) for tick in odds_ticks])
    colorbar.set_ticklabels([f"{tick:g}" for tick in odds_ticks])
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


def transform_odds_tick_for_color_scale(tick: float) -> float:
    """Return a colorbar tick location in the configured odds color scale."""

    if COLORBAR_SCALE == "log":
        return math.log(tick)
    if COLORBAR_SCALE == "linear":
        return tick
    raise ValueError(f"Unsupported COLORBAR_SCALE: {COLORBAR_SCALE}")


if __name__ == "__main__":
    main()
