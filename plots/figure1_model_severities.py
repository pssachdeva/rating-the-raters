"""Plot Figure 1 model severities with an AA Intelligence Index scatter panel."""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_lego.labels import apply_subplot_labels
import numpy as np
import pandas as pd

from mhs_llms.facets.postprocess import parse_facets_score_file
from mhs_llms.labels import infer_provider, model_id_to_plot_label, provider_display_name
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR
from mhs_llms.plotting import (
    PROVIDER_COLORS,
    apply_plot_style,
    build_gaussian_kde_curve,
    format_plot_text,
    get_provider_color,
    save_figure,
)


MODEL_IDS = [
    "openai_gpt-5.4_medium",
    "google_gemini-3.1-pro-preview_medium",
    "anthropic_claude-opus-4-6_medium",
    "xai_grok-4-1-fast-reasoning",
    "deepseek_deepseek-v4-pro",
    "openrouter_minimax_minimax-m2.5",
    "moonshot_kimi-k2.5",
    "together_openai_gpt-oss-120b",
    "together_meta-llama_llama-3.3-70b-instruct-turbo",
]
HUMAN_SCORE_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.2.txt"
JUDGE_SCORE_PATHS = [FACETS_DIR / "full_run_models_reference_set" / "judges_scores.csv"]
AA_SEVERITY_PATH = Path("data/reference_set_model_severities_latest.csv")
OUTPUT_PATH = ARTIFACTS_DIR / "figure1_model_severities.pdf"

FIGURE_WIDTH = 15.0
FIGURE_HEIGHT = 5.3
SAVE_DPI = 300
GRID_HEIGHT_RATIOS = [0.9, 1.9]
GRID_WIDTH_RATIOS = [1.85, 1.55]
GRID_HSPACE = 0.02
GRID_WSPACE = 0.02

SEVERITY_X_LIMITS = (-1.5, 1.5)
SEVERITY_X_PADDING = 0.0
BAR_ALPHA = 0.95
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 1.1
ERROR_LINE_WIDTH = 1.3
ERROR_CAP_SIZE = 3.2
ERROR_COLOR = "#202020"
BOTTOM_LABEL_FONT_SIZE = 10.0
VALUE_LABEL_FONT_SIZE = 8.0
LEFT_TICK_LABEL_SIZE = 10.0
LEFT_AXIS_LABEL_SIZE = 12.0
MODEL_LABEL_PAD_FRACTION = 0.012
VALUE_LABEL_PAD_FRACTION = 0.010

HUMAN_DISTRIBUTION_COLOR = "#8CA2CF"
HUMAN_DENSITY_LINE_WIDTH = 2.6
HUMAN_DENSITY_FILL_ALPHA = 0.32
HUMAN_MODEL_LINE_WIDTH = 1.6
ZERO_LINE_COLOR = "#202020"
ZERO_LINE_WIDTH = 1.1
LEFT_GRID_ALPHA = 0.18
AXIS_BACKGROUND_COLOR = "#F7F7F7"
HUMAN_Y_LABEL = "Human Annotator\nSeverity Density"
MODEL_Y_LABEL = "Models"
MODEL_X_LABEL = "Severity"

DIRECTION_LABEL_LEFT = "More Likely to\nLabel as Hateful"
DIRECTION_LABEL_RIGHT = "Less Likely to\nLabel as Hateful"
DIRECTION_LABEL_FONT_SIZE = 8.0
DIRECTION_LABEL_Y_POSITION = 0.95
DIRECTION_LABEL_INSET_FRACTION = 0.10
DIRECTION_LABEL_BOX = {
    "boxstyle": "round,pad=0.42,rounding_size=0.28",
    "facecolor": "white",
    "edgecolor": "#D0D0D0",
    "linewidth": 0.8,
    "alpha": 1,
}

AA_VARIANT_SUFFIXES = ("_minimal", "_low", "_medium", "_high", "_xhigh", "_none")
AA_X_COLUMN = "aa_intelligence_index"
AA_Y_COLUMN = "measure"
AA_OTHER_PROVIDER_SLUGS = {"qwen", "xiaomi", "zai"}
AA_MARKER_SIZE = 38.0
AA_MARKER_ALPHA = 0.82
AA_MARKER_EDGE_COLOR = "#222222"
AA_MARKER_EDGE_WIDTH = 0.45
AA_TREND_COLOR = "#222222"
AA_TREND_LINE_WIDTH = 1.4
AA_ZERO_LINE_COLOR = "#777777"
AA_ZERO_LINE_WIDTH = 0.8
AA_ZERO_LINE_ALPHA = 0.55
AA_GRID_ALPHA = 0.22
AA_GRID_LINE_WIDTH = 0.7
AA_LEGEND_FONT_SIZE = 10.0
AA_LEGEND_LOCATION = "lower right"
AA_LEGEND_COLUMNS = 3
AA_LEGEND_HANDLE_TEXT_PAD = 0.35
AA_LEGEND_COLUMN_SPACING = 0.8
AA_LEGEND_FRAME_ON = True
AA_AXIS_LABEL_SIZE = 12.0
AA_TICK_LABEL_SIZE = 10.0
AA_X_LABEL = "AA Intelligence Index"
AA_Y_LABEL = r"Severity ($\alpha_j$)"
AA_BOX_ASPECT = 1.0
AA_PROVIDER_ORDER = [
    "anthropic",
    "deepseek",
    "google",
    "meta",
    "minimax",
    "moonshotai",
    "openai",
    "xai",
    "other",
]
SUBPLOT_LABEL_FONT_SIZE = 14.0
SUBPLOT_LABEL_A_X = -0.03
SUBPLOT_LABEL_B_X = 0.05
SUBPLOT_LABEL_Y = 1.06
SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT = "left"
SUBPLOT_LABEL_VERTICAL_ALIGNMENT = "top"


def main() -> None:
    """Build and save the Figure 1 model severity plot."""
    apply_plot_style()

    human_severity_frame = load_human_judge_severities(HUMAN_SCORE_PATH)
    model_severity_frame = load_model_judge_severities(JUDGE_SCORE_PATHS)
    model_severity_frame = select_model_panel(model_severity_frame, MODEL_IDS)
    aa_severity_frame = load_medium_effort_aa_severities(AA_SEVERITY_PATH)

    figure = build_figure(human_severity_frame, model_severity_frame, aa_severity_frame)
    output_path = save_figure(figure, OUTPUT_PATH, dpi=SAVE_DPI)
    plt.close(figure)

    print(f"output={output_path.resolve()}")
    print(f"aa_scatter_n={len(aa_severity_frame)}")


def build_figure(
    human_severity_frame: pd.DataFrame,
    model_severity_frame: pd.DataFrame,
    aa_severity_frame: pd.DataFrame,
) -> plt.Figure:
    """Build the complete Figure 1 layout with severity and AA scatter panels."""
    plot_min, plot_max = build_severity_plot_bounds(human_severity_frame, model_severity_frame)
    human_measures = human_severity_frame["measure"].astype(float).tolist()
    density_x, density_y = build_gaussian_kde_curve(human_measures, plot_min, plot_max)

    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=GRID_HEIGHT_RATIOS,
        width_ratios=GRID_WIDTH_RATIOS,
        hspace=GRID_HSPACE,
        wspace=GRID_WSPACE,
    )
    left_label_axis = figure.add_subplot(grid[:, 0])
    left_label_axis.axis("off")
    scatter_label_axis = figure.add_subplot(grid[:, 1])
    scatter_label_axis.axis("off")
    top_axis = figure.add_subplot(grid[0, 0])
    bottom_axis = figure.add_subplot(grid[1, 0], sharex=top_axis)
    scatter_axis = figure.add_subplot(grid[:, 1])

    plot_human_density_panel(top_axis, model_severity_frame, density_x, density_y)
    plot_model_bar_panel(bottom_axis, model_severity_frame, plot_min, plot_max)
    plot_aa_scatter_panel(scatter_axis, aa_severity_frame)
    add_subplot_labels(left_label_axis, scatter_label_axis)

    figure.align_ylabels([top_axis, bottom_axis])
    return figure


def load_human_judge_severities(judges_path: Path) -> pd.DataFrame:
    """Load FACETS human judge severity measures for the density panel."""
    score_frame = read_facets_score_table(judges_path)
    validate_required_columns(score_frame, judges_path, {"measure"})
    return score_frame.loc[:, ["measure"]].copy()


def load_model_judge_severities(judges_paths: list[Path]) -> pd.DataFrame:
    """Load, label, and sort FACETS model judge severities."""
    severity_frames = []
    for judges_path in judges_paths:
        score_frame = read_facets_score_table(judges_path)
        validate_required_columns(score_frame, judges_path, {"facet_label", "measure", "s_e"})
        severity_frames.append(score_frame.loc[:, ["facet_label", "measure", "s_e"]].copy())

    combined = pd.concat(severity_frames, ignore_index=True)
    if combined["facet_label"].duplicated().any():
        duplicate_ids = sorted(set(combined.loc[combined["facet_label"].duplicated(), "facet_label"]))
        raise ValueError(f"Duplicate model ids found across judge score files: {duplicate_ids}")

    combined["provider"] = combined["facet_label"].map(infer_provider)
    combined["provider_label"] = combined["provider"].map(provider_display_name)
    combined["display_label"] = combined["facet_label"].map(model_id_to_plot_label)
    return combined.sort_values(["measure", "display_label"], ascending=[False, True]).reset_index(
        drop=True
    )


def read_facets_score_table(score_path: Path) -> pd.DataFrame:
    """Read one FACETS score table from either raw text or processed CSV."""
    if score_path.suffix.lower() == ".csv":
        return pd.read_csv(score_path)
    return parse_facets_score_file(score_path)


def validate_required_columns(
    score_frame: pd.DataFrame,
    score_path: Path,
    required_columns: set[str],
) -> None:
    """Validate that a loaded table contains the required columns."""
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {score_path}: {missing_list}")


def select_model_panel(model_severity_frame: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    """Validate and sort the configured full-run model panel by severity."""
    requested_ids = list(dict.fromkeys(model_ids))
    available_ids = set(model_severity_frame["facet_label"].astype(str).tolist())
    missing_ids = [model_id for model_id in requested_ids if model_id not in available_ids]
    if missing_ids:
        missing_list = ", ".join(missing_ids)
        raise ValueError(f"Model severity frame is missing requested models: {missing_list}")

    selected = model_severity_frame.loc[model_severity_frame["facet_label"].isin(requested_ids)].copy()
    return selected.sort_values(
        ["measure", "display_label"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def load_medium_effort_aa_severities(data_path: Path) -> pd.DataFrame:
    """Load AA rows and keep medium effort rows when effort variants repeat."""
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

    data = data.dropna(subset=[AA_X_COLUMN, AA_Y_COLUMN]).copy()
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


def strip_effort_suffix(judge_id: str) -> str:
    """Return a model-family id without the local reasoning-effort suffix."""
    for suffix in AA_VARIANT_SUFFIXES:
        if judge_id.endswith(suffix):
            return judge_id[: -len(suffix)]
    return judge_id


def infer_scatter_provider(row: pd.Series) -> str:
    """Return the provider grouping used by the AA scatter legend."""
    judge_id = str(row["judge_id"])
    if judge_id.startswith("together_openai_"):
        return "openai"
    if judge_id.startswith("together_meta-llama_"):
        return "meta"

    provider = str(row["provider"])
    if provider in AA_OTHER_PROVIDER_SLUGS:
        return "other"
    return provider


def build_severity_plot_bounds(
    human_severity_frame: pd.DataFrame,
    model_severity_frame: pd.DataFrame,
) -> tuple[float, float]:
    """Build the severity x-axis limits for the left panels."""
    if human_severity_frame.empty:
        raise ValueError("Human severity frame is empty")
    if model_severity_frame.empty:
        raise ValueError("Model severity frame is empty")

    if SEVERITY_X_LIMITS is not None:
        return SEVERITY_X_LIMITS

    human_min = float(human_severity_frame["measure"].min())
    human_max = float(human_severity_frame["measure"].max())
    model_min = float((model_severity_frame["measure"] - model_severity_frame["s_e"]).min())
    model_max = float((model_severity_frame["measure"] + model_severity_frame["s_e"]).max())
    plot_min = min(human_min, model_min)
    plot_max = max(human_max, model_max)
    padding = max((plot_max - plot_min) * 0.08, 0.12)
    return plot_min - padding, plot_max + padding


def plot_human_density_panel(
    axis: plt.Axes,
    model_severity_frame: pd.DataFrame,
    density_x: list[float],
    density_y: list[float],
) -> None:
    """Plot the human annotator severity density panel."""
    axis.set_facecolor(AXIS_BACKGROUND_COLOR)
    axis.plot(density_x, density_y, color=HUMAN_DISTRIBUTION_COLOR, linewidth=HUMAN_DENSITY_LINE_WIDTH)
    axis.fill_between(
        density_x,
        density_y,
        color=HUMAN_DISTRIBUTION_COLOR,
        alpha=HUMAN_DENSITY_FILL_ALPHA,
    )
    for row in model_severity_frame.itertuples(index=False):
        axis.axvline(
            row.measure,
            color=get_provider_color(row.provider),
            linewidth=HUMAN_MODEL_LINE_WIDTH,
            alpha=0.85,
            linestyle="--",
            zorder=3,
        )

    axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=4)
    axis.set_ylabel(format_plot_text(HUMAN_Y_LABEL), fontsize=LEFT_TICK_LABEL_SIZE)
    axis.set_xlim(*SEVERITY_X_LIMITS)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=LEFT_GRID_ALPHA)
    axis.tick_params(axis="y", labelsize=LEFT_TICK_LABEL_SIZE)
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def plot_model_bar_panel(
    axis: plt.Axes,
    model_severity_frame: pd.DataFrame,
    plot_min: float,
    plot_max: float,
) -> None:
    """Plot the horizontal model severity bar panel."""
    y_positions = list(range(len(model_severity_frame)))
    colors = [get_provider_color(provider) for provider in model_severity_frame["provider"].tolist()]
    axis.set_facecolor(AXIS_BACKGROUND_COLOR)
    axis.barh(
        y_positions,
        model_severity_frame["measure"],
        xerr=model_severity_frame["s_e"],
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        alpha=BAR_ALPHA,
        error_kw={"elinewidth": ERROR_LINE_WIDTH, "capsize": ERROR_CAP_SIZE, "ecolor": ERROR_COLOR},
    )

    axis.set_yticks(y_positions)
    axis.set_yticklabels([])
    axis.tick_params(axis="y", which="both", right=False, left=False)
    axis.tick_params(axis="x", labelsize=LEFT_TICK_LABEL_SIZE)
    axis.invert_yaxis()
    axis.set_ylim(len(model_severity_frame) - 0.1, -0.90)
    axis.axvline(0.0, color="#444444", linewidth=ZERO_LINE_WIDTH)
    axis.set_xlim(plot_min - SEVERITY_X_PADDING, plot_max + SEVERITY_X_PADDING)
    axis.set_xlabel(format_plot_text(MODEL_X_LABEL), fontsize=LEFT_AXIS_LABEL_SIZE)
    axis.set_ylabel(format_plot_text(MODEL_Y_LABEL), fontsize=LEFT_AXIS_LABEL_SIZE)
    axis.yaxis.labelpad = -8
    axis.grid(axis="x", alpha=LEFT_GRID_ALPHA)
    add_model_bar_labels(axis, model_severity_frame, plot_min, plot_max)
    add_direction_labels(axis, plot_min, plot_max)


def add_model_bar_labels(
    axis: plt.Axes,
    model_severity_frame: pd.DataFrame,
    plot_min: float,
    plot_max: float,
) -> None:
    """Add model names and severity values to the model severity bars."""
    x_range = plot_max - plot_min
    label_pad = x_range * MODEL_LABEL_PAD_FRACTION
    value_pad = x_range * VALUE_LABEL_PAD_FRACTION

    for y_position, row in enumerate(model_severity_frame.itertuples(index=False)):
        value = float(row.measure)
        standard_error = float(row.s_e)
        model_label_x = label_pad if value <= 0.0 else -label_pad
        model_label_alignment = "left" if value <= 0.0 else "right"
        value_x = (
            value - standard_error - value_pad
            if value <= 0.0
            else value + standard_error + value_pad
        )
        value_alignment = "right" if value <= 0.0 else "left"
        axis.text(
            model_label_x,
            y_position,
            format_plot_text(row.display_label),
            va="center",
            ha=model_label_alignment,
            fontsize=BOTTOM_LABEL_FONT_SIZE,
        )
        axis.text(
            value_x,
            y_position,
            format_plot_text(f"{value:.3f}"),
            va="center",
            ha=value_alignment,
            fontsize=VALUE_LABEL_FONT_SIZE,
        )


def add_direction_labels(axis: plt.Axes, plot_min: float, plot_max: float) -> None:
    """Add directional interpretation labels to the model severity panel."""
    inset = (plot_max - plot_min) * DIRECTION_LABEL_INSET_FRACTION
    axis.text(
        plot_min + inset,
        DIRECTION_LABEL_Y_POSITION,
        format_plot_text(DIRECTION_LABEL_LEFT),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=DIRECTION_LABEL_FONT_SIZE,
        color="#555555",
        bbox=DIRECTION_LABEL_BOX,
    )
    axis.text(
        plot_max - inset,
        DIRECTION_LABEL_Y_POSITION,
        format_plot_text(DIRECTION_LABEL_RIGHT),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=DIRECTION_LABEL_FONT_SIZE,
        color="#555555",
        bbox=DIRECTION_LABEL_BOX,
    )


def plot_aa_scatter_panel(axis: plt.Axes, data: pd.DataFrame) -> None:
    """Plot AA Intelligence Index against FACETS severity."""
    axis.set_facecolor(AXIS_BACKGROUND_COLOR)
    for provider in AA_PROVIDER_ORDER:
        provider_data = data.loc[data["scatter_provider"] == provider]
        if provider_data.empty:
            continue
        axis.scatter(
            provider_data[AA_X_COLUMN],
            provider_data[AA_Y_COLUMN],
            s=AA_MARKER_SIZE,
            alpha=AA_MARKER_ALPHA,
            color=PROVIDER_COLORS.get(provider, PROVIDER_COLORS["unknown"]),
            edgecolor=AA_MARKER_EDGE_COLOR,
            linewidth=AA_MARKER_EDGE_WIDTH,
            label=provider_data["scatter_provider_label"].iloc[0],
        )

    add_aa_trend_line(axis, data)
    axis.axhline(
        0,
        color=AA_ZERO_LINE_COLOR,
        linewidth=AA_ZERO_LINE_WIDTH,
        alpha=AA_ZERO_LINE_ALPHA,
        zorder=0,
    )
    axis.set_xlabel(format_plot_text(AA_X_LABEL), fontsize=AA_AXIS_LABEL_SIZE)
    axis.set_ylabel(format_plot_text(AA_Y_LABEL), fontsize=AA_AXIS_LABEL_SIZE)
    axis.set_box_aspect(AA_BOX_ASPECT)
    axis.tick_params(labelsize=AA_TICK_LABEL_SIZE)
    axis.grid(True, alpha=AA_GRID_ALPHA, linewidth=AA_GRID_LINE_WIDTH)
    axis.legend(
        frameon=AA_LEGEND_FRAME_ON,
        fontsize=AA_LEGEND_FONT_SIZE,
        loc=AA_LEGEND_LOCATION,
        ncol=AA_LEGEND_COLUMNS,
        handletextpad=AA_LEGEND_HANDLE_TEXT_PAD,
        columnspacing=AA_LEGEND_COLUMN_SPACING,
    )


def add_aa_trend_line(axis: plt.Axes, data: pd.DataFrame) -> None:
    """Add an ordinary least-squares trend line to the AA scatter panel."""
    x_values = data[AA_X_COLUMN].to_numpy(dtype=float)
    y_values = data[AA_Y_COLUMN].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    axis.plot(line_x, slope * line_x + intercept, color=AA_TREND_COLOR, linewidth=AA_TREND_LINE_WIDTH)


def add_subplot_labels(left_axis: plt.Axes, right_axis: plt.Axes) -> None:
    """Add subplot labels to the left panel group and right scatter panel."""
    apply_subplot_labels(
        left_axis,
        labels=["a"],
        bold=True,
        x=SUBPLOT_LABEL_A_X,
        y=SUBPLOT_LABEL_Y,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
        size=SUBPLOT_LABEL_FONT_SIZE,
    )
    apply_subplot_labels(
        right_axis,
        labels=["b"],
        bold=True,
        x=SUBPLOT_LABEL_B_X,
        y=SUBPLOT_LABEL_Y,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
        size=SUBPLOT_LABEL_FONT_SIZE,
    )

if __name__ == "__main__":
    main()
