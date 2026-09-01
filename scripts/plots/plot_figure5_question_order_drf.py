"""Generate Figure 5: question-order differential rater functioning."""

from pathlib import Path
import math

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style
import pandas as pd

from rating_raters.facets.order_effect_plot import load_order_shift_comparison, load_pooled_order_delta
from rating_raters.paths import ARTIFACTS_DIR, DATA_DIR, FACETS_DIR
from rating_raters.plotting import get_provider_color


ORIGINAL_JUDGES_PATH = FACETS_DIR / "full_run_models_reference_set" / "judges_scores.csv"
REVERSE_JUDGES_PATH = FACETS_DIR / "question_order_full_run_reverse_matched" / "judges_scores.csv"
POOLED_ORDER_CONTRAST_PATH = DATA_DIR / "question_order_full_run_pooled_effect_order_contrast.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "figure5_question_order_drf.pdf"

COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]

FIGURE_WIDTH = 10.8
FIGURE_MIN_HEIGHT = 1.8
FIGURE_ROW_HEIGHT = 0.31
FIGURE_HEIGHT_PADDING = 0.85
SUBPLOT_WIDTH_RATIOS = [1.6, 0.9]
SUBPLOT_WSPACE = 0.08
DPI = 300
SAVE_PAD_INCHES = 0.02

MARKER_FORMAT = "o"
SEVERITY_CONDITION_Y_OFFSET = 0.20
SEVERITY_MARKER_SIZE = 8.
SEVERITY_ERROR_COLOR = "#303030"
SEVERITY_ERROR_CAPSIZE = 2.5
SEVERITY_MARKER_EDGE_WIDTH = 1.3
SEVERITY_REVERSE_MARKER_FACE = "white"
SEVERITY_ORIGINAL_ZORDER = 3
SEVERITY_REVERSE_ZORDER = 4
SEVERITY_NULL_LINE_VALUE = 0.0
SEVERITY_NULL_LINE_COLOR = "#888888"
SEVERITY_NULL_LINE_WIDTH = 1.0
SEVERITY_NULL_LINE_STYLE = "-"
SEVERITY_XLABEL = r"Severity ($\alpha_j$)"
SEVERITY_XLABEL_SIZE = 16

ODDS_RATIO_MARKER_SIZE = 8.5
ODDS_RATIO_ERROR_COLOR = "#303030"
ODDS_RATIO_ERROR_CAPSIZE = 2.8
ODDS_RATIO_MARKER_EDGE_COLOR = "white"
ODDS_RATIO_MARKER_EDGE_WIDTH = 0.8
ODDS_RATIO_ZORDER = 3
ODDS_RATIO_NULL_VALUE = 1.0
ODDS_RATIO_NULL_LINE_COLOR = "#444444"
ODDS_RATIO_NULL_LINE_WIDTH = 1.0
ODDS_RATIO_XLABEL = "Change in Odds Ratio"
ODDS_RATIO_XLABEL_SIZE = 16
ODDS_RATIO_X_LIMITS = (0.7, 2.2)
ODDS_RATIO_X_SCALE = "log"

POOLED_BAND_COLOR = "#666666"
POOLED_BAND_ALPHA = 0.12
POOLED_BAND_ZORDER = 0
POOLED_LINE_COLOR = "#202020"
POOLED_LINE_STYLE = "--"
POOLED_LINE_WIDTH = 1.1
POOLED_LINE_ZORDER = 3

LEFT_BAND_COLOR = "#F3F3F3"
LEFT_BAND_HALF_HEIGHT = 0.5
LEFT_BAND_ZORDER = 0
LEFT_BAND_ROW_INTERVAL = 2
LEFT_BAND_ROW_REMAINDER = 0
RIGHT_FACE_COLOR = "#FAFAFA"
RIGHT_SPINES_VISIBLE = True
RIGHT_Y_TICKS_VISIBLE = False

GRID_AXIS = "x"
GRID_ALPHA = 0.18
X_TICK_LABEL_SIZE = 14
Y_TICK_LABEL_SIZE = 15
EMPTY_Y_TICK_LABEL = ""

PANEL_A_LABEL = "a"
PANEL_B_LABEL = "b"
PANEL_A_LABEL_X = -0.05
PANEL_B_LABEL_X = -0.08
PANEL_LABEL_Y = 1.02
SUBPLOT_LABEL_SIZE = 16
SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT = "left"
SUBPLOT_LABEL_VERTICAL_ALIGNMENT = "bottom"
DIRECTION_LEFT_LABEL = "Reversal Lowers\nHate Threshold"
DIRECTION_RIGHT_LABEL = "Reversal Raises\nHate Threshold"
DIRECTION_LEFT_X = 0.01
DIRECTION_RIGHT_X = 0.99
DIRECTION_LABEL_Y = 1.12
DIRECTION_LABEL_SIZE = 11
DIRECTION_LEFT_HORIZONTAL_ALIGNMENT = "left"
DIRECTION_RIGHT_HORIZONTAL_ALIGNMENT = "right"
DIRECTION_VERTICAL_ALIGNMENT = "top"

LEGEND_ORIGINAL_LABEL = "Original"
LEGEND_REVERSE_LABEL = "Reverse"
LEGEND_MARKER_COLOR = "#555555"
LEGEND_LINE_COORDINATES = [0]
LEGEND_LINE_COLOR = "none"
LEGEND_MARKER_SCALE = 1.25
LEGEND_LOCATION = "upper left"
LEGEND_FONT_SIZE = 14
LEGEND_FRAME_ON = True

RIGHT_TICK_VALUES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
RIGHT_TICK_LABEL_FORMAT = "{value:g}"

SIGNIFICANCE_MARKER_X = 1.18
SIGNIFICANCE_MARKER_X_OFFSET_POINTS = -25
SIGNIFICANCE_MARKER_Y_OFFSET_POINTS = 0
SIGNIFICANCE_MARKER_TEXT_COORDS = "offset points"
SIGNIFICANCE_MARKER_FONT_SIZE = 10
SIGNIFICANCE_MARKER_COLOR = "#202020"
SIGNIFICANCE_MARKER_HORIZONTAL_ALIGNMENT = "center"
SIGNIFICANCE_MARKER_VERTICAL_ALIGNMENT = "center"
SIGNIFICANCE_MARKER_CLIP_ON = False
SIGNIFICANCE_MARKER_ZORDER = 4
SIGNIFICANCE_P001_MARKER = "***"
SIGNIFICANCE_P05_MARKER = "**"
SIGNIFICANCE_P10_MARKER = "*"
SIGNIFICANCE_P001_THRESHOLD = 0.001
SIGNIFICANCE_P05_THRESHOLD = 0.05
SIGNIFICANCE_P10_THRESHOLD = 0.1


def main() -> None:
    """Build and save the Figure 5 question-order DRF plot."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    comparison = load_order_shift_comparison(
        original_judges_path=ORIGINAL_JUDGES_PATH,
        reverse_judges_path=REVERSE_JUDGES_PATH,
    )
    pooled_delta, pooled_delta_se = _load_optional_pooled_delta(POOLED_ORDER_CONTRAST_PATH)

    figure_height = max(FIGURE_MIN_HEIGHT, len(comparison) * FIGURE_ROW_HEIGHT + FIGURE_HEIGHT_PADDING)
    figure, (severity_axis, delta_axis) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, figure_height),
        gridspec_kw={"width_ratios": SUBPLOT_WIDTH_RATIOS, "wspace": SUBPLOT_WSPACE},
    )

    _plot_severity_panel(severity_axis, comparison)
    _plot_odds_ratio_panel(delta_axis, comparison, pooled_delta, pooled_delta_se)
    _format_shared_y_axes(severity_axis, delta_axis, comparison)
    _add_panel_labels(severity_axis, delta_axis)
    _add_legend(severity_axis)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"output={OUTPUT_PATH.resolve()}")


def _load_optional_pooled_delta(order_contrast_path: Path) -> tuple[float | None, float | None]:
    """Load the pooled order effect when the contrast file exists."""

    if not order_contrast_path.exists():
        return None, None
    return load_pooled_order_delta(order_contrast_path)


def _plot_severity_panel(axis: plt.Axes, comparison: pd.DataFrame) -> None:
    """Draw original and reverse model severities on the left panel."""

    y_positions = list(range(len(comparison)))
    colors = [get_provider_color(provider) for provider in comparison["provider"].tolist()]
    _add_row_bands(axis, y_positions)

    for y_position, row, color in zip(y_positions, comparison.itertuples(index=False), colors, strict=True):
        axis.errorbar(
            row.original_measure,
            y_position - SEVERITY_CONDITION_Y_OFFSET,
            xerr=row.original_s_e,
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=SEVERITY_ERROR_COLOR,
            capsize=SEVERITY_ERROR_CAPSIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=SEVERITY_MARKER_SIZE,
            zorder=SEVERITY_ORIGINAL_ZORDER,
        )
        axis.errorbar(
            row.reverse_measure,
            y_position + SEVERITY_CONDITION_Y_OFFSET,
            xerr=row.reverse_s_e,
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=SEVERITY_ERROR_COLOR,
            capsize=SEVERITY_ERROR_CAPSIZE,
            markerfacecolor=SEVERITY_REVERSE_MARKER_FACE,
            markeredgecolor=color,
            markeredgewidth=SEVERITY_MARKER_EDGE_WIDTH,
            markersize=SEVERITY_MARKER_SIZE,
            zorder=SEVERITY_REVERSE_ZORDER,
        )

    axis.axvline(
        SEVERITY_NULL_LINE_VALUE,
        color=SEVERITY_NULL_LINE_COLOR,
        linewidth=SEVERITY_NULL_LINE_WIDTH,
        linestyle=SEVERITY_NULL_LINE_STYLE,
    )
    axis.set_xlabel(bold_text(SEVERITY_XLABEL), fontsize=SEVERITY_XLABEL_SIZE)
    axis.grid(axis=GRID_AXIS, alpha=GRID_ALPHA)
    axis.tick_params(axis=GRID_AXIS, labelsize=X_TICK_LABEL_SIZE)


def _plot_odds_ratio_panel(
    axis: plt.Axes,
    comparison: pd.DataFrame,
    pooled_delta: float | None,
    pooled_delta_se: float | None,
) -> None:
    """Draw reverse-versus-original odds ratios on the right panel."""

    y_positions = list(range(len(comparison)))
    colors = [get_provider_color(provider) for provider in comparison["provider"].tolist()]
    odds_ratio = comparison["severity_delta"].map(math.exp)
    lower_ratio = (comparison["severity_delta"] - comparison["delta_se_independent"]).map(math.exp)
    upper_ratio = (comparison["severity_delta"] + comparison["delta_se_independent"]).map(math.exp)
    x_errors = [odds_ratio - lower_ratio, upper_ratio - odds_ratio]

    for y_position, ratio, error, color in zip(
        y_positions,
        odds_ratio,
        zip(x_errors[0], x_errors[1], strict=True),
        colors,
        strict=True,
    ):
        axis.errorbar(
            ratio,
            y_position,
            xerr=[[error[0]], [error[1]]],
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=ODDS_RATIO_ERROR_COLOR,
            capsize=ODDS_RATIO_ERROR_CAPSIZE,
            markersize=ODDS_RATIO_MARKER_SIZE,
            markeredgecolor=ODDS_RATIO_MARKER_EDGE_COLOR,
            markeredgewidth=ODDS_RATIO_MARKER_EDGE_WIDTH,
            zorder=ODDS_RATIO_ZORDER,
        )

    _add_significance_markers(axis, comparison)
    axis.axvline(ODDS_RATIO_NULL_VALUE, color=ODDS_RATIO_NULL_LINE_COLOR, linewidth=ODDS_RATIO_NULL_LINE_WIDTH)
    axis.set_xscale(ODDS_RATIO_X_SCALE)
    axis.set_xlim(*ODDS_RATIO_X_LIMITS)
    axis.set_xticks(RIGHT_TICK_VALUES)
    axis.set_xticklabels([RIGHT_TICK_LABEL_FORMAT.format(value=value) for value in RIGHT_TICK_VALUES])
    axis.set_xlabel(bold_text(ODDS_RATIO_XLABEL), fontsize=ODDS_RATIO_XLABEL_SIZE)
    axis.grid(axis=GRID_AXIS, alpha=GRID_ALPHA)
    axis.tick_params(axis=GRID_AXIS, labelsize=X_TICK_LABEL_SIZE)
    axis.set_facecolor(RIGHT_FACE_COLOR)
    for spine in axis.spines.values():
        spine.set_visible(RIGHT_SPINES_VISIBLE)


def _add_row_bands(axis: plt.Axes, y_positions: list[int]) -> None:
    """Add alternating row bands behind model rows."""

    for y_position in y_positions:
        if y_position % LEFT_BAND_ROW_INTERVAL == LEFT_BAND_ROW_REMAINDER:
            axis.axhspan(
                y_position - LEFT_BAND_HALF_HEIGHT,
                y_position + LEFT_BAND_HALF_HEIGHT,
                color=LEFT_BAND_COLOR,
                zorder=LEFT_BAND_ZORDER,
            )


def _add_pooled_order_effect(axis: plt.Axes, pooled_delta: float | None, pooled_delta_se: float | None) -> None:
    """Add the pooled order-effect reference line and uncertainty band."""

    if pooled_delta is None:
        return
    pooled_ratio = math.exp(pooled_delta)
    if pooled_delta_se is not None:
        axis.axvspan(
            math.exp(pooled_delta - pooled_delta_se),
            math.exp(pooled_delta + pooled_delta_se),
            color=POOLED_BAND_COLOR,
            alpha=POOLED_BAND_ALPHA,
            zorder=POOLED_BAND_ZORDER,
        )
    axis.axvline(
        pooled_ratio,
        color=POOLED_LINE_COLOR,
        linestyle=POOLED_LINE_STYLE,
        linewidth=POOLED_LINE_WIDTH,
        zorder=POOLED_LINE_ZORDER,
    )


def _format_shared_y_axes(severity_axis: plt.Axes, delta_axis: plt.Axes, comparison: pd.DataFrame) -> None:
    """Apply shared model labels and y-limits to both panels."""

    row_count = len(comparison)
    y_positions = list(range(row_count))
    shared_y_limits = (row_count - LEFT_BAND_HALF_HEIGHT, -LEFT_BAND_HALF_HEIGHT)
    severity_axis.set_yticks(y_positions)
    severity_axis.set_yticklabels(bold_text(comparison["display_label"].tolist()), fontsize=Y_TICK_LABEL_SIZE)
    delta_axis.set_yticks(y_positions)
    delta_axis.set_yticklabels([EMPTY_Y_TICK_LABEL for _ in y_positions])
    delta_axis.tick_params(axis="y", left=RIGHT_Y_TICKS_VISIBLE)
    severity_axis.set_ylim(*shared_y_limits)
    delta_axis.set_ylim(*shared_y_limits)


def _add_panel_labels(severity_axis: plt.Axes, delta_axis: plt.Axes) -> None:
    """Add panel letters and direction labels."""

    severity_axis.text(
        PANEL_A_LABEL_X,
        PANEL_LABEL_Y,
        bold_text(PANEL_A_LABEL),
        transform=severity_axis.transAxes,
        fontsize=SUBPLOT_LABEL_SIZE,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        PANEL_B_LABEL_X,
        PANEL_LABEL_Y,
        bold_text(PANEL_B_LABEL),
        transform=delta_axis.transAxes,
        fontsize=SUBPLOT_LABEL_SIZE,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        DIRECTION_LEFT_X,
        DIRECTION_LABEL_Y,
        bold_text(DIRECTION_LEFT_LABEL),
        transform=delta_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        ha=DIRECTION_LEFT_HORIZONTAL_ALIGNMENT,
        va=DIRECTION_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        DIRECTION_RIGHT_X,
        DIRECTION_LABEL_Y,
        bold_text(DIRECTION_RIGHT_LABEL),
        transform=delta_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        ha=DIRECTION_RIGHT_HORIZONTAL_ALIGNMENT,
        va=DIRECTION_VERTICAL_ALIGNMENT,
    )


def _add_legend(axis: plt.Axes) -> None:
    """Add the original/reverse marker legend to the severity panel."""

    legend_handles = [
        Line2D(
            LEGEND_LINE_COORDINATES,
            LEGEND_LINE_COORDINATES,
            marker=MARKER_FORMAT,
            color=LEGEND_LINE_COLOR,
            markerfacecolor=LEGEND_MARKER_COLOR,
            markeredgecolor=LEGEND_MARKER_COLOR,
            label=bold_text(LEGEND_ORIGINAL_LABEL),
            markersize=SEVERITY_MARKER_SIZE,
        ),
        Line2D(
            LEGEND_LINE_COORDINATES,
            LEGEND_LINE_COORDINATES,
            marker=MARKER_FORMAT,
            color=LEGEND_LINE_COLOR,
            markerfacecolor=SEVERITY_REVERSE_MARKER_FACE,
            markeredgecolor=LEGEND_MARKER_COLOR,
            markeredgewidth=SEVERITY_MARKER_EDGE_WIDTH,
            label=bold_text(LEGEND_REVERSE_LABEL),
            markersize=SEVERITY_MARKER_SIZE,
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_MARKER_SCALE,
        frameon=LEGEND_FRAME_ON,
    )


def _add_significance_markers(axis: plt.Axes, comparison: pd.DataFrame) -> None:
    """Annotate significant item-order shifts in the odds-ratio panel."""

    for y_position, row in enumerate(comparison.itertuples(index=False)):
        marker = _significance_marker(float(row.severity_delta), float(row.delta_se_independent))
        if not marker:
            continue
        axis.annotate(
            bold_text(marker),
            xy=(SIGNIFICANCE_MARKER_X, y_position),
            xycoords=axis.get_yaxis_transform(),
            xytext=(SIGNIFICANCE_MARKER_X_OFFSET_POINTS, SIGNIFICANCE_MARKER_Y_OFFSET_POINTS),
            textcoords=SIGNIFICANCE_MARKER_TEXT_COORDS,
            ha=SIGNIFICANCE_MARKER_HORIZONTAL_ALIGNMENT,
            va=SIGNIFICANCE_MARKER_VERTICAL_ALIGNMENT,
            fontsize=SIGNIFICANCE_MARKER_FONT_SIZE,
            color=SIGNIFICANCE_MARKER_COLOR,
            clip_on=SIGNIFICANCE_MARKER_CLIP_ON,
            zorder=SIGNIFICANCE_MARKER_ZORDER,
        )


def _significance_marker(delta: float, standard_error: float) -> str:
    """Return a star marker from a two-sided normal approximation."""

    if standard_error <= 0:
        return ""
    p_value = math.erfc(abs(delta / standard_error) / math.sqrt(2.0))
    if p_value < SIGNIFICANCE_P001_THRESHOLD:
        return SIGNIFICANCE_P001_MARKER
    if p_value < SIGNIFICANCE_P05_THRESHOLD:
        return SIGNIFICANCE_P05_MARKER
    if p_value < SIGNIFICANCE_P10_THRESHOLD:
        return SIGNIFICANCE_P10_MARKER
    return ""


if __name__ == "__main__":
    main()
