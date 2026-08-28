from pathlib import Path
import re

from cycler import cycler
import matplotlib.pyplot as plt
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import pandas as pd

from rating_raters.plotting import save_figure


FACETS_OUTPUT_PATH = Path("facets/full_set_all_models_llm_only/full_set_all_models_llm_only_output.txt")
OUTPUT_PNG_PATH = Path("artifacts/figure5_category_diagnostics.png")
OUTPUT_PDF_PATH = Path("artifacts/figure5_category_diagnostics.pdf")

FIGURE_SIZE = (3.05, 2.75)
DPI = 300
SAVE_DPI = 300
SAVE_PAD_INCHES = 0.02
X_LIMITS = (-4.0, 4.0)
X_TICKS = [-4, -2, 0, 2, 4]
STEP_OFFSETS = {
    "0->1": 0.27,
    "1->2": 0.09,
    "2->3": -0.09,
    "3->4": -0.27,
}
TICK_HALF_HEIGHT = 0.09
CONNECTOR_LINE_WIDTH = 0.75
THRESHOLD_LINE_WIDTH = 1.35
ORDER_MARKER_SIZE = 15
ZERO_LINE_WIDTH = 0.75
GRID_LINE_WIDTH = 0.45
AXIS_LINE_WIDTH = 0.7
X_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
ITEM_LABEL_SIZE = 6.7
LEGEND_LABEL_SIZE = 5.6
THRESHOLD_COLOR = "#202020"
ORDERED_COLOR = "#009E73"
DISORDERED_COLOR = "#D55E00"
CONNECTOR_COLOR = "#6B7280"
GRID_COLOR = "#D1D5DB"
ZERO_COLOR = "#9CA3AF"
BAND_COLOR = "#F3F4F6"
X_LABEL = r"Rasch-Andrich Threshold ($\tau_k$)"
ORDERED_LABEL = "Ordered step"
DISORDERED_LABEL = "Disordered step"
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]
ITEM_ORDER = [
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
MODEL_LINE_PATTERN = re.compile(r"Model = .*?; Items: (.+)")
THRESHOLD_ROW_PATTERN = re.compile(
    r"^\|\s*(\d+)\s+[-\d]+\s+[-\d]+\s+\d+%\s+\d+%\|.*?\|\s*"
    r"([-+]?\d*\.\d+|[-+]?\d+)\s+(\.\d+|\d*\.\d+|\d+)\|"
)


def main() -> None:
    """Build and save the threshold staircase plot."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    thresholds = parse_thresholds(FACETS_OUTPUT_PATH)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    plot_thresholds(axis, thresholds)
    format_axis(axis)
    add_legend(axis)

    figure.tight_layout(pad=0.35)
    save_figure(figure, OUTPUT_PNG_PATH, dpi=SAVE_DPI)
    save_figure(figure, OUTPUT_PDF_PATH, dpi=SAVE_DPI)


def parse_thresholds(path: Path) -> pd.DataFrame:
    """Parse FACETS Rasch-Andrich thresholds from the original LLM output."""

    records: list[dict[str, object]] = []
    current_item: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        item_match = MODEL_LINE_PATTERN.search(line)
        if item_match:
            current_item = item_match.group(1).strip()
            continue

        if current_item is None:
            continue

        row_match = THRESHOLD_ROW_PATTERN.match(line)
        if row_match is None:
            continue

        score = int(row_match.group(1))
        records.append(
            {
                "item": current_item,
                "step": f"{score - 1}->{score}",
                "step_to": score,
                "threshold": float(row_match.group(2)),
            }
        )

    dataframe = pd.DataFrame(records)
    dataframe["item"] = pd.Categorical(dataframe["item"], categories=ITEM_ORDER, ordered=True)
    return dataframe.sort_values(["item", "step_to"], kind="stable").reset_index(drop=True)


def plot_thresholds(axis: plt.Axes, thresholds: pd.DataFrame) -> None:
    """Plot vertically offset threshold ticks for each item."""

    y_positions = item_y_positions()
    for item in ITEM_ORDER:
        item_thresholds = thresholds.loc[thresholds["item"].astype(str) == item].copy()
        if item_thresholds.empty:
            continue

        base_y = y_positions[item]
        item_thresholds["y"] = item_thresholds["step"].map(STEP_OFFSETS).astype(float) + base_y
        threshold_rows = list(item_thresholds.itertuples(index=False))
        for row in threshold_rows:
            axis.vlines(
                row.threshold,
                row.y - TICK_HALF_HEIGHT,
                row.y + TICK_HALF_HEIGHT,
                color=THRESHOLD_COLOR,
                linewidth=THRESHOLD_LINE_WIDTH,
                zorder=3,
            )
        for previous_row, current_row in zip(threshold_rows, threshold_rows[1:], strict=False):
            plot_step_connector(axis, previous_row, current_row)


def plot_step_connector(axis: plt.Axes, previous_row: object, current_row: object) -> None:
    """Draw an endpoint-to-endpoint connector with an ordering marker."""

    is_ordered = current_row.threshold >= previous_row.threshold
    start_y = previous_row.y - TICK_HALF_HEIGHT
    end_y = current_row.y + TICK_HALF_HEIGHT
    axis.plot(
        [previous_row.threshold, current_row.threshold],
        [start_y, end_y],
        color=CONNECTOR_COLOR,
        linewidth=CONNECTOR_LINE_WIDTH,
        alpha=0.85,
        zorder=2,
    )
    axis.scatter(
        [(previous_row.threshold + current_row.threshold) / 2.0],
        [(start_y + end_y) / 2.0],
        marker=">" if is_ordered else "<",
        s=ORDER_MARKER_SIZE,
        color=ORDERED_COLOR if is_ordered else DISORDERED_COLOR,
        edgecolor="none",
        zorder=4,
    )


def item_y_positions() -> dict[str, int]:
    """Return descending y-axis positions keyed by item name."""

    return {item: len(ITEM_ORDER) - index for index, item in enumerate(ITEM_ORDER)}


def format_axis(axis: plt.Axes) -> None:
    """Apply axis labels, ticks, grid lines, and plot cleanup."""

    y_positions = item_y_positions()
    add_row_banding(axis, y_positions)
    axis.axvline(0.0, color=ZERO_COLOR, linewidth=ZERO_LINE_WIDTH, linestyle="--", zorder=1)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(X_TICKS)
    axis.set_ylim(0.45, len(ITEM_ORDER) + 0.55)
    axis.set_yticks([y_positions[item] for item in ITEM_ORDER])
    axis.set_yticklabels(
        [fix_labels_for_tex_style(ITEM_LABELS[item]) for item in ITEM_ORDER],
        fontsize=ITEM_LABEL_SIZE,
    )
    axis.set_xlabel(bold_text(X_LABEL), fontsize=X_LABEL_SIZE)
    axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.grid(axis="x", color=GRID_COLOR, alpha=0.6, linewidth=GRID_LINE_WIDTH)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)


def add_row_banding(axis: plt.Axes, y_positions: dict[str, int]) -> None:
    """Add alternating gray bands behind item rows."""

    for index, item in enumerate(ITEM_ORDER):
        if index % 2:
            continue
        y_position = y_positions[item]
        axis.axhspan(y_position - 0.5, y_position + 0.5, color=BAND_COLOR, zorder=0)


def add_legend(axis: plt.Axes) -> None:
    """Add a compact legend explaining disordered threshold steps."""

    ordered = axis.scatter([], [], marker=">", s=ORDER_MARKER_SIZE, color=ORDERED_COLOR)
    disordered = axis.scatter([], [], marker="<", s=ORDER_MARKER_SIZE, color=DISORDERED_COLOR)
    axis.legend(
        [ordered, disordered],
        [bold_text(ORDERED_LABEL), bold_text(DISORDERED_LABEL)],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_LABEL_SIZE,
        handlelength=1.2,
        handletextpad=0.25,
        columnspacing=0.65,
        borderpad=0.1,
    )


if __name__ == "__main__":
    main()
