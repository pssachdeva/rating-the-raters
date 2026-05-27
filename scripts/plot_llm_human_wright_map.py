from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import pandas as pd

from mhs_llms.labels import infer_provider, model_id_to_plot_label
from mhs_llms.plotting import build_gaussian_kde_curve, get_provider_color, save_figure


COMMENTS_SCORE_PATH = Path("facets/full_set_all_models_llm_only_recoded/comments_scores.csv")
JUDGES_SCORE_PATH = Path("facets/full_set_all_models_llm_only_recoded/judges_scores.csv")
LLM_ITEMS_SCORE_PATH = Path("facets/full_set_all_models_llm_only_recoded/items_scores.csv")
HUMAN_ITEMS_SCORE_PATH = Path("facets/human_baseline/items_scores.csv")
OUTPUT_PNG_PATH = Path("artifacts/full_set_llm_human_wright_map.png")
OUTPUT_PDF_PATH = Path("artifacts/full_set_llm_human_wright_map.pdf")

FIGURE_SIZE = (7.8, 4.1)
DPI = 300
SAVE_DPI = 300
SAVE_PAD_INCHES = 0.06
X_LIMITS = (-6.5, 6.0)
Y_LIMITS = (2.85, 4.5)
KDE_POINTS = 700
KDE_SCALE = 0.75
X_AXIS_Y = 3.50
KDE_BASE_Y = X_AXIS_Y
MODEL_Y = 4.25
MODEL_Y_JITTER = 0.025
ITEM_CALLOUT_LINE_WIDTH = 1.3
MODEL_MARKER_SIZE = 54
DEFAULT_MODEL_MARKER = "o"
MODEL_MARKER_OVERRIDES = {
    "GPT-5.4": "D",
}
KDE_LINE_WIDTH = 1.7
ZERO_LINE_WIDTH = 0.85
ZERO_LINE_Y_MIN = 3.0
ZERO_LINE_Y_MAX = 4.50
AXIS_LINE_WIDTH = 0.8
ITEM_LABEL_SIZE = 10.0
X_TICK_LABEL_SIZE = 7.5
SHOW_Y_TICKS = False
Y_TICK_STEP = 0.25
Y_TICK_LABEL_SIZE = 6.0
LEGEND_LOC = "upper center"
LEGEND_BBOX_ANCHOR = (0.90, 1.00)
LEGEND_N_COLUMNS = 1
LEGEND_FRAME_ON = False
LEGEND_FONT_SIZE = 10.0
LEGEND_HANDLE_TEXT_PAD = 0.2
LEGEND_COLUMN_SPACING = 0.45
LEGEND_MARKER_SIZE = 26
ITEM_LEGEND_LOC = "lower right"
ITEM_LEGEND_BBOX_ANCHOR = (1.05, 0.09)
ITEM_LEGEND_N_COLUMNS = 1
ITEM_LEGEND_FRAME_ON = False
ITEM_LEGEND_FONT_SIZE = 10.0
ITEM_LEGEND_LINE_WIDTH = 2.0
ITEM_LEGEND_HANDLE_LENGTH = 1.2
ITEM_LEGEND_HANDLE_TEXT_PAD = 0.4
ITEM_LEGEND_LABELS = {
    "llm": "LLM scale",
    "human": "Human scale",
}
SENTIMENT_ITEM = "sentiment"
MODEL_JITTER_PATTERN = [-2, 1, -1, 2, 0, -2, 1, -1, 2]
MODEL_LABEL_OVERRIDES = {
    "GPT-OSS 120B": "GPT-OSS",
    "Llama 3.3 70B": "Llama",
    "DeepSeek V4 Pro": "DeepSeek",
    "Kimi K2.5": "Kimi",
    "Claude Opus 4.6": "Claude",
    "Gemini 3.1 Pro": "Gemini",
    "GPT-5.4": "GPT-5.4",
    "MiniMax M2.5": "MiniMax",
    "Grok 4.1": "Grok",
}
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]
COMMENT_FILL_COLOR = "#8CA2CF"
COMMENT_LINE_COLOR = "#4169A8"
ITEM_COLOR = "#202020"
HUMAN_ITEM_COLOR = "#8B5E5A"
BACKGROUND_COLOR = "#FFFFFF"
GRID_COLOR = "#D1D5DB"
ZERO_LINE_COLOR = "#9CA3AF"
SHOW_ZERO_LINE = False

ITEM_LABELS = {
    "sentiment": "Sentiment",
    "respect": "Respect",
    "insult": "Insult",
    "humiliate": "Humiliate",
    "status": "Status",
    "dehumanize": "Dehumanize",
    "violence": "Violence",
    "genocide": "Genocide",
    "attack_defend": "Attack",
    "hate_speech": "Hate speech",
}
LLM_ITEM_LABEL_LAYOUT = {
    "sentiment": {"x": -4.25, "y": 3.70, "ha": "center", "va": "center"},
    "respect": {"x": -3.14, "y": 3.80, "ha": "center", "va": "center"},
    "attack_defend": {"x": -2.50, "y": 3.60, "ha": "center", "va": "center"},
    "insult": {"x": -1.470, "y": 3.750, "ha": "center", "va": "center"},
    "status": {"x": -0.800, "y": 3.95, "ha": "center", "va": "center"},
    "hate_speech": {"x": 0.20, "y": 3.80, "ha": "center", "va": "center"},
    "humiliate": {"x": 0.13, "y": 3.650, "ha": "center", "va": "center"},
    "dehumanize": {"x": 2.340, "y": 3.700, "ha": "center", "va": "center"},
    "violence": {"x": 3.680, "y": 3.63, "ha": "center", "va": "center"},
    "genocide": {"x": 5.410, "y": 3.650, "ha": "center", "va": "center"},
}
HUMAN_ITEM_LABEL_LAYOUT = {
    "sentiment": {"x": -4.25, "y": 3.30, "ha": "center", "va": "center"},
    "respect": {"x": -3.14, "y": 3.20, "ha": "center", "va": "center"},
    "attack_defend": {"x": -2.750, "y": 3.100, "ha": "center", "va": "center"},
    "insult": {"x": -1.820, "y": 2.9500, "ha": "center", "va": "center"},
    "status": {"x": -0.80, "y": 3.250, "ha": "center", "va": "center"},
    "dehumanize": {"x": -2, "y": 2.850, "ha": "center", "va": "center"},
    "humiliate": {"x": 0.800, "y": 2.85, "ha": "center", "va": "center"},
    "hate_speech": {"x": 1.500, "y": 3.0, "ha": "center", "va": "center"},
    "violence": {"x": 1.36, "y": 3.15, "ha": "center", "va": "center"},
    "genocide": {"x": 2.250, "y": 3.25, "ha": "center", "va": "center"},
}


def main() -> None:
    """Build and save the horizontal Wright-map style figure."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    comment_scores = load_score_table(COMMENTS_SCORE_PATH)
    model_scores = load_model_scores(JUDGES_SCORE_PATH)
    llm_items = load_item_scores(LLM_ITEMS_SCORE_PATH)
    human_items = load_shifted_human_items(HUMAN_ITEMS_SCORE_PATH, llm_items)
    x_min, x_max = build_x_limits()

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    axis.set_facecolor(BACKGROUND_COLOR)

    plot_comment_distribution(axis, comment_scores, x_min, x_max)
    plot_model_markers(axis, model_scores)
    plot_item_offshoots(
        axis=axis,
        llm_items=llm_items,
        human_items=human_items,
    )
    format_axis(axis, x_min, x_max)
    add_legend(axis)

    figure.tight_layout(pad=0.4)
    save_figure(figure, OUTPUT_PNG_PATH, dpi=SAVE_DPI)
    save_figure(figure, OUTPUT_PDF_PATH, dpi=SAVE_DPI)


def load_score_table(path: Path) -> pd.DataFrame:
    """Load a processed FACETS score CSV and validate the measure column."""

    dataframe = pd.read_csv(path)
    if "measure" not in dataframe.columns:
        raise ValueError(f"{path} is missing measure")
    return dataframe


def load_model_scores(path: Path) -> pd.DataFrame:
    """Load model scores with provider colors and plot labels."""

    dataframe = load_score_table(path)
    required_columns = {"facet_label", "measure"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing_text}")

    selected = dataframe.loc[:, ["facet_label", "measure"]].copy()
    selected["provider"] = selected["facet_label"].map(infer_provider)
    selected["plot_label"] = selected["facet_label"].map(model_id_to_plot_label)
    selected["plot_label"] = selected["plot_label"].replace(MODEL_LABEL_OVERRIDES)
    selected = selected.sort_values(["measure", "plot_label"], kind="stable").reset_index(drop=True)
    return selected


def load_item_scores(path: Path) -> pd.DataFrame:
    """Load item scores and add display labels."""

    dataframe = load_score_table(path)
    selected = dataframe.loc[:, ["facet_label", "measure"]].copy()
    selected["plot_label"] = selected["facet_label"].map(ITEM_LABELS)
    return selected


def load_shifted_human_items(path: Path, llm_items: pd.DataFrame) -> pd.DataFrame:
    """Shift human item measures so human sentiment equals LLM sentiment."""

    human_items = load_item_scores(path)
    human_sentiment = float(
        human_items.loc[human_items["facet_label"] == SENTIMENT_ITEM, "measure"].iloc[0]
    )
    llm_sentiment = float(
        llm_items.loc[llm_items["facet_label"] == SENTIMENT_ITEM, "measure"].iloc[0]
    )
    shifted = human_items.copy()
    shifted["measure"] = shifted["measure"] + (llm_sentiment - human_sentiment)
    return shifted


def build_x_limits() -> tuple[float, float]:
    """Return the fixed x-axis limits for the compact Wright-map figure."""

    return X_LIMITS


def plot_comment_distribution(
    axis: plt.Axes,
    comment_scores: pd.DataFrame,
    x_min: float,
    x_max: float,
) -> None:
    """Plot the LLM comment severity distribution as a horizontal density ridge."""

    measures = comment_scores["measure"].astype(float).tolist()
    x_values, density_values = build_gaussian_kde_curve(
        measures,
        x_min=x_min,
        x_max=x_max,
        point_count=KDE_POINTS,
    )
    max_density = max(density_values)
    scaled_density = [KDE_BASE_Y + KDE_SCALE * value / max_density for value in density_values]
    axis.fill_between(
        x_values,
        [KDE_BASE_Y] * len(x_values),
        scaled_density,
        color=COMMENT_FILL_COLOR,
        alpha=0.38,
        linewidth=0.0,
        zorder=1,
    )
    axis.plot(x_values, scaled_density, color=COMMENT_LINE_COLOR, linewidth=KDE_LINE_WIDTH, zorder=2)


def plot_model_markers(axis: plt.Axes, model_scores: pd.DataFrame) -> None:
    """Plot provider-colored model severity points with deterministic vertical jitter."""

    for index, row in enumerate(model_scores.itertuples(index=False)):
        color = get_provider_color(row.provider)
        jitter = MODEL_JITTER_PATTERN[index % len(MODEL_JITTER_PATTERN)] * MODEL_Y_JITTER
        axis.scatter(
            row.measure,
            MODEL_Y + jitter,
            s=MODEL_MARKER_SIZE,
            marker=MODEL_MARKER_OVERRIDES.get(row.plot_label, DEFAULT_MODEL_MARKER),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=row.plot_label,
        )


def plot_item_offshoots(
    axis: plt.Axes,
    llm_items: pd.DataFrame,
    human_items: pd.DataFrame,
) -> None:
    """Draw item difficulty labels as elbow callouts to the main x-axis."""

    plot_item_callouts(
        axis,
        llm_items,
        label_layout=LLM_ITEM_LABEL_LAYOUT,
        color=ITEM_COLOR,
    )
    plot_item_callouts(
        axis,
        human_items,
        label_layout=HUMAN_ITEM_LABEL_LAYOUT,
        color=HUMAN_ITEM_COLOR,
    )


def plot_item_callouts(
    axis: plt.Axes,
    items: pd.DataFrame,
    label_layout: dict[str, dict[str, float | str]],
    color: str,
) -> None:
    """Draw horizontal item labels with elbow arrows to their item measures."""

    sorted_items = items.sort_values(["measure", "plot_label"], kind="stable").reset_index(drop=True)
    for row in sorted_items.itertuples(index=False):
        layout = label_layout[row.facet_label]
        axis.annotate(
            bold_text(fix_labels_for_tex_style(row.plot_label)),
            xy=(row.measure, X_AXIS_Y),
            xytext=(layout["x"], layout["y"]),
            ha=layout["ha"],
            va=layout["va"],
            fontsize=ITEM_LABEL_SIZE,
            color=color,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": ITEM_CALLOUT_LINE_WIDTH,
                "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
                "mutation_scale": 5.0,
                "shrinkA": 2.0,
                "shrinkB": 2.0,
            },
            zorder=4,
        )


def format_axis(axis: plt.Axes, x_min: float, x_max: float) -> None:
    """Apply final axis limits, row labels, and visual cleanup."""

    if SHOW_ZERO_LINE:
        axis.vlines(
            0.0,
            ymin=ZERO_LINE_Y_MIN,
            ymax=ZERO_LINE_Y_MAX,
            color=ZERO_LINE_COLOR,
            linewidth=ZERO_LINE_WIDTH,
            linestyle="--",
            zorder=0,
        )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlabel("")
    if SHOW_Y_TICKS:
        y_ticks = build_y_ticks(*Y_LIMITS, step=Y_TICK_STEP)
        axis.set_yticks(y_ticks)
        axis.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH)
    else:
        axis.set_yticks([])
    axis.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH)
    axis.grid(axis="both" if SHOW_Y_TICKS else "x", color=GRID_COLOR, alpha=0.42, linewidth=0.55)
    for spine in ("left", "right", "top"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_position(("data", X_AXIS_Y))
    axis.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)


def build_y_ticks(y_min: float, y_max: float, step: float) -> list[float]:
    """Return evenly spaced y ticks across the configured plot range."""

    ticks: list[float] = []
    current_tick = y_min
    while current_tick <= y_max + 1e-9:
        ticks.append(round(current_tick, 2))
        current_tick += step
    return ticks


def add_legend(axis: plt.Axes) -> None:
    """Add model and item-scale legends."""

    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    model_legend = axis.legend(
        unique.values(),
        [fix_labels_for_tex_style(label) for label in unique.keys()],
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_ANCHOR,
        ncol=LEGEND_N_COLUMNS,
        frameon=LEGEND_FRAME_ON,
        fontsize=LEGEND_FONT_SIZE,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        columnspacing=LEGEND_COLUMN_SPACING,
    )
    for handle in model_legend.legend_handles:
        handle.set_sizes([LEGEND_MARKER_SIZE])
    axis.add_artist(model_legend)

    item_handles = [
        Line2D([0], [0], color=ITEM_COLOR, linewidth=ITEM_LEGEND_LINE_WIDTH),
        Line2D([0], [0], color=HUMAN_ITEM_COLOR, linewidth=ITEM_LEGEND_LINE_WIDTH),
    ]
    axis.legend(
        item_handles,
        [
            bold_text(ITEM_LEGEND_LABELS["llm"]),
            bold_text(ITEM_LEGEND_LABELS["human"]),
        ],
        loc=ITEM_LEGEND_LOC,
        bbox_to_anchor=ITEM_LEGEND_BBOX_ANCHOR,
        ncol=ITEM_LEGEND_N_COLUMNS,
        frameon=ITEM_LEGEND_FRAME_ON,
        fontsize=ITEM_LEGEND_FONT_SIZE,
        handlelength=ITEM_LEGEND_HANDLE_LENGTH,
        handletextpad=ITEM_LEGEND_HANDLE_TEXT_PAD,
    )


if __name__ == "__main__":
    main()
