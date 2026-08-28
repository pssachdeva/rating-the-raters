"""Plot the item-by-item severity change against AA Intelligence Index."""

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style
import numpy as np
import pandas as pd

from rating_raters.labels import infer_provider, model_id_to_plot_label, provider_display_name
from rating_raters.plotting import PROVIDER_COLORS, save_figure


ORIGINAL_SCORES_PATH = Path("facets/reference_set_all/judges_scores.csv")
SPLIT_SCORES_PATH = Path(
    "facets/item_by_item_preliminary_2026_07_12/judges_scores.csv"
)
AA_INDEX_PATH = Path("data/reference_set_model_severities_latest.csv")
OUTPUT_DATA_PATH = Path("data/item_by_item/severity_change_vs_aa_index.csv")
OUTPUT_PATH = Path("artifacts/severity_change_vs_aa_index.png")

FIGURE_SIZE = (6.4, 4.4)
DPI = 300
MARKER_SIZE = 62
MARKER_ALPHA = 0.88
MARKER_EDGE_COLOR = "#222222"
MARKER_EDGE_WIDTH = 0.5
TREND_COLOR = "#333333"
TREND_WIDTH = 1.5
ZERO_LINE_COLOR = "#777777"
ZERO_LINE_WIDTH = 0.9
ANNOTATION_SIZE = 7.5
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 7.5
LABEL_THRESHOLD = 0.175
LABEL_OFFSETS = {
    "openai_gpt-4o": (5, -14),
    "openrouter_deepseek_deepseek-v3.2": (5, 5),
    "openai_gpt-5.2_medium": (5, 5),
    "openai_gpt-5.4-mini_medium": (5, 5),
    "openai_gpt-5.4-nano_medium": (5, 5),
}
PROVIDER_ORDER = ("anthropic", "openai", "google", "deepseek")
AXIS_BACKGROUND_COLOR = "#F7F7F7"


def load_comparison_data() -> pd.DataFrame:
    """Join shared-model severity changes to the saved AA capability index."""

    original = pd.read_csv(ORIGINAL_SCORES_PATH).loc[:, ["facet_label", "measure"]]
    original = original.rename(
        columns={"facet_label": "judge_id", "measure": "severity_original"}
    )
    split = pd.read_csv(SPLIT_SCORES_PATH).loc[:, ["facet_label", "measure"]]
    split = split.rename(
        columns={"facet_label": "judge_id", "measure": "severity_split"}
    )
    aa_index = pd.read_csv(AA_INDEX_PATH).loc[
        :, ["judge_id", "aa_intelligence_index", "aa_retrieved_at_utc"]
    ]

    comparison = original.merge(split, on="judge_id", validate="one_to_one")
    comparison = comparison.merge(aa_index, on="judge_id", validate="one_to_one")
    comparison["severity_change"] = (
        comparison["severity_split"] - comparison["severity_original"]
    )
    comparison["provider"] = comparison["judge_id"].map(infer_provider)
    comparison["provider_label"] = comparison["provider"].map(provider_display_name)
    comparison["model_label"] = comparison["judge_id"].map(model_id_to_plot_label)
    return comparison.sort_values("aa_intelligence_index").reset_index(drop=True)


def main() -> None:
    """Build and save the capability-versus-severity-change scatterplot."""

    use_latex_style()
    data = load_comparison_data()
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DATA_PATH, index=False)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    axis.set_facecolor(AXIS_BACKGROUND_COLOR)

    for provider in PROVIDER_ORDER:
        selected = data.loc[data["provider"] == provider]
        if selected.empty:
            continue
        axis.scatter(
            selected["aa_intelligence_index"],
            selected["severity_change"],
            s=MARKER_SIZE,
            alpha=MARKER_ALPHA,
            color=PROVIDER_COLORS[provider],
            edgecolor=MARKER_EDGE_COLOR,
            linewidth=MARKER_EDGE_WIDTH,
            label=selected["provider_label"].iloc[0],
            zorder=3,
        )

    # Show the signed linear relationship without treating it as a causal estimate.
    x_values = data["aa_intelligence_index"].to_numpy(dtype=float)
    y_values = data["severity_change"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    axis.plot(
        line_x,
        slope * line_x + intercept,
        color=TREND_COLOR,
        linewidth=TREND_WIDTH,
        zorder=2,
    )

    axis.axhline(
        0.0,
        color=ZERO_LINE_COLOR,
        linewidth=ZERO_LINE_WIDTH,
        linestyle="--",
        zorder=1,
    )
    for row in data.loc[data["severity_change"].abs() >= LABEL_THRESHOLD].itertuples(
        index=False
    ):
        x_offset, y_offset = LABEL_OFFSETS.get(row.judge_id, (5, 4))
        axis.annotate(
            row.model_label,
            (row.aa_intelligence_index, row.severity_change),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
        )

    correlation = data["aa_intelligence_index"].corr(data["severity_change"])
    axis.text(
        0.97,
        0.96,
        f"$r = {correlation:.2f}$; $n = {len(data)}$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_SIZE,
    )
    axis.text(
        0.02,
        0.04,
        "More severe under split prompts",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
        color="#555555",
    )
    axis.set_xlabel(bold_text("AA Intelligence Index"), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(
        bold_text("Severity Change (Split $-$ Original)"),
        fontsize=AXIS_LABEL_SIZE,
    )
    axis.tick_params(labelsize=TICK_LABEL_SIZE)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.legend(
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    figure.tight_layout()
    output_path = save_figure(figure, OUTPUT_PATH, dpi=DPI)
    plt.close(figure)
    print(f"output={output_path.resolve()}")
    print(f"data={OUTPUT_DATA_PATH.resolve()}")
    print(f"correlation={correlation:.3f}")
    print(f"n={len(data)}")


if __name__ == "__main__":
    main()
