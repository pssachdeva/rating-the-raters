"""Plot preliminary item-by-item model severities on the human-anchored scale."""

from pathlib import Path

from mpl_lego.style import use_latex_style

from mhs_llms.facets.model_severity_figure import (
    load_human_judge_severities,
    load_model_judge_severities,
    plot_model_severity_figure,
)


HUMAN_JUDGES_PATH = Path("facets/human_baseline/judges_scores.csv")
MODEL_JUDGES_PATH = Path(
    "facets/item_by_item_preliminary_2026_07_12/judges_scores.csv"
)
OUTPUT_PATH = Path(
    "artifacts/item_by_item_preliminary_2026_07_12_model_severity.png"
)
TITLE = "Preliminary Item-by-Item Model Severities"
FIGURE_WIDTH = 8.5
FIGURE_HEIGHT = 9.0


def main() -> None:
    """Load the human and model FACETS scores and save the severity figure."""

    use_latex_style()
    human_severities = load_human_judge_severities(HUMAN_JUDGES_PATH)
    model_severities = load_model_judge_severities([MODEL_JUDGES_PATH])
    plot_model_severity_figure(
        human_severity_frame=human_severities,
        model_severity_frame=model_severities,
        output_path=OUTPUT_PATH,
        title=TITLE,
        figure_width=FIGURE_WIDTH,
        figure_height=FIGURE_HEIGHT,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
