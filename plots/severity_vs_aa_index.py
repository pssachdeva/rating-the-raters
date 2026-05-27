"""Plot model severity estimates against AA Intelligence Index values."""

from pathlib import Path

import matplotlib.pyplot as plt

from mhs_llms.aa_scatter import load_medium_effort_aa_severities, plot_aa_severity_scatter
from mhs_llms.paths import ARTIFACTS_DIR
from mhs_llms.plotting import apply_plot_style, save_figure


DATA_PATH = Path("data/reference_set_model_severities_latest.csv")
OUTPUT_PATH = ARTIFACTS_DIR / "severity_vs_aa_index_scatter.png"

FIGURE_SIZE = (6.2, 4.2)
DPI = 300
MARKER_SIZE = 54
LEGEND_FONT_SIZE = 7
TICK_LABEL_SIZE = 8
AXIS_LABEL_SIZE = 10
ANNOTATION_SIZE = 8
SAVE_DPI = 300


def main() -> None:
    """Build and save the severity by AA Intelligence Index scatter plot."""
    apply_plot_style()

    data = load_medium_effort_aa_severities(DATA_PATH)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    plot_aa_severity_scatter(
        axis=ax,
        data=data,
        marker_size=MARKER_SIZE,
        legend_font_size=LEGEND_FONT_SIZE,
        axis_label_size=AXIS_LABEL_SIZE,
        tick_label_size=TICK_LABEL_SIZE,
        annotation_size=ANNOTATION_SIZE,
    )

    fig.tight_layout()
    output_path = save_figure(fig, OUTPUT_PATH, dpi=SAVE_DPI)
    print(f"output={output_path.resolve()}")
    print(f"n={len(data)}")


if __name__ == "__main__":
    main()
