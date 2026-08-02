from pathlib import Path
import re


FACETS_OUTPUT_PATH = Path("facets/full_set_all_models_llm_only/full_set_all_models_llm_only_output.txt")
OUTPUT_TEX_PATH = Path("tables/llm_no_recoding_andrich_thresholds_with_deltas.tex")

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
    "attack_defend": "Attack/defend",
    "hate_speech": "Hate speech",
}
THRESHOLD_STEPS = ["0->1", "1->2", "2->3", "3->4"]
TABLE_CAPTION = "Rasch-Andrich thresholds for the original LLM scale without response recoding."
TABLE_LABEL = "tab:llm-no-recoding-andrich-thresholds"

MODEL_LINE_PATTERN = re.compile(r"Model = .*?; Items: (.+)")
THRESHOLD_ROW_PATTERN = re.compile(
    r"^\|\s*(\d+)\s+[-\d]+\s+[-\d]+\s+\d+%\s+\d+%\|.*?\|\s*"
    r"([-+]?\d*\.\d+|[-+]?\d+)\s+(\.\d+|\d*\.\d+|\d+)\|"
)


def main() -> None:
    """Export the no-recoding LLM Rasch-Andrich threshold table to LaTeX."""

    thresholds = parse_thresholds(FACETS_OUTPUT_PATH)
    latex_table = build_latex_table(thresholds)
    OUTPUT_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX_PATH.write_text(latex_table)
    print(f"Wrote {OUTPUT_TEX_PATH}")


def parse_thresholds(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Parse item threshold steps from a FACETS output report."""

    thresholds: dict[str, list[tuple[str, float]]] = {}
    current_item: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        item_match = MODEL_LINE_PATTERN.search(line)
        if item_match:
            current_item = item_match.group(1).strip()
            thresholds.setdefault(current_item, [])
            continue

        if current_item is None:
            continue

        row_match = THRESHOLD_ROW_PATTERN.match(line)
        if row_match is None:
            continue

        score = int(row_match.group(1))
        threshold = float(row_match.group(2))
        thresholds[current_item].append((f"{score - 1}->{score}", threshold))

    return thresholds


def build_latex_table(thresholds: dict[str, list[tuple[str, float]]]) -> str:
    """Build a LaTeX tabular table with threshold deltas in parentheses."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{TABLE_CAPTION}}}",
        rf"\label{{{TABLE_LABEL}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\hline",
        r"Item & $0\to1$ & $1\to2$ & $2\to3$ & $3\to4$ \\",
        r"\hline",
    ]
    for item in ITEM_ORDER:
        item_thresholds = dict(thresholds.get(item, []))
        cells = [ITEM_LABELS[item]]
        previous_threshold: float | None = None
        for step in THRESHOLD_STEPS:
            threshold = item_thresholds.get(step)
            if threshold is None:
                cells.append("")
                continue
            cells.append(format_threshold_cell(threshold, previous_threshold))
            previous_threshold = threshold
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular*}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def format_threshold_cell(threshold: float, previous_threshold: float | None) -> str:
    """Format one threshold cell with a delta from the previous threshold."""

    if previous_threshold is None:
        return rf"${threshold:.2f}$"
    delta = threshold - previous_threshold
    return rf"${threshold:.2f}$ (${delta:+.2f}$)"


if __name__ == "__main__":
    main()
