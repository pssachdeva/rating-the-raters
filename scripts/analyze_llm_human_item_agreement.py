"""Compare primary LLM ratings with median human ratings by MHS item."""

from pathlib import Path

import pandas as pd

from mhs_llms.annotator_agreement import build_llm_human_consensus_agreement
from mhs_llms.dataset import load_mhs_dataframe


LLM_PATH = Path("data/full_run_models_reference_set_processed.csv")
DETAIL_PATH = Path("data/reference_set_llm_human_item_agreement_by_model.csv")
SUMMARY_PATH = Path("data/reference_set_llm_human_item_agreement_summary.csv")


def main() -> None:
    """Run and save item-level LLM-to-human-consensus agreement analyses."""

    llm_annotations = pd.read_csv(LLM_PATH)
    human_annotations = load_mhs_dataframe("ucberkeley-dlab/measuring-hate-speech")
    reference_comment_ids = sorted(llm_annotations["comment_id"].astype(int).unique().tolist())
    detail, summary = build_llm_human_consensus_agreement(
        llm_annotations=llm_annotations,
        human_annotations=human_annotations,
        reference_comment_ids=reference_comment_ids,
    )
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
