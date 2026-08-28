from pathlib import Path

from rating_raters.aa_index import add_aa_index


def main() -> None:
    """Add AA Intelligence Index columns to the latest reference-set severity table."""
    add_aa_index(
        input_path=Path("data/reference_set_model_severities_latest.csv"),
        output_path=Path("data/reference_set_model_severities_latest.csv"),
        audit_path=Path("data/aa_intelligence_index_model_matches.csv"),
    )


if __name__ == "__main__":
    main()
