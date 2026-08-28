# Rating the Raters: Rasch Measurement Theory for LLM Evaluation

<p align="center">
  Pratik S. Sachdeva &nbsp;·&nbsp; Nathan Boudol<br>
  <strong>Empirical Methods in Natural Language Processing (EMNLP), 2026</strong>
</p>
LLMs now sit on every side of evaluation: as examinees scored on benchmarks, judges of other models' outputs, and raters of human-generated content. Each paradigm can be viewed as a measurement problem, where a latent property of an object is probed with items from an instrument (e.g., benchmark) by raters. Standard evaluation practices often neglect the contributions of each core component to the end result, limiting our understanding of what is being measured. Rasch measurement theory (RMT) is well-suited to this kind of problem. RMT decomposes ordinal ratings into separable facets on a common scale. It further provides a battery of diagnostics that can identify miscalibrated measurements and rater biases. We present a case study of RMT applied to the LLM-as-rater paradigm using the \textit{Measuring Hate Speech} corpus, whose construct was itself built under RMT. We fit a series of many-facet Rasch models to annotations from nine LLMs spanning families and capability levels. Our analyses show that LLMs systematically differ from human raters in severity, item-level calibration, question-order robustness, target-identity sensitivity, and rating scale use, which all would be obscured by standard evaluation practice. Overall, we argue that RMT belongs in the toolkit for evaluating LLM-as-examinee, -judge, and -rater paradigms.

## Setup

Requirements:

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/)
- FACETS on Windows for fitting the generated measurement models

Install the project and development dependencies:

```bash
uv sync --group dev
uv run python -c "import rating_raters; print(rating_raters.__version__)"
```

The MHS dataset is downloaded from Hugging Face when a workflow first needs it.

## Repository layout

- `src/rating_raters/`: reusable dataset, prompting, response-processing, FACETS, and plotting code
- `prompts/`: versioned MHS survey prompts and item definitions
- `configs/`: reproducible experiment and FACETS configurations
- `scripts/`: command-line entry points for pipeline stages
- `plots/`: figure-generation entry points
- `data/`: downloaded, cleaned, cached, and processed tabular outputs
- `facets/`: generated FACETS specifications, inputs, and run outputs
- `artifacts/`: generated figures and other image-like outputs
- `tests/`: unit and integration tests
