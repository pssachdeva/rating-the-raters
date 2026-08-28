import json
from pathlib import Path

from rating_raters.config import (
    BatchModelConfig,
    BatchPromptConfig,
    BatchStorageConfig,
    ModelBatchConfig,
)
from rating_raters.retry_direct import (
    _parse_openai_compatible_streaming_lines,
    _parse_together_streaming_lines,
    _merge_processed_rows,
    _merge_processing_errors,
    _merge_raw_results,
    _rebuild_itemwise_original_run,
    _select_retry_manifest_rows,
)


def test_select_retry_manifest_rows_keeps_manifest_order() -> None:
    manifest_rows = [
        {"custom_id": "comment-1", "comment_id": 1, "text": "one"},
        {"custom_id": "comment-2", "comment_id": 2, "text": "two"},
        {"custom_id": "comment-3", "comment_id": 3, "text": "three"},
    ]
    error_rows = [
        {"custom_id": "comment-3", "error": "bad"},
        {"custom_id": "comment-1", "error": "bad"},
    ]

    selected_rows = _select_retry_manifest_rows(
        manifest_rows=manifest_rows,
        error_rows=error_rows,
    )

    assert [row["custom_id"] for row in selected_rows] == ["comment-1", "comment-3"]


def test_merge_processed_rows_prefers_retry_rows_and_preserves_manifest_order() -> None:
    original_rows = [
        {"comment_id": 1, "judge_id": "model-a", "sentiment": "A"},
        {"comment_id": 3, "judge_id": "model-a", "sentiment": "C"},
    ]
    retry_rows = [
        {"comment_id": 2, "judge_id": "model-a", "sentiment": "B"},
        {"comment_id": 3, "judge_id": "model-a", "sentiment": "D"},
    ]
    manifest_rows = [
        {"comment_id": 1, "custom_id": "comment-1", "text": "one"},
        {"comment_id": 2, "custom_id": "comment-2", "text": "two"},
        {"comment_id": 3, "custom_id": "comment-3", "text": "three"},
    ]

    merged_rows = _merge_processed_rows(
        original_rows=original_rows,
        retry_rows=retry_rows,
        manifest_rows=manifest_rows,
    )

    assert [row["comment_id"] for row in merged_rows] == [1, 2, 3]
    assert merged_rows[2]["sentiment"] == "D"


def test_merge_raw_results_prefers_retry_rows_and_preserves_manifest_order() -> None:
    original_rows = [
        {"custom_id": "comment-1", "response_text": "old-1"},
        {"custom_id": "comment-3", "response_text": "old-3"},
    ]
    retry_rows = [
        {"custom_id": "comment-2", "response_text": "new-2"},
        {"custom_id": "comment-3", "response_text": "new-3"},
    ]
    manifest_rows = [
        {"custom_id": "comment-1", "comment_id": 1},
        {"custom_id": "comment-2", "comment_id": 2},
        {"custom_id": "comment-3", "comment_id": 3},
    ]

    merged_rows = _merge_raw_results(
        original_rows=original_rows,
        retry_rows=retry_rows,
        manifest_rows=manifest_rows,
    )

    assert [row["custom_id"] for row in merged_rows] == [
        "comment-1",
        "comment-2",
        "comment-3",
    ]
    assert merged_rows[2]["response_text"] == "new-3"


def test_merge_processing_errors_keeps_only_unresolved_retry_failures() -> None:
    original_error_rows = [
        {"custom_id": "comment-1", "error": "old-1"},
        {"custom_id": "comment-2", "comment_id": 2, "error": "old-2"},
        {"custom_id": "comment-3", "comment_id": 3, "error": "old-3"},
    ]
    retry_rows = [
        {"comment_id": 1, "judge_id": "model-a"},
        {"comment_id": 2, "judge_id": "model-a"},
    ]
    retry_errors = [
        {"custom_id": "comment-3", "comment_id": 3, "error": "still-bad"},
    ]

    remaining_rows = _merge_processing_errors(
        original_error_rows=original_error_rows,
        retry_rows=retry_rows,
        retry_errors=retry_errors,
    )

    assert remaining_rows == retry_errors


def test_rebuild_itemwise_original_run_repairs_one_failed_item(tmp_path: Path) -> None:
    config = ModelBatchConfig(
        name="itemwise",
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_itemwise_v1.txt"),
            user_prompt_template="",
            mode="item_by_item",
            items_path=Path("prompts/mhs_survey_items_v1.yaml"),
        ),
        model=BatchModelConfig(provider="openai", name="gpt-5.4", id="openai_gpt-5.4"),
        batches=BatchStorageConfig(
            run_dir=tmp_path,
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )
    item_names = [
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
    manifest = [
        {
            "custom_id": f"comment-1--{item_name}",
            "comment_id": 1,
            "item_name": item_name,
            "text": "example",
        }
        for item_name in item_names
    ]
    original_raw = [
        {
            "custom_id": row["custom_id"],
            "response_text": json.dumps({"target_groups": ["A"], row["item_name"]: "A"}),
        }
        for row in manifest
    ]
    original_raw[-1]["response_text"] = "not json"
    retry_raw = [
        {
            "custom_id": manifest[-1]["custom_id"],
            "response_text": json.dumps({"target_groups": ["A"], "hate_speech": "B"}),
        }
    ]

    raw_path = tmp_path / "raw_results.jsonl"
    processed_path = tmp_path / "processed_records.jsonl"
    csv_path = tmp_path / "processed_records.csv"
    errors_path = tmp_path / "processing_errors.jsonl"
    metadata_path = tmp_path / "batch_job.json"
    raw_path.write_text("\n".join(json.dumps(row) for row in original_raw) + "\n")
    processed_path.write_text("")
    errors_path.write_text(
        json.dumps({"custom_id": manifest[-1]["custom_id"], "error": "not json"}) + "\n"
    )
    metadata_path.write_text(json.dumps({"batch_id": "batch-1"}))

    _, rows, errors = _rebuild_itemwise_original_run(
        config=config,
        manifest_rows=manifest,
        original_raw_results_path=raw_path,
        original_processed_records_path=processed_path,
        original_processed_csv_path=csv_path,
        original_errors_path=errors_path,
        original_metadata_path=metadata_path,
        raw_retry_results=retry_raw,
        retry_metadata={"retried_count": 1},
        include_all_cols=False,
    )

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["hate_speech"] == "B"


def test_parse_together_streaming_lines_combines_delta_content() -> None:
    provider_response, response_text = _parse_together_streaming_lines(
        [
            b'data: {"choices":[{"delta":{"content":"{\\"target_groups\\":"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" [\\"I\\"]}"}}]}\n',
            b"data: [DONE]\n",
        ]
    )

    assert response_text == '{"target_groups": ["I"]}'
    assert provider_response["stream"] is True
    assert provider_response["choices"][0]["message"]["content"] == response_text


def test_parse_openai_compatible_streaming_lines_combines_delta_content() -> None:
    provider_response, response_text = _parse_openai_compatible_streaming_lines(
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"{\\"hate_speech\\":"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" \\"B\\"}"}}]}\n',
            b"data: [DONE]\n",
        ]
    )

    assert response_text == '{"hate_speech": "B"}'
    assert provider_response["stream"] is True
    assert provider_response["choices"][0]["message"]["content"] == response_text
