import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"
LMSPEED_AA_MIRROR_URL = "https://lmspeed.net/leaderboard/best-intelligence-index-models"

AA_MODEL_PATTERN = re.compile(
    r'\\"id\\":\\"(?P<id>[^\\]+)\\",'
    r'\\"name\\":\\"(?P<name>(?:[^\\]|\\.)*?)\\",'
    r'\\"shortName\\":\\"(?P<short_name>(?:[^\\]|\\.)*?)\\",'
    r'\\"slug\\":\\"(?P<slug>[^\\]+)\\"'
    r'.*?\\"intelligenceIndex\\":(?P<intelligence_index>-?\d+(?:\.\d+)?|null)',
    re.S,
)

FALLBACK_MIRROR_ROWS = {
    "claude-sonnet-4-5": ("Claude Sonnet 4.5", 37.1),
    "claude-opus-4-1": ("Claude Opus 4.1", 36.0),
    "claude-opus-4": ("Claude Opus 4", 33.0),
    "claude-sonnet-4": ("Claude Sonnet 4", 33.0),
}

AA_MATCHES = {
    "openai_gpt-5.4-nano_medium": ("gpt-5-4-nano-medium", "exact", ""),
    "openai_gpt-5.4-nano_high": (
        "gpt-5-4-nano",
        "nearest_effort",
        "AA reports xhigh but not high for this model.",
    ),
    "openai_gpt-5.4-nano_low": (
        "gpt-5-4-nano-medium",
        "nearest_effort",
        "AA reports medium but not low for this model.",
    ),
    "openai_gpt-5.4-mini_xhigh": ("gpt-5-4-mini", "exact", ""),
    "openai_gpt-5.4-nano_xhigh": ("gpt-5-4-nano", "exact", ""),
    "openai_gpt-5.4-mini_high": (
        "gpt-5-4-mini",
        "nearest_effort",
        "AA reports xhigh but not high for this model.",
    ),
    "openai_gpt-5.4-mini_low": (
        "gpt-5-4-mini-medium",
        "nearest_effort",
        "AA reports medium but not low for this model.",
    ),
    "openai_gpt-5.4-mini_none": ("gpt-5-4-mini-non-reasoning", "exact", ""),
    "openai_gpt-5.4-mini_medium": ("gpt-5-4-mini-medium", "exact", ""),
    "openrouter_deepseek_deepseek-v3.2": (
        "deepseek-v3-2",
        "exact_model",
        "Mapped to AA non-reasoning row because the run has no reasoning suffix.",
    ),
    "openai_gpt-5.2_high": (
        "gpt-5-2",
        "nearest_effort",
        "AA reports xhigh but not high for this model.",
    ),
    "openai_gpt-5.4-nano_none": ("gpt-5-4-nano-non-reasoning", "exact", ""),
    "openai_gpt-5.4_high": (
        "gpt-5-4",
        "nearest_effort",
        "AA reports xhigh but not high for this model.",
    ),
    "openai_gpt-5.2_medium": ("gpt-5-2-medium", "exact", ""),
    "openai_gpt-5.2_xhigh": ("gpt-5-2", "exact", ""),
    "openai_gpt-5.4_medium": (
        "gpt-5-4-low",
        "nearest_effort",
        "AA reports low and xhigh but not medium for this model.",
    ),
    "openai_gpt-5.4_low": ("gpt-5-4-low", "exact", ""),
    "google_gemini-3.1-pro-preview_high": (
        "gemini-3-1-pro-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Pro Preview score without effort splits.",
    ),
    "openrouter_z-ai_glm-5-turbo": ("glm-5-turbo", "exact", ""),
    "openai_gpt-5.2_low": (
        "gpt-5-2-medium",
        "nearest_effort",
        "AA reports medium but not low for this model.",
    ),
    "google_gemini-3.1-pro-preview_medium": (
        "gemini-3-1-pro-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Pro Preview score without effort splits.",
    ),
    "openai_gpt-5.4_xhigh": ("gpt-5-4", "exact", ""),
    "google_gemini-3-flash-preview_low": (
        "gemini-3-flash-reasoning",
        "nearest_effort",
        "AA reports reasoning and non-reasoning rows, not low/medium/high splits.",
    ),
    "openai_gpt-5.4_none": ("gpt-5-4-non-reasoning", "exact", ""),
    "openai_gpt-5.2_none": ("gpt-5-2-non-reasoning", "exact", ""),
    "anthropic_claude-opus-4-6_high": ("claude-opus-4-6", "exact", ""),
    "anthropic_claude-opus-4": ("claude-opus-4", "mirror_fallback", ""),
    "anthropic_claude-opus-4-5_high": (
        "claude-opus-4-5-thinking",
        "exact_model",
        "Mapped to AA reasoning row.",
    ),
    "anthropic_claude-sonnet-4-6_high": ("claude-sonnet-4-6", "exact", ""),
    "anthropic_claude-sonnet-4-6_medium": (
        "claude-sonnet-4-6",
        "nearest_effort",
        "AA reports high and low non-reasoning rows but not medium.",
    ),
    "google_gemini-3.1-pro-preview_low": (
        "gemini-3-1-pro-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Pro Preview score without effort splits.",
    ),
    "openrouter_minimax_minimax-m2.5": ("minimax-m2-5", "exact", ""),
    "google_gemini-3.1-flash-lite-preview_medium": (
        "gemini-3-1-flash-lite-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Flash-Lite Preview score without effort splits.",
    ),
    "google_gemini-3-flash-preview_medium": (
        "gemini-3-flash-reasoning",
        "nearest_effort",
        "AA reports reasoning and non-reasoning rows, not low/medium/high splits.",
    ),
    "moonshot_kimi-k2.5": ("kimi-k2-5", "exact_model", "Mapped to AA reasoning row."),
    "xai_grok-4-fast-non-reasoning": ("grok-4-fast", "exact", ""),
    "anthropic_claude-sonnet-4-5": ("claude-sonnet-4-5", "mirror_fallback", ""),
    "google_gemini-3-flash-preview_minimal": (
        "gemini-3-flash",
        "nearest_effort",
        "AA reports non-reasoning rather than a minimal-effort row.",
    ),
    "anthropic_claude-opus-4-6_low": (
        "claude-opus-4-6",
        "nearest_effort",
        "AA reports high non-reasoning and max adaptive rows but not low.",
    ),
    "google_gemini-3.1-flash-lite-preview_low": (
        "gemini-3-1-flash-lite-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Flash-Lite Preview score without effort splits.",
    ),
    "anthropic_claude-opus-4-1": ("claude-opus-4-1", "mirror_fallback", ""),
    "google_gemini-3-flash-preview_high": (
        "gemini-3-flash-reasoning",
        "nearest_effort",
        "AA reports reasoning and non-reasoning rows, not low/medium/high splits.",
    ),
    "anthropic_claude-opus-4-5_low": (
        "claude-opus-4-5",
        "nearest_effort",
        "Mapped to AA non-reasoning row because no low-effort row is published.",
    ),
    "anthropic_claude-sonnet-4-6_low": (
        "claude-sonnet-4-6-non-reasoning-low-effort",
        "exact",
        "",
    ),
    "anthropic_claude-opus-4-6_medium": (
        "claude-opus-4-6",
        "nearest_effort",
        "AA reports high non-reasoning and max adaptive rows but not medium.",
    ),
    "google_gemini-3.1-flash-lite-preview_high": (
        "gemini-3-1-flash-lite-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Flash-Lite Preview score without effort splits.",
    ),
    "openrouter_qwen_qwen3.5-122b-a10b": ("qwen3-5-122b-a10b", "exact", ""),
    "xai_grok-4-1-fast-non-reasoning": ("grok-4-1-fast", "exact", ""),
    "anthropic_claude-opus-4-5_medium": (
        "claude-opus-4-5-thinking",
        "exact_model",
        "Mapped to AA reasoning row.",
    ),
    "deepseek_deepseek-v4-pro": ("deepseek-v4-pro", "exact", ""),
    "openrouter_moonshotai_kimi-k2.5": (
        "kimi-k2-5",
        "exact_model",
        "Mapped to AA reasoning row.",
    ),
    "xai_grok-4-fast-reasoning": ("grok-4-fast-reasoning", "exact", ""),
    "anthropic_claude-sonnet-4": ("claude-sonnet-4", "mirror_fallback", ""),
    "openrouter_xiaomi_mimo-v2-pro": ("mimo-v2-pro", "exact", ""),
    "google_gemini-3.1-flash-lite-preview_minimal": (
        "gemini-3-1-flash-lite-preview",
        "exact_model",
        "AA reports one Gemini 3.1 Flash-Lite Preview score without effort splits.",
    ),
    "xai_grok-4-1-fast-reasoning": ("grok-4-1-fast-reasoning", "exact", ""),
    "google_gemini-2.5-pro": ("gemini-2-5-pro", "exact", ""),
    "anthropic_claude-haiku-4-5": (
        "claude-4-5-haiku",
        "exact_model",
        "Mapped to AA non-reasoning row because the run has no reasoning suffix.",
    ),
    "together_openai_gpt-oss-120b": (
        "gpt-oss-120b",
        "exact_model",
        "AA slug matches the base model and reports the high-effort row.",
    ),
    "google_gemini-2.5-flash": (
        "gemini-2-5-flash",
        "exact_model",
        "Mapped to AA non-reasoning row because the run has no reasoning suffix.",
    ),
    "openai_gpt-4.1": ("gpt-4-1", "exact", ""),
    "xai_grok-3": ("grok-3", "exact", ""),
    "openai_gpt-4o": ("gpt-4o", "exact_model", "Mapped to AA generic GPT-4o row."),
    "together_meta-llama_llama-3.3-70b-instruct-turbo": (
        "llama-3-3-instruct-70b",
        "exact",
        "",
    ),
}


def fetch_aa_models() -> dict[str, dict[str, object]]:
    """Fetch and parse Artificial Analysis model rows from the current leaderboard page."""
    request = Request(AA_LEADERBOARD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")

    models = {}
    for match in AA_MODEL_PATTERN.finditer(html):
        row = match.groupdict()
        if "\\" in row["name"] or "},{" in row["name"]:
            continue
        index = row["intelligence_index"]
        if index == "null":
            continue
        models[row["slug"]] = {
            "aa_model_name": json.loads(f'"{row["name"]}"'),
            "aa_model_slug": row["slug"],
            "aa_intelligence_index": float(index),
            "aa_source_url": AA_LEADERBOARD_URL,
        }
    return models


def add_aa_index(input_path: Path, output_path: Path, audit_path: Path) -> None:
    """Add AA Intelligence Index fields to a severity dataset and write match audit rows."""
    models = fetch_aa_models()
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    df = pd.read_csv(input_path)
    audit_rows = []

    for index, row in df.iterrows():
        judge_id = row["judge_id"]
        if judge_id not in AA_MATCHES:
            raise KeyError(f"No AA match configured for {judge_id}")
        slug, match_type, notes = AA_MATCHES[judge_id]

        if match_type == "mirror_fallback":
            model_name, aa_index = FALLBACK_MIRROR_ROWS[slug]
            match = {
                "aa_model_name": model_name,
                "aa_model_slug": slug,
                "aa_intelligence_index": aa_index,
                "aa_source_url": LMSPEED_AA_MIRROR_URL,
            }
            notes = "Not present in the current AA page scrape; using LMSpeed's AA mirror."
        else:
            if slug not in models:
                raise KeyError(f"AA slug not found: {slug}")
            match = models[slug]

        for column, value in match.items():
            df.loc[index, column] = value
        df.loc[index, "aa_match_type"] = match_type
        df.loc[index, "aa_match_notes"] = notes
        df.loc[index, "aa_retrieved_at_utc"] = retrieved_at

        audit_rows.append(
            {
                "judge_id": judge_id,
                "model_label": row["model_label"],
                **match,
                "aa_match_type": match_type,
                "aa_match_notes": notes,
                "aa_retrieved_at_utc": retrieved_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
