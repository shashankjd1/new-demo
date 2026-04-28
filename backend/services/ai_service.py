import json
import os
from typing import Any

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

from models.schemas import AnalysisResponse, AnalyzeRequest


load_dotenv()


ANALYSIS_SYSTEM_PROMPT = """You are a consumer technology truth analysis engine.
Return ONLY valid JSON matching this exact schema:
{
  "truth_score": number,
  "dimensions": {
    "transparency": number,
    "verifiability": number,
    "comparability": number,
    "consistency": number
  },
  "tldr": [string, string, string],
  "flagged_claims": [
    {
      "claim": string,
      "classification": "VERIFIED | CONDITIONAL | MISLEADING | NON-VERIFIABLE",
      "severity": "LOW | MEDIUM | HIGH",
      "reason": string,
      "realistic_interpretation": string
    }
  ],
  "normalized_specs": [
    {
      "term": string,
      "meaning": string
    }
  ],
  "tradeoffs": [string],
  "real_world": [
    {
      "feature": string,
      "insight": string,
      "confidence": "HIGH | MEDIUM | LOW"
    }
  ],
  "verdict": string
}

Rules:
- Output JSON only. No markdown, no prose outside JSON.
- Score every dimension from 0 to 100.
- truth_score must also be 0 to 100 and reflect the dimensions.
- Always return exactly 3 TL;DR bullets.
- Do not hallucinate missing specs or benchmarks.
- If specs are missing, say so inside relevant fields instead of inventing details.
- Be critical but fair, grounded only in the provided product details.
- Flag vague phrases like "up to", "best-in-class", "industry-leading", "revolutionary", unnamed comparisons, and other non-verifiable marketing language when present."""

CHAT_SYSTEM_PROMPT = """You are a consumer tech truth assistant.
Answer ONLY using the provided analysis.
Do not hallucinate.
Be concise and practical."""

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 1800


def _get_client() -> OpenAI:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Treat sk-or-v1 keys as OpenRouter keys for convenience.
    if openrouter_api_key or (openai_api_key and openai_api_key.startswith("sk-or-v1-")):
        api_key = openrouter_api_key or openai_api_key
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    if openai_api_key:
        return OpenAI(api_key=openai_api_key)

    raise ValueError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file.")


def _get_model_name() -> str:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openrouter_api_key or (openai_api_key and openai_api_key.startswith("sk-or-v1-")):
        return os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def _get_max_output_tokens() -> int:
    configured = os.getenv("LLM_MAX_OUTPUT_TOKENS")
    if configured and configured.isdigit():
        return int(configured)
    return DEFAULT_MAX_OUTPUT_TOKENS


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON.")
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise ValueError("Unable to parse model JSON output.") from exc


def _validate_analysis_payload(payload: dict[str, Any]) -> AnalysisResponse:
    normalized = dict(payload)
    tldr = list(normalized.get("tldr", []))
    if len(tldr) < 3:
        tldr.extend(["Insufficient information to generate this takeaway."] * (3 - len(tldr)))
    normalized["tldr"] = tldr[:3]
    return AnalysisResponse.model_validate(normalized)


def _analysis_messages(data: AnalyzeRequest, strict_json_retry: bool = False) -> list[dict[str, str]]:
    retry_instruction = ""
    if strict_json_retry:
        retry_instruction = (
            "\nYour previous response was invalid JSON or did not match schema. "
            "Return corrected JSON only."
        )

    specs_text = data.specs.strip() if data.specs.strip() else "Specs not provided."
    return [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT + retry_instruction},
        {
            "role": "user",
            "content": (
                f"Product name: {data.product_name}\n"
                f"Marketing text:\n{data.marketing_text.strip()}\n\n"
                f"Technical specifications:\n{specs_text}"
            ),
        },
    ]


def generate_analysis(data: AnalyzeRequest) -> AnalysisResponse:
    client = _get_client()
    model_name = _get_model_name()
    max_output_tokens = _get_max_output_tokens()
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = client.responses.create(
                model=model_name,
                input=_analysis_messages(data, strict_json_retry=attempt > 0),
                max_output_tokens=max_output_tokens,
            )
        except AuthenticationError as exc:
            raise ValueError(
                "Invalid LLM API key. Update OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
            ) from exc
        except APIConnectionError as exc:
            raise ValueError("Unable to reach the LLM provider. Check your internet connection and try again.") from exc
        except APIStatusError as exc:
            if exc.status_code == 402:
                raise ValueError(
                    "Your OpenRouter account does not have enough credits for this request. Add credits or lower the model cost."
                ) from exc
            raise ValueError(f"LLM provider error: {exc.status_code}") from exc

        raw_text = response.output_text.strip()

        try:
            payload = _extract_json_payload(raw_text)
            return _validate_analysis_payload(payload)
        except (ValueError, ValidationError) as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError("Failed to generate valid structured analysis.") from last_error
    raise ValueError("Failed to generate analysis.")


def analyze_product(data: AnalyzeRequest) -> AnalysisResponse:
    return generate_analysis(data)


def generate_chat_response(query: str, context: AnalysisResponse) -> str:
    client = _get_client()
    model_name = _get_model_name()
    max_output_tokens = min(_get_max_output_tokens(), 600)
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Analysis JSON:\n{context.model_dump_json(indent=2)}\n\n"
                        f"User question: {query.strip()}"
                    ),
                },
            ],
            max_output_tokens=max_output_tokens,
        )
    except AuthenticationError as exc:
        raise ValueError(
            "Invalid LLM API key. Update OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
        ) from exc
    except APIConnectionError as exc:
        raise ValueError("Unable to reach the LLM provider. Check your internet connection and try again.") from exc
    except APIStatusError as exc:
        if exc.status_code == 402:
            raise ValueError(
                "Your OpenRouter account does not have enough credits for this request. Add credits or lower the model cost."
            ) from exc
        raise ValueError(f"LLM provider error: {exc.status_code}") from exc

    answer = response.output_text.strip()
    if not answer:
        raise ValueError("Failed to generate chat response.")
    return answer
