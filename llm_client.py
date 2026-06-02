from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

YamlDict = Dict[str, Any]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _normalize_azure_endpoint(raw_value: str) -> str:
    cleaned = raw_value.strip().rstrip("/")
    marker = ".cognitiveservices.azure.com"
    if marker in cleaned:
        base = cleaned.split(marker, 1)[0] + marker
        return base
    return cleaned


def get_llm_runtime_config() -> YamlDict:
    
    load_local_env()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPEN_BASE_URL")
    resolved_model = (
        os.getenv("OPENAI_DEPLOYMENT")
    )

    is_azure = bool(base_url and "cognitiveservices.azure.com" in base_url)
    provider = "azure_openai" if is_azure else "openai"
    normalized_base_url = _normalize_azure_endpoint(base_url) if base_url else "https://api.openai.com/v1"

    return {
        "provider": provider,
        "model": resolved_model,
        "base_url": normalized_base_url,
        "api_version": os.getenv("OPENAI_API_VERSION", "2025-04-01-preview") if is_azure else None,
        "api_key_present": bool(api_key),
    }


_cached_model_name: Optional[str] = None


def get_default_model_name() -> str:
    global _cached_model_name
    if _cached_model_name is None:
        load_local_env()
        _cached_model_name = str(get_llm_runtime_config()["model"])
    return _cached_model_name


def build_openai_client() -> Any:
    
    load_local_env()

    runtime_config = get_llm_runtime_config()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is not set")

    if runtime_config["provider"] == "azure_openai":
        from openai import AzureOpenAI

        api_version = str(runtime_config["api_version"])
        azure_endpoint = str(runtime_config["base_url"])
        return AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=str(runtime_config["base_url"]))


def call_json_prompt(
    client: Any,
    *,
    model: Optional[str] = None,
    system_text: str,
    user_payload: YamlDict,
    max_tokens: Optional[int] = None,
) -> str:
    resolved_model = model or get_default_model_name()
    user_text = json.dumps(
        user_payload,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )
    kwargs: YamlDict = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


from typing import Iterator


def call_prompt_stream(
    client: Any,
    *,
    model: Optional[str] = None,
    system_text: str,
    user_text: str,
) -> Iterator[str]:
    """Stream response chunks from OpenAI chat completions API."""
    resolved_model = model or get_default_model_name()
    stream = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


__all__ = [
    "build_openai_client",
    "call_json_prompt",
    "call_prompt_stream",
    "get_default_model_name",
    "get_llm_runtime_config",
    "load_local_env",
]
