"""The Meta Model API client: the official OpenAI Python SDK, pointed at api.meta.ai.

    from kernel_agent.client import make_client
    client = make_client()
    response = client.responses.create(model="muse-spark-1.3", input="Hello")

The API key stays inside the client; traces never see it. The SDK retries rate
limits (429) and overloads (503) on its own, waiting as long as the API asks.
"""

from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES = 4
TIMEOUT_SECONDS = 180


class MissingKeyError(RuntimeError):
    pass


def make_client() -> OpenAI:
    """An OpenAI client for the Meta Model API, from MODEL_API_KEY and MODEL_API_BASE_URL."""
    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise MissingKeyError("MODEL_API_KEY is not set: add it to .env")
    return OpenAI(
        api_key=key,
        base_url=os.environ.get("MODEL_API_BASE_URL") or DEFAULT_BASE_URL,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT_SECONDS,
    )


def make_critic_client() -> OpenAI | None:
    """An OpenAI client for the critic API, or None when it is not configured.

    Reads GEMINI_API_KEY and GEMINI_API_BASE_URL (default: Google's
    OpenAI-compatible endpoint). No key means no critic: the run works with
    the driver model only.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    return OpenAI(
        api_key=key,
        base_url=os.environ.get("GEMINI_API_BASE_URL") or GEMINI_DEFAULT_BASE_URL,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT_SECONDS,
    )
