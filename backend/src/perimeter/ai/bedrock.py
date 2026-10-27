"""Bedrock runtime client, authenticated with the Bedrock API key (`BEDROCK_API_KEY`).

The key is a bearer token. boto3 reads it from `AWS_BEARER_TOKEN_BEDROCK`, and only Bedrock
services use that variable - the S3 evidence store keeps signing with the regular AWS keys
(verified with both present in one process). The key may belong to a different AWS account
than the evidence bucket, which is why S3 reads by Bedrock need their own permission.
"""

from __future__ import annotations

import json
import os
from typing import Any

from perimeter.settings import AISettings

EMBED_MAX_CHARS = 2000


class AIUnavailableError(RuntimeError):
    """AI features are switched off or have no Bedrock API key."""


def runtime_client(ai: AISettings) -> Any:
    if not ai.enabled:
        raise AIUnavailableError("AI features are disabled (ai.enabled is false)")
    if not ai.api_key:
        raise AIUnavailableError("BEDROCK_API_KEY is not set in .env")

    import boto3
    from botocore.config import Config

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = ai.api_key
    return boto3.client(
        "bedrock-runtime",
        region_name=ai.region,
        config=Config(
            connect_timeout=10,
            # Video analysis takes ~5-10 s per 10 s clip (measured); leave room for longer clips.
            read_timeout=180,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def embed_text(client: Any, ai: AISettings, text: str) -> list[float]:
    """A normalised `embed_model` (Titan Text v2) vector for `text`."""
    response = client.invoke_model(
        modelId=ai.embed_model,
        body=json.dumps({
            "inputText": text[:EMBED_MAX_CHARS],
            "dimensions": ai.embed_dims,
            "normalize": True,
        }),
        contentType="application/json",
        accept="application/json",
    )
    return [float(v) for v in json.loads(response["body"].read())["embedding"]]


def converse_text(
    client: Any,
    ai: AISettings,
    content: list[dict[str, Any]],
    max_tokens: int,
    system: str | None = None,
) -> str:
    """One non-streaming `chat_model` reply to a single user turn, as plain text."""
    kwargs: dict[str, Any] = {
        "modelId": ai.chat_model,
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    response = client.converse(**kwargs)
    parts = response["output"]["message"]["content"]
    return "".join(part.get("text", "") for part in parts)


class LazyClient:
    """The Bedrock client, created on first use - so a service can be built (and tested)
    without AWS, and several services share one client."""

    def __init__(self, ai: AISettings, client: Any = None) -> None:
        self.ai = ai
        self._client = client

    def get(self) -> Any:
        if self._client is None:
            self._client = runtime_client(self.ai)
        return self._client
