"""OpenAI-compatible client for the local OmniRoute gateway."""

from __future__ import annotations

import os
from collections.abc import Iterator

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only in minimal environments
    OpenAI = None  # type: ignore[assignment,misc]

from retrievers.config import OMNIROUTE_DEFAULT_BASE_URL

DEFAULT_LOCAL_API_KEY = "sk-omniroute-local"
MAX_RESPONSE_TOKENS = 4_096
MAX_EMPTY_OUTPUT_ATTEMPTS = 3


class OmniRouteError(RuntimeError):
    """Raised when OmniRoute returns no usable text."""


def omniroute_base_url() -> str:
    """Return the configured OpenAI-compatible endpoint."""
    configured_url = os.getenv("OMNIROUTE_BASE_URL", OMNIROUTE_DEFAULT_BASE_URL).strip()
    return configured_url.rstrip("/") or OMNIROUTE_DEFAULT_BASE_URL


def omniroute_api_key() -> str:
    """Return the configured key, or a local-only placeholder for no-auth mode."""
    return (
        os.getenv("OMNIROUTE_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or DEFAULT_LOCAL_API_KEY
    )


class OmniRouteClient:
    """Small synchronous client for OmniRoute chat completions."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _client(self) -> OpenAI:
        if OpenAI is None:
            raise OmniRouteError(
                "OmniRoute requires the openai package. Run: pip install openai."
            )
        return OpenAI(
            api_key=omniroute_api_key(),
            base_url=omniroute_base_url(),
        )

    def _request_payload(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
    ) -> dict:
        return {
            "model": self._model,
            "messages": self._messages(system, user),
            "temperature": temperature,
            "max_tokens": MAX_RESPONSE_TOKENS,
        }

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> str:
        """Send one request and return its text content."""
        client = self._client()
        for _ in range(MAX_EMPTY_OUTPUT_ATTEMPTS):
            completion = client.chat.completions.create(
                **self._request_payload(system=system, user=user, temperature=temperature)
            )
            choices = getattr(completion, "choices", [])
            if not choices:
                raise OmniRouteError("OmniRoute returned no choices.")

            content = getattr(getattr(choices[0], "message", None), "content", None)
            if isinstance(content, str) and content.strip():
                return content

        raise OmniRouteError(
            "OmniRoute returned empty output after "
            f"{MAX_EMPTY_OUTPUT_ATTEMPTS} attempts."
        )

    def chat_stream(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """Yield text deltas from a streaming chat completion."""
        client = self._client()
        for _ in range(MAX_EMPTY_OUTPUT_ATTEMPTS):
            emitted_text = False
            stream = client.chat.completions.create(
                **self._request_payload(system=system, user=user, temperature=temperature),
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue
                delta = getattr(getattr(choices[0], "delta", None), "content", None)
                if isinstance(delta, str) and delta:
                    emitted_text = True
                    yield delta
            if emitted_text:
                return

        raise OmniRouteError(
            "OmniRoute returned empty output after "
            f"{MAX_EMPTY_OUTPUT_ATTEMPTS} attempts."
        )
