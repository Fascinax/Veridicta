from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.omniroute_client import (
    DEFAULT_LOCAL_API_KEY,
    MAX_EMPTY_OUTPUT_ATTEMPTS,
    MAX_RESPONSE_TOKENS,
    OmniRouteClient,
    OmniRouteError,
    omniroute_api_key,
    omniroute_base_url,
)


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_omniroute_configuration_reads_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://gateway.test/v1/")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "sk-test")

    assert (omniroute_base_url(), omniroute_api_key()) == (
        "http://gateway.test/v1",
        "sk-test",
    )


def test_omniroute_uses_local_placeholder_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert omniroute_api_key() == DEFAULT_LOCAL_API_KEY


def test_chat_returns_openai_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "sk-test")
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _completion("Réponse [Source 1]")

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client):
        response = OmniRouteClient("auto").chat(system="system", user="question")

    assert response == "Réponse [Source 1]"


def test_chat_uses_openai_compatible_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("OMNIROUTE_API_KEY", "sk-test")
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _completion("Réponse")

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client) as openai:
        OmniRouteClient("auto").chat(system="system", user="question")

    openai.assert_called_once_with(
        api_key="sk-test",
        base_url="http://gateway.test/v1",
    )
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="auto",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        temperature=0.1,
        max_tokens=MAX_RESPONSE_TOKENS,
    )


def test_chat_retries_empty_output_until_text_is_returned() -> None:
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        _completion(""),
        _completion("Réponse après retry"),
    ]

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client):
        response = OmniRouteClient("auto").chat(system="system", user="question")

    assert response == "Réponse après retry"
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_chat_raises_after_empty_output_attempts_are_exhausted() -> None:
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        _completion("") for _ in range(MAX_EMPTY_OUTPUT_ATTEMPTS)
    ]

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client):
        with pytest.raises(OmniRouteError, match="after 3 attempts"):
            OmniRouteClient("auto").chat(system="system", user="question")

    assert mock_openai_client.chat.completions.create.call_count == MAX_EMPTY_OUTPUT_ATTEMPTS


def test_chat_stream_yields_text_deltas() -> None:
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = iter(
        [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Réponse "))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="streamée"))]),
        ]
    )

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client):
        chunks = list(OmniRouteClient("auto").chat_stream(system="system", user="question"))

    assert chunks == ["Réponse ", "streamée"]


def test_chat_stream_retries_when_first_stream_is_empty() -> None:
    empty_stream = iter([SimpleNamespace(choices=[])])
    text_stream = iter(
        [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Réponse"))])]
    )
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [empty_stream, text_stream]

    with patch("tools.omniroute_client.OpenAI", return_value=mock_openai_client):
        chunks = list(OmniRouteClient("auto").chat_stream(system="system", user="question"))

    assert chunks == ["Réponse"]
    assert mock_openai_client.chat.completions.create.call_count == 2
