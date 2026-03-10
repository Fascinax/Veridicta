"""Tests for tools/copilot_client.py — SDK-based Copilot client.

Covers mojibake detection/repair, token resolution, client construction,
chat() and chat_stream() methods with fully mocked SDK internals.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.copilot_client import (
    COPILOT_DEFAULT_MODEL,
    BridgeError,
    CopilotClient,
    _resolve_token,
    _build_sdk_client,
)


# ── Mojibake scoring ──────────────────────────────────────────────────────

class TestMojibakeScoring:

    def test_zero_for_clean_text(self) -> None:
        assert CopilotClient._mojibake_score("Les conditions du droit monégasque.") == 0

    def test_counts_markers(self) -> None:
        text = "monÃ©gasque et lâ€™employeur"
        score = CopilotClient._mojibake_score(text)
        assert score >= 2

    def test_counts_multiple_occurrences(self) -> None:
        assert CopilotClient._mojibake_score("Ã Ã Ã") == 3

    def test_empty_string(self) -> None:
        assert CopilotClient._mojibake_score("") == 0


# ── Mojibake repair ──────────────────────────────────────────────────────

class TestMojibakeRepair:

    def test_fixes_latin1_encoding(self) -> None:
        repaired = CopilotClient._repair_mojibake("monÃ©gasque")
        assert "é" in repaired
        assert "Ã" not in repaired

    def test_fixes_quote_corruption(self) -> None:
        repaired = CopilotClient._repair_mojibake("lâ€™employeur")
        assert "â€™" not in repaired
        assert "employeur" in repaired

    def test_returns_clean_text_unchanged(self) -> None:
        clean = "Le contrat de travail monégasque."
        assert CopilotClient._repair_mojibake(clean) == clean

    def test_empty_string(self) -> None:
        assert CopilotClient._repair_mojibake("") == ""

    def test_unfixable_text_returns_string(self) -> None:
        result = CopilotClient._repair_mojibake("Ã random noise")
        assert isinstance(result, str)


# ── Token resolution ─────────────────────────────────────────────────────

class TestResolveToken:

    def test_returns_github_pat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_PAT", "pat_abc")
        for key in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        assert _resolve_token() == "pat_abc"

    def test_returns_gh_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("GITHUB_PAT", "COPILOT_GITHUB_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GH_TOKEN", "gh_xyz")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert _resolve_token() == "gh_xyz"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("GITHUB_PAT", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        assert _resolve_token() is None

    def test_priority_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_PAT", "first")
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "second")
        monkeypatch.setenv("GH_TOKEN", "third")
        monkeypatch.setenv("GITHUB_TOKEN", "fourth")
        assert _resolve_token() == "first"


# ── Client construction ──────────────────────────────────────────────────

class TestCopilotClientInit:

    def test_default_model(self) -> None:
        client = CopilotClient()
        assert client.model == COPILOT_DEFAULT_MODEL

    def test_custom_model(self) -> None:
        client = CopilotClient(model="o3-mini")
        assert client.model == "o3-mini"

    def test_context_manager(self) -> None:
        with CopilotClient() as client:
            assert isinstance(client, CopilotClient)

    def test_close_is_noop(self) -> None:
        client = CopilotClient()
        client.close()


# ── _build_sdk_client ────────────────────────────────────────────────────

class TestBuildSdkClient:

    @patch("tools.copilot_client._resolve_token", return_value="tok_123")
    @patch("tools.copilot_client._SdkClient")
    def test_passes_token_when_available(self, mock_sdk: MagicMock, _mock_tok: MagicMock) -> None:
        _build_sdk_client()
        call_args = mock_sdk.call_args[0][0]
        assert call_args["github_token"] == "tok_123"
        assert call_args["use_logged_in_user"] is False

    @patch("tools.copilot_client._resolve_token", return_value=None)
    @patch("tools.copilot_client._SdkClient")
    def test_no_token_omits_github_token_key(self, mock_sdk: MagicMock, _mock_tok: MagicMock) -> None:
        _build_sdk_client()
        call_args = mock_sdk.call_args[0][0]
        assert "github_token" not in call_args


# ── helpers to build mocked SDK sessions ─────────────────────────────────

def _make_fake_session(response_text: str):
    """Mock session that emits assistant.message then session.idle."""
    session = AsyncMock()
    _handler_box: list = []

    def sync_on(handler):
        _handler_box.append(handler)

    async def fake_send(payload):
        handler = _handler_box[0]
        msg_event = MagicMock()
        msg_event.type.value = "assistant.message"
        msg_event.data.content = response_text
        handler(msg_event)
        idle_event = MagicMock()
        idle_event.type.value = "session.idle"
        handler(idle_event)

    session.on = MagicMock(side_effect=sync_on)
    session.send = AsyncMock(side_effect=fake_send)
    session.disconnect = AsyncMock()
    return session


def _make_fake_client(session):
    """Mock SDK client returning the given session."""
    client = AsyncMock()
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.create_session = AsyncMock(return_value=session)
    return client


# ── chat() ───────────────────────────────────────────────────────────────

class TestCopilotClientChat:

    @patch("tools.copilot_client._build_sdk_client")
    def test_returns_response_text(self, mock_build: MagicMock) -> None:
        session = _make_fake_session("Réponse juridique")
        mock_build.return_value = _make_fake_client(session)

        result = CopilotClient().chat(system="System", user="Question")
        assert result == "Réponse juridique"

    @patch("tools.copilot_client._build_sdk_client")
    def test_repairs_mojibake_in_response(self, mock_build: MagicMock) -> None:
        session = _make_fake_session("monÃ©gasque")
        mock_build.return_value = _make_fake_client(session)

        result = CopilotClient().chat(system="S", user="U")
        assert "é" in result
        assert "Ã" not in result

    @patch("tools.copilot_client._build_sdk_client")
    def test_raises_on_empty_response(self, mock_build: MagicMock) -> None:
        session = _make_fake_session("")
        mock_build.return_value = _make_fake_client(session)

        with pytest.raises(BridgeError, match="empty output"):
            CopilotClient().chat(system="S", user="U")

    @patch("tools.copilot_client._build_sdk_client")
    def test_raises_bridge_error_on_sdk_exception(self, mock_build: MagicMock) -> None:
        fake_client = AsyncMock()
        fake_client.start = AsyncMock(side_effect=RuntimeError("SDK crash"))
        fake_client.stop = AsyncMock()
        mock_build.return_value = fake_client

        with pytest.raises(BridgeError, match="SDK error"):
            CopilotClient().chat(system="S", user="U")

    @patch("tools.copilot_client._build_sdk_client")
    def test_concatenates_multiple_message_events(self, mock_build: MagicMock) -> None:
        session = AsyncMock()
        _handler_box: list = []

        def sync_on(handler):
            _handler_box.append(handler)

        async def fake_send(payload):
            handler = _handler_box[0]
            for part in ["Part 1. ", "Part 2."]:
                ev = MagicMock()
                ev.type.value = "assistant.message"
                ev.data.content = part
                handler(ev)
            idle = MagicMock()
            idle.type.value = "session.idle"
            handler(idle)

        session.on = MagicMock(side_effect=sync_on)
        session.send = AsyncMock(side_effect=fake_send)
        session.disconnect = AsyncMock()
        mock_build.return_value = _make_fake_client(session)

        result = CopilotClient().chat(system="S", user="U")
        assert result == "Part 1. Part 2."


# ── chat_stream() ────────────────────────────────────────────────────────

class TestCopilotClientStream:

    @patch("tools.copilot_client._build_sdk_client")
    def test_yields_token_deltas(self, mock_build: MagicMock) -> None:
        session = AsyncMock()
        _handler_box: list = []

        def sync_on(handler):
            _handler_box.append(handler)

        async def fake_send(payload):
            handler = _handler_box[0]
            for delta in ["Tok1", "Tok2", "Tok3"]:
                ev = MagicMock()
                ev.type.value = "assistant.message_delta"
                ev.data.delta_content = delta
                handler(ev)
            idle = MagicMock()
            idle.type.value = "session.idle"
            handler(idle)

        session.on = MagicMock(side_effect=sync_on)
        session.send = AsyncMock(side_effect=fake_send)
        session.disconnect = AsyncMock()
        mock_build.return_value = _make_fake_client(session)

        tokens = list(CopilotClient().chat_stream(system="S", user="U"))
        assert tokens == ["Tok1", "Tok2", "Tok3"]

    @patch("tools.copilot_client._build_sdk_client")
    def test_stream_raises_on_sdk_error(self, mock_build: MagicMock) -> None:
        fake_client = AsyncMock()
        fake_client.start = AsyncMock(side_effect=RuntimeError("Stream crash"))
        fake_client.stop = AsyncMock()
        mock_build.return_value = fake_client

        with pytest.raises(BridgeError, match="stream error"):
            list(CopilotClient().chat_stream(system="S", user="U"))
