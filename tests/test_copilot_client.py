"""Tests for tools/copilot_client.py — mojibake repair and basic client logic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.copilot_client import BridgeError, CopilotClient


class TestMojibakeDetection:
    """Test mojibake detection and scoring."""

    def test_mojibake_score_counts_markers_correctly(self) -> None:
        text = "Les conditions du licenciement en droit monÃ©gasque et lâ€™employeur."
        score = CopilotClient._mojibake_score(text)
        assert score >= 2  # Contains "Ã" and "â€™"

    def test_mojibake_score_zero_for_clean_text(self) -> None:
        text = "Les conditions du licenciement en droit monégasque."
        score = CopilotClient._mojibake_score(text)
        assert score == 0

    def test_mojibake_score_counts_multiple_occurrences(self) -> None:
        text = "Ã Ã Ã"
        score = CopilotClient._mojibake_score(text)
        assert score == 3


class TestMojibakeRepair:
    """Test mojibake repair logic."""

    def test_repair_mojibake_fixes_latin1_encoding(self) -> None:
        # "monégasque" encoded as latin-1 then decoded as utf-8 → "monÃ©gasque"
        corrupted = "monÃ©gasque"
        repaired = CopilotClient._repair_mojibake(corrupted)
        assert "é" in repaired
        assert "Ã" not in repaired

    def test_repair_mojibake_fixes_quote_corruption(self) -> None:
        corrupted = "lâ€™employeur"
        repaired = CopilotClient._repair_mojibake(corrupted)
        # Accept various apostrophe forms (', ', etc.)
        assert repaired != corrupted
        assert "employeur" in repaired
        assert "â€™" not in repaired

    def test_repair_mojibake_returns_unchanged_if_clean(self) -> None:
        clean_text = "Le contrat de travail monégasque."
        repaired = CopilotClient._repair_mojibake(clean_text)
        assert repaired == clean_text

    def test_repair_mojibake_handles_empty_string(self) -> None:
        repaired = CopilotClient._repair_mojibake("")
        assert repaired == ""

    def test_repair_mojibake_handles_no_valid_repair(self) -> None:
        # Text with markers but no valid latin-1/cp1252 repair
        text = "Test Ã with invalid repair"
        repaired = CopilotClient._repair_mojibake(text)
        # Should return either original or best attempt
        assert isinstance(repaired, str)


class TestCopilotClientConstruction:
    """Test client initialization."""

    @patch("tools.copilot_client._BRIDGE_SCRIPT", Path("/fake/bridge.mjs"))
    def test_init_raises_if_bridge_script_missing(self) -> None:
        with pytest.raises(BridgeError, match="Bridge script not found"):
            CopilotClient()

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    def test_init_succeeds_with_valid_bridge(self, mock_bridge: MagicMock) -> None:
        mock_bridge.exists.return_value = True
        client = CopilotClient(model="test-model")
        assert client.model == "test-model"

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    def test_init_uses_default_model_if_not_specified(self, mock_bridge: MagicMock) -> None:
        mock_bridge.exists.return_value = True
        client = CopilotClient()
        assert client.model == "gpt-4.1"


class TestCopilotClientChat:
    """Test chat() method with mocked subprocess."""

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run")
    def test_chat_returns_content_from_json_response(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"content": "Test response"}),
            stderr="",
        )

        client = CopilotClient()
        result = client.chat(system="System", user="User")

        assert result == "Test response"
        mock_run.assert_called_once()

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run")
    def test_chat_repairs_mojibake_in_response(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"content": "monÃ©gasque"}),
            stderr="",
        )

        client = CopilotClient()
        result = client.chat(system="System", user="User")

        assert "é" in result
        assert "Ã" not in result

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run")
    def test_chat_raises_bridge_error_on_nonzero_exit(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Node error",
        )

        client = CopilotClient()
        with pytest.raises(BridgeError, match="exited with code 1"):
            client.chat(system="System", user="User")

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run")
    def test_chat_raises_bridge_error_on_empty_output(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        client = CopilotClient()
        with pytest.raises(BridgeError, match="empty output"):
            client.chat(system="System", user="User")

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_chat_raises_bridge_error_if_node_not_found(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True

        client = CopilotClient()
        with pytest.raises(BridgeError, match="Node.js not found"):
            client.chat(system="System", user="User")

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("node", 300))
    def test_chat_raises_bridge_error_on_timeout(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True

        client = CopilotClient()
        with pytest.raises(BridgeError, match="timed out"):
            client.chat(system="System", user="User")

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    @patch("subprocess.run")
    def test_chat_handles_non_json_response_gracefully(
        self, mock_run: MagicMock, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.exists.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Plain text response",
            stderr="",
        )

        client = CopilotClient()
        result = client.chat(system="System", user="User")

        assert result == "Plain text response"


class TestCopilotClientContextManager:
    """Test context manager protocol."""

    @patch("tools.copilot_client._BRIDGE_SCRIPT")
    def test_client_works_as_context_manager(self, mock_bridge: MagicMock) -> None:
        mock_bridge.exists.return_value = True

        with CopilotClient() as client:
            assert client.model == "gpt-4.1"
