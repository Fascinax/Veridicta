"""Copilot SDK client — GitHub Copilot LLM backend for Veridicta.

Calls ``copilot-bridge.mjs`` as a Node.js subprocess, passing system + user
prompt as JSON on stdin and reading the generated content from stdout.

This lets Veridicta use GitHub Copilot models (gpt-4.1, claude-sonnet-4, o3-mini, etc.)
via the @github/copilot-sdk — requires a GitHub PAT with ``copilot`` scope
or ``gh auth login``.

Activate via ``LLM_BACKEND=copilot`` in ``.env``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

_BRIDGE_SCRIPT = Path(__file__).resolve().parent.parent / "copilot-bridge.mjs"

COPILOT_DEFAULT_MODEL = "gpt-4.1"


class BridgeError(Exception):
    """Raised when the Node.js copilot-bridge.mjs subprocess fails."""


class CopilotClient:
    """LLM client that delegates to the Node.js copilot-bridge.mjs subprocess."""

    def __init__(self, model: str = COPILOT_DEFAULT_MODEL) -> None:
        self._model = model
        if not _BRIDGE_SCRIPT.exists():
            raise BridgeError(
                f"Bridge script not found: {_BRIDGE_SCRIPT}\n"
                "Run 'npm install' in the project root first."
            )

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> str:
        """Send a system + user prompt to Copilot and return the response text."""
        payload = json.dumps({"system": system, "user": user})
        timeout_seconds = 300

        try:
            result = subprocess.run(
                ["node", str(_BRIDGE_SCRIPT), self._model],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ},
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                "Node.js not found. Install Node.js 18+ to use the Copilot backend."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BridgeError(
                f"Copilot bridge timed out after {timeout_seconds}s."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise BridgeError(
                f"Copilot bridge exited with code {result.returncode}.\n{stderr}"
            )

        raw = result.stdout.strip()
        if not raw:
            raise BridgeError("Copilot bridge returned empty output.")

        try:
            data = json.loads(raw)
            return data.get("content", raw)
        except json.JSONDecodeError:
            return raw

    def close(self) -> None:
        """No persistent resources to release."""

    def __enter__(self) -> CopilotClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
