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
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

_BRIDGE_SCRIPT = Path(__file__).resolve().parent.parent / "copilot-bridge.mjs"

COPILOT_DEFAULT_MODEL = "gpt-4.1"

_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€™",
    "â€œ",
    "â€",
)


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

    @staticmethod
    def _mojibake_score(text: str) -> int:
        return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)

    @classmethod
    def _repair_mojibake(cls, text: str) -> str:
        baseline_score = cls._mojibake_score(text)
        if baseline_score == 0:
            return text

        repaired_candidates: list[str] = []
        for source_encoding in ("latin-1", "cp1252"):
            try:
                repaired = text.encode(source_encoding).decode("utf-8")
            except UnicodeError:
                continue
            repaired_candidates.append(repaired)

        if not repaired_candidates:
            return text

        best_candidate = min(repaired_candidates, key=cls._mojibake_score)
        if cls._mojibake_score(best_candidate) < baseline_score:
            logger.warning("Detected mojibake in Copilot response; repaired output text.")
            return best_candidate
        return text

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
                encoding="utf-8",
                errors="replace",
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
            return self._repair_mojibake(data.get("content", raw))
        except json.JSONDecodeError:
            return self._repair_mojibake(raw)

    def chat_stream(self, *, system: str, user: str) -> Iterator[str]:
        """Stream token deltas from Copilot via the bridge in --stream mode.

        Yields incremental text chunks as they arrive from the model.
        """
        payload = json.dumps({"system": system, "user": user})
        try:
            proc = subprocess.Popen(
                ["node", str(_BRIDGE_SCRIPT), self._model, "--stream"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ},
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                "Node.js not found. Install Node.js 18+ to use the Copilot backend."
            ) from exc

        proc.stdin.write(payload)
        proc.stdin.close()

        for raw_line in proc.stdout:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            delta = data.get("partial", "")
            if delta:
                yield self._repair_mojibake(delta)

        proc.stdout.close()
        stderr_output = proc.stderr.read()
        proc.stderr.close()
        proc.wait()
        if proc.returncode != 0:
            raise BridgeError(
                f"Copilot bridge exited with code {proc.returncode}.\n"
                f"{(stderr_output or '').strip()}"
            )

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
