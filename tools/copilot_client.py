"""Copilot SDK client — GitHub Copilot LLM backend for Veridicta.

Uses the official ``github-copilot-sdk`` Python package (``copilot``) which
communicates with the bundled Copilot CLI via JSON-RPC.

This lets Veridicta use GitHub Copilot models (gpt-4.1, claude-sonnet-4, o3-mini, etc.)
— requires a GitHub PAT with ``copilot`` scope or ``gh auth login``.

Activate via ``LLM_BACKEND=copilot`` in ``.env``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from collections.abc import Iterator
from types import TracebackType

from copilot import CopilotClient as _SdkClient, PermissionHandler

from retrievers.config import COPILOT_DEFAULT_MODEL

logger = logging.getLogger(__name__)

_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€™",
    "â€œ",
    "â€",
)

_SENTINEL = object()

_TOKEN_ENV_KEYS = (
    "GITHUB_PAT",
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)


class BridgeError(Exception):
    """Raised when the Copilot SDK call fails."""


def _resolve_token() -> str | None:
    for key in _TOKEN_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _build_sdk_client() -> _SdkClient:
    token = _resolve_token()
    opts: dict = {"log_level": "warning"}
    if token:
        opts["github_token"] = token
        opts["use_logged_in_user"] = False
    return _SdkClient(opts)


class CopilotClient:
    """LLM client backed by the official ``github-copilot-sdk`` Python package."""

    def __init__(self, model: str = COPILOT_DEFAULT_MODEL) -> None:
        self._model = model

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

        async def _run() -> str:
            client = _build_sdk_client()
            await client.start()
            try:
                session = await client.create_session({
                    "model": self._model,
                    "system_message": {"mode": "replace", "content": system},
                    "available_tools": [],
                    "infinite_sessions": {"enabled": False},
                    "on_permission_request": PermissionHandler.approve_all,
                })
                done = asyncio.Event()
                result_parts: list[str] = []

                def on_event(event):
                    if event.type.value == "assistant.message":
                        result_parts.append(event.data.content or "")
                    elif event.type.value == "session.idle":
                        done.set()

                session.on(on_event)
                await session.send({"prompt": user})
                await asyncio.wait_for(done.wait(), timeout=300)
                await session.disconnect()
                return "".join(result_parts)
            finally:
                await client.stop()

        try:
            raw = asyncio.run(_run())
        except asyncio.TimeoutError as exc:
            raise BridgeError("Copilot SDK timed out after 300s.") from exc
        except Exception as exc:
            raise BridgeError(f"Copilot SDK error: {exc}") from exc

        if not raw:
            raise BridgeError("Copilot SDK returned empty output.")
        return self._repair_mojibake(raw)

    def chat_stream(self, *, system: str, user: str) -> Iterator[str]:
        """Stream token deltas from Copilot via the Python SDK.

        Yields incremental text chunks as they arrive from the model.
        """
        token_queue: queue.Queue[object] = queue.Queue()

        async def _stream() -> None:
            client = _build_sdk_client()
            await client.start()
            try:
                session = await client.create_session({
                    "model": self._model,
                    "system_message": {"mode": "replace", "content": system},
                    "streaming": True,
                    "available_tools": [],
                    "infinite_sessions": {"enabled": False},
                    "on_permission_request": PermissionHandler.approve_all,
                })
                done = asyncio.Event()

                def on_event(event):
                    if event.type.value == "assistant.message_delta":
                        delta = event.data.delta_content or ""
                        if delta:
                            token_queue.put(delta)
                    elif event.type.value == "session.idle":
                        done.set()

                session.on(on_event)
                await session.send({"prompt": user})
                await asyncio.wait_for(done.wait(), timeout=300)
                await session.disconnect()
            finally:
                await client.stop()
                token_queue.put(_SENTINEL)

        error_holder: list[Exception] = []

        def _run_in_thread() -> None:
            try:
                asyncio.run(_stream())
            except Exception as exc:
                error_holder.append(exc)
                token_queue.put(_SENTINEL)

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            token = token_queue.get()
            if token is _SENTINEL:
                break
            yield self._repair_mojibake(token)

        thread.join(timeout=5)
        if error_holder:
            raise BridgeError(f"Copilot SDK stream error: {error_holder[0]}") from error_holder[0]

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
