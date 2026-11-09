"""Isolated local case workspaces and memo export helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import textwrap
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from functools import wraps

from filelock import FileLock

RESTORE_MARKER = ".restore-pending.json"


def _require_available(root: Path) -> None:
    if (root / RESTORE_MARKER).exists():
        raise RuntimeError("Workspace scope has an interrupted restore; recover into a new scope")


def _locked(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            _require_available(self.root)
            return method(self, *args, **kwargs)
    return synchronized


def _contained(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("path escapes workspace storage")
    return target


def _valid_workspace_id(workspace_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", workspace_id):
        raise ValueError("invalid workspace id")
    return workspace_id


@dataclass(frozen=True)
class PinnedSource:
    source_id: str
    title: str
    passage: str
    stable_url: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class PrivateNote:
    note_id: str
    text: str
    created_at: str


@dataclass
class CaseWorkspace:
    workspace_id: str
    name: str
    created_at: str
    updated_at: str
    archived: bool = False
    tags: list[str] = field(default_factory=list)
    conversations: list[dict[str, Any]] = field(default_factory=list)
    pinned_sources: list[PinnedSource] = field(default_factory=list)
    private_notes: list[PrivateNote] = field(default_factory=list)


class WorkspaceStore:
    """Persist each workspace below its own directory with atomic writes."""

    def __init__(self, root: str | Path, owner_id: str | None = None) -> None:
        self.root = Path(root).resolve()
        if owner_id is not None:
            if not owner_id.strip():
                raise ValueError("owner must not be empty")
            scope = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
            self.root = _contained(self.root, "owners", scope)
        self.root.mkdir(parents=True, exist_ok=True)
        self.scope_key = str(self.root)
        self._lock = FileLock(str(self.root / ".workspace.lock"), timeout=20)
        self.workspace_root = _contained(self.root, "workspaces")
        self.audit_root = _contained(self.root, "audit")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.audit_root.mkdir(parents=True, exist_ok=True)

    @_locked
    def create(self, name: str, tags: list[str] | None = None) -> CaseWorkspace:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("workspace name must not be empty")
        now = _now()
        workspace = CaseWorkspace(
            workspace_id=uuid.uuid4().hex,
            name=clean_name,
            created_at=now,
            updated_at=now,
            tags=_clean_tags(tags or []),
        )
        self._write(workspace)
        return workspace

    @_locked
    def list(self, include_archived: bool = False) -> list[CaseWorkspace]:
        workspaces = [self.get(path.parent.name) for path in self.workspace_root.glob("*/workspace.json")]
        visible = workspaces if include_archived else [item for item in workspaces if not item.archived]
        return sorted(visible, key=lambda item: (item.archived, item.name.casefold(), item.workspace_id))

    @_locked
    def get(self, workspace_id: str) -> CaseWorkspace:
        workspace = self._read(self._workspace_path(workspace_id))
        if workspace.workspace_id != workspace_id:
            raise ValueError("workspace ID mismatch")
        return workspace

    @_locked
    def rename(self, workspace_id: str, name: str) -> CaseWorkspace:
        workspace = self.get(workspace_id)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("workspace name must not be empty")
        workspace.name = clean_name
        workspace.updated_at = _now()
        self._write(workspace)
        return workspace

    @_locked
    def archive(self, workspace_id: str, archived: bool = True) -> CaseWorkspace:
        workspace = self.get(workspace_id)
        workspace.archived = archived
        workspace.updated_at = _now()
        self._write(workspace)
        return workspace

    @_locked
    def delete(self, workspace_id: str) -> None:
        workspace_path = self._workspace_path(workspace_id)
        audit_path = self._audit_path(workspace_id)
        if not workspace_path.exists():
            raise FileNotFoundError(f"Unknown workspace: {workspace_id}")
        shutil.rmtree(workspace_path.parent)
        if audit_path.exists():
            shutil.rmtree(audit_path)

    @_locked
    def append_message(
        self,
        workspace_id: str,
        role: str,
        content: str,
        *,
        trace: dict[str, object] | None = None,
        sources: list[dict[str, object]] | None = None,
    ) -> CaseWorkspace:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        if not content.strip():
            raise ValueError("message content must not be empty")
        workspace = self.get(workspace_id)
        workspace.conversations.append(
            {
                "role": role,
                "content": content,
                "trace": trace or {},
                "sources": sources or [],
                "created_at": _now(),
            }
        )
        workspace.updated_at = _now()
        self._write(workspace)
        return workspace

    @_locked
    def pin_source(
        self,
        workspace_id: str,
        source_id: str,
        title: str,
        passage: str,
        *,
        stable_url: str | None = None,
        trace_id: str | None = None,
    ) -> CaseWorkspace:
        if not source_id.strip() or not title.strip() or not passage.strip():
            raise ValueError("a pinned source needs an id, title and passage")
        workspace = self.get(workspace_id)
        workspace.pinned_sources = [
            source for source in workspace.pinned_sources if source.source_id != source_id
        ]
        workspace.pinned_sources.append(
            PinnedSource(source_id, title.strip(), passage.strip(), stable_url, trace_id)
        )
        workspace.updated_at = _now()
        self._write(workspace)
        return workspace

    @_locked
    def add_private_note(self, workspace_id: str, text: str) -> CaseWorkspace:
        if not text.strip():
            raise ValueError("private note must not be empty")
        workspace = self.get(workspace_id)
        workspace.private_notes.append(PrivateNote(uuid.uuid4().hex, text.strip(), _now()))
        workspace.updated_at = _now()
        self._write(workspace)
        return workspace

    @_locked
    def export_markdown(self, workspace_id: str, generated_at: str | None = None) -> str:
        workspace = self.get(workspace_id)
        timestamp = generated_at or _now()
        lines = [
            f"# Mémo juridique — {workspace.name}",
            "",
            f"- Dossier : `{workspace.workspace_id}`",
            f"- Généré le : {timestamp}",
            f"- Tags : {', '.join(workspace.tags) or 'aucun'}",
            "",
            "## Analyse du dossier",
            "",
        ]
        lines.extend(_conversation_markdown(workspace.conversations))
        lines.extend(["", "## Sources épinglées", ""])
        if workspace.pinned_sources:
            for source in workspace.pinned_sources:
                lines.append(f"### {source.title} — `{source.source_id}`")
                if source.stable_url:
                    lines.append(f"Lien stable : {source.stable_url}")
                if source.trace_id:
                    lines.append(f"Trace : `{source.trace_id}`")
                lines.extend(["", "> " + source.passage.replace("\n", "\n> "), ""])
        else:
            lines.append("Aucune source épinglée.")
        lines.extend(["", "## Notes privées", ""])
        if workspace.private_notes:
            lines.extend(f"- {note.text}" for note in workspace.private_notes)
        else:
            lines.append("Aucune note privée.")
        return "\n".join(lines).rstrip() + "\n"

    def export_pdf(self, workspace_id: str, generated_at: str | None = None) -> bytes:
        """Export a dependency-free, readable PDF using the memo text."""

        return _build_pdf(self.export_markdown(workspace_id, generated_at))

    def _workspace_path(self, workspace_id: str) -> Path:
        return _contained(self.root, "workspaces", _valid_workspace_id(workspace_id), "workspace.json")

    def _audit_path(self, workspace_id: str) -> Path:
        return _contained(self.root, "audit", _valid_workspace_id(workspace_id))

    @_locked
    def append_audit(self, workspace_id: str | None, **event) -> Path | None:
        from retrievers.traceability import append_audit_event

        if workspace_id is not None:
            self.get(workspace_id)
            directory = self._audit_path(workspace_id)
        else:
            directory = _contained(self.root, "audit", "unassigned")
        return append_audit_event(**event, audit_dir=directory)

    def _write(self, workspace: CaseWorkspace) -> None:
        directory = self._workspace_path(workspace.workspace_id).parent
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "workspace.json"
        temporary = directory / f".workspace-{uuid.uuid4().hex}.tmp"
        payload = _workspace_to_dict(workspace)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _read(path: Path) -> CaseWorkspace:
        if not path.exists():
            raise FileNotFoundError(f"Unknown workspace: {path.parent.name}")
        return _workspace_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _conversation_markdown(conversations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in conversations:
        role = "Question" if message.get("role") == "user" else "Conclusion / raisonnement"
        lines.extend([f"### {role}", "", str(message.get("content", "")), ""])
        trace = message.get("trace") or {}
        if message.get("role") == "assistant":
            lines.extend(
                [
                    "**Citations et incertitude**",
                    "",
                    f"- Trace : `{trace.get('trace_id', 'n/a')}`",
                    f"- Sources citées : {', '.join(map(str, trace.get('cited_source_numbers', []))) or 'aucune'}",
                    f"- Incertitude : {trace.get('failure_reason') or _orchestration_failure(trace) or 'non signalée'}",
                    "",
                    "**Hypothèses**",
                    "",
                    "- Les sources injectées sont celles associées à cette réponse.",
                    "",
                ]
            )
    return lines


def _orchestration_failure(trace: dict[str, object]) -> str | None:
    orchestration = trace.get("orchestration")
    return orchestration.get("failure_reason") if isinstance(orchestration, dict) else None


def _workspace_to_dict(workspace: CaseWorkspace) -> dict[str, object]:
    payload = asdict(workspace)
    payload["pinned_sources"] = [asdict(source) for source in workspace.pinned_sources]
    payload["private_notes"] = [asdict(note) for note in workspace.private_notes]
    return payload


def _workspace_from_dict(raw: dict[str, object]) -> CaseWorkspace:
    return CaseWorkspace(
        workspace_id=str(raw["workspace_id"]),
        name=str(raw["name"]),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        archived=bool(raw.get("archived", False)),
        tags=[str(tag) for tag in raw.get("tags", [])],
        conversations=list(raw.get("conversations", [])),
        pinned_sources=[PinnedSource(**source) for source in raw.get("pinned_sources", [])],
        private_notes=[PrivateNote(**note) for note in raw.get("private_notes", [])],
    )


def _clean_tags(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_pdf(text: str) -> bytes:
    lines: list[str] = []
    for raw_line in text.splitlines():
        lines.extend(textwrap.wrap(raw_line, width=96) or [""])
    page_lines = [lines[index : index + 48] for index in range(0, len(lines), 48)] or [[""]]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    page_ids: list[int] = []
    for page in page_lines:
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_ids.append(page_id)
        commands = ["BT /F1 9 Tf 48 752 Td"]
        for line in page:
            encoded = line.encode("cp1252", errors="replace")
            escaped = encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            commands.append(f"({escaped.decode('latin-1')}) Tj 0 -14 Td")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii"),
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
            ]
        )
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode("ascii")
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage a local Veridicta case workspace.")
    parser.add_argument("--root", default=os.getenv("VERIDICTA_WORKSPACE_DIR", "data/cases"))
    parser.add_argument("--demo", action="store_true", help="Create and export a small isolated demo case.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    store = WorkspaceStore(args.root)
    if not args.demo:
        print(f"Workspace store ready: {store.root}")
        return
    workspace = store.create("Demo dossier", ["demo"])
    store.append_message(workspace.workspace_id, "user", "Question de démonstration")
    store.append_message(
        workspace.workspace_id,
        "assistant",
        "Conclusion de démonstration.",
        trace={"trace_id": "demo-trace", "cited_source_numbers": [1]},
    )
    store.pin_source(workspace.workspace_id, "source-1", "Source demo", "Passage exact de démonstration.")
    print(store.export_markdown(workspace.workspace_id))


if __name__ == "__main__":
    main()
