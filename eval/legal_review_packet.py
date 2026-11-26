"""Prepare private human-authored references; never generate legal answers or approvals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from filelock import FileLock

from eval.review_sets import question_key
from retrievers.snapshots import contained_file, file_digest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
EXPOSURE_PATHS = (ROOT / "eval/test_questions.json", ROOT / "eval/datasets/legal_review_intake.json",
                  ROOT / "eval/results/stage0/annotation_packet.jsonl",
                  ROOT / "eval/results/stage0/annotation_packet_ambiguous30.jsonl")
THEMES = ("contrats", "rupture", "remuneration", "conges", "temps-travail",
          "relations-collectives", "sante-securite", "formation", "versions-et-dates", "limites-des-sources")
CASE_FIELDS = ("id", "theme", "question", "as_of", "question_author", "expected_behavior",
               "reference_answer", "abstention_reason", "applicability_rationale",
               "review_status", "reviewed_by", "reviewed_at")
IMMUTABLE_FILES = ("plan.json", "source-chunks.jsonl", "exposure-keys.json")
SEALED_FILES = (*IMMUTABLE_FILES, "packet.manifest.json", "cases.jsonl", "independence.json")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_cases(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _separate_output(destination: Path, protected: Path) -> None:
    destination, protected = destination.resolve(), protected.resolve()
    if destination.is_relative_to(protected) or protected.is_relative_to(destination):
        raise ValueError("Output must be separate from the protected input")
    if any((parent / name).exists() for parent in destination.parents
           for name in ("snapshot.manifest.json", "packet.manifest.json", "reference.manifest.json")):
        raise ValueError("Output cannot be inside a corpus or review input")


def _question_digest(text: str) -> str:
    return hashlib.sha256(question_key(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PacketRequest:
    snapshot: Path
    destination: Path
    slots: int = 50


def prepare_packet(request: PacketRequest, exposure_paths=EXPOSURE_PATHS) -> dict:
    if not isinstance(request.slots, int) or isinstance(request.slots, bool) or not 50 <= request.slots <= 100:
        raise ValueError("A reference packet needs 50 to 100 slots")
    _separate_output(request.destination, request.snapshot.parent)
    snapshot = verify_manifest(request.snapshot)
    source_name = "index/chunks_map.jsonl"
    if source_name not in snapshot["files"]:
        raise ValueError("An indexed source snapshot is required")
    snapshot_digest = file_digest(request.snapshot)
    exposure_keys, exposed_files = set(), {}
    for path in exposure_paths:
        rows = _read_jsonl(path) if path.suffix == ".jsonl" else _read_json(path)
        exposure_keys.update(_question_digest(row["question"]) for row in rows
                             if isinstance(row.get("question"), str) and question_key(row["question"]))
        exposed_files[path.name] = file_digest(path)
    root = request.destination
    root.mkdir(parents=True, exist_ok=False)
    source = contained_file(request.snapshot.parent, source_name)
    shutil.copyfile(source, root / "source-chunks.jsonl")
    if file_digest(root / "source-chunks.jsonl") != snapshot["files"][source_name]["sha256"]:
        raise ValueError("Source chunk bytes changed during copy")
    slots = [f"independent-{index:03d}" for index in range(1, request.slots + 1)]
    _write_json(root / "plan.json", {"schema_version": "1.0", "slot_ids": slots, "planned_themes": list(THEMES)})
    _write_json(root / "exposure-keys.json", {"normalization": "NFKD-casefold-words-no-accents-v1",
                                            "question_keys": sorted(exposure_keys), "source_files": exposed_files})
    rows = []
    for index, slot in enumerate(slots):
        row = dict.fromkeys(CASE_FIELDS, "")
        row.update(id=slot, theme=THEMES[index % len(THEMES)], review_status="pending", supporting_passages=[])
        rows.append(row)
    _write_cases(root / "cases.jsonl", rows)
    _write_json(root / "independence.json", {"status": "pending", "attested_by": "", "attested_at": "",
                                            "human_authored": False, "not_used_for_development": False,
                                            "not_derived_from_exposed_questions": False})
    manifest = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_manifest_sha256": snapshot_digest, "snapshot_collected_at": snapshot["collected_at"],
                "slots": request.slots, "files": {name: file_digest(root / name) for name in IMMUTABLE_FILES}}
    if file_digest(request.snapshot) != snapshot_digest:
        raise ValueError("Source snapshot changed during packet preparation")
    _write_json(root / "packet.manifest.json", manifest)
    return {"status": "prepared", "slots": request.slots, "authored_questions": 0, "reviewed": 0,
            "reference_ready": False, "legal_release_ready": False, "packet_manifest_sha256": file_digest(root / "packet.manifest.json")}


def _packet(root: Path) -> tuple[dict, list[dict]]:
    sealed = None
    if (root / "reference.manifest.json").exists():
        sealed = _read_json(contained_file(root, "reference.manifest.json"))
        if (not isinstance(sealed, dict) or sealed.get("schema_version") != "1.0" or sealed.get("status") != "sealed_references"
                or sealed.get("reference_ready") is not True or sealed.get("legal_release_ready") is not False
                or set(sealed.get("files", {})) != set(SEALED_FILES)):
            raise ValueError("Invalid sealed reference manifest")
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if actual - {".cases.lock"} != set(SEALED_FILES) | {"reference.manifest.json"}:
            raise ValueError("Sealed references contain undeclared files")
        for name, digest in sealed["files"].items():
            if file_digest(contained_file(root, name)) != digest:
                raise ValueError("Sealed reference bytes changed")
    manifest = _read_json(contained_file(root, "packet.manifest.json"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0"
            or not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(IMMUTABLE_FILES)
            or not isinstance(manifest.get("slots"), int) or isinstance(manifest["slots"], bool)
            or not 50 <= manifest["slots"] <= 100):
        raise ValueError("Invalid packet manifest")
    if (sealed is not None
            and sealed.get("snapshot_manifest_sha256") != manifest.get("snapshot_manifest_sha256")):
        raise ValueError("Sealed reference snapshot identity changed")
    for name, digest in manifest["files"].items():
        if file_digest(contained_file(root, name)) != digest:
            raise ValueError("Immutable review input changed")
    rows = _read_jsonl(contained_file(root, "cases.jsonl"))
    plan = _read_json(contained_file(root, "plan.json"))
    if [row.get("id") for row in rows] != plan["slot_ids"] or len(rows) != manifest["slots"]:
        raise ValueError("Case identities or their order changed")
    return manifest, rows


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: object, now: datetime) -> bool:
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None
        return instant is not None and instant.tzinfo is not None and instant <= now
    except ValueError:
        return False


def _review_issues(row: dict, sources: dict, now: datetime) -> list[str]:
    issues = [field for field in ("question", "theme", "question_author", "reference_answer", "reviewed_by", "applicability_rationale")
              if not _has_text(row.get(field))]
    try:
        if date.fromisoformat(row.get("as_of", "")).isoformat() != row["as_of"]:
            issues.append("as_of")
    except (ValueError, TypeError):
        issues.append("as_of")
    if not _timestamp(row.get("reviewed_at"), now):
        issues.append("reviewed_at")
    passages = row.get("supporting_passages")
    if not isinstance(passages, list):
        return [*issues, "supporting_passages"]
    if row.get("expected_behavior") == "answer":
        if not passages:
            issues.append("supporting_passages")
    elif row.get("expected_behavior") == "abstain":
        if not _has_text(row.get("abstention_reason")):
            issues.append("abstention_reason")
    else:
        issues.append("expected_behavior")
    seen = set()
    for passage in passages:
        if not isinstance(passage, dict):
            issues.append("passage_shape")
            continue
        chunk_id = passage.get("chunk_id")
        chunk = sources.get(chunk_id) if isinstance(chunk_id, str) else None
        quote = passage.get("text")
        if (chunk is None or passage.get("source_url") != chunk.get("source")
                or not _has_text(quote) or quote not in chunk["text"]):
            issues.append("passage_not_in_snapshot")
            continue
        identity = (passage["chunk_id"], quote)
        if identity in seen:
            issues.append("duplicate_passage")
        seen.add(identity)
    return sorted(set(issues))


def validate_packet(root: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Validation time needs a timezone")
    before = {name: file_digest(contained_file(root, name)) for name in ("cases.jsonl", "independence.json")}
    manifest, rows = _packet(root)
    source_rows = _read_jsonl(contained_file(root, "source-chunks.jsonl"))
    sources = {}
    for row in source_rows:
        if (not all(_has_text(row.get(key)) for key in ("chunk_id", "source", "text"))
                or row["chunk_id"] in sources):
            raise ValueError("Source chunk identities must be unique and nonempty")
        sources[row["chunk_id"]] = row
    if not sources:
        raise ValueError("Source chunk identities must be unique and nonempty")
    excluded = set(_read_json(contained_file(root, "exposure-keys.json"))["question_keys"])
    seen, issues = set(), []
    reviewed = authored = 0
    for row in rows:
        errors = []
        question = row.get("question")
        if _has_text(question):
            authored += 1
            key = _question_digest(question)
            if not question_key(question) or key in seen or key in excluded:
                errors.append("duplicate_or_exposed_question")
            seen.add(key)
        elif question != "":
            errors.append("question_type")
        if row.get("review_status") == "reviewed":
            errors.extend(_review_issues(row, sources, now))
            reviewed += not errors
        elif row.get("review_status") != "pending":
            errors.append("review_status")
        if errors:
            issues.append({"id": row["id"], "fields": sorted(set(errors))})
    attestation = _read_json(contained_file(root, "independence.json"))
    if not isinstance(attestation, dict):
        raise ValueError("Invalid independence attestation")
    attested = (attestation.get("status") == "attested" and _has_text(attestation.get("attested_by"))
                and _timestamp(attestation.get("attested_at"), now)
                and all(attestation.get(key) is True for key in ("human_authored", "not_used_for_development", "not_derived_from_exposed_questions")))
    complete = not issues and reviewed == len(rows) and attested
    if any(file_digest(contained_file(root, name)) != digest for name, digest in before.items()):
        raise ValueError("Annotations changed during validation")
    return {"schema_version": "1.0", "status": "references_complete" if complete else "pending",
            "checked_at": now.isoformat(), "slots": len(rows), "authored_questions": authored,
            "reviewed": reviewed, "pending": len(rows) - reviewed, "issues": issues,
            "independence_attested": bool(attested), "independence_verified": False,
            "reference_ready": bool(complete), "legal_release_ready": False,
            "snapshot_manifest_sha256": manifest["snapshot_manifest_sha256"],
            "packet_manifest_sha256": file_digest(root / "packet.manifest.json"),
            "cases_sha256": before["cases.jsonl"],
            "independence_sha256": before["independence.json"],
            "semantics": "Reference preparation only; attestations do not verify identity, legal correctness or semantic independence."}


def _sheet_value(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Review sheet fields must be strings")
    if value.startswith("'") or value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def _sheet_text(value: str) -> str:
    if value.startswith("'") and _sheet_value(value[1:]) == value:
        return value[1:]
    return value


def export_sheets(root: Path, destination: Path) -> dict:
    _separate_output(destination, root)
    with FileLock(str(root / ".cases.lock")):
        _, rows = _packet(root)
        exported = {"schema_version": "1.0", "packet_manifest_sha256": file_digest(root / "packet.manifest.json"),
                    "base_cases_sha256": file_digest(root / "cases.jsonl")}
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "cases.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows({field: _sheet_value(row.get(field, "")) for field in CASE_FIELDS} for row in rows)
    with (destination / "passages.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "chunk_id", "source_url", "text"))
        writer.writeheader()
        for row in rows:
            writer.writerows({key: _sheet_value(value) for key, value in {"case_id": row["id"], **passage}.items()}
                              for passage in row["supporting_passages"])
    _write_json(destination / "export.manifest.json", exported)
    return {"status": "exported", "slots": len(rows), **exported}


def import_sheets(root: Path, sheets: Path) -> dict:
    if (root / "reference.manifest.json").exists():
        raise ValueError("Sealed references cannot be edited")
    _packet(root)
    exported = _read_json(contained_file(sheets, "export.manifest.json"))
    with FileLock(str(root / ".cases.lock")):
        if (exported.get("schema_version") != "1.0"
                or exported.get("packet_manifest_sha256") != file_digest(root / "packet.manifest.json")
                or exported.get("base_cases_sha256") != file_digest(contained_file(root, "cases.jsonl"))):
            raise ValueError("Stale export or another review packet")
        with contained_file(sheets, "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CASE_FIELDS):
                raise ValueError("Unexpected case sheet columns")
            raw_rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in raw_rows):
                raise ValueError("Malformed case sheet row")
            rows = [dict({key: _sheet_text(value) for key, value in row.items()}, supporting_passages=[]) for row in raw_rows]
        ids = {row["id"]: row for row in rows}
        plan = _read_json(root / "plan.json")
        if [row["id"] for row in rows] != plan["slot_ids"] or any(None in row or None in row.values() for row in rows):
            raise ValueError("Case sheet identities, order or fields changed")
        with contained_file(sheets, "passages.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["case_id", "chunk_id", "source_url", "text"]:
                raise ValueError("Unexpected passage sheet columns")
            for passage in reader:
                if None in passage or any(value is None for value in passage.values()):
                    raise ValueError("Malformed passage sheet row")
                passage = {key: _sheet_text(value) for key, value in passage.items()}
                case_id = passage.pop("case_id")
                if case_id not in ids or any(value is None for value in passage.values()):
                    raise ValueError("Passage sheet refers to an unknown case or malformed row")
                ids[case_id]["supporting_passages"].append(passage)
        _write_cases(root / "cases.jsonl", rows)
    return validate_packet(root)


@dataclass(frozen=True)
class SourceSearch:
    query: str
    case_id: str
    limit: int = 20


def find_sources(root: Path, request: SourceSearch, destination: Path) -> dict:
    _separate_output(destination, root)
    _, rows = _packet(root)
    query = question_key(request.query)
    if (not query or request.case_id not in {row["id"] for row in rows}
            or not isinstance(request.limit, int) or isinstance(request.limit, bool) or not 1 <= request.limit <= 100):
        raise ValueError("Source search needs a query, a packet case ID and a limit from 1 to 100")
    destination.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with destination.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "chunk_id", "source_url", "text"))
        writer.writeheader()
        with contained_file(root, "source-chunks.jsonl").open(encoding="utf-8") as source:
            for line in source:
                chunk = json.loads(line)
                if query not in question_key(chunk["text"]):
                    continue
                row = {"case_id": request.case_id, "chunk_id": chunk["chunk_id"],
                       "source_url": chunk["source"], "text": chunk["text"]}
                writer.writerow({key: _sheet_value(value) for key, value in row.items()})
                matched += 1
                if matched == request.limit:
                    break
    return {"status": "source_matches_exported", "matches": matched, "limit": request.limit,
            "selection_order": "frozen_source_file_order_not_relevance", "citations_automatically_added": False}


def seal_packet(root: Path, destination: Path) -> dict:
    _separate_output(destination, root)
    with FileLock(str(root / ".cases.lock")):
        report = validate_packet(root)
        if not report["reference_ready"]:
            raise ValueError("Reference sealing blocked: human annotations or attestation incomplete")
        expected = {name: file_digest(contained_file(root, name)) for name in SEALED_FILES}
        destination.mkdir(parents=True, exist_ok=False)
        for name, digest in expected.items():
            shutil.copyfile(contained_file(root, name), destination / name)
            if file_digest(destination / name) != digest:
                raise ValueError("Review file changed during sealing")
        copied = validate_packet(destination)
        if (not copied["reference_ready"] or copied["cases_sha256"] != report["cases_sha256"]
                or copied["independence_sha256"] != report["independence_sha256"]):
            raise ValueError("Annotations changed during sealing")
        sealed = {"schema_version": "1.0", "status": "sealed_references", "sealed_at": datetime.now(timezone.utc).isoformat(),
                  "files": expected, "snapshot_manifest_sha256": report["snapshot_manifest_sha256"],
                  "reference_ready": True, "legal_release_ready": False}
        _write_json(destination / "reference.manifest.json", sealed)
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--slots", type=int, default=50)
    search = commands.add_parser("find-sources")
    search.add_argument("--packet", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--case-id", required=True)
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--out", type=Path, required=True)
    for name in ("validate", "export", "import", "seal"):
        command = commands.add_parser(name)
        command.add_argument("--packet", type=Path, required=True)
        if name == "import":
            command.add_argument("--sheets", type=Path, required=True)
        elif name != "validate":
            command.add_argument("--out", type=Path, required=True)
        else:
            command.add_argument("--require-complete", action="store_true")
    prepare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_packet(PacketRequest(args.snapshot, args.out, args.slots))
    elif args.command == "validate":
        result = validate_packet(args.packet)
    elif args.command == "export":
        result = export_sheets(args.packet, args.out)
    elif args.command == "import":
        result = import_sheets(args.packet, args.sheets)
    elif args.command == "find-sources":
        result = find_sources(args.packet, SourceSearch(args.query, args.case_id, args.limit), args.out)
    else:
        result = seal_packet(args.packet, args.out)
    print(json.dumps(result))
    if args.command == "validate" and args.require_complete and not result["reference_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
