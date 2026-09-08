"""Offline production-readiness checks for corpus and generation boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re


DEFAULT_MAX_CORPUS_AGE_DAYS = 30.0
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMEOUT_NAMES = frozenset(
    {"APITimeoutError", "ConnectTimeout", "ReadTimeout", "TimeoutError"}
)


@dataclass(frozen=True)
class ReadinessCheck:
    """One public, metadata-only readiness result."""

    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ReadinessContext:
    """Inputs for an offline readiness report."""

    manifest_path: Path
    source_count: int
    provider_error: Exception | None = None
    max_corpus_age_days: float = DEFAULT_MAX_CORPUS_AGE_DAYS
    now: datetime | None = None


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate readiness state without source text or credentials."""

    checks: tuple[ReadinessCheck, ...]

    @property
    def status(self) -> str:
        statuses = {check.status for check in self.checks}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"

    @property
    def is_ready(self) -> bool:
        return self.status != "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status": self.status,
            "ready": self.is_ready,
            "checks": [check.to_dict() for check in self.checks],
        }


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, TimeoutError) or type(error).__name__ in _TIMEOUT_NAMES


def public_error_message(error: Exception) -> str:
    """Return a user-safe error without provider text, credentials or source data."""
    if _is_timeout(error):
        return "Le fournisseur n’a pas répondu à temps."
    if isinstance(error, EnvironmentError):
        return "La configuration ou la clé du fournisseur est indisponible."
    return "Erreur de génération. Consultez les journaux d’audit."


def check_source_count(source_count: int) -> ReadinessCheck:
    """Require at least one retrieved source before generation."""
    if source_count > 0:
        return ReadinessCheck("retrieved_sources", "pass", f"{source_count} source(s) available")
    return ReadinessCheck(
        "retrieved_sources",
        "fail",
        "No source was retrieved; generation must abstain.",
    )


def _parse_generated_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def check_corpus_manifest(
    manifest_path: Path,
    max_age_days: float = DEFAULT_MAX_CORPUS_AGE_DAYS,
    now: datetime | None = None,
) -> ReadinessCheck:
    """Validate corpus provenance, integrity fields and freshness."""
    if max_age_days <= 0:
        return ReadinessCheck("corpus_manifest", "fail", "max_age_days must be positive")
    if not manifest_path.exists():
        return ReadinessCheck("corpus_manifest", "fail", "Corpus manifest is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ReadinessCheck("corpus_manifest", "fail", "Corpus manifest is unreadable")
    if not isinstance(payload, dict):
        return ReadinessCheck("corpus_manifest", "fail", "Corpus manifest is not an object")

    for field in ("input_sha256", "output_sha256"):
        if not _DIGEST_PATTERN.fullmatch(str(payload.get(field, ""))):
            return ReadinessCheck(
                "corpus_manifest",
                "fail",
                f"Corpus manifest has an invalid {field}",
            )
    if not isinstance(payload.get("source_filename"), str) or not payload["source_filename"].strip():
        return ReadinessCheck("corpus_manifest", "fail", "Corpus source provenance is missing")
    if not isinstance(payload.get("objects_written"), int) or payload["objects_written"] <= 0:
        return ReadinessCheck("corpus_manifest", "fail", "Corpus manifest contains no records")

    generated_at = _parse_generated_at(payload.get("generated_at_utc"))
    if generated_at is None:
        return ReadinessCheck("corpus_manifest", "fail", "Corpus refresh timestamp is missing or invalid")
    reference_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = (reference_time - generated_at).total_seconds() / 86_400
    if age_days < -0.01:
        return ReadinessCheck("corpus_manifest", "fail", "Corpus refresh timestamp is in the future")
    if age_days > max_age_days:
        return ReadinessCheck(
            "corpus_manifest",
            "fail",
            f"Corpus is stale ({age_days:.1f} days; maximum {max_age_days:g})",
        )
    return ReadinessCheck(
        "corpus_manifest",
        "pass",
        f"Corpus is current ({age_days:.1f} days old; source provenance and digests present)",
    )


def check_provider_error(error: Exception | None) -> ReadinessCheck:
    """Classify a provider failure without serializing its message."""
    if error is None:
        return ReadinessCheck("provider_boundary", "pass", "No provider failure recorded")
    if _is_timeout(error):
        return ReadinessCheck(
            "provider_boundary",
            "fail",
            "Provider timeout is handled as a safe generation failure",
        )
    return ReadinessCheck(
        "provider_boundary",
        "fail",
        f"Provider failure is handled as a safe generation failure ({type(error).__name__})",
    )


def build_readiness_report(context: ReadinessContext) -> ReadinessReport:
    """Build the readiness report used by CI and operational smoke checks."""
    checks = (
        check_corpus_manifest(
            context.manifest_path,
            context.max_corpus_age_days,
            context.now,
        ),
        check_source_count(context.source_count),
        check_provider_error(context.provider_error),
    )
    return ReadinessReport(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=1)
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_CORPUS_AGE_DAYS)
    args = parser.parse_args()
    report = build_readiness_report(
        ReadinessContext(
            manifest_path=args.manifest,
            source_count=args.source_count,
            max_corpus_age_days=args.max_age_days,
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.is_ready else 1)


if __name__ == "__main__":
    main()
