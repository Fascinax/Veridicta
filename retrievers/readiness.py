"""Offline production-readiness checks for corpus and generation boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from retrievers.snapshots import verify_manifest
from retrievers.release_evidence import verify_index_bundle, verify_release_review, verify_provider_smoke


DEFAULT_MAX_CORPUS_AGE_DAYS = 30.0
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
    release_review_path: Path | None = None
    provider_smoke_path: Path | None = None
    expected_backend: str | None = None
    expected_model: str | None = None
    expected_base_url: str | None = None


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




def check_corpus_manifest(
    manifest_path: Path,
    max_age_days: float = DEFAULT_MAX_CORPUS_AGE_DAYS,
    now: datetime | None = None,
) -> ReadinessCheck:
    """Check actual file digests and collection freshness before release."""
    try:
        verify_manifest(manifest_path, max_age_days, now)
    except (OSError, ValueError, KeyError, TypeError):
        return ReadinessCheck(
            "corpus_manifest", "fail",
            "Corpus manifest is missing, stale or invalid; a verified v2 snapshot is required",
        )
    return ReadinessCheck(
        "corpus_manifest", "pass",
        "Corpus collection is current and all declared file digests match",
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
    corpus = check_corpus_manifest(context.manifest_path, context.max_corpus_age_days, context.now)
    index_bundle = (_check_evidence("index_bundle", lambda: verify_index_bundle(context.manifest_path))
                    if corpus.status == "pass" else ReadinessCheck(
                        "index_bundle", "fail", "Index is not loaded because corpus integrity could not be verified"))
    checks = (
        corpus,
        check_source_count(context.source_count),
        check_provider_error(context.provider_error),
        index_bundle,
        _check_evidence("release_review", lambda: verify_release_review(
            context.manifest_path, context.release_review_path, context.now or datetime.now(timezone.utc))),
        _check_evidence("provider_smoke", lambda: verify_provider_smoke(
            context.provider_smoke_path, (context.expected_backend, context.expected_model, context.expected_base_url),
            context.now or datetime.now(timezone.utc))),
    )
    return ReadinessReport(checks)


def _check_evidence(name: str, verify) -> ReadinessCheck:
    try:
        verify()
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return ReadinessCheck(name, "fail", "Required evidence is missing, invalid or does not match the release")
    return ReadinessCheck(name, "pass", "Evidence verified for the selected release")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--release-review", type=Path)
    parser.add_argument("--provider-smoke", type=Path)
    parser.add_argument("--backend", choices=("copilot", "cerebras", "omniroute"))
    parser.add_argument("--model")
    parser.add_argument("--base-url", help="Expected OpenAI-compatible endpoint for the smoke")
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_CORPUS_AGE_DAYS)
    args = parser.parse_args()
    report = build_readiness_report(
        ReadinessContext(
            manifest_path=args.manifest,
            source_count=args.source_count,
            max_corpus_age_days=args.max_age_days,
            release_review_path=args.release_review,
            provider_smoke_path=args.provider_smoke,
            expected_backend=args.backend,
            expected_model=args.model,
            expected_base_url=args.base_url,
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.is_ready else 1)


if __name__ == "__main__":
    main()
