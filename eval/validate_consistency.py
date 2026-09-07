"""Check that the evaluation contract, baseline artifact and docs agree."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from eval.contract import (
    ContractValidationError,
    EvaluationContract,
    load_contract,
    validate_questions_file,
    validate_results_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "eval" / "evaluation_contract.json"
REQUIRED_AUTOEVAL_METRICS = (
    "bertscore_f1",
    "judge_score",
    "citation_faithfulness",
    "context_coverage",
    "latency_s",
    "cost_usd",
)


class ConsistencyValidationError(ValueError):
    """Raised when a versioned evaluation surface has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConsistencyValidationError(f"Required file is missing: {path}") from exc


def _require_text(text: str, expected: str, source: str) -> None:
    if expected not in text:
        raise ConsistencyValidationError(
            f"{source} is missing the canonical value: {expected}"
        )


def _baseline_metrics(artifact_path: Path) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "keyword_recall": statistics.fmean(row["keyword_recall"] for row in rows),
        "word_f1": statistics.fmean(row["word_f1"] for row in rows),
        "bertscore_f1": statistics.fmean(row["bertscore_f1"] for row in rows),
        "judge_score": statistics.fmean(row["judge_score"] for row in rows),
        "citation_faithfulness": statistics.fmean(
            row["citation_faithfulness"] for row in rows
        ),
        "context_coverage": statistics.fmean(row["context_coverage"] for row in rows),
        "latency_s": statistics.fmean(row["latency_s"] for row in rows),
    }


def _validate_baseline_artifact(
    contract: EvaluationContract,
    root: Path,
) -> tuple[Path, dict[str, float]]:
    artifact_path = (root / contract.baseline.artifact).resolve()
    if not artifact_path.exists():
        raise ConsistencyValidationError(
            f"Canonical baseline artifact is missing: {contract.baseline.artifact}"
        )
    actual_hash = _sha256(artifact_path)
    if actual_hash != contract.baseline.artifact_sha256:
        raise ConsistencyValidationError(
            "Canonical baseline artifact SHA-256 differs from the contract: "
            f"expected {contract.baseline.artifact_sha256}, got {actual_hash}."
        )
    try:
        validate_results_file(artifact_path, contract)
    except ContractValidationError as exc:
        raise ConsistencyValidationError(
            f"Canonical baseline artifact fails the result contract: {exc}"
        ) from exc
    return artifact_path, _baseline_metrics(artifact_path)


def _validate_metric_roles(
    contract: EvaluationContract,
    readme: str,
    contract_doc: str,
    autoeval_program: str,
) -> None:
    thresholds = {
        "citation_faithfulness": "0.99",
        "context_coverage": "0.60",
        "bertscore_f1": "0.75",
        "judge_score": "0.60",
    }
    for metric, threshold in thresholds.items():
        _require_text(contract_doc, threshold, "eval/EVALUATION_CONTRACT.md")
        _require_text(readme, threshold, "README.md")
        _require_text(autoeval_program, threshold, "autoeval/program.md")

    diagnostic_marker = "Word F1 is diagnostic-only"
    _require_text(contract_doc, "Word F1 | Diagnostic only", "eval/EVALUATION_CONTRACT.md")
    _require_text(readme, diagnostic_marker, "README.md")
    _require_text(autoeval_program, diagnostic_marker, "autoeval/program.md")


def _validate_autoeval_source(experiment: str) -> None:
    _require_text(experiment, "validate_questions_file", "autoeval/experiment.py")
    for metric in REQUIRED_AUTOEVAL_METRICS:
        _require_text(experiment, metric, "autoeval/experiment.py")


def validate_consistency(root: Path = ROOT) -> None:
    """Raise an actionable error if docs, AutoEval or artifacts drift."""
    contract_path = root / "eval" / "evaluation_contract.json"
    try:
        contract = load_contract(contract_path)
        validate_questions_file(
            root / contract.regression_set.path,
            contract,
        )
        artifact_path, metrics = _validate_baseline_artifact(contract, root)
        readme = _read_text(root / "README.md")
        contract_doc = _read_text(root / "eval" / "EVALUATION_CONTRACT.md")
        autoeval_program = _read_text(root / "autoeval" / "program.md")
        experiment = _read_text(root / "autoeval" / "experiment.py")

        for document_path, document in (
            ("README.md", readme),
            ("eval/EVALUATION_CONTRACT.md", contract_doc),
            ("autoeval/program.md", autoeval_program),
        ):
            _require_text(document, contract.baseline.retriever, document_path)
            _require_text(document, f"k={contract.baseline.k}", document_path)
            _require_text(document, contract.baseline.model, document_path)
        _require_text(readme, contract.baseline.command, "README.md")
        _require_text(contract_doc, contract.baseline.command, "eval/EVALUATION_CONTRACT.md")
        _require_text(readme, contract.baseline.artifact, "README.md")
        _require_text(contract_doc, contract.baseline.artifact, "eval/EVALUATION_CONTRACT.md")

        _validate_metric_roles(contract, readme, contract_doc, autoeval_program)
        _validate_autoeval_source(experiment)

        expected_numbers = {
            "keyword_recall": f"{metrics['keyword_recall']:.4f}",
            "word_f1": f"{metrics['word_f1']:.4f}",
            "bertscore_f1": f"{metrics['bertscore_f1']:.4f}",
            "judge_score": f"{metrics['judge_score']:.4f}",
            "citation_faithfulness": f"{metrics['citation_faithfulness']:.4f}",
            "context_coverage": f"{metrics['context_coverage']:.4f}",
            "latency_s": f"{metrics['latency_s']:.2f}",
        }
        for metric, value in expected_numbers.items():
            _require_text(readme, value, "README.md")
        print(
            "Consistency validation: OK "
            f"(contract {contract.version}, {contract.regression_set.expected_count} questions, "
            f"artifact {artifact_path.name})"
        )
    except (ContractValidationError, ConsistencyValidationError) as exc:
        raise SystemExit(f"ERROR: evaluation consistency: {exc}") from exc


if __name__ == "__main__":
    validate_consistency()
