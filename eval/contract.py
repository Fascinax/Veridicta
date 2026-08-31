"""Versioned evaluation contract and dataset validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT_PATH = Path("eval/evaluation_contract.json")
P95_QUANTILE = 0.95
HUMAN_LABELS = frozenset({"correct", "incomplete", "unsupported", "wrong"})


class ContractValidationError(ValueError):
    """Raised when a contract, dataset, annotation, or result is invalid."""


@dataclass(frozen=True)
class RegressionSetSpec:
    path: str
    expected_count: int
    sha256: str


@dataclass(frozen=True)
class HumanGoldSetSpec:
    packet_path: str
    labels_path: str
    expected_count: int
    min_count: int
    max_count: int
    required_labels: frozenset[str]


@dataclass(frozen=True)
class MetricGate:
    role: str
    minimum: float | None = None
    optional: bool = False


@dataclass(frozen=True)
class EvaluationContract:
    project_root: Path
    version: str
    regression_set: RegressionSetSpec
    human_gold_set: HumanGoldSetSpec
    quality_gates: dict[str, MetricGate]
    diagnostic_metrics: tuple[str, ...]
    required_result_fields: tuple[str, ...]


@dataclass(frozen=True)
class DatasetFingerprint:
    path: Path
    count: int
    sha256: str


@dataclass(frozen=True)
class HumanGoldReport:
    packet_count: int
    annotation_count: int
    labeled_count: int
    pending_count: int


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    observed: float | None
    minimum: float | None
    passed: bool | None


@dataclass(frozen=True)
class ResultsReport:
    count: int
    mean_latency_s: float
    p95_latency_s: float
    known_cost_usd: float | None
    cost_rows: int
    quality_checks: tuple[MetricCheck, ...]
    warnings: tuple[str, ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractValidationError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ContractValidationError(f"File not found: {path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"Invalid JSONL in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ContractValidationError(
                f"JSONL row {line_number} in {path} must be an object."
            )
        rows.append(row)
    return rows


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{context} must be an object.")
    return value


def _required_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _required_positive_int(payload: dict[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{context}.{key} must be a positive integer.")
    return value


def _optional_unit_float(payload: dict[str, Any], key: str, context: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{context}.{key} must be a number or null.")
    numeric_value = float(value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ContractValidationError(f"{context}.{key} must be between 0 and 1.")
    return numeric_value


def _parse_metric_gate(name: str, payload: Any) -> MetricGate:
    section = _mapping(payload, f"quality_gates.{name}")
    role = _required_string(section, "role", f"quality_gates.{name}")
    minimum = _optional_unit_float(section, "minimum", f"quality_gates.{name}")
    optional = section.get("optional", False)
    if not isinstance(optional, bool):
        raise ContractValidationError(f"quality_gates.{name}.optional must be boolean.")
    return MetricGate(role=role, minimum=minimum, optional=optional)


def load_contract(path: Path) -> EvaluationContract:
    """Load and structurally validate a JSON evaluation contract."""
    contract_path = path.expanduser().resolve()
    payload = _mapping(_read_json(contract_path), "contract")
    version = _required_string(payload, "contract_version", "contract")

    regression_payload = _mapping(payload.get("regression_set"), "regression_set")
    regression_set = RegressionSetSpec(
        path=_required_string(regression_payload, "path", "regression_set"),
        expected_count=_required_positive_int(
            regression_payload, "expected_count", "regression_set"
        ),
        sha256=_required_string(regression_payload, "sha256", "regression_set").lower(),
    )
    if len(regression_set.sha256) != 64 or any(
        character not in "0123456789abcdef" for character in regression_set.sha256
    ):
        raise ContractValidationError("regression_set.sha256 must be a SHA-256 hex digest.")

    human_payload = _mapping(payload.get("human_gold_set"), "human_gold_set")
    required_labels = human_payload.get("required_labels")
    if not isinstance(required_labels, list) or set(required_labels) != HUMAN_LABELS:
        raise ContractValidationError(
            "human_gold_set.required_labels must contain exactly the four supported labels."
        )
    min_count = _required_positive_int(human_payload, "min_count", "human_gold_set")
    max_count = _required_positive_int(human_payload, "max_count", "human_gold_set")
    expected_count = _required_positive_int(
        human_payload, "expected_count", "human_gold_set"
    )
    if min_count > expected_count or expected_count > max_count:
        raise ContractValidationError(
            "human_gold_set counts must satisfy min_count <= expected_count <= max_count."
        )
    human_gold_set = HumanGoldSetSpec(
        packet_path=_required_string(human_payload, "packet_path", "human_gold_set"),
        labels_path=_required_string(human_payload, "labels_path", "human_gold_set"),
        expected_count=expected_count,
        min_count=min_count,
        max_count=max_count,
        required_labels=frozenset(required_labels),
    )

    quality_payload = _mapping(payload.get("quality_gates"), "quality_gates")
    quality_gates = {
        name: _parse_metric_gate(name, metric_payload)
        for name, metric_payload in quality_payload.items()
    }
    if not quality_gates:
        raise ContractValidationError("quality_gates must contain at least one metric.")

    diagnostics = payload.get("diagnostic_metrics")
    if not isinstance(diagnostics, list) or not all(
        isinstance(metric, str) and metric.strip() for metric in diagnostics
    ):
        raise ContractValidationError("diagnostic_metrics must be a list of metric names.")

    required_fields = payload.get("required_result_fields")
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) and field.strip() for field in required_fields
    ):
        raise ContractValidationError(
            "required_result_fields must be a list of non-empty field names."
        )
    if len(set(required_fields)) != len(required_fields):
        raise ContractValidationError("required_result_fields must not contain duplicates.")

    project_root = (
        contract_path.parent.parent
        if contract_path.parent.name.lower() == "eval"
        else contract_path.parent
    )
    return EvaluationContract(
        project_root=project_root,
        version=version,
        regression_set=regression_set,
        human_gold_set=human_gold_set,
        quality_gates=quality_gates,
        diagnostic_metrics=tuple(diagnostics),
        required_result_fields=tuple(required_fields),
    )


def _resolve_contract_path(path_value: str, contract: EvaluationContract) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (contract.project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_question_ids(path: Path) -> tuple[str, ...]:
    """Load question IDs and reject malformed or duplicate entries."""
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ContractValidationError(f"Question file {path} must be a non-empty JSON list.")

    question_ids: list[str] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ContractValidationError(f"Question {index} in {path} must be an object.")
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ContractValidationError(
                f"Question {index} in {path} has no non-empty string id."
            )
        question_ids.append(question_id.strip())

    if len(set(question_ids)) != len(question_ids):
        raise ContractValidationError(f"Question IDs are not unique in {path}.")
    return tuple(question_ids)


def validate_questions_file(
    questions_path: Path,
    contract: EvaluationContract,
    allow_custom: bool = False,
) -> DatasetFingerprint:
    """Validate the canonical regression set or an explicitly allowed custom set."""
    actual_path = questions_path.expanduser().resolve()
    if not actual_path.exists():
        raise ContractValidationError(f"Question file not found: {actual_path}")

    question_ids = load_question_ids(actual_path)
    actual_hash = _sha256(actual_path)
    expected_path = _resolve_contract_path(contract.regression_set.path, contract)
    if actual_path != expected_path:
        if not allow_custom:
            raise ContractValidationError(
                f"{actual_path} is outside the fixed regression set. "
                "Pass --allow-custom-questions for a diagnostic subset."
            )
        return DatasetFingerprint(actual_path, len(question_ids), actual_hash)

    expected = contract.regression_set
    if len(question_ids) != expected.expected_count:
        raise ContractValidationError(
            f"Regression set contains {len(question_ids)} questions; "
            f"expected {expected.expected_count}."
        )
    if actual_hash.lower() != expected.sha256:
        raise ContractValidationError(
            "Regression set SHA-256 does not match the versioned contract. "
            "Update the contract deliberately when changing the set."
        )
    return DatasetFingerprint(actual_path, len(question_ids), actual_hash)


def _unique_ids(rows: list[dict[str, Any]], field: str, path: Path) -> set[str]:
    identifiers: list[str] = []
    for index, row in enumerate(rows, 1):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ContractValidationError(f"Row {index} in {path} has no valid {field}.")
        identifiers.append(value.strip())
    if len(set(identifiers)) != len(identifiers):
        raise ContractValidationError(f"{field} values are not unique in {path}.")
    return set(identifiers)


def validate_human_gold(
    contract: EvaluationContract,
    require_labels: bool = False,
) -> HumanGoldReport:
    """Validate the annotation packet and its human-label overlay."""
    packet_path = _resolve_contract_path(contract.human_gold_set.packet_path, contract)
    labels_path = _resolve_contract_path(contract.human_gold_set.labels_path, contract)
    packet_rows = _read_jsonl(packet_path)
    label_rows = _read_jsonl(labels_path)
    spec = contract.human_gold_set

    if not spec.min_count <= len(packet_rows) <= spec.max_count:
        raise ContractValidationError(
            f"Human packet contains {len(packet_rows)} rows; expected "
            f"between {spec.min_count} and {spec.max_count}."
        )
    if len(packet_rows) != spec.expected_count or len(label_rows) != len(packet_rows):
        raise ContractValidationError(
            "Human packet and label overlay must both contain the versioned expected count "
            f"of {spec.expected_count} rows."
        )

    packet_ids = _unique_ids(packet_rows, "question_id", packet_path)
    label_ids = _unique_ids(label_rows, "question_id", labels_path)
    if packet_ids != label_ids:
        raise ContractValidationError("Human packet IDs and label overlay IDs differ.")

    required_fields = {
        "question_id",
        "human_label",
        "rationale",
        "annotator_id",
        "annotated_at",
    }
    pending_count = 0
    for index, row in enumerate(label_rows, 1):
        missing_fields = required_fields.difference(row)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ContractValidationError(f"Annotation row {index} is missing: {missing}.")

        label = row["human_label"]
        if label is None:
            pending_count += 1
            continue
        if not isinstance(label, str) or label not in spec.required_labels:
            raise ContractValidationError(
                f"Annotation row {index} has unsupported human_label {label!r}."
            )
        for field in ("rationale", "annotator_id", "annotated_at"):
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(
                    f"Annotation row {index} requires {field} when human_label is set."
                )

    if require_labels and pending_count:
        raise ContractValidationError(
            f"{pending_count} human annotations are still pending."
        )
    return HumanGoldReport(
        packet_count=len(packet_rows),
        annotation_count=len(label_rows),
        labeled_count=len(label_rows) - pending_count,
        pending_count=pending_count,
    )


def _numeric_values(
    rows: list[dict[str, Any]],
    field: str,
    allow_none: bool,
) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows, 1):
        value = row.get(field)
        if value is None and allow_none:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError(
                f"Result row {index} has a non-numeric {field}: {value!r}."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ContractValidationError(
                f"Result row {index} has a non-finite {field}: {value!r}."
            )
        values.append(numeric_value)
    return values


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ContractValidationError("Cannot calculate a percentile without values.")
    ordered_values = sorted(values)
    index = min(len(ordered_values) - 1, math.ceil(len(ordered_values) * quantile) - 1)
    return ordered_values[index]


def validate_results_file(
    results_path: Path,
    contract: EvaluationContract,
) -> ResultsReport:
    """Validate a full regression run and calculate contract-level summaries."""
    actual_path = results_path.expanduser().resolve()
    rows = _read_jsonl(actual_path)
    if len(rows) != contract.regression_set.expected_count:
        raise ContractValidationError(
            f"Results contain {len(rows)} rows; expected "
            f"{contract.regression_set.expected_count} regression results."
        )

    expected_ids = set(
        load_question_ids(_resolve_contract_path(contract.regression_set.path, contract))
    )
    result_ids = _unique_ids(rows, "question_id", actual_path)
    if result_ids != expected_ids:
        raise ContractValidationError("Result question IDs do not match the regression set.")

    for index, row in enumerate(rows, 1):
        missing_fields = set(contract.required_result_fields).difference(row)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ContractValidationError(f"Result row {index} is missing: {missing}.")

    latency_values = _numeric_values(rows, "latency_s", allow_none=False)
    cost_values = _numeric_values(rows, "cost_usd", allow_none=True)
    if any(value < 0 for value in latency_values + cost_values):
        raise ContractValidationError("latency_s and cost_usd must not be negative.")

    warnings: list[str] = []
    if not cost_values:
        warnings.append(
            "cost_usd is null for every result; the provider did not expose a cost."
        )
    elif len(cost_values) != len(rows):
        warnings.append(
            f"cost_usd is known for {len(cost_values)}/{len(rows)} results; "
            "the reported total is partial."
        )

    quality_checks: list[MetricCheck] = []
    for metric, gate in contract.quality_gates.items():
        values = _numeric_values(rows, metric, allow_none=True)
        if not values:
            if gate.optional:
                warnings.append(f"{metric} was not computed for this run.")
                quality_checks.append(MetricCheck(metric, None, gate.minimum, None))
                continue
            raise ContractValidationError(f"Required quality metric {metric} is absent.")
        observed = statistics.fmean(values)
        passed = gate.minimum is None or observed >= gate.minimum
        quality_checks.append(
            MetricCheck(metric, round(observed, 4), gate.minimum, passed)
        )

    return ResultsReport(
        count=len(rows),
        mean_latency_s=round(statistics.fmean(latency_values), 4),
        p95_latency_s=round(_percentile(latency_values, P95_QUANTILE), 4),
        known_cost_usd=round(sum(cost_values), 6) if cost_values else None,
        cost_rows=len(cost_values),
        quality_checks=tuple(quality_checks),
        warnings=tuple(warnings),
    )


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "EvaluationContract",
    "ContractValidationError",
    "DatasetFingerprint",
    "HumanGoldReport",
    "MetricCheck",
    "ResultsReport",
    "load_contract",
    "load_question_ids",
    "validate_human_gold",
    "validate_questions_file",
    "validate_results_file",
]
