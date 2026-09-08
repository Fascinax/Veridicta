"""Versioned generation-model matrix and deterministic task routing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "eval/test_questions.json"
POLICY_VERSION = "2026-09-08-model-routing-v1"


@dataclass(frozen=True)
class ModelSpec:
    """One provider/model candidate in the evaluation matrix."""

    backend: str
    model: str
    role: str

    @property
    def key(self) -> str:
        return f"{self.backend}/{self.model}"


@dataclass(frozen=True)
class GenerationRoute:
    """Task class and selected model with an auditable candidate order."""

    task_class: str
    selected: ModelSpec
    candidates: tuple[ModelSpec, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_class": self.task_class,
            "selected": self.selected.key,
            "backend": self.selected.backend,
            "model": self.selected.model,
            "role": self.selected.role,
            "candidates": [candidate.key for candidate in self.candidates],
            "reason": self.reason,
            "policy_version": POLICY_VERSION,
        }


DEFAULT_MODEL_MATRIX = (
    ModelSpec("copilot", "gpt-4.1", "quality"),
    ModelSpec("copilot", "gpt-4.1-mini", "latency"),
    ModelSpec("cerebras", "gpt-oss-120b", "quality"),
    ModelSpec("cerebras", "llama3.1-8b", "latency"),
    ModelSpec("omniroute", "auto", "local"),
)

_COMPLEX_TERMS = (
    "comparer",
    "conditions",
    "exception",
    "exceptions",
    "indemnité",
    "indemnités",
    "jurisprudence",
    "recours",
    "calcul",
    "différence",
    "mise à la retraite",
)


def classify_generation_task(question: str) -> str:
    """Classify a question for model selection without an LLM call."""
    normalized = " ".join(question.casefold().split())
    if len(normalized) >= 100 or any(term in normalized for term in _COMPLEX_TERMS):
        return "complex_legal_analysis"
    return "factual_lookup"


def _priority(task_class: str, specs: Iterable[ModelSpec]) -> list[ModelSpec]:
    role_order = {
        "complex_legal_analysis": ("quality", "local", "latency"),
        "factual_lookup": ("latency", "local", "quality"),
    }[task_class]
    rank = {role: index for index, role in enumerate(role_order)}
    return sorted(specs, key=lambda spec: (rank.get(spec.role, len(rank)), spec.key))


def route_generation_model(
    question: str,
    available_specs: Iterable[ModelSpec] = DEFAULT_MODEL_MATRIX,
) -> GenerationRoute:
    """Select a model from the available matrix using a fixed policy."""
    specs = tuple(available_specs)
    if not specs:
        raise ValueError("Generation model matrix is empty")
    task_class = classify_generation_task(question)
    candidates = tuple(_priority(task_class, specs))
    selected = candidates[0]
    reason = (
        "complex questions prefer quality models; latency/local candidates remain fallbacks"
        if task_class == "complex_legal_analysis"
        else "factual questions prefer latency/local models; quality candidates remain fallbacks"
    )
    return GenerationRoute(task_class, selected, candidates, reason)


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Question set must be a JSON list: {path}")
    rows: list[dict] = []
    seen: set[str] = set()
    for row in payload:
        question_id = str(row.get("id", "")).strip()
        if not question_id or question_id in seen:
            raise ValueError(f"Question IDs must be non-empty and unique: {path}")
        seen.add(question_id)
        rows.append(row)
    return rows


def build_matrix_plan(
    questions_path: Path,
    available_specs: Iterable[ModelSpec] = DEFAULT_MODEL_MATRIX,
) -> list[dict[str, object]]:
    """Create one deterministic model-routing row per question."""
    specs = tuple(available_specs)
    plan: list[dict[str, object]] = []
    for question in load_questions(questions_path):
        question_text = str(question.get("question", ""))
        route = route_generation_model(question_text, specs)
        plan.append(
            {
                "question_id": str(question["id"]),
                "question": question_text,
                **route.to_dict(),
            }
        )
    return plan


class MatrixArtifactError(ValueError):
    """Raised when a model artifact cannot be compared on the fixed set."""


def _load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise MatrixArtifactError(f"Empty model artifact: {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise MatrixArtifactError(f"Model artifact rows must be objects: {path}")
    return rows


def _artifact_index(path: Path) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in _load_jsonl(path):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id or question_id in indexed:
            raise MatrixArtifactError(f"Duplicate or missing question_id in {path}")
        indexed[question_id] = row
    return indexed


def _mean(rows: Iterable[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _p95(rows: Iterable[dict], field: str) -> float | None:
    values = sorted(float(row[field]) for row in rows if row.get(field) is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(len(values) * 0.95) - 1))
    return round(values[index], 4)


def _summarize_artifact(label: str, path: Path, rows: dict[str, dict]) -> dict[str, object]:
    ordered = [rows[key] for key in sorted(rows)]
    return {
        "label": label,
        "artifact": path.as_posix(),
        "question_count": len(ordered),
        "keyword_recall": _mean(ordered, "keyword_recall"),
        "context_coverage": _mean(ordered, "context_coverage"),
        "citation_faithfulness": _mean(ordered, "citation_faithfulness"),
        "bertscore_f1": _mean(ordered, "bertscore_f1"),
        "judge_score": _mean(ordered, "judge_score"),
        "latency_mean_s": _mean(ordered, "latency_s"),
        "latency_p95_s": _p95(ordered, "latency_s"),
        "cost_mean_usd": _mean(ordered, "cost_usd"),
        "unsupported_claim_rate": _mean(ordered, "unsupported_claim_rate"),
    }


def build_matrix_report(
    artifacts: dict[str, Path],
    questions_path: Path = DEFAULT_QUESTIONS,
) -> dict[str, object]:
    """Aggregate comparable model artifacts and enforce question alignment."""
    expected_ids = {str(row["id"]) for row in load_questions(questions_path)}
    summaries: list[dict[str, object]] = []
    for label, path in sorted(artifacts.items()):
        indexed = _artifact_index(path)
        if set(indexed) != expected_ids:
            raise MatrixArtifactError(f"Artifact {label!r} is not aligned to the fixed question set")
        summaries.append(_summarize_artifact(label, path, indexed))
    return {
        "schema_version": "1.0",
        "policy_version": POLICY_VERSION,
        "question_contract": questions_path.as_posix(),
        "question_count": len(expected_ids),
        "artifacts": summaries,
        "selection_policy": "quality metrics first, latency and cost as tie-breakers; no automatic legal decision",
    }


def _parse_artifact(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, required=True, help="JSONL routing plan")
    parser.add_argument("--report", type=Path, help="Optional JSON model comparison report")
    parser.add_argument("--results", type=_parse_artifact, action="append", default=[])
    args = parser.parse_args()

    plan = build_matrix_plan(args.questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in plan) + "\n",
        encoding="utf-8",
    )
    print(f"Model routing plan written to {args.out} ({len(plan)} questions)")

    if args.report:
        report = build_matrix_report(dict(args.results), args.questions)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Model matrix report written to {args.report}")


if __name__ == "__main__":
    main()
