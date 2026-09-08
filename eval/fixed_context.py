"""Validate fixed-context packets for the Stage 2 generation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.contract import HUMAN_LABELS, load_contract
from eval.model_matrix import load_questions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_QUESTIONS = ROOT / "eval/test_questions_stage2_gen_bad6.json"
REQUIRED_PROMPTS = frozenset({"v1", "v2", "v3"})
REQUIRED_METRICS = frozenset(
    {
        "bertscore_f1",
        "citation_faithfulness",
        "context_coverage",
        "cost_usd",
        "judge_score",
        "latency_s",
    }
)
FAILURE_CLASSES = frozenset(
    {"omission", "unsupported_synthesis", "citation_placement", "verbosity", "metric_only"}
)


class FixedContextPacketError(ValueError):
    """Raised when a Stage 2 packet is not comparable or auditable."""


@dataclass(frozen=True)
class FixedContextReport:
    """Metadata-only validation result for one fixed-context packet."""

    contract_version: str
    question_count: int
    case_ids: tuple[str, ...]
    context_digests: dict[str, str]
    prompt_count: int
    answer_count: int
    human_reviewed_count: int
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "contract_version": self.contract_version,
            "question_count": self.question_count,
            "case_ids": list(self.case_ids),
            "context_digests": self.context_digests,
            "prompt_count": self.prompt_count,
            "answer_count": self.answer_count,
            "human_reviewed_count": self.human_reviewed_count,
            "complete": self.complete,
        }


def context_sha256(context: list[dict[str, Any]]) -> str:
    """Return a stable digest for the exact ordered context payload."""
    canonical = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedContextPacketError(f"Cannot read fixed-context packet: {path}") from exc
    if not isinstance(payload, dict):
        raise FixedContextPacketError("Fixed-context packet must be a JSON object")
    return payload


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixedContextPacketError(f"{field} must be a non-empty string")
    return value


def _validate_answer_matrix(answers: object, prompts: dict[str, object]) -> int:
    if not isinstance(answers, dict):
        raise FixedContextPacketError("answers must be an object keyed by prompt version")
    if set(answers) != set(prompts):
        raise FixedContextPacketError("answers and prompts must have identical keys")
    for prompt_key, raw_answer in answers.items():
        if not isinstance(raw_answer, dict):
            raise FixedContextPacketError(f"answers[{prompt_key}] must be an object")
        _require_text(raw_answer.get("text"), f"answers[{prompt_key}].text")
        metrics = raw_answer.get("metrics")
        if not isinstance(metrics, dict):
            raise FixedContextPacketError(f"answers[{prompt_key}].metrics must be an object")
        missing_metrics = sorted(REQUIRED_METRICS - set(metrics))
        if missing_metrics:
            raise FixedContextPacketError(
                f"answers[{prompt_key}].metrics missing: {', '.join(missing_metrics)}"
            )
    return len(answers)


def validate_fixed_context_packet(
    packet_path: Path,
    questions_path: Path = DEFAULT_STAGE2_QUESTIONS,
    *,
    require_answers: bool = True,
) -> FixedContextReport:
    """Validate a six-case packet against the versioned evaluation contract."""
    packet = _load_packet(packet_path)
    questions = load_questions(questions_path)
    expected_ids = tuple(str(question["id"]) for question in questions)
    if len(expected_ids) != 6:
        raise FixedContextPacketError("Stage 2 fixed-context question set must contain six cases")

    contract = load_contract(ROOT / "eval/evaluation_contract.json")
    if packet.get("contract_version") != contract.version:
        raise FixedContextPacketError(
            f"Packet contract version {packet.get('contract_version')!r} does not match {contract.version!r}"
        )
    cases = packet.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_ids):
        raise FixedContextPacketError("Packet must contain exactly the six fixed cases")

    question_by_id = {str(question["id"]): question for question in questions}
    case_ids: list[str] = []
    context_digests: dict[str, str] = {}
    prompt_count = 0
    answer_count = 0
    human_reviewed_count = 0
    for case in cases:
        if not isinstance(case, dict):
            raise FixedContextPacketError("Each fixed-context case must be an object")
        question_id = _require_text(case.get("question_id"), "question_id")
        if question_id in case_ids:
            raise FixedContextPacketError(f"Duplicate fixed-context case: {question_id}")
        if question_id not in question_by_id:
            raise FixedContextPacketError(f"Unexpected fixed-context case: {question_id}")
        if case.get("question") != question_by_id[question_id]["question"]:
            raise FixedContextPacketError(f"Question text mismatch for {question_id}")
        context = case.get("context")
        if not isinstance(context, list) or not context:
            raise FixedContextPacketError(f"Context is empty for {question_id}")
        if any(not isinstance(chunk, dict) or not str(chunk.get("text", "")).strip() for chunk in context):
            raise FixedContextPacketError(f"Every context chunk needs text for {question_id}")
        digest = context_sha256(context)
        if case.get("context_sha256") != digest:
            raise FixedContextPacketError(f"Context digest mismatch for {question_id}")

        prompts = case.get("prompts")
        if not isinstance(prompts, dict) or not REQUIRED_PROMPTS.issubset(prompts):
            raise FixedContextPacketError(
                f"Prompts v1, v2 and v3 are required for {question_id}"
            )
        for prompt_key, prompt in prompts.items():
            _require_text(prompt, f"prompts[{prompt_key}]")
        prompt_count += len(prompts)

        classification = _require_text(case.get("failure_classification"), "failure_classification")
        if classification not in FAILURE_CLASSES:
            raise FixedContextPacketError(f"Unsupported failure classification: {classification}")
        human_label = _require_text(case.get("human_label"), "human_label")
        if human_label not in HUMAN_LABELS:
            raise FixedContextPacketError(f"Unsupported human label: {human_label}")
        _require_text(case.get("decision"), "decision")
        human_reviewed_count += 1

        answers = case.get("answers")
        if answers is None:
            if require_answers:
                raise FixedContextPacketError(f"Answers are missing for {question_id}")
        else:
            answer_count += _validate_answer_matrix(answers, prompts)
        case_ids.append(question_id)
        context_digests[question_id] = digest

    if tuple(case_ids) != expected_ids:
        raise FixedContextPacketError("Fixed-context cases must follow the question-set order")
    return FixedContextReport(
        contract_version=contract.version,
        question_count=len(case_ids),
        case_ids=tuple(case_ids),
        context_digests=context_digests,
        prompt_count=prompt_count,
        answer_count=answer_count,
        human_reviewed_count=human_reviewed_count,
        complete=answer_count == prompt_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_STAGE2_QUESTIONS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-pending-answers",
        action="store_true",
        help="Validate the fixed context and prompts before provider runs are available",
    )
    args = parser.parse_args()
    try:
        report = validate_fixed_context_packet(
            args.packet,
            args.questions,
            require_answers=not args.allow_pending_answers,
        )
    except FixedContextPacketError as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"schema_version": "1.0", "status": "fail", "error": str(exc)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    status = "pass" if report.complete else "pending"
    args.out.write_text(
        json.dumps({"status": status, **report.to_dict()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if report.complete:
        print(f"Fixed-context packet is valid: {args.out}")
        return
    print(f"Fixed-context packet is pending provider answers: {args.out}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
