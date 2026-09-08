from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.fixed_context import (
    FixedContextPacketError,
    context_sha256,
    validate_fixed_context_packet,
)
from eval.model_matrix import load_questions


QUESTIONS_PATH = Path("eval/test_questions_stage2_gen_bad6.json")


def _packet(include_answers: bool = True) -> dict:
    cases = []
    for question in load_questions(QUESTIONS_PATH):
        context = [{"chunk_id": f"{question['id']}-chunk-1", "text": "Contexte juridique de test."}]
        prompts = {version: f"Prompt {version}" for version in ("v1", "v2", "v3")}
        case = {
            "question_id": question["id"],
            "question": question["question"],
            "context": context,
            "context_sha256": context_sha256(context),
            "prompts": prompts,
            "failure_classification": "omission",
            "human_label": "incomplete",
            "decision": "prompt",
        }
        if include_answers:
            case["answers"] = {
                version: {
                    "text": f"Réponse {version}.",
                    "metrics": {
                        "bertscore_f1": 0.5,
                        "citation_faithfulness": 1.0,
                        "context_coverage": 1.0,
                        "cost_usd": None,
                        "judge_score": 0.5,
                        "latency_s": 1.0,
                    },
                }
                for version in prompts
            }
        cases.append(case)
    return {"contract_version": "1.1.0", "cases": cases}


def test_fixed_context_packet_requires_exact_context_and_metric_matrix(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")

    report = validate_fixed_context_packet(packet_path)

    assert report.question_count == 6
    assert report.prompt_count == 18
    assert report.answer_count == 18
    assert report.complete
    assert len(report.context_digests) == 6


def test_fixed_context_packet_can_be_prepared_before_provider_runs(tmp_path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet(include_answers=False)), encoding="utf-8")

    report = validate_fixed_context_packet(packet_path, require_answers=False)

    assert report.prompt_count == 18
    assert report.answer_count == 0
    assert not report.complete
    with pytest.raises(FixedContextPacketError, match="Answers are missing"):
        validate_fixed_context_packet(packet_path)


def test_fixed_context_packet_rejects_digest_drift(tmp_path) -> None:
    packet = _packet()
    packet["cases"][0]["context_sha256"] = "0" * 64
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    with pytest.raises(FixedContextPacketError, match="Context digest mismatch"):
        validate_fixed_context_packet(packet_path)
