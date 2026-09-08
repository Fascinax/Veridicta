from __future__ import annotations

import json

import pytest

from eval.model_matrix import (
    MatrixArtifactError,
    ModelSpec,
    build_matrix_plan,
    build_matrix_report,
    route_generation_model,
)


def test_complex_questions_prefer_quality_model() -> None:
    route = route_generation_model(
        "Comparez les conditions, exceptions et indemnités applicables au départ à la retraite.",
        (
            ModelSpec("omniroute", "auto", "local"),
            ModelSpec("copilot", "gpt-4.1", "quality"),
            ModelSpec("cerebras", "llama3.1-8b", "latency"),
        ),
    )

    assert route.task_class == "complex_legal_analysis"
    assert route.selected.key == "copilot/gpt-4.1"
    assert route.to_dict()["policy_version"]


def test_factual_questions_prefer_latency_model() -> None:
    route = route_generation_model(
        "Quel est le délai de préavis ?",
        (
            ModelSpec("copilot", "gpt-4.1", "quality"),
            ModelSpec("cerebras", "llama3.1-8b", "latency"),
        ),
    )

    assert route.task_class == "factual_lookup"
    assert route.selected.key == "cerebras/llama3.1-8b"


def test_build_matrix_plan_is_deterministic(tmp_path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([
            {"id": "q1", "question": "Quel est le délai de préavis ?"},
            {"id": "q2", "question": "Comparez les exceptions applicables."},
        ]),
        encoding="utf-8",
    )

    plan = build_matrix_plan(questions)

    assert [row["question_id"] for row in plan] == ["q1", "q2"]
    assert plan[0]["task_class"] == "factual_lookup"
    assert plan[1]["task_class"] == "complex_legal_analysis"


def test_build_matrix_report_requires_fixed_question_alignment(tmp_path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1"}]), encoding="utf-8")
    artifact = tmp_path / "model.jsonl"
    artifact.write_text(
        json.dumps({
            "question_id": "q2",
            "keyword_recall": 1.0,
            "context_coverage": 1.0,
            "citation_faithfulness": 1.0,
            "latency_s": 1.0,
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MatrixArtifactError, match="fixed question set"):
        build_matrix_report({"model": artifact}, questions)
