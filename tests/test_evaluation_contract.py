from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.contract import (
    ContractValidationError,
    load_contract,
    load_question_ids,
    validate_human_gold,
    validate_questions_file,
    validate_results_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "eval" / "evaluation_contract.json"


def test_fixed_regression_set_matches_contract() -> None:
    contract = load_contract(CONTRACT_PATH)

    fingerprint = validate_questions_file(
        ROOT / "eval" / "test_questions.json",
        contract,
    )

    assert fingerprint.count == 100
    assert fingerprint.sha256 == contract.regression_set.sha256


def test_human_gold_template_exposes_pending_rows() -> None:
    contract = load_contract(CONTRACT_PATH)

    report = validate_human_gold(contract)

    assert report.annotation_count == 40
    assert report.labeled_count + report.pending_count == report.annotation_count


def test_strict_human_gold_validation_rejects_pending_rows() -> None:
    contract = load_contract(CONTRACT_PATH)
    report = validate_human_gold(contract)
    if report.pending_count == 0:
        pytest.skip("The local gold set is already fully reviewed.")

    with pytest.raises(
        ContractValidationError,
        match=f"{report.pending_count} human annotations",
    ):
        validate_human_gold(contract, require_labels=True)


def test_custom_question_set_requires_explicit_opt_in(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT_PATH)
    custom_path = tmp_path / "custom_questions.json"
    custom_path.write_text(
        json.dumps(
            [
                {
                    "id": "custom-001",
                    "question": "Question de diagnostic",
                    "reference_answer": "Reponse",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError, match="outside the fixed regression set"):
        validate_questions_file(custom_path, contract)

    fingerprint = validate_questions_file(custom_path, contract, allow_custom=True)

    assert fingerprint.count == 1


def test_results_contract_requires_quality_latency_and_cost_fields(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT_PATH)
    question_ids = load_question_ids(ROOT / "eval" / "test_questions.json")
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "question_id": question_id,
            "question": "Question",
            "answer": "Reponse [Source 1]",
            "keyword_recall": 0.8,
            "word_f1": 0.2,
            "citation_faithfulness": 1.0,
            "context_coverage": 0.8,
            "bertscore_f1": 0.8,
            "judge_score": 0.8,
            "latency_s": 1.5,
            "cost_usd": None,
        }
        for question_id in question_ids
    ]
    results_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = validate_results_file(results_path, contract)

    assert report.count == 100
    assert report.mean_latency_s == pytest.approx(1.5)
    assert report.p95_latency_s == pytest.approx(1.5)
    assert report.known_cost_usd is None
    assert report.cost_rows == 0
    assert all(check.passed is True for check in report.quality_checks)


def test_results_contract_rejects_missing_cost_field(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT_PATH)
    question_ids = load_question_ids(ROOT / "eval" / "test_questions.json")
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "question_id": question_id,
            "question": "Question",
            "answer": "Reponse",
            "keyword_recall": 0.8,
            "word_f1": 0.2,
            "citation_faithfulness": 1.0,
            "context_coverage": 0.8,
            "bertscore_f1": None,
            "judge_score": None,
            "latency_s": 1.5,
            "cost_usd": None,
        }
        for question_id in question_ids
    ]
    del rows[0]["cost_usd"]
    results_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError, match="missing: cost_usd"):
        validate_results_file(results_path, contract)
