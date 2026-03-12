from __future__ import annotations

import pytest

from data_ingest.data_processor import _overlap_tail
from eval.evaluate import _parse_args, _parse_judge_response, keyword_recall
from retrievers.config import get_context_budget_tokens, resolve_llm_backend


def test_overlap_tail_keeps_word_boundaries() -> None:
    tail = _overlap_tail("alpha beta gamma delta", 10)

    assert tail == "delta"
    assert not tail.startswith("amma")


def test_keyword_recall_matches_french_morphology() -> None:
    prediction = "Le salarié a été congédié sans préavis."

    score = keyword_recall(prediction, ["congédiement", "préavis"])

    assert score == pytest.approx(1.0)


def test_resolve_llm_backend_rejects_typos() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM_BACKEND"):
        resolve_llm_backend("copliot")


def test_context_budget_is_model_aware() -> None:
    copilot_budget = get_context_budget_tokens("copilot", "gpt-4.1")
    cerebras_budget = get_context_budget_tokens("cerebras", "gpt-oss-120b")

    assert copilot_budget > cerebras_budget
    assert cerebras_budget >= 1024


def test_parse_args_bertscore_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate.py"])

    args = _parse_args()

    assert args.bertscore is False
    assert args.bertscore_model == "distilbert-base-multilingual-cased"
    assert args.bertscore_lang == "fr"
    assert args.bertscore_batch_size == 16
    assert args.bertscore_device is None


def test_parse_args_bertscore_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--bertscore",
            "--bertscore-model",
            "bert-base-multilingual-cased",
            "--bertscore-lang",
            "fr",
            "--bertscore-batch-size",
            "4",
            "--bertscore-device",
            "cpu",
        ],
    )

    args = _parse_args()

    assert args.bertscore is True
    assert args.bertscore_model == "bert-base-multilingual-cased"
    assert args.bertscore_lang == "fr"
    assert args.bertscore_batch_size == 4
    assert args.bertscore_device == "cpu"


def test_parse_args_judge_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate.py"])

    args = _parse_args()

    assert args.judge is False
    assert args.judge_backend == "copilot"
    assert args.judge_model is None


def test_parse_args_judge_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--judge",
            "--judge-backend",
            "cerebras",
            "--judge-model",
            "llama3.1-8b",
        ],
    )

    args = _parse_args()

    assert args.judge is True
    assert args.judge_backend == "cerebras"
    assert args.judge_model == "llama3.1-8b"


def test_parse_judge_response_accepts_fenced_json() -> None:
    verdict = _parse_judge_response(
        """```json
        {"score": 0.82, "verdict": "acceptable", "reason": "Bonne regle, formulation differente."}
        ```"""
    )

    assert verdict.error is None
    assert verdict.score == pytest.approx(0.82)
    assert verdict.label == "acceptable"
    assert verdict.reason == "Bonne regle, formulation differente."


def test_parse_judge_response_rejects_non_json() -> None:
    verdict = _parse_judge_response("not json at all")

    assert verdict.score is None
    assert verdict.label is None
    assert verdict.error == "Judge returned non-JSON output"
