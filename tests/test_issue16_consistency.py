from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from autoeval.experiment import _TSV_HEADERS, _compute_metrics
from eval.validate_consistency import validate_consistency


ROOT = Path(__file__).resolve().parents[1]


def test_contract_documentation_and_autoeval_stay_aligned() -> None:
    validate_consistency(ROOT)


def test_autoeval_aggregates_contract_metrics_without_word_f1_gate() -> None:
    results = [
        SimpleNamespace(
            keyword_recall=0.5,
            word_f1=0.4,
            bertscore_f1=0.8,
            judge_score=0.7,
            citation_faithfulness=1.0,
            context_coverage=0.6,
            latency_s=0.5,
            cost_usd=None,
        ),
        SimpleNamespace(
            keyword_recall=0.7,
            word_f1=None,
            bertscore_f1=0.6,
            judge_score=None,
            citation_faithfulness=0.9,
            context_coverage=0.8,
            latency_s=2.0,
            cost_usd=0.2,
        ),
    ]

    metrics = _compute_metrics(results)

    assert metrics["BERTScore"] == 0.7
    assert metrics["Judge"] == 0.7
    assert metrics["LatP95"] == 2.0
    assert metrics["Cost"] == 0.2
    assert "WordF1" in _TSV_HEADERS
    assert "BERTScore" in _TSV_HEADERS
    assert "Judge" in _TSV_HEADERS
