from dataclasses import asdict
import json

import pytest

from eval.benchmark_rerankers import (
    BenchmarkConfig, BenchmarkDependencies, EvalQuestion, RawRetrievalCase,
    RERANKER_SPECS, run_benchmark, validate_ranking,
)
from eval.reranker_replay import check_output, load_pool, save_pool
from retrievers.snapshots import file_digest


def _case():
    return RawRetrievalCase(EvalQuestion("q1", "Question synthétique", "Réponse", ["preuve"]),
                            [{"chunk_id": "a", "text": "Autre sujet."},
                             {"chunk_id": "b", "text": "Une preuve synthétique."}], 0.01)


def _pool(tmp_path):
    case = _case()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([asdict(case.question)]), encoding="utf-8")
    root = tmp_path / "frozen"
    save_pool(root, [case], {"questions_sha256": file_digest(questions), "candidate_k": 2, "retriever": "faiss"})
    return root, questions


def test_pool_round_trip_keeps_candidate_order_text_and_questions(tmp_path):
    root, questions = _pool(tmp_path)
    assert load_pool(root, questions)[0] == [_case()]


def test_pool_cannot_overwrite_an_existing_run(tmp_path):
    with pytest.raises(FileExistsError):
        save_pool(tmp_path, [_case()], {})


def test_output_cannot_change_a_frozen_input_directory(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        check_output(tmp_path / "pool/run", tmp_path / "pool")


def test_output_cannot_be_nested_inside_an_unrelated_sealed_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot"):
        check_output(snapshot / "raw/new", tmp_path / "pool")


def test_empty_question_id_does_not_enter_a_frozen_pool(tmp_path):
    case = _case()
    case.question.id = ""
    with pytest.raises(ValueError, match="question IDs"):
        save_pool(tmp_path / "pool", [case], {})


def test_changed_candidate_bytes_are_rejected(tmp_path):
    root, questions = _pool(tmp_path)
    path = root / "candidates.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("Autre sujet", "Altération"), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_pool(root, questions)


def test_changed_question_file_is_rejected(tmp_path):
    root, questions = _pool(tmp_path)
    questions.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_pool(root, questions)


def test_declared_question_order_cannot_drift_even_with_rehashed_pool(tmp_path):
    root, questions = _pool(tmp_path)
    path = root / "candidates.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["question"]["reference_keywords"] = ["autre"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = root / "pool.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidates_sha256"] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="questions differ"):
        load_pool(root, questions)


@pytest.mark.parametrize("ranked", [
    [{"chunk_id": "a", "text": "Autre sujet."}],
    [{"chunk_id": "a", "text": "Autre sujet."}] * 2,
    [{"chunk_id": "a", "text": "Modifié"}, {"chunk_id": "b", "text": "Une preuve synthétique."}],
    [{"chunk_id": "a", "text": "Autre sujet.", "rerank_score": float("nan")},
     {"chunk_id": "b", "text": "Une preuve synthétique."}],
])
def test_incomplete_duplicate_modified_or_nonfinite_rankings_fail(ranked):
    with pytest.raises(ValueError):
        validate_ranking(_case().candidates, ranked)


def test_mrr_at_k_does_not_credit_a_passage_outside_the_injected_window():
    result = run_benchmark([_case()], [RERANKER_SPECS["none"]],
                           BenchmarkConfig(candidate_pools=(2,), top_ks=(1, 2)))
    assert [(r.schema_version, r.top_k, r.mrr) for r in result.summaries] == [("2.0", 1, 0.0), ("2.0", 2, 0.5)]


def test_adapter_mutation_cannot_change_the_shared_input_pool():
    case = _case()

    class MutatingAdapter:
        def rank(self, query, candidates):
            candidates.reverse()
            return candidates

    run_benchmark([case], [RERANKER_SPECS["none"]], BenchmarkConfig(candidate_pools=(2,), top_ks=(1,)),
                   BenchmarkDependencies(adapter_factory=lambda _: MutatingAdapter()))
    assert case == _case()


def test_cross_encoder_uses_pinned_weights_and_explicit_inference_settings(monkeypatch):
    from types import SimpleNamespace
    from eval.benchmark_rerankers import CrossEncoderRuntime, LocalCrossEncoderAdapter

    recorded = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            recorded.update(name=name, **kwargs)

        def predict(self, pairs, **kwargs):
            recorded.update(prediction=kwargs)
            return [-1.0, 2.0]

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    adapter = LocalCrossEncoderAdapter(RERANKER_SPECS["qwen3"], CrossEncoderRuntime("cpu", 4, True))
    result = adapter.rank("Question", _case().candidates)
    assert (recorded["revision"], recorded["trust_remote_code"], recorded["local_files_only"],
            recorded["prediction"]["batch_size"], result[0]["chunk_id"]) == (
        RERANKER_SPECS["qwen3"].revision, False, True, 4, "b")
