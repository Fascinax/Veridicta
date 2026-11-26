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


def test_empty_reranker_workload_leaves_no_partial_run(tmp_path, monkeypatch):
    from argparse import Namespace
    from eval import reranker_replay

    args = Namespace(out=tmp_path / "run", pool=tmp_path / "pool", questions=tmp_path / "questions")
    cases = [RawRetrievalCase(_case().question, [], 0)]
    monkeypatch.setattr(reranker_replay, "load_pool", lambda *_: (cases, {}))
    with pytest.raises(ValueError, match="no candidates"):
        reranker_replay.replay(args)
    assert not args.out.exists()


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


@pytest.mark.parametrize("different_model", [False, True])
def test_collect_uses_sealed_lancedb_instead_of_active_store(tmp_path, monkeypatch, different_model):
    import sys
    from datetime import datetime, timezone

    import lancedb

    from eval import benchmark_rerankers, reranker_replay
    from retrievers import lancedb_rag
    from retrievers.config import EMBEDDING_CONFIG
    from retrievers.snapshots import create_manifest

    snapshot = tmp_path / "sealed"
    index_dir = snapshot / "index"
    index_dir.mkdir(parents=True)
    active = tmp_path / "active"
    canonical = {
        "chunk_id": "shared", "doc_id": "doc", "text": "Sealed text",
        "source": "", "title": "",
    }
    for db_dir, text in ((index_dir / "lancedb", canonical["text"]), (active, "Active text")):
        row = canonical | {"text": text, "metadata_json": "{}", "vector": [1.0] + [0.0] * (EMBEDDING_CONFIG.dimension - 1)}
        lancedb.connect(str(db_dir)).create_table("chunks", [row])
    (index_dir / "chunks_map.jsonl").write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    metadata = {"embed_model": "different-model" if different_model else EMBEDDING_CONFIG.model_name,
                "embed_revision": EMBEDDING_CONFIG.revision, "embedding_dimension": EMBEDDING_CONFIG.dimension,
                "embed_query_prefix": EMBEDDING_CONFIG.query_prefix}
    (index_dir / "embedding_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = snapshot / "snapshot.manifest.json"
    manifest.write_text(
        json.dumps(create_manifest(snapshot, datetime.now(timezone.utc).isoformat(), "fixture")),
        encoding="utf-8",
    )
    question = _case().question
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([asdict(question)]), encoding="utf-8")
    out = tmp_path / "pool"

    monkeypatch.setattr(lancedb_rag, "active_lancedb_dir", lambda: active)
    monkeypatch.setattr(benchmark_rerankers, "_load_embedder", object)
    monkeypatch.setattr(
        reranker_replay, "collect_raw_retrieval",
        lambda questions, pipeline, config: [RawRetrievalCase(questions[0], pipeline.chunks, 0.0)],
    )
    monkeypatch.setattr(sys, "argv", [
        "reranker_replay", "collect", "--snapshot", str(manifest), "--retriever", "lancedb",
        "--questions", str(questions), "--out", str(out), "--candidate-k", "1", "--allow-custom-questions",
    ])
    if different_model:
        with pytest.raises(ValueError, match="query encoder"):
            reranker_replay.main()
        assert not out.exists()
        return
    reranker_replay.main()

    assert load_pool(out, questions)[0][0].candidates == [canonical]
