"""Freeze retrieval candidates and replay one local reranker per isolated process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
import zipfile

from eval.benchmark_rerankers import (
    BenchmarkConfig, BenchmarkDependencies, CrossEncoderRuntime, EvalQuestion,
    FlashRankAdapter, LocalCrossEncoderAdapter, RawRetrievalCase, RERANKER_SPECS,
    RetrievalCollectionConfig, RetrievalOrderAdapter, _load_pipeline, _memory_rss_mb,
    _release_retrieval_resources, _validate_questions, collect_raw_retrieval,
    load_questions, run_benchmark, write_jsonl,
)
from retrievers.snapshots import file_digest, verify_manifest

FLASHRANK_REVISION = "858a1ac046a05663a35367eac852d7f76feeefdd"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_candidates(cases: list[RawRetrievalCase]) -> None:
    ids = [case.question.id for case in cases]
    if (not ids or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Expected distinct nonempty question IDs")
    for case in cases:
        chunk_ids = [candidate.get("chunk_id") for candidate in case.candidates]
        if (any(not isinstance(value, str) or not value for value in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids)
                or any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in case.candidates)
                or not math.isfinite(case.retrieval_latency_s) or case.retrieval_latency_s < 0):
            raise ValueError("Invalid frozen candidate case")


def check_output(destination: Path, protected: Path) -> None:
    destination, protected = destination.resolve(), protected.resolve()
    if destination.is_relative_to(protected) or protected.is_relative_to(destination):
        raise ValueError("Output must be separate from frozen inputs")
    if any((parent / "snapshot.manifest.json").exists() for parent in destination.parents):
        raise ValueError("Output cannot be inside a corpus snapshot")


def save_pool(destination: Path, cases: list[RawRetrievalCase], provenance: dict) -> dict:
    _validate_candidates(cases)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "candidates.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
                "candidates_sha256": file_digest(path), "question_count": len(cases),
                "provenance": provenance}
    _write_json(destination / "pool.manifest.json", manifest)
    return manifest


def load_pool(root: Path, questions_path: Path) -> tuple[list[RawRetrievalCase], dict]:
    manifest = json.loads((root / "pool.manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "1.0"
            or file_digest(root / "candidates.jsonl") != manifest.get("candidates_sha256")
            or file_digest(questions_path) != manifest["provenance"]["questions_sha256"]):
        raise ValueError("Frozen pool or question file changed")
    cases = []
    with (root / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cases.append(RawRetrievalCase(EvalQuestion(**row["question"]), row["candidates"], row["retrieval_latency_s"]))
    if [case.question for case in cases] != load_questions(questions_path) or len(cases) != manifest["question_count"]:
        raise ValueError("Frozen questions differ from the declared reference set")
    _validate_candidates(cases)
    return cases, manifest


def collect(args: argparse.Namespace) -> None:
    check_output(args.out, args.snapshot.parent)
    _validate_questions(args, args.questions)
    manifest = verify_manifest(args.snapshot)
    snapshot_digest = file_digest(args.snapshot)
    policy_date = date.today().isoformat()
    args.index_dir = args.snapshot.parent / "index"
    pipeline = _load_pipeline(args)
    try:
        cases = collect_raw_retrieval(load_questions(args.questions), pipeline,
                                      RetrievalCollectionConfig(args.retriever, args.candidate_k))
    finally:
        _release_retrieval_resources(pipeline)
    selected = {row["chunk_id"]: row["text"] for case in cases for row in case.candidates}
    matched = set()
    with (args.index_dir / "chunks_map.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["chunk_id"] in selected:
                if selected[row["chunk_id"]] != row["text"]:
                    raise ValueError("Candidate text differs from sealed corpus")
                matched.add(row["chunk_id"])
    verify_manifest(args.snapshot)
    if len(matched) != len(selected) or file_digest(args.snapshot) != snapshot_digest or policy_date != date.today().isoformat():
        raise ValueError("Corpus identity or applicability date changed during retrieval")
    provenance = {"snapshot_manifest_sha256": snapshot_digest,
                  "snapshot_collected_at": manifest["collected_at"],
                  "questions_sha256": file_digest(args.questions), "retriever": args.retriever,
                  "candidate_k": args.candidate_k, "query_expansion": False,
                  "applicability_policy_date": policy_date,
                  "embedding_config": json.loads((args.index_dir / "embedding_config.json").read_text(encoding="utf-8"))}
    result = save_pool(args.out, cases, provenance)
    print(json.dumps({"status": "frozen", "questions": len(cases), "pool_sha256": result["candidates_sha256"]}))


def local_adapter(args: argparse.Namespace):
    from huggingface_hub import hf_hub_download, snapshot_download

    spec = RERANKER_SPECS[args.model]
    if args.model == "none":
        return RetrievalOrderAdapter(), {}
    if args.model == "flashrank":
        archive = Path(hf_hub_download("prithivida/flashrank", "ms-marco-MultiBERT-L-12.zip",
                                      revision=FLASHRANK_REVISION, local_files_only=True))
        cache = (args.out / "model-cache").resolve()
        cache.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            if any(not (cache / member.filename).resolve().is_relative_to(cache) for member in zipped.infolist()):
                raise ValueError("Model archive path escapes cache")
            zipped.extractall(cache)
        adapter = FlashRankAdapter(spec, cache)
        artifacts = {"repo": "prithivida/flashrank", "revision": FLASHRANK_REVISION,
                     "archive_sha256": file_digest(archive)}
    else:
        root = Path(snapshot_download(spec.model_name, revision=spec.revision, local_files_only=True))
        artifacts = {"repo": spec.model_name, "revision": spec.revision,
                     "files": {path.relative_to(root).as_posix(): file_digest(path)
                               for path in sorted(root.rglob("*")) if path.is_file()}}
        adapter = LocalCrossEncoderAdapter(spec, CrossEncoderRuntime(args.device, args.batch_size, True))
    return adapter, artifacts


def replay(args: argparse.Namespace) -> None:
    check_output(args.out, args.pool)
    cases, manifest = load_pool(args.pool, args.questions)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        import torch

        torch.set_num_threads(args.torch_threads)
        adapter, artifacts = local_adapter(args)
        warm_case = next(case for case in cases if case.candidates)
        started = time.perf_counter()
        adapter.rank(warm_case.question.question, warm_case.candidates[:2])
        warmup_s = time.perf_counter() - started
        rss_loaded = _memory_rss_mb()
        config = BenchmarkConfig(candidate_pools=(manifest["provenance"]["candidate_k"],),
                                 top_ks=tuple(args.top_k), retriever=manifest["provenance"]["retriever"])
        result = run_benchmark(cases, [RERANKER_SPECS[args.model]], config,
                               BenchmarkDependencies(adapter_factory=lambda _: adapter))
        write_jsonl(result.summaries, args.out / "summary.jsonl")
        write_jsonl(result.observations, args.out / "observations.jsonl")
        import onnxruntime

        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: version(name) for name in ("torch", "sentence-transformers", "transformers", "flashrank", "onnxruntime")},
                   "requested_torch_device": args.device, "batch_size": args.batch_size, "max_length": 512,
                   "warmup_seconds": warmup_s, "rss_after_warmup_mb": rss_loaded,
                   "onnx_available_providers": onnxruntime.get_available_providers(),
                   "torch_cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
        if torch.cuda.is_available():
            runtime.update(cuda_device_name=torch.cuda.get_device_name(),
                           cuda_peak_allocated_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
                           cuda_peak_reserved_mb=torch.cuda.max_memory_reserved() / (1024 * 1024))
        if args.model == "flashrank":
            runtime["actual_providers"] = adapter._get_model().session.get_providers()
        elif args.model != "none":
            model = adapter._get_model()
            parameter = next(model.parameters())
            runtime.update(actual_device=str(parameter.device), dtype=str(parameter.dtype),
                           prompts=model.prompts, default_prompt_name=model.default_prompt_name)
        report = {"status": "completed", "schema_version": "1.0", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "model": asdict(RERANKER_SPECS[args.model]), "artifacts": artifacts, "runtime": runtime,
                  "pool_manifest_sha256": file_digest(args.pool / "pool.manifest.json"),
                  "pool_sha256": manifest["candidates_sha256"], "provenance": manifest["provenance"],
                  "summaries": [asdict(summary) for summary in result.summaries],
                  "candidate_counts": {"min": min(len(case.candidates) for case in cases),
                                       "max": max(len(case.candidates) for case in cases),
                                       "total": sum(len(case.candidates) for case in cases),
                                       "fewer_than_k": {str(k): sum(len(case.candidates) < k for case in cases) for k in args.top_k}},
                  "semantics": "Historical visible regression; keyword proxies only, no generation, no independent legal evaluation.",
                  "release_ready": False}
        _write_json(args.out / "run.json", report)
    except Exception as error:
        _write_json(args.out / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    print(json.dumps({"status": "completed", "model": args.model, "questions": len(cases)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--snapshot", type=Path, required=True)
    capture.add_argument("--retriever", choices=("faiss", "hybrid"), default="faiss")
    capture.add_argument("--candidate-k", type=int, default=20)
    capture.add_argument("--contract", default="eval/evaluation_contract.json")
    capture.add_argument("--allow-custom-questions", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--pool", type=Path, required=True)
    run.add_argument("--model", choices=("none", "flashrank", "bge", "qwen3"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--torch-threads", type=int, default=4)
    run.add_argument("--top-k", type=int, nargs="+", default=(5, 10))
    for command in (capture, run):
        command.add_argument("--questions", type=Path, default=Path("eval/test_questions.json"))
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.candidate_k < 1:
            parser.error("candidate-k must be positive")
        collect(args)
    else:
        if args.batch_size < 1 or args.torch_threads < 1 or any(value < 1 for value in args.top_k):
            parser.error("batch-size, torch-threads and top-k must be positive")
        replay(args)


if __name__ == "__main__":
    main()
