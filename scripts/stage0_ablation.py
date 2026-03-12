from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import _build_retrieval_only_results, _load_optional_retrievers, _retrieve_contexts, _write_results_file, load_questions
from retrievers.baseline_rag import _load_embedder, load_index


@dataclass(frozen=True)
class DefaultPaths:
    source_run: Path = Path("eval/results/copilot-lancedb-graph/eval_20260311_003251.jsonl")
    questions: Path = Path("eval/test_questions.json")
    subset: Path = Path("eval/test_questions_stage0_bottom40.json")
    chunks: Path = Path("data/processed/chunks.jsonl")
    index_dir: Path = Path("data/index")


DEFAULT_PATHS = DefaultPaths()


def _read_jsonl(file_path: Path) -> list[dict]:
    with file_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_json(file_path: Path, payload: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(file_path: Path, rows: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    file_path.write_text(f"{content}\n", encoding="utf-8")


def _latest_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("eval_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No eval_*.jsonl found in {directory}")
    return candidates[-1]


def _question_id_by_text(questions: list[dict]) -> dict[str, str]:
    return {question["question"]: question["id"] for question in questions}


def make_subset(source_run: Path, questions_path: Path, output_path: Path, count: int) -> None:
    scored_rows = _read_jsonl(source_run)
    worst_ids = {
        row["question_id"]
        for row in sorted(
            scored_rows,
            key=lambda row: row["word_f1"] if row.get("word_f1") is not None else 999.0,
        )[:count]
    }
    with questions_path.open(encoding="utf-8") as handle:
        all_questions = json.load(handle)
    subset = [question for question in all_questions if question["id"] in worst_ids]
    _write_json(output_path, subset)
    print(f"Subset written -> {output_path} ({len(subset)} questions)")


def export_topk(subset_path: Path, retriever: str, output_path: Path, k: int, index_dir: Path) -> None:
    questions = load_questions(subset_path)
    index, chunks = load_index(index_dir)
    embedder = _load_embedder()
    args = argparse.Namespace(retriever=retriever)
    bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(args, index_dir)
    retrieved_all = _retrieve_contexts(
        questions,
        index,
        chunks,
        embedder,
        k=k,
        bm25=bm25,
        neo4j_mgr=neo4j_mgr,
        lancedb_table=lancedb_table,
    )
    rows = [
        {
            "question_id": question.id,
            "question": question.question,
            "chunks": retrieved,
        }
        for question, retrieved in zip(questions, retrieved_all)
    ]
    _write_jsonl(output_path, rows)
    print(f"Top-{k} snapshot written -> {output_path}")


def run_retrieval_only(subset_path: Path, retriever: str, output_dir: Path, k: int, index_dir: Path) -> None:
    questions = load_questions(subset_path)
    index, chunks = load_index(index_dir)
    embedder = _load_embedder()
    args = argparse.Namespace(retriever=retriever)
    bm25, neo4j_mgr, lancedb_table = _load_optional_retrievers(args, index_dir)
    retrieved_all = _retrieve_contexts(
        questions,
        index,
        chunks,
        embedder,
        k=k,
        bm25=bm25,
        neo4j_mgr=neo4j_mgr,
        lancedb_table=lancedb_table,
    )
    results = _build_retrieval_only_results(questions, retrieved_all)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.jsonl"
    _write_results_file(results, output_path)
    print(f"Retrieval-only results written -> {output_path}")


def _load_chunk_texts(chunks_path: Path) -> dict[str, dict]:
    chunk_index: dict[str, dict] = {}
    for row in _read_jsonl(chunks_path):
        chunk_index[row["chunk_id"]] = {
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "titre": row.get("titre", row.get("title", "")),
            "text": row.get("text", ""),
            "date": row.get("date", ""),
            "type": row.get("type", ""),
        }
    return chunk_index


def _merge_chunk_text(chunk_meta: dict, chunk_index: dict[str, dict]) -> dict:
    merged = dict(chunk_meta)
    chunk_id = chunk_meta.get("chunk_id")
    merged["text"] = chunk_index.get(chunk_id, {}).get("text", "")
    return merged


def build_packet(
    subset_path: Path,
    results_dir: Path,
    audit_path: Path,
    topk_path: Path,
    chunks_path: Path,
    output_path: Path,
) -> None:
    with subset_path.open(encoding="utf-8") as handle:
        subset_questions = json.load(handle)
    questions_by_id = {question["id"]: question for question in subset_questions}
    question_ids_by_text = _question_id_by_text(subset_questions)
    chunk_index = _load_chunk_texts(chunks_path)
    latest_results = _latest_jsonl(results_dir)
    results = {row["question_id"]: row for row in _read_jsonl(latest_results)}
    topk = {row["question_id"]: row["chunks"] for row in _read_jsonl(topk_path)}

    packet_rows: list[dict] = []
    for audit in _read_jsonl(audit_path):
        query_text = audit.get("query", {}).get("text", "")
        question_id = question_ids_by_text.get(query_text)
        if question_id is None:
            continue
        used_chunks = [
            _merge_chunk_text(chunk_meta, chunk_index)
            for chunk_meta in audit["retrieval"]["chunks"]
            if chunk_meta.get("used_in_prompt")
        ]
        question = questions_by_id[question_id]
        result = results[question_id]
        packet_rows.append(
            {
                "question_id": question_id,
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "reference_keywords": question["reference_keywords"],
                "prediction": result["answer"],
                "word_f1": result["word_f1"],
                "keyword_recall": result["keyword_recall"],
                "citation_faithfulness": result["citation_faithfulness"],
                "context_coverage": result["context_coverage"],
                "top20_chunks": topk.get(question_id, []),
                "used_chunks": used_chunks,
            }
        )
    _write_jsonl(output_path, packet_rows)
    print(f"Annotation packet written -> {output_path} ({len(packet_rows)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0 ablation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset_parser = subparsers.add_parser("make-subset")
    subset_parser.add_argument("--source-run", type=Path, default=DEFAULT_PATHS.source_run)
    subset_parser.add_argument("--questions", type=Path, default=DEFAULT_PATHS.questions)
    subset_parser.add_argument("--out", type=Path, default=DEFAULT_PATHS.subset)
    subset_parser.add_argument("--count", type=int, default=40)

    topk_parser = subparsers.add_parser("export-topk")
    topk_parser.add_argument("--subset", type=Path, default=DEFAULT_PATHS.subset)
    topk_parser.add_argument("--retriever", choices=["faiss", "hybrid", "graph", "hybrid_graph", "lancedb", "lancedb_graph"], required=True)
    topk_parser.add_argument("--out", type=Path, required=True)
    topk_parser.add_argument("--k", type=int, default=20)
    topk_parser.add_argument("--index-dir", type=Path, default=DEFAULT_PATHS.index_dir)

    retrieval_only_parser = subparsers.add_parser("retrieval-only")
    retrieval_only_parser.add_argument("--subset", type=Path, default=DEFAULT_PATHS.subset)
    retrieval_only_parser.add_argument("--retriever", choices=["faiss", "hybrid", "graph", "hybrid_graph", "lancedb", "lancedb_graph"], required=True)
    retrieval_only_parser.add_argument("--out-dir", type=Path, required=True)
    retrieval_only_parser.add_argument("--k", type=int, default=5)
    retrieval_only_parser.add_argument("--index-dir", type=Path, default=DEFAULT_PATHS.index_dir)

    packet_parser = subparsers.add_parser("build-packet")
    packet_parser.add_argument("--subset", type=Path, default=DEFAULT_PATHS.subset)
    packet_parser.add_argument("--results-dir", type=Path, required=True)
    packet_parser.add_argument("--audit", type=Path, required=True)
    packet_parser.add_argument("--topk", type=Path, required=True)
    packet_parser.add_argument("--chunks", type=Path, default=DEFAULT_PATHS.chunks)
    packet_parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-subset":
        make_subset(args.source_run, args.questions, args.out, args.count)
        return
    if args.command == "export-topk":
        export_topk(args.subset, args.retriever, args.out, args.k, args.index_dir)
        return
    if args.command == "retrieval-only":
        run_retrieval_only(args.subset, args.retriever, args.out_dir, args.k, args.index_dir)
        return
    build_packet(args.subset, args.results_dir, args.audit, args.topk, args.chunks, args.out)


if __name__ == "__main__":
    main()