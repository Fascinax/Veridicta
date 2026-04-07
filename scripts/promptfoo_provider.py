from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrievers.baseline_rag import _load_embedder, answer, load_index
from retrievers.pipeline import RetrievalPipeline

try:
    from retrievers.hybrid_rag import load_bm25_index
except ImportError:
    load_bm25_index = None

try:
    from retrievers.graph_rag import load_neo4j_manager
except ImportError:
    load_neo4j_manager = None

try:
    from retrievers.lancedb_rag import load_lancedb_index
except ImportError:
    load_lancedb_index = None


class PromptfooProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.embedder = _load_embedder()
        self.index = None
        self.chunks = None
        self.bm25 = None
        self.neo4j_manager = None
        self.lancedb_table = None

        if args.retriever in {"faiss", "hybrid", "graph", "hybrid_graph"}:
            self.index, self.chunks = load_index(Path(args.index_dir))

        if args.retriever in {"hybrid", "hybrid_graph"}:
            if load_bm25_index is None:
                raise RuntimeError("Retriever 'hybrid' unavailable. Install bm25s + PyStemmer.")
            self.bm25 = load_bm25_index(Path(args.index_dir))

        if args.retriever in {"graph", "hybrid_graph", "lancedb_graph"}:
            if load_neo4j_manager is None:
                raise RuntimeError("Retriever with graph support unavailable.")
            self.neo4j_manager = load_neo4j_manager()
            if self.neo4j_manager is None:
                raise RuntimeError(
                    "Neo4j unreachable. Check NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD."
                )

        if args.retriever in {"lancedb", "lancedb_graph"}:
            if load_lancedb_index is None:
                raise RuntimeError("Retriever 'lancedb' unavailable. Install lancedb.")
            self.lancedb_table = load_lancedb_index()

        self.pipeline = RetrievalPipeline(
            embedder=self.embedder,
            index=self.index,
            chunks=self.chunks,
            bm25=self.bm25,
            neo4j_manager=self.neo4j_manager,
            lancedb_table=self.lancedb_table,
        )

    def run(self, query: str) -> str:
        if self.args.targeted_patch:
            os.environ["VERIDICTA_PROMPT_V3_TARGETED_PATCH"] = "1"
        else:
            os.environ.pop("VERIDICTA_PROMPT_V3_TARGETED_PATCH", None)

        context_chunks = self.pipeline.retrieve(
            query,
            retriever=self.args.retriever,
            k=self.args.k,
            query_expansion=self.args.query_expansion,
            use_reranker=self.args.reranker,
            reranker_candidate_multiplier=self.args.reranker_candidate_multiplier,
            reranker_min_score=self.args.reranker_min_score,
            hybrid_faiss_weight=self.args.hybrid_faiss_weight,
            hybrid_bm25_weight=self.args.hybrid_bm25_weight,
        )
        return answer(
            query,
            context_chunks,
            model=self.args.model,
            backend=self.args.backend,
            prompt_version=self.args.prompt_version,
        )


def _extract_prompt(stdin_payload: str) -> str:
    raw = stdin_payload.strip()
    if not raw:
        return ""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("prompt", "input", "question"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        vars_payload = payload.get("vars")
        if isinstance(vars_payload, dict):
            question = vars_payload.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()
            nested_vars = vars_payload.get("vars")
            if isinstance(nested_vars, dict):
                nested_question = nested_vars.get("question")
                if isinstance(nested_question, str) and nested_question.strip():
                    return nested_question.strip()
        test_payload = payload.get("test")
        if isinstance(test_payload, dict):
            test_vars = test_payload.get("vars")
            if isinstance(test_vars, dict):
                nested_question = test_vars.get("question")
                if isinstance(nested_question, str) and nested_question.strip():
                    return nested_question.strip()
                nested_vars = test_vars.get("vars")
                if isinstance(nested_vars, dict):
                    nested_question_2 = nested_vars.get("question")
                    if isinstance(nested_question_2, str) and nested_question_2.strip():
                        return nested_question_2.strip()
    return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promptfoo exec provider for Veridicta RAG")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument(
        "--retriever",
        default="lancedb_graph",
        choices=["faiss", "hybrid", "graph", "hybrid_graph", "lancedb", "lancedb_graph"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-expansion", action="store_true", default=True)
    parser.add_argument("--reranker", action="store_true", default=False)
    parser.add_argument("--reranker-candidate-multiplier", type=int, default=4)
    parser.add_argument("--reranker-min-score", type=float, default=None)
    parser.add_argument("--hybrid-faiss-weight", type=float, default=None)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=None)
    parser.add_argument("--prompt-version", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--backend", default=None, choices=["copilot", "cerebras"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--targeted-patch", action="store_true")
    return parser


def _extract_fallback_prompt_from_args(extra_args: list[str]) -> str:
    if not extra_args:
        return ""
    # Promptfoo exec can pass one or more JSON payload args after the command.
    for arg in extra_args:
        extracted = _extract_prompt(arg)
        if extracted:
            return extracted
    return ""


def main() -> None:
    parser = _build_parser()
    args, extra_args = parser.parse_known_args()

    stdin_payload = "" if sys.stdin.isatty() else sys.stdin.read()
    query = _extract_prompt(stdin_payload)
    if not query:
        query = _extract_fallback_prompt_from_args(extra_args)
    if not query:
        raise SystemExit("Empty prompt input received from Promptfoo")

    provider = PromptfooProvider(args)
    response = provider.run(query)
    print(response)


if __name__ == "__main__":
    main()
