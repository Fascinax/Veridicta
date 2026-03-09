from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from retrievers.baseline_rag import _embed_query, _load_embedder, load_index, retrieve
from retrievers.hybrid_rag import hybrid_retrieve, load_bm25_index


def print_top_hot_paths(profile: cProfile.Profile, title: str, top_n: int = 20) -> None:
    print(f"\n=== {title} ===")
    stats = pstats.Stats(profile).strip_dirs().sort_stats("cumulative")
    stats.print_stats(top_n)


def main() -> None:
    out_dir = Path("eval/results/profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    query = "Quelles sont les indemnités de licenciement en droit monégasque?"

    embedder = _load_embedder()
    index, chunks = load_index(Path("data/index"))
    bm25 = load_bm25_index(Path("data/index"))

    embedding_profile_path = out_dir / "embedding_query.prof"
    embedding_profile = cProfile.Profile()
    embedding_profile.enable()
    for _ in range(20):
        _embed_query(query, embedder)
    embedding_profile.disable()
    embedding_profile.dump_stats(str(embedding_profile_path))
    print_top_hot_paths(embedding_profile, "TOP HOT PATHS: EMBEDDING (_embed_query x20)")

    faiss_profile_path = out_dir / "retrieval_faiss.prof"
    faiss_profile = cProfile.Profile()
    faiss_profile.enable()
    for _ in range(20):
        retrieve(query, index, chunks, embedder, k=8)
    faiss_profile.disable()
    faiss_profile.dump_stats(str(faiss_profile_path))
    print_top_hot_paths(faiss_profile, "TOP HOT PATHS: RETRIEVAL FAISS (retrieve x20)")

    hybrid_profile_path = out_dir / "retrieval_hybrid.prof"
    hybrid_profile = cProfile.Profile()
    hybrid_profile.enable()
    for _ in range(20):
        hybrid_retrieve(
            query,
            faiss_index=index,
            bm25=bm25,
            chunks=chunks,
            embedder=embedder,
            k=8,
        )
    hybrid_profile.disable()
    hybrid_profile.dump_stats(str(hybrid_profile_path))
    print_top_hot_paths(hybrid_profile, "TOP HOT PATHS: RETRIEVAL HYBRID (hybrid_retrieve x20)")

    print("\nSaved profiles:")
    print(f"- {embedding_profile_path}")
    print(f"- {faiss_profile_path}")
    print(f"- {hybrid_profile_path}")


if __name__ == "__main__":
    main()
