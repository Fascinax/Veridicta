"""Diagnostic ablation for error attribution across the RAG pipeline.

This script compares three aligned runs:
1) Retrieval-only baseline (raw retrieval)
2) Retrieval-only with reranker
3) Full RAG with reranker (generation enabled)

It annotates 30-50 failing questions with a taxonomy:
- retrieval_miss: quality already lost at raw retrieval stage
- rank_miss: reranking degrades otherwise acceptable retrieval
- gen_bad: retrieval/ranking acceptable but final generation still poor

Outputs:
- CSV annotation table
- Markdown breakdown summary
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    kw_gate: float
    f1_gate: float
    cov_gate: float
    rank_drop_tol: float


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _resolve_jsonl(path_or_dir: Path) -> Path:
    if path_or_dir.is_file():
        return path_or_dir
    candidates = sorted(path_or_dir.glob("eval_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No eval_*.jsonl found in: {path_or_dir}")
    return candidates[-1]


def _by_question_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows}


def _classify_error(raw_kw: float, rerank_kw: float, thresholds: Thresholds) -> str:
    if raw_kw < thresholds.kw_gate:
        return "retrieval_miss"

    if rerank_kw + thresholds.rank_drop_tol < raw_kw:
        return "rank_miss"

    if rerank_kw < thresholds.kw_gate <= raw_kw:
        return "rank_miss"

    return "gen_bad"


def _severity(full_row: dict, thresholds: Thresholds) -> float:
    full_kw = float(full_row.get("keyword_recall", 0.0) or 0.0)
    full_f1 = float(full_row.get("word_f1", 0.0) or 0.0)
    full_cov = float(full_row.get("context_coverage", 0.0) or 0.0)
    full_cit = float(full_row.get("citation_faithfulness", 0.0) or 0.0)

    kw_penalty = max(0.0, thresholds.kw_gate - full_kw)
    f1_penalty = max(0.0, thresholds.f1_gate - full_f1)
    cov_penalty = max(0.0, thresholds.cov_gate - full_cov)
    cit_penalty = max(0.0, 1.0 - full_cit)
    return (2.0 * kw_penalty) + (3.0 * f1_penalty) + cov_penalty + cit_penalty


def _is_error(full_row: dict, thresholds: Thresholds) -> bool:
    full_kw = float(full_row.get("keyword_recall", 0.0) or 0.0)
    full_f1 = float(full_row.get("word_f1", 0.0) or 0.0)
    full_cov = float(full_row.get("context_coverage", 0.0) or 0.0)
    full_cit = float(full_row.get("citation_faithfulness", 0.0) or 0.0)
    return (
        full_kw < thresholds.kw_gate
        or full_f1 < thresholds.f1_gate
        or full_cov < thresholds.cov_gate
        or full_cit < 1.0
    )


def _build_annotations(
    raw_by_id: dict[str, dict],
    rerank_by_id: dict[str, dict],
    full_by_id: dict[str, dict],
    thresholds: Thresholds,
    max_errors: int,
    min_errors: int,
) -> list[dict]:
    aligned_ids = sorted(set(raw_by_id) & set(rerank_by_id) & set(full_by_id))
    if not aligned_ids:
        raise ValueError("No overlapping question_id across the 3 input runs.")

    candidates: list[tuple[float, str]] = []
    backups: list[tuple[float, str]] = []
    for question_id in aligned_ids:
        full_row = full_by_id[question_id]
        sev = _severity(full_row, thresholds)
        backups.append((sev, question_id))
        if _is_error(full_row, thresholds):
            candidates.append((sev, question_id))

    candidates.sort(reverse=True)
    selected_ids = [question_id for _, question_id in candidates[:max_errors]]

    if len(selected_ids) < min_errors:
        backups.sort(reverse=True)
        for _, question_id in backups:
            if question_id in selected_ids:
                continue
            selected_ids.append(question_id)
            if len(selected_ids) >= min_errors:
                break

    selected_ids = selected_ids[:max_errors]

    annotations: list[dict] = []
    for question_id in selected_ids:
        raw_row = raw_by_id[question_id]
        rerank_row = rerank_by_id[question_id]
        full_row = full_by_id[question_id]

        raw_kw = float(raw_row.get("keyword_recall", 0.0) or 0.0)
        rerank_kw = float(rerank_row.get("keyword_recall", 0.0) or 0.0)
        full_kw = float(full_row.get("keyword_recall", 0.0) or 0.0)
        full_f1 = float(full_row.get("word_f1", 0.0) or 0.0)
        full_cov = float(full_row.get("context_coverage", 0.0) or 0.0)
        full_cit = float(full_row.get("citation_faithfulness", 0.0) or 0.0)

        taxonomy = _classify_error(raw_kw, rerank_kw, thresholds)

        annotations.append(
            {
                "question_id": question_id,
                "topic": str(full_row.get("topic", "general")),
                "taxonomy": taxonomy,
                "raw_kw": round(raw_kw, 4),
                "rerank_kw": round(rerank_kw, 4),
                "full_kw": round(full_kw, 4),
                "full_f1": round(full_f1, 4),
                "full_cov": round(full_cov, 4),
                "full_citation_faithfulness": round(full_cit, 4),
                "delta_raw_to_rerank": round(rerank_kw - raw_kw, 4),
                "delta_rerank_to_full": round(full_kw - rerank_kw, 4),
            }
        )

    annotations.sort(key=lambda row: (row["taxonomy"], row["question_id"]))
    return annotations


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "topic",
        "taxonomy",
        "raw_kw",
        "rerank_kw",
        "full_kw",
        "full_f1",
        "full_cov",
        "full_citation_faithfulness",
        "delta_raw_to_rerank",
        "delta_rerank_to_full",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_breakdown_md(rows: list[dict], csv_path: Path, settings: Thresholds) -> str:
    total = len(rows)
    counts = {
        "retrieval_miss": sum(1 for row in rows if row["taxonomy"] == "retrieval_miss"),
        "rank_miss": sum(1 for row in rows if row["taxonomy"] == "rank_miss"),
        "gen_bad": sum(1 for row in rows if row["taxonomy"] == "gen_bad"),
    }

    def pct(value: int) -> float:
        return round((100.0 * value / total), 1) if total else 0.0

    lines = [
        "# Ablation diagnostique — repartition des erreurs",
        "",
        f"- Nombre d'erreurs annotees: {total}",
        f"- Seuils: kw<{settings.kw_gate}, f1<{settings.f1_gate}, cov<{settings.cov_gate}",
        f"- Source des annotations: `{csv_path.as_posix()}`",
        "",
        "| Taxonomie | Nombre | Pourcentage |",
        "|---|---:|---:|",
        f"| retrieval_miss | {counts['retrieval_miss']} | {pct(counts['retrieval_miss'])}% |",
        f"| rank_miss | {counts['rank_miss']} | {pct(counts['rank_miss'])}% |",
        f"| gen_bad | {counts['gen_bad']} | {pct(counts['gen_bad'])}% |",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic ablation: retrieval -> reranking -> generation")
    parser.add_argument(
        "--retrieval-base",
        type=Path,
        default=Path("eval/results/diag_ablation/retrieval_base"),
        help="JSONL file or folder for retrieval-only baseline run",
    )
    parser.add_argument(
        "--retrieval-rerank",
        type=Path,
        default=Path("eval/results/diag_ablation/retrieval_rerank"),
        help="JSONL file or folder for retrieval-only reranker run",
    )
    parser.add_argument(
        "--full-rag",
        type=Path,
        default=Path("eval/results/copilot-hybrid-bm25s-promptv3-k8-reranker/eval_20260309_130240.jsonl"),
        help="JSONL file or folder for full RAG run (with generation)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("eval/results/diag_ablation/error_annotations.csv"),
        help="Output CSV annotations",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("eval/results/diag_ablation/error_breakdown.md"),
        help="Output Markdown breakdown",
    )
    parser.add_argument("--kw-gate", type=float, default=0.6)
    parser.add_argument("--f1-gate", type=float, default=0.25)
    parser.add_argument("--cov-gate", type=float, default=0.55)
    parser.add_argument("--rank-drop-tol", type=float, default=0.2)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--min-errors", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        kw_gate=args.kw_gate,
        f1_gate=args.f1_gate,
        cov_gate=args.cov_gate,
        rank_drop_tol=args.rank_drop_tol,
    )

    raw_path = _resolve_jsonl(args.retrieval_base)
    rerank_path = _resolve_jsonl(args.retrieval_rerank)
    full_path = _resolve_jsonl(args.full_rag)

    raw_rows = _load_jsonl(raw_path)
    rerank_rows = _load_jsonl(rerank_path)
    full_rows = _load_jsonl(full_path)

    annotations = _build_annotations(
        raw_by_id=_by_question_id(raw_rows),
        rerank_by_id=_by_question_id(rerank_rows),
        full_by_id=_by_question_id(full_rows),
        thresholds=thresholds,
        max_errors=max(1, args.max_errors),
        min_errors=max(1, args.min_errors),
    )

    _write_csv(annotations, args.out_csv)

    markdown = _build_breakdown_md(annotations, args.out_csv, thresholds)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")

    counts = {
        "retrieval_miss": sum(1 for row in annotations if row["taxonomy"] == "retrieval_miss"),
        "rank_miss": sum(1 for row in annotations if row["taxonomy"] == "rank_miss"),
        "gen_bad": sum(1 for row in annotations if row["taxonomy"] == "gen_bad"),
    }
    print("Diagnostic ablation complete")
    print(f"  retrieval_base:   {raw_path}")
    print(f"  retrieval_rerank: {rerank_path}")
    print(f"  full_rag:         {full_path}")
    print(f"  annotations:      {args.out_csv} ({len(annotations)} rows)")
    print(f"  breakdown:        {args.out_md}")
    print(
        "  counts: "
        f"retrieval_miss={counts['retrieval_miss']}, "
        f"rank_miss={counts['rank_miss']}, "
        f"gen_bad={counts['gen_bad']}"
    )


if __name__ == "__main__":
    main()
