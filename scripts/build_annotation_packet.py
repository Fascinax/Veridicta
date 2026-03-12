"""Build Stage 0 annotation packet from eval results and top20 snapshots.

Joins:
  - eval/test_questions_stage0_bottom40.json        → reference answers
  - eval/results/stage0/lancedb_graph_full/*.jsonl  → LG metrics + answer
  - eval/results/stage0/hybrid_graph_full/*.jsonl   → HG metrics + answer
  - eval/results/stage0/lancedb_graph_top20.jsonl   → top-20 chunks LG
  - eval/results/stage0/hybrid_graph_top20.jsonl    → top-20 chunks HG

Output: eval/results/stage0/annotation_packet.jsonl
        eval/results/stage0/annotation_packet_review.md  (human-readable)

Chunk taxonomy fields added:
  used_in_prompt: bool  → retrieval_rank <= K (default k=5)
"""

import json
import glob
import pathlib
import sys

K = 5
BASE = pathlib.Path(__file__).parent.parent

QUESTIONS_FILE = BASE / "eval" / "test_questions_stage0_bottom40.json"
LG_FULL_GLOB   = str(BASE / "eval/results/stage0/lancedb_graph_full/eval_*.jsonl")
HG_FULL_GLOB   = str(BASE / "eval/results/stage0/hybrid_graph_full/eval_*.jsonl")
LG_TOP20_FILE  = BASE / "eval/results/stage0/lancedb_graph_top20.jsonl"
HG_TOP20_FILE  = BASE / "eval/results/stage0/hybrid_graph_top20.jsonl"
OUT_JSONL      = BASE / "eval/results/stage0/annotation_packet.jsonl"
OUT_MD         = BASE / "eval/results/stage0/annotation_packet_review.md"


def load_jsonl(path, key_field):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records[d[key_field]] = d
    return records


def load_latest_jsonl_glob(pattern, key_field):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No file found for pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    return load_jsonl(files[-1], key_field)


def annotate_chunks(chunks):
    for c in chunks:
        c["used_in_prompt"] = c["retrieval_rank"] <= K
    return chunks


def build_packet():
    questions = json.load(open(QUESTIONS_FILE, encoding="utf-8"))
    lg_full   = load_latest_jsonl_glob(LG_FULL_GLOB, "question_id")
    hg_full   = load_latest_jsonl_glob(HG_FULL_GLOB, "question_id")
    lg_top20  = load_jsonl(str(LG_TOP20_FILE), "question_id")
    hg_top20  = load_jsonl(str(HG_TOP20_FILE), "question_id")

    packets = []
    missing_lg = []
    missing_hg = []

    for q in questions:
        qid = q["id"]
        lg  = lg_full.get(qid, {})
        hg  = hg_full.get(qid, {})
        lt  = lg_top20.get(qid, {})
        ht  = hg_top20.get(qid, {})

        if not lg:
            missing_lg.append(qid)
        if not hg:
            missing_hg.append(qid)

        packet = {
            "question_id":       qid,
            "question":          q["question"],
            "reference_answer":  q["reference_answer"],
            "reference_keywords": q.get("reference_keywords", []),
            "lancedb_graph": {
                "word_f1":            lg.get("word_f1"),
                "keyword_recall":     lg.get("keyword_recall"),
                "context_coverage":   lg.get("context_coverage"),
                "citation_faithfulness": lg.get("citation_faithfulness"),
                "n_retrieved":        lg.get("n_retrieved"),
                "answer":             lg.get("answer"),
                "sources_titles":     lg.get("sources_titles", []),
                "top20_chunks":       annotate_chunks(lt.get("chunks", [])),
            },
            "hybrid_graph": {
                "word_f1":            hg.get("word_f1"),
                "keyword_recall":     hg.get("keyword_recall"),
                "context_coverage":   hg.get("context_coverage"),
                "citation_faithfulness": hg.get("citation_faithfulness"),
                "n_retrieved":        hg.get("n_retrieved"),
                "answer":             hg.get("answer"),
                "sources_titles":     hg.get("sources_titles", []),
                "top20_chunks":       annotate_chunks(ht.get("chunks", [])),
            },
            # Annotation fields — to be filled manually
            "error_label":       "",   # retrieval_absent | retrieval_present_wrong_passage | ranking_bad_order | injected_but_bad_use | semantically_ok_but_metric_penalty
            "error_notes":       "",
        }
        packets.append(packet)

    if missing_lg:
        print(f"[WARN] {len(missing_lg)} questions missing from lancedb_graph full run: {missing_lg[:5]}")
    if missing_hg:
        print(f"[WARN] {len(missing_hg)} questions missing from hybrid_graph full run: {missing_hg[:5]}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[OK] annotation_packet.jsonl written: {len(packets)} records → {OUT_JSONL}")
    return packets


def build_review_md(packets):
    lines = [
        "# Stage 0 — Annotation Packet Review",
        "",
        f"**{len(packets)} questions** — worst-40 subset (lancedb_graph baseline)",
        "",
        "| # | Question | LG F1 | HG F1 | LG top1 title | HG top1 title |",
        "|---|----------|-------|-------|---------------|---------------|",
    ]
    for i, p in enumerate(packets, 1):
        lg = p["lancedb_graph"]
        hg = p["hybrid_graph"]
        q_short = p["question"][:60].replace("|", "/")
        lg_f1 = f"{lg['word_f1']:.3f}" if lg["word_f1"] is not None else "—"
        hg_f1 = f"{hg['word_f1']:.3f}" if hg["word_f1"] is not None else "—"
        lg_top1 = (lg["top20_chunks"][0]["titre"] or lg["top20_chunks"][0]["title"] or "?")[:40] if lg["top20_chunks"] else "—"
        hg_top1 = (hg["top20_chunks"][0]["titre"] or hg["top20_chunks"][0]["title"] or "?")[:40] if hg["top20_chunks"] else "—"
        lines.append(f"| {i} | {q_short} | {lg_f1} | {hg_f1} | {lg_top1} | {hg_top1} |")

    lines += [
        "",
        "## Error taxonomy",
        "",
        "- `retrieval_absent` — none of the top-20 chunks contain the answer",
        "- `retrieval_present_wrong_passage` — relevant doc exists but not retrieved at all",
        "- `ranking_bad_order` — relevant chunk present in top-20 but at rank > 5 (not injected)",
        "- `injected_but_bad_use` — relevant chunk was in top-5 but LLM answer ignored/hallucinated",
        "- `semantically_ok_but_metric_penalty` — answer is correct but word_f1 penalises formulation",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] annotation_packet_review.md written → {OUT_MD}")


if __name__ == "__main__":
    packets = build_packet()
    build_review_md(packets)
