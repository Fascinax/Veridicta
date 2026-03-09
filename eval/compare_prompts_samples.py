"""Compare sample answers between prompt v1, v2, and v3."""
import jsonlines
from pathlib import Path


def main():
    # Load eval results
    v1 = Path("eval/results/copilot-hybrid-bm25s/eval_20260309_005557.jsonl")
    v2 = Path("eval/results/copilot-hybrid-bm25s-promptv2/eval_20260309_011026.jsonl")
    v3 = Path("eval/results/copilot-hybrid-bm25s-promptv3/eval_20260309_113848.jsonl")

    with jsonlines.open(v1) as r:
        v1_data = list(r)
    with jsonlines.open(v2) as r:
        v2_data = list(r)
    with jsonlines.open(v3) as r:
        v3_data = list(r)

    # Take 3 sample questions to compare answers
    print("=" * 80)
    print("SAMPLE COMPARISON — Prompt v1 vs v2 vs v3")
    print("=" * 80)
    for i in [0, 10, 20]:
        q1 = v1_data[i]
        q2 = v2_data[i]
        q3 = v3_data[i]

        question_text = q1["question"][:80]
        print(f"\n[Question {i+1}] {question_text}...")

        print(f"\nv1: KW={q1['keyword_recall']:.2f} F1={q1['word_f1']:.3f} | {len(q1['answer'])} chars")
        print("Answer:", q1["answer"][:250].replace("\n", " "))

        print(f"\nv2: KW={q2['keyword_recall']:.2f} F1={q2['word_f1']:.3f} | {len(q2['answer'])} chars")
        print("Answer:", q2["answer"][:250].replace("\n", " "))

        print(f"\nv3: KW={q3['keyword_recall']:.2f} F1={q3['word_f1']:.3f} | {len(q3['answer'])} chars")
        print("Answer:", q3["answer"][:250].replace("\n", " "))

        print("-" * 80)

    # Overall stats
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    v1_avg_kw = sum(r['keyword_recall'] for r in v1_data) / len(v1_data)
    v2_avg_kw = sum(r['keyword_recall'] for r in v2_data) / len(v2_data)
    v3_avg_kw = sum(r['keyword_recall'] for r in v3_data) / len(v3_data)

    v1_avg_f1 = sum(r['word_f1'] for r in v1_data) / len(v1_data)
    v2_avg_f1 = sum(r['word_f1'] for r in v2_data) / len(v2_data)
    v3_avg_f1 = sum(r['word_f1'] for r in v3_data) / len(v3_data)

    v1_avg_len = sum(len(r['answer']) for r in v1_data) / len(v1_data)
    v2_avg_len = sum(len(r['answer']) for r in v2_data) / len(v2_data)
    v3_avg_len = sum(len(r['answer']) for r in v3_data) / len(v3_data)

    print(f"\nPrompt v1 (original):           KW={v1_avg_kw:.3f}, F1={v1_avg_f1:.3f}, Avg={v1_avg_len:.0f} chars")
    print(f"Prompt v2 (structured):         KW={v2_avg_kw:.3f}, F1={v2_avg_f1:.3f}, Avg={v2_avg_len:.0f} chars")
    print(f"Prompt v3 (exhaustive+concise): KW={v3_avg_kw:.3f}, F1={v3_avg_f1:.3f}, Avg={v3_avg_len:.0f} chars")

    print(f"\n[v2 vs v1] KW: {(v2_avg_kw/v1_avg_kw-1)*100:+.1f}%, F1: {(v2_avg_f1/v1_avg_f1-1)*100:+.1f}%")
    print(f"[v3 vs v1] KW: {(v3_avg_kw/v1_avg_kw-1)*100:+.1f}%, F1: {(v3_avg_f1/v1_avg_f1-1)*100:+.1f}%")
    print(f"[v3 vs v2] KW: {(v3_avg_kw/v2_avg_kw-1)*100:+.1f}%, F1: {(v3_avg_f1/v2_avg_f1-1)*100:+.1f}%")


if __name__ == "__main__":
    main()

