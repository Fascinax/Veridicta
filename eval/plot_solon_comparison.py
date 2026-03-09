"""
Generate comparison charts: Solon 1024d vs Baseline MiniLM 384d.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jsonlines


def avg(data, key):
    vals = [d.get(key, 0) for d in data]
    return sum(vals) / len(vals)


def main():
    # Load results
    solon = Path("eval/results/solon-full/eval_20260309_112112.jsonl")
    baseline = Path("eval/results/copilot-hybrid-bm25s/eval_20260309_005557.jsonl")

    with jsonlines.open(solon) as r:
        solon_data = list(r)
    with jsonlines.open(baseline) as r:
        base_data = list(r)

    # Compute averages
    metrics = {
        "Baseline\n(MiniLM 384d)": {
            "KW": avg(base_data, "keyword_recall"),
            "F1": avg(base_data, "word_f1"),
            "CitFaith": avg(base_data, "citation_faithfulness"),
            "CtxCov": avg(base_data, "context_coverage"),
            "Lat": avg(base_data, "latency_s"),
        },
        "Solon\n(1024d)": {
            "KW": avg(solon_data, "keyword_recall"),
            "F1": avg(solon_data, "word_f1"),
            "CitFaith": avg(solon_data, "citation_faithfulness"),
            "CtxCov": avg(solon_data, "context_coverage"),
            "Lat": avg(solon_data, "latency_s"),
        },
    }

    # Chart 1: Grouped bars + latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Phase 16: Solon embeddings vs Baseline (100Q, hybrid retriever)",
        fontsize=14,
        fontweight="bold",
    )

    labels = list(metrics.keys())
    kw = [metrics[l]["KW"] for l in labels]
    f1 = [metrics[l]["F1"] for l in labels]
    cit = [metrics[l]["CitFaith"] for l in labels]
    ctx = [metrics[l]["CtxCov"] for l in labels]

    x = np.arange(len(labels))
    width = 0.2

    bars1 = ax1.bar(x - 1.5 * width, kw, width, label="KW Recall", color="#3498db")
    bars2 = ax1.bar(x - 0.5 * width, f1, width, label="Word F1", color="#2ecc71")
    bars3 = ax1.bar(x + 0.5 * width, cit, width, label="Cit. Faith", color="#e74c3c")
    bars4 = ax1.bar(x + 1.5 * width, ctx, width, label="Ctx Cov", color="#f39c12")

    ax1.set_ylabel("Score")
    ax1.set_title("Quality metrics (higher = better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Chart 2: Latency comparison
    lats = [metrics[l]["Lat"] for l in labels]
    bars = ax2.bar(labels, lats, color=["#95a5a6", "#9b59b6"])
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_title("Average latency per question")
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.3,
            f"{h:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_dir = Path("eval/charts/solon-comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "solon_vs_baseline.png", dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out_dir / 'solon_vs_baseline.png'}")

    # Print summary
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY (100 questions, hybrid retriever, copilot/gpt-4.1)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline (MiniLM)':<20} {'Solon (1024d)':<20} {'Delta':<15}")
    print("-" * 70)
    for metric in ["KW", "F1", "CitFaith", "CtxCov", "Lat"]:
        base_val = metrics["Baseline\n(MiniLM 384d)"][metric]
        solon_val = metrics["Solon\n(1024d)"][metric]
        delta = ((solon_val / base_val) - 1) * 100
        print(f"{metric:<20} {base_val:<20.3f} {solon_val:<20.3f} {delta:+.1f}%")


if __name__ == "__main__":
    main()
