"""Comprehensive architecture comparison charts for Veridicta.

Produces 6 charts in eval/charts/architectures/:
  1. Grouped bar chart — all quality metrics, 5 key configs
  2. Radar chart — multi-dim profile, 4 copilot architectures
  3. Quality × Latency scatter — all configs, bubble size = Word F1
  4. Per-question scatter — KW Recall vs Context Coverage, main 4
  5. Progression chart — step-by-step gain FAISS → Hybrid → Graph → Hybrid+Graph
  6. Per-topic heatmap — KW Recall by topic, main 4 copilot architectures

Usage:
    python -m eval.plot_architectures
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval/results")
OUTPUT_DIR = Path("eval/charts/architectures")

# ── Config registry ───────────────────────────────────────────────────────────

# (label, dir, color, marker)
ALL_CONFIGS: list[tuple[str, str, str, str]] = [
    ("FAISS",             "copilot-faiss",             "#4A90D9", "o"),
    ("Hybrid\n(BM25+FAISS)", "copilot-hybrid-v3corpus",  "#E8833A", "s"),
    ("Graph\n(Neo4j)",    "copilot-graph",             "#59A14F", "^"),
    ("Hybrid+Graph",      "copilot-hybrid-graph",      "#9B59B6", "D"),
    ("LightRAG\n(copilot)", "copilot-graph-lightrag",  "#76B7B2", "P"),
    ("Cerebras\nHybrid",  "cerebras-hybrid-tuned",     "#E15759", "v"),
]

# Subset used in focused copilot-only charts
COPILOT_CONFIGS = [c for c in ALL_CONFIGS if "Cerebras" not in c[0] and "LightRAG" not in c[0]]

METRICS = [
    ("keyword_recall",        "KW Recall",        True),
    ("word_f1",               "Word F1",          True),
    ("citation_faithfulness", "Citation Faith.",  True),
    ("context_coverage",      "Context Coverage", True),
    ("hallucination_risk",    "Hallucin. Risk",   False),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#161b27",
    "axes.labelcolor": "#c9d1e0",
    "axes.titlecolor": "#e8d5a3",
    "xtick.color": "#8892ab",
    "ytick.color": "#8892ab",
    "text.color": "#c9d1e0",
    "legend.facecolor": "#1a1d2e",
    "legend.edgecolor": "#2a2f47",
    "grid.color": "#2a2f47",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_rows(dirname: str) -> list[dict]:
    pattern = str(RESULTS_DIR / dirname / "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL in eval/results/{dirname}/")
    rows = []
    with open(files[-1], encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("question_id") != "OVERALL"]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary(rows: list[dict]) -> dict[str, float]:
    s: dict[str, float] = {}
    for key, _, _ in METRICS:
        s[key] = _avg([r[key] for r in rows if key in r])
    s["latency_s"] = _avg([r["latency_s"] for r in rows if "latency_s" in r])
    return s


def _load_all(configs: list[tuple]) -> dict[str, tuple[list[dict], dict[str, float]]]:
    result = {}
    for label, dirname, color, marker in configs:
        try:
            rows = _load_rows(dirname)
            result[label] = (rows, _summary(rows))
            print(f"  {label.replace(chr(10),' '):25s}  N={len(rows):3d}  KW={result[label][1]['keyword_recall']:.3f}  F1={result[label][1]['word_f1']:.3f}  Lat={result[label][1]['latency_s']:.1f}s")
        except FileNotFoundError as exc:
            print(f"  SKIP {dirname}: {exc}")
    return result


# ── Chart 1: Grouped bars ─────────────────────────────────────────────────────

def plot_grouped_bars(data: dict, configs: list[tuple]) -> None:
    labels_in_data = [c[0] for c in configs if c[0] in data]
    colors_map = {c[0]: c[2] for c in configs}

    metric_labels = [m[1] for m in METRICS]
    n_metrics = len(METRICS)
    n_configs = len(labels_in_data)
    x = np.arange(n_metrics)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, label in enumerate(labels_in_data):
        _, summ = data[label]
        vals = [summ.get(m[0], 0) for m in METRICS]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=label.replace("\n", " "),
            color=colors_map[label],
            edgecolor="#0f1117", linewidth=0.6, alpha=0.9,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=colors_map[label],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Veridicta — Comparaison des architectures de retrieval\n"
        "Copilot / gpt-4.1 · k=5 · 100 questions",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "1_grouped_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Chart 2: Radar ────────────────────────────────────────────────────────────

def plot_radar(data: dict, configs: list[tuple]) -> None:
    labels_in_data = [c[0] for c in configs if c[0] in data]
    colors_map = {c[0]: c[2] for c in configs}

    all_summaries = {label: data[label][1] for label in labels_in_data}
    max_lat = max(s["latency_s"] for s in all_summaries.values()) or 1

    radar_labels = [m[1] for m in METRICS] + ["Rapidité"]
    n = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b27")

    for label in labels_in_data:
        s = all_summaries[label]
        vals = []
        for m_key, _, higher_better in METRICS:
            v = s.get(m_key, 0)
            vals.append(v if higher_better else 1 - v)
        vals.append(1 - s["latency_s"] / (max_lat * 1.25))
        vals += vals[:1]
        color = colors_map[label]
        ax.plot(angles, vals, "o-", linewidth=2, label=label.replace("\n", " "), color=color)
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10, color="#c9d1e0")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#8892ab")
    ax.set_title(
        "Profil multi-dimensionnel des architectures\n(copilot / gpt-4.1)",
        fontsize=12, fontweight="bold", pad=22, color="#e8d5a3",
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.32, 1.12),
        fontsize=9, framealpha=0.9,
    )
    # Spiderweb grid styling
    ax.grid(color="#2a2f47", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#2a2f47")

    fig.tight_layout()
    out = OUTPUT_DIR / "2_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Chart 3: Quality × Latency bubble scatter ─────────────────────────────────

def plot_quality_vs_latency(data: dict, configs: list[tuple]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for label, dirname, color, marker in configs:
        if label not in data:
            continue
        _, s = data[label]
        kw = s.get("keyword_recall", 0)
        lat = s.get("latency_s", 0)
        f1 = s.get("word_f1", 0)
        size = 200 + f1 * 1600   # bubble size encodes F1

        ax.scatter(
            lat, kw, s=size, color=color, marker=marker,
            edgecolors="white", linewidths=0.8, alpha=0.88, zorder=3,
            label=label.replace("\n", " "),
        )
        ax.annotate(
            label.replace("\n", " "),
            (lat, kw), textcoords="offset points", xytext=(8, 5),
            fontsize=8.5, color=color, fontweight="bold",
        )

    ax.set_xlabel("Latence moyenne (s)", fontsize=11)
    ax.set_ylabel("KW Recall moyen", fontsize=11)
    ax.set_title(
        "Qualité vs Latence — toutes architectures\n"
        "(taille des bulles = Word F1)",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Bubble size legend
    for f1_val, lbl in [(0.25, "F1=0.25"), (0.35, "F1=0.35")]:
        ax.scatter([], [], s=200 + f1_val * 1600, c="gray", alpha=0.5,
                   label=lbl)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "3_quality_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Chart 4: Per-question scatter KW vs CtxCov ──────────────────────────────

def plot_per_question_scatter(data: dict, configs: list[tuple]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for label, dirname, color, marker in configs:
        if label not in data:
            continue
        rows, _ = data[label]
        kw = [r.get("keyword_recall", 0) for r in rows]
        ctx = [r.get("context_coverage", 0) for r in rows]
        ax.scatter(
            kw, ctx, alpha=0.45, s=28, color=color, marker=marker,
            edgecolors="none", label=label.replace("\n", " "), zorder=2,
        )

    ax.set_xlabel("Keyword Recall (par question)", fontsize=11)
    ax.set_ylabel("Context Coverage (par question)", fontsize=11)
    ax.set_title(
        "Distribution par question — KW Recall vs Couverture contextuelle\n"
        "(chaque point = 1 question)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "4_per_question_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Chart 5: Progression (delta vs FAISS) ────────────────────────────────────

def plot_progression(data: dict, configs: list[tuple]) -> None:
    """Horizontal grouped bar showing % gain/loss vs FAISS baseline."""
    labels_in_data = [c[0] for c in configs if c[0] in data]
    colors_map = {c[0]: c[2] for c in configs}

    if "FAISS" not in data:
        print("  SKIP progression: FAISS baseline missing")
        return

    baseline = data["FAISS"][1]
    others = [l for l in labels_in_data if l != "FAISS"]

    progress_metrics = [
        ("keyword_recall",        "KW Recall",        True),
        ("word_f1",               "Word F1",          True),
        ("citation_faithfulness", "Citation Faith.",  True),
        ("context_coverage",      "Context Coverage", True),
        ("hallucination_risk",    "Hallucin. Risk",   False),
        ("latency_s",             "Latence (coût)",   False),
    ]

    n_metrics = len(progress_metrics)
    n_others = len(others)
    y = np.arange(n_metrics)
    height = 0.7 / n_others

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvline(0, color="#8892ab", linewidth=1, linestyle="--", zorder=1)

    for i, label in enumerate(others):
        _, summ = data[label]
        deltas = []
        for m_key, _, higher_is_better in progress_metrics:
            base_val = baseline.get(m_key, 0) or 1e-9
            delta_pct = (summ.get(m_key, 0) - base_val) / abs(base_val) * 100
            if not higher_is_better:
                delta_pct = -delta_pct
            deltas.append(delta_pct)
        offset = (i - (n_others - 1) / 2) * height
        bars = ax.barh(
            y + offset, deltas, height,
            label=label.replace("\n", " "),
            color=colors_map[label],
            edgecolor="#0f1117", linewidth=0.5, alpha=0.9,
        )
        for bar, val in zip(bars, deltas):
            x_pos = val + (0.8 if val >= 0 else -0.8)
            ax.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color=colors_map[label], fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in progress_metrics], fontsize=10)
    ax.set_xlabel("Variation vs FAISS baseline (%)", fontsize=11)
    ax.set_title(
        "Gain par rapport au FAISS de référence\n"
        "(positif = meilleur, négatif = moins bon)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = OUTPUT_DIR / "5_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Chart 6: Per-topic heatmap ────────────────────────────────────────────────

def plot_topic_heatmap(data: dict, configs: list[tuple]) -> None:
    labels_in_data = [c[0] for c in configs if c[0] in data]

    topics_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for label in labels_in_data:
        rows, _ = data[label]
        for r in rows:
            topic = str(r.get("topic") or r.get("question_id", "?").rsplit("-", 1)[0])
            topics_data[topic][label].append(r.get("keyword_recall", 0))

    topics = sorted(topics_data.keys())
    n_topics = len(topics)
    n_configs = len(labels_in_data)
    clean_labels = [l.replace("\n", " ") for l in labels_in_data]

    matrix = np.zeros((n_topics, n_configs))
    for i, topic in enumerate(topics):
        for j, label in enumerate(labels_in_data):
            vals = topics_data[topic].get(label, [0])
            matrix[i, j] = _avg(vals)

    fig_h = max(5, n_topics * 0.38)
    fig, ax = plt.subplots(figsize=(max(8, n_configs * 2.2), fig_h))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(clean_labels, fontsize=10)
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_title(
        "KW Recall par thème et architecture",
        fontsize=13, fontweight="bold",
    )

    for i in range(n_topics):
        for j in range(n_configs):
            v = matrix[i, j]
            txt_color = "white" if v < 0.35 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="KW Recall")
    cbar.ax.yaxis.set_tick_params(color="#8892ab")

    fig.tight_layout()
    out = OUTPUT_DIR / "6_topic_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===")
    all_data = _load_all(ALL_CONFIGS)
    copilot_data = {k: v for k, v in all_data.items()
                    if k in {c[0] for c in COPILOT_CONFIGS}}

    print("\n=== Generating charts ===")
    plot_grouped_bars(all_data, ALL_CONFIGS)
    plot_radar(copilot_data, COPILOT_CONFIGS)
    plot_quality_vs_latency(all_data, ALL_CONFIGS)
    plot_per_question_scatter(copilot_data, COPILOT_CONFIGS)
    plot_progression(all_data, ALL_CONFIGS)
    plot_topic_heatmap(copilot_data, COPILOT_CONFIGS)

    print(f"\nDone. Charts saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
