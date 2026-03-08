"""Run all quick-win experiment configurations and save results.

Each config runs 100 questions with copilot/gpt-4.1, workers=4.
Results are saved to eval/results/<label>/.

Usage:
    python -m eval.run_quickwins
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

EXPERIMENTS: list[dict] = [
    {
        "label": "copilot-hybrid-k8",
        "args": ["--retriever", "hybrid", "--k", "8"],
    },
    {
        "label": "copilot-hybrid-k10",
        "args": ["--retriever", "hybrid", "--k", "10"],
    },
    {
        "label": "copilot-hybrid-reranker",
        "args": ["--retriever", "hybrid", "--reranker"],
    },
    {
        "label": "copilot-hybrid-promptv2",
        "args": ["--retriever", "hybrid", "--prompt-version", "2"],
    },
    {
        "label": "copilot-hybrid-reranker-promptv2",
        "args": ["--retriever", "hybrid", "--reranker", "--prompt-version", "2"],
    },
]

COMMON_ARGS = [
    "--backend", "copilot",
    "--model", "gpt-4.1",
    "--workers", "4",
    "--questions", "eval/test_questions.json",
]


def main() -> None:
    total = len(EXPERIMENTS)
    for i, exp in enumerate(EXPERIMENTS, 1):
        label = exp["label"]
        out_dir = f"eval/results/{label}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            PYTHON, "-m", "eval.evaluate",
            *COMMON_ARGS,
            "--out", out_dir,
            *exp["args"],
        ]

        print(f"\n{'='*70}")
        print(f"  [{i}/{total}] {label}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'='*70}\n")

        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        elapsed = time.monotonic() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\n  -> {label}: {status}  ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print("  All experiments complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
