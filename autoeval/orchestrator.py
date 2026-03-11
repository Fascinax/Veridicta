"""autoeval/orchestrator.py — Autonomous optimization loop.

Sends program.md + results history + experiment.py to Copilot (via SDK),
asks the LLM to propose new parameter values, applies them, runs the
experiment, and repeats until constraints are met or max iterations reached.

Usage:
    python autoeval/orchestrator.py --pat ghp_XXXX
    python autoeval/orchestrator.py --pat ghp_XXXX --max-iter 30
    python autoeval/orchestrator.py --pat ghp_XXXX --full
    python autoeval/orchestrator.py                  # uses GITHUB_PAT env var
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOEVAL_DIR = Path(__file__).resolve().parent
EXPERIMENT_PY = AUTOEVAL_DIR / "experiment.py"
PROGRAM_MD = AUTOEVAL_DIR / "program.md"
RESULTS_TSV = AUTOEVAL_DIR / "results.tsv"

PARAM_BLOCK_START = "# === TUNABLE PARAMETERS ==="
PARAM_BLOCK_END = "# === END TUNABLE PARAMETERS ==="

SUPERVISOR_MODEL = "gpt-4.1"

PARAM_SCHEMA = {
    "RETRIEVER": {"type": str, "choices": ["faiss", "hybrid", "lancedb", "graph", "hybrid_graph", "lancedb_graph"]},
    "K": {"type": int, "min": 3, "max": 15},
    "VECTOR_WEIGHT": {"type": float, "min": 0.1, "max": 0.9},
    "FTS_WEIGHT": {"type": float, "min": 0.1, "max": 0.9},
    "HYBRID_FAISS_WEIGHT": {"type": (float, type(None)), "min": 0.1, "max": 0.9},
    "HYBRID_BM25_WEIGHT": {"type": (float, type(None)), "min": 0.1, "max": 0.9},
    "USE_RERANKER": {"type": bool},
    "RERANKER_CANDIDATE_MULTIPLIER": {"type": int, "min": 2, "max": 8},
    "RERANKER_MIN_SCORE": {"type": (float, type(None))},
    "QUERY_EXPANSION": {"type": bool},
    "PROMPT_VERSION": {"type": int, "choices": [1, 2, 3]},
    "NOTE": {"type": str},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoEval orchestrator — autonomous optimization loop")
    parser.add_argument("--pat", type=str, default=None,
                        help="GitHub PAT with copilot scope (or set GITHUB_PAT env var)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Maximum number of optimization iterations (default: 20)")
    parser.add_argument("--full", action="store_true",
                        help="Run experiments in full RAG mode (slower, ~15 min per iter)")
    parser.add_argument("--model", type=str, default=SUPERVISOR_MODEL,
                        help=f"Copilot model for the supervisor LLM (default: {SUPERVISOR_MODEL})")
    return parser.parse_args()


def _resolve_pat(cli_pat: str | None) -> str:
    if cli_pat:
        return cli_pat
    for key in ("GITHUB_PAT", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    sys.exit("ERROR: No GitHub PAT provided. Use --pat or set GITHUB_PAT env var.")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_results_tail(max_rows: int = 15) -> str:
    if not RESULTS_TSV.exists():
        return "(no experiments run yet)"
    lines = RESULTS_TSV.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return "(no experiments run yet)"
    header = lines[0]
    rows = lines[1:]
    tail = rows[-max_rows:]
    return header + "\n" + "\n".join(tail)


def _extract_param_block() -> str:
    src = EXPERIMENT_PY.read_text(encoding="utf-8")
    start = src.index(PARAM_BLOCK_START)
    end = src.index(PARAM_BLOCK_END) + len(PARAM_BLOCK_END)
    return src[start:end]


def _build_system_prompt() -> str:
    return _read_text(PROGRAM_MD)


def _build_user_prompt(last_output: str, iteration: int) -> str:
    param_block = _extract_param_block()
    results_tail = _read_results_tail()
    return textwrap.dedent(f"""\
        ## Iteration {iteration}

        ### Current experiment.py parameters
        ```python
        {param_block}
        ```

        ### Recent results (last 15 experiments)
        ```
        {results_tail}
        ```

        ### Output of last experiment
        ```
        {last_output if last_output else "(first iteration — no output yet)"}
        ```

        ---

        Based on the results so far, propose the next set of tunable parameters
        to try.  Explain your reasoning in 2-3 sentences, then output EXACTLY
        one JSON block with the new values:

        ```json
        {{
          "RETRIEVER": "...",
          "K": ...,
          "VECTOR_WEIGHT": ...,
          "FTS_WEIGHT": ...,
          "HYBRID_FAISS_WEIGHT": null,
          "HYBRID_BM25_WEIGHT": null,
          "USE_RERANKER": false,
          "RERANKER_CANDIDATE_MULTIPLIER": 4,
          "RERANKER_MIN_SCORE": null,
          "QUERY_EXPANSION": false,
          "PROMPT_VERSION": 3,
          "NOTE": "one-line hypothesis"
        }}
        ```

        Rules:
        - Include ALL parameters in the JSON, even unchanged ones.
        - Only use valid values per the parameter space in program.md.
        - The NOTE should describe your hypothesis for this experiment.
    """)


def _call_copilot(system: str, user: str, pat: str, model: str) -> str:
    from tools.copilot_client import CopilotClient, BridgeError

    old_pat = os.environ.get("GITHUB_PAT")
    os.environ["GITHUB_PAT"] = pat
    try:
        client = CopilotClient(model=model)
        return client.chat(system=system, user=user, temperature=0.3)
    except BridgeError as exc:
        return f"LLM_ERROR: {exc}"
    finally:
        if old_pat is not None:
            os.environ["GITHUB_PAT"] = old_pat
        elif "GITHUB_PAT" in os.environ:
            del os.environ["GITHUB_PAT"]


def _extract_json(response: str) -> dict | None:
    pattern = r"```json\s*\n(.+?)\n\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        pattern = r"\{[^{}]*\"RETRIEVER\"[^{}]*\}"
        match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1) if "```" in match.group(0) else match.group(0))
    except json.JSONDecodeError:
        return None


def _validate_params(params: dict) -> dict | None:
    validated = {}
    for key, schema in PARAM_SCHEMA.items():
        if key not in params:
            return None
        val = params[key]

        if val is None and type(None) in (schema["type"] if isinstance(schema["type"], tuple) else (schema["type"],)):
            validated[key] = None
            continue

        if val is None:
            return None

        expected = schema["type"] if isinstance(schema["type"], tuple) else (schema["type"],)
        if not isinstance(val, expected):
            if int in expected and isinstance(val, float) and val == int(val):
                val = int(val)
            elif float in expected and isinstance(val, int):
                val = float(val)
            else:
                return None

        if "choices" in schema and val not in schema["choices"]:
            return None
        if "min" in schema and val is not None and val < schema["min"]:
            return None
        if "max" in schema and val is not None and val > schema["max"]:
            return None

        validated[key] = val
    return validated


def _apply_params(params: dict) -> None:
    src = EXPERIMENT_PY.read_text(encoding="utf-8")
    start_idx = src.index(PARAM_BLOCK_START)
    end_idx = src.index(PARAM_BLOCK_END) + len(PARAM_BLOCK_END)

    def _fmt(key: str, val) -> str:
        if val is None:
            return "None"
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    new_block = textwrap.dedent(f"""\
        # ============================================================================
        # === TUNABLE PARAMETERS === (edit this section, then run the script)
        # ============================================================================

        RETRIEVER = {_fmt("RETRIEVER", params["RETRIEVER"])}
        K = {_fmt("K", params["K"])}

        # RRF weights — LanceDB retrievers only
        VECTOR_WEIGHT = {_fmt("VECTOR_WEIGHT", params["VECTOR_WEIGHT"])}
        FTS_WEIGHT = {_fmt("FTS_WEIGHT", params["FTS_WEIGHT"])}

        # Hybrid weights — hybrid / hybrid_graph only
        HYBRID_FAISS_WEIGHT = {_fmt("HYBRID_FAISS_WEIGHT", params["HYBRID_FAISS_WEIGHT"])}
        HYBRID_BM25_WEIGHT = {_fmt("HYBRID_BM25_WEIGHT", params["HYBRID_BM25_WEIGHT"])}

        # Reranker
        USE_RERANKER = {_fmt("USE_RERANKER", params["USE_RERANKER"])}
        RERANKER_CANDIDATE_MULTIPLIER = {_fmt("RERANKER_CANDIDATE_MULTIPLIER", params["RERANKER_CANDIDATE_MULTIPLIER"])}
        RERANKER_MIN_SCORE = {_fmt("RERANKER_MIN_SCORE", params["RERANKER_MIN_SCORE"])}

        # Query expansion
        QUERY_EXPANSION = {_fmt("QUERY_EXPANSION", params["QUERY_EXPANSION"])}

        # LLM generation (only used with --full)
        PROMPT_VERSION = {_fmt("PROMPT_VERSION", params["PROMPT_VERSION"])}
        BACKEND = "copilot"
        MODEL = "gpt-4.1"
        WORKERS = 4

        # Experiment note (one line describing your hypothesis)
        NOTE = {_fmt("NOTE", params["NOTE"])}

        # ============================================================================
        # === END TUNABLE PARAMETERS ===""")

    new_src = src[:start_idx] + new_block + src[end_idx:]
    EXPERIMENT_PY.write_text(new_src, encoding="utf-8")


def _run_experiment(full: bool) -> str:
    cmd = [sys.executable, str(EXPERIMENT_PY)]
    if full:
        cmd.append("--full")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=1200,
    )
    output = result.stdout + result.stderr
    return output[-3000:] if len(output) > 3000 else output


def _check_constraints_met() -> bool:
    if not RESULTS_TSV.exists():
        return False
    import csv
    with open(RESULTS_TSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        return False
    last = rows[-1]
    try:
        kw = float(last.get("KW", 0))
        f1_raw = last.get("F1", "")
        f1 = float(f1_raw) if f1_raw else 0.0
        cit = float(last.get("CitFaith", 0))
    except ValueError:
        return False
    return kw >= 0.67 and f1 >= 0.30 and cit >= 0.95


def main() -> None:
    args = _parse_args()
    pat = _resolve_pat(args.pat)
    system_prompt = _build_system_prompt()

    print(f"\n{'='*70}")
    print("  AUTOEVAL ORCHESTRATOR")
    print(f"  Supervisor model: {args.model}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Mode: {'full RAG' if args.full else 'retrieval-only'}")
    print(f"{'='*70}\n")

    last_output = ""

    for iteration in range(1, args.max_iter + 1):
        print(f"\n{'—'*70}")
        print(f"  ITERATION {iteration}/{args.max_iter} — asking supervisor LLM ...")
        print(f"{'—'*70}\n")

        user_prompt = _build_user_prompt(last_output, iteration)
        response = _call_copilot(system_prompt, user_prompt, pat, args.model)

        if response.startswith("LLM_ERROR:"):
            print(f"  ❌ Copilot error: {response}")
            print("  Retrying next iteration ...\n")
            continue

        print("  Supervisor response:")
        for line in response.splitlines():
            print(f"    {line}")

        params = _extract_json(response)
        if params is None:
            print("\n  ❌ Could not extract JSON parameters from LLM response.")
            print("  Retrying next iteration ...\n")
            continue

        validated = _validate_params(params)
        if validated is None:
            print(f"\n  ❌ Invalid parameter values: {params}")
            print("  Retrying next iteration ...\n")
            continue

        print(f"\n  ✅ Applying: RETRIEVER={validated['RETRIEVER']} K={validated['K']} "
              f"VW={validated['VECTOR_WEIGHT']} FTS={validated['FTS_WEIGHT']} "
              f"reranker={validated['USE_RERANKER']} qe={validated['QUERY_EXPANSION']}")
        print(f"     Note: {validated['NOTE']}")

        _apply_params(validated)

        print(f"\n  ▶ Running experiment ({'--full' if args.full else 'retrieval-only'}) ...\n")
        try:
            last_output = _run_experiment(args.full)
        except subprocess.TimeoutExpired:
            last_output = "TIMEOUT: experiment exceeded 1200s"
            print(f"  ❌ {last_output}")
            continue

        summary_lines = [l for l in last_output.splitlines() if "EXPERIMENT #" in l or "KW=" in l or "SCORE" in l or "🎉" in l or "⚠️" in l]
        for line in summary_lines:
            print(f"  {line.strip()}")

        if _check_constraints_met():
            print(f"\n{'='*70}")
            print("  🎉 ALL CONSTRAINTS SATISFIED!")
            print(f"  Optimization complete after {iteration} iterations.")
            print(f"{'='*70}\n")
            return

    print(f"\n{'='*70}")
    print(f"  Reached max iterations ({args.max_iter}).")
    print("  Review autoeval/results.tsv for the best config found.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
