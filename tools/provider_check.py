"""Check a provider with an explicit synthetic, non-sensitive request."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("copilot", "cerebras", "omniroute"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Explicit local OpenAI-compatible endpoint")
    args = parser.parse_args()
    load_dotenv()
    if args.base_url:
        if args.backend != "omniroute":
            parser.error("--base-url requires omniroute")
        os.environ["OMNIROUTE_BASE_URL"] = args.base_url
    logging.disable(logging.CRITICAL)
    from retrievers.baseline_rag import answer

    started = time.perf_counter()
    report = {"backend": args.backend, "model": args.model, "input_class": "synthetic"}
    try:
        response, trace = answer(
            "Quel délai le document de test indique-t-il ?",
            [{"text": "Le document de test indique un délai de 30 jours.", "titre": "Document synthétique"}],
            backend=args.backend, model=args.model, return_trace=True,
        )
        report.update(status="ok", answer=response, grounding=trace.get("grounding"))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
    report["latency_s"] = round(time.perf_counter() - started, 3)
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
