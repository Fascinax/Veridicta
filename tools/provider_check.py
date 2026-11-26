"""Check a provider with an explicit synthetic, non-sensitive request."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv


def supported_synthetic_answer(trace: dict) -> bool:
    grounding = trace.get("grounding") or {}
    return (grounding.get("supported_count", 0) >= 1
            and grounding.get("unsupported_count", 0) == 0
            and not trace.get("failure_reason"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("copilot", "cerebras", "omniroute"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Explicit local OpenAI-compatible endpoint")
    parser.add_argument("--out", type=Path, help="Persist the timestamped synthetic report")
    args = parser.parse_args()
    load_dotenv()
    if args.base_url:
        if args.backend != "omniroute":
            parser.error("--base-url requires omniroute")
        os.environ["OMNIROUTE_BASE_URL"] = args.base_url
    logging.disable(logging.CRITICAL)
    from retrievers.baseline_rag import answer

    started = time.perf_counter()
    report = {"schema_version": "1.0", "backend": args.backend, "model": args.model, "input_class": "synthetic"}
    if args.backend == "omniroute":
        from tools.omniroute_client import omniroute_base_url
        from retrievers.release_evidence import endpoint_fingerprint
        report["endpoint_sha256"] = endpoint_fingerprint(omniroute_base_url())
    try:
        response, trace = answer(
            "Quel délai le document de test indique-t-il ?",
            [{"text": "Le document de test indique un délai de 30 jours.", "titre": "Document synthétique"}],
            backend=args.backend, model=args.model, return_trace=True,
        )
        report.update(status="ok" if supported_synthetic_answer(trace) else "failed",
                      answer=response, grounding=trace.get("grounding"))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
    report["latency_s"] = round(time.perf_counter() - started, 3)
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
