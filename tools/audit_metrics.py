"""Summarize scoped audit logs without emitting their query or answer content."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def summarize_audits(root: Path) -> dict:
    latencies: list[float] = []
    stages: dict[str, list[float]] = {}
    errors = invalid = count = 0
    for path in root.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                latency = float(record["latency_s"])
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError("invalid latency")
            except (ValueError, TypeError, KeyError):
                invalid += 1
                continue
            count += 1
            latencies.append(latency)
            errors += bool((record.get("failure_classification") or {}).get("generation_error"))
            for name, value in (record.get("stage_latencies_s") or {}).items():
                if name in {"retrieval", "generation"} and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    stages.setdefault(name, []).append(float(value))
    return {"request_count": count, "invalid_row_count": invalid, "generation_error_count": errors,
            "latency_s": _distribution(latencies),
            "stage_latencies_s": {name: _distribution(values) for name, values in stages.items()}}


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p95": None}
    ordered = sorted(values)
    return {"count": len(values), "median": median(ordered), "p95": ordered[math.ceil(len(ordered) * .95) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_audits(args.root), indent=2))


if __name__ == "__main__":
    main()
