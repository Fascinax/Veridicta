"""Measure deterministic guard regressions without claiming legal accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from retrievers.grounding import enforce_grounding, verify_grounding


def evaluate_cases(cases: list[dict], verifier=verify_grounding, enforcer=enforce_grounding) -> dict:
    if not isinstance(cases, list) or not cases:
        raise ValueError("A guard benchmark requires at least one case")
    results = []
    for case in cases:
        report = verifier(case["answer"], case["sources"])
        accepted = report.supported_count > 0 and report.unsupported_count == 0
        abstained = report.abstained_count > 0 and not (report.supported_count or report.unsupported_count)
        verifier_pass = accepted == case["expected_supported"] and abstained == case.get("expected_abstained", False)
        row = {"id": case["id"], "expected_supported": case["expected_supported"],
               "accepted": accepted, "abstained": abstained, "verifier_pass": verifier_pass, "pass": verifier_pass}
        if "expected_excerpt" in case and not accepted:
            rendered, _ = enforcer(case["answer"], case["sources"])
            lines = rendered.splitlines()
            excerpt_returned = bool(lines and lines[0] == case["expected_excerpt"] and all(
                all(claim.status == "abstained" for claim in verifier(line, []).claims) for line in lines[1:]
            ))
            row["excerpt_returned"] = excerpt_returned
            row["pass"] = excerpt_returned and not abstained
        results.append(row)
    accepted = [row for row in results if row["accepted"]]
    return {
        "scope": "synthetic_guard_regression_only",
        "verifier_passed": sum(row["verifier_pass"] for row in results),
        "cases": len(results), "passed": sum(row["pass"] for row in results),
        "acceptance_rate": len(accepted) / len(results) if results else 0,
        "false_acceptance_rate": sum(not row["expected_supported"] for row in accepted) / len(accepted) if accepted else None,
        "results": results,
    }


def historical_guard(revision: str):
    source = subprocess.run(["git", "show", f"{revision}:retrievers/grounding.py"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    module = types.ModuleType("_historical_grounding")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:retrievers/grounding.py", "exec"), module.__dict__)
    return module.verify_grounding, module.enforce_grounding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/datasets/guard_regressions.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-revision")
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases = json.loads(raw)
    report = {"fixture_sha256": hashlib.sha256(raw).hexdigest(), "current": evaluate_cases(cases)}
    if args.compare_revision:
        report["comparison_revision"] = args.compare_revision
        report["historical"] = evaluate_cases(cases, *historical_guard(args.compare_revision))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic guard regressions: {report['current']['passed']}/{len(cases)}")
    raise SystemExit(0 if report["current"]["passed"] == len(cases) else 1)


if __name__ == "__main__":
    main()
