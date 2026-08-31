"""Validate the versioned Veridicta evaluation contract and its artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.contract import (
    DEFAULT_CONTRACT_PATH,
    ContractValidationError,
    EvaluationContract,
    load_contract,
    validate_human_gold,
    validate_questions_file,
    validate_results_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve_cli_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Veridicta's fixed regression set, human gold packet, and results."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Versioned contract JSON (default: eval/evaluation_contract.json)",
    )
    parser.add_argument(
        "--strict-human-labels",
        action="store_true",
        help="Fail until every annotation row has a reviewed human label.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional full-run JSONL to validate and summarise.",
    )
    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Return a failure when a computed quality gate is below its threshold.",
    )
    return parser.parse_args()


def _print_contract_summary(contract: EvaluationContract) -> None:
    print(f"Contract: {contract.version}")
    print(
        "Regression set: "
        f"{contract.regression_set.path} ({contract.regression_set.expected_count} questions)"
    )
    print(
        "Human gold packet: "
        f"{contract.human_gold_set.packet_path} ({contract.human_gold_set.expected_count} rows)"
    )


def _print_human_summary(report) -> None:
    print(
        "Human labels: "
        f"{report.labeled_count}/{report.annotation_count} reviewed; "
        f"{report.pending_count} pending"
    )


def _print_results_summary(report) -> None:
    print(
        "Results: "
        f"{report.count} rows, latency mean={report.mean_latency_s:.4f}s, "
        f"p95={report.p95_latency_s:.4f}s"
    )
    cost = "n/a" if report.known_cost_usd is None else f"{report.known_cost_usd:.6f} USD"
    print(f"Cost: {cost} ({report.cost_rows}/{report.count} rows with a value)")
    for check in report.quality_checks:
        if check.observed is None:
            status = "not computed"
        else:
            status = "PASS" if check.passed else "FAIL"
        threshold = "n/a" if check.minimum is None else f">={check.minimum:.2f}"
        observed = "n/a" if check.observed is None else f"{check.observed:.4f}"
        print(f"Quality gate {check.metric}: {observed} {threshold} [{status}]")
    for warning in report.warnings:
        print(f"WARNING: {warning}")


def main() -> None:
    args = _parse_args()
    try:
        contract = load_contract(_resolve_cli_path(args.contract))
        fingerprint = validate_questions_file(
            _resolve_cli_path(Path(contract.regression_set.path)),
            contract,
        )
        human_report = validate_human_gold(
            contract,
            require_labels=args.strict_human_labels,
        )
        result_report = None
        if args.results is not None:
            result_report = validate_results_file(_resolve_cli_path(args.results), contract)
    except ContractValidationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    _print_contract_summary(contract)
    print(f"Regression SHA-256: {fingerprint.sha256}")
    _print_human_summary(human_report)
    if result_report is not None:
        _print_results_summary(result_report)
        if args.fail_on_quality_gate and any(
            check.passed is False for check in result_report.quality_checks
        ):
            raise SystemExit("ERROR: one or more quality gates failed.")
    print("Contract validation: OK")


if __name__ == "__main__":
    main()
