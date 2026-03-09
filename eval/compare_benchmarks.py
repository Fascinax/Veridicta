"""Compare two pytest-benchmark JSON snapshots.

Usage:
    python -m eval.compare_benchmarks --old path/to/old.json --new path/to/new.json

If --old/--new are omitted, the latest two files from eval/results/benchmarks are used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BENCH_DIR = Path("eval/results/benchmarks")
DEFAULT_PATTERN = "perf_full_*.json"
MS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class BenchmarkStats:
    mean_s: float
    stddev_s: float
    median_s: float
    min_s: float
    max_s: float
    rounds: int


@dataclass(frozen=True)
class BenchmarkDelta:
    name: str
    old_mean_ms: float
    new_mean_ms: float
    delta_ms: float
    delta_pct: float | None


def _load_benchmarks(path: Path) -> dict[str, BenchmarkStats]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, BenchmarkStats] = {}
    for entry in data.get("benchmarks", []):
        test_name = entry.get("name", "").split("::")[-1]
        stats = entry.get("stats", {})
        results[test_name] = BenchmarkStats(
            mean_s=float(stats.get("mean", 0.0)),
            stddev_s=float(stats.get("stddev", 0.0)),
            median_s=float(stats.get("median", 0.0)),
            min_s=float(stats.get("min", 0.0)),
            max_s=float(stats.get("max", 0.0)),
            rounds=int(stats.get("rounds", 0)),
        )
    return results


def _select_latest_files(bench_dir: Path, pattern: str) -> tuple[Path, Path]:
    files = sorted(bench_dir.glob(pattern))
    if len(files) < 2:
        raise SystemExit(
            f"Need at least two benchmark files in {bench_dir} matching {pattern}."
        )
    return files[-2], files[-1]


def _build_deltas(old_stats: dict[str, BenchmarkStats], new_stats: dict[str, BenchmarkStats]) -> list[BenchmarkDelta]:
    deltas: list[BenchmarkDelta] = []
    for name in sorted(set(old_stats) & set(new_stats)):
        old_mean_ms = old_stats[name].mean_s * MS_PER_SECOND
        new_mean_ms = new_stats[name].mean_s * MS_PER_SECOND
        delta_ms = new_mean_ms - old_mean_ms
        delta_pct = (delta_ms / old_mean_ms * 100.0) if old_mean_ms else None
        deltas.append(
            BenchmarkDelta(
                name=name,
                old_mean_ms=old_mean_ms,
                new_mean_ms=new_mean_ms,
                delta_ms=delta_ms,
                delta_pct=delta_pct,
            )
        )
    return deltas


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _print_table(deltas: list[BenchmarkDelta], old_path: Path, new_path: Path) -> None:
    print()
    print("=" * 100)
    print(f"Old: {old_path}")
    print(f"New: {new_path}")
    print("=" * 100)
    print(f"  {'Test':<36} {'Old (ms)':>10} {'New (ms)':>10} {'Delta (ms)':>12} {'Delta %':>10}")
    print("-" * 100)
    for item in deltas:
        print(
            f"  {item.name:<36} "
            f"{item.old_mean_ms:>10.3f} "
            f"{item.new_mean_ms:>10.3f} "
            f"{item.delta_ms:>12.3f} "
            f"{_format_pct(item.delta_pct):>10}"
        )
    print("=" * 100)

    regressions = sum(1 for item in deltas if (item.delta_pct or 0.0) > 0.0)
    improvements = sum(1 for item in deltas if (item.delta_pct or 0.0) < 0.0)
    unchanged = len(deltas) - regressions - improvements
    print(f"Regressions: {regressions}  Improvements: {improvements}  Flat: {unchanged}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare pytest-benchmark JSON snapshots and report mean deltas.")
    parser.add_argument("--old", type=Path, help="Path to older benchmark JSON.")
    parser.add_argument("--new", type=Path, help="Path to newer benchmark JSON.")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=DEFAULT_BENCH_DIR,
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for benchmark files (default: {DEFAULT_PATTERN}).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    old_path = args.old
    new_path = args.new

    if old_path is None or new_path is None:
        old_path, new_path = _select_latest_files(args.bench_dir, args.pattern)

    old_stats = _load_benchmarks(old_path)
    new_stats = _load_benchmarks(new_path)
    deltas = _build_deltas(old_stats, new_stats)
    _print_table(deltas, old_path, new_path)


if __name__ == "__main__":
    main()
