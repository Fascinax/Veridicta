"""Collect LegiMonaco into a new snapshot without promoting or replacing an index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from data_ingest import legimonaco_scraper as scraper
from retrievers.snapshots import create_manifest


COLLECTORS = {
    "legislation": scraper.collect_legislation,
    "jurisprudence": scraper.collect_jurisprudence,
    "regulations": scraper.collect_regulations,
    "jurisprudence_courts": scraper.collect_cross_court_jurisprudence,
    "traites_internationaux": scraper.collect_traites_internationaux,
    "projets_loi": scraper.collect_projets_loi,
}


def collect_snapshot(root: Path, collectors=None) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    collected_at = datetime.now(timezone.utc).isoformat()
    counts = {}
    status_path = root / "collection-status.json"
    try:
        for name, collect in (collectors or COLLECTORS).items():
            counts[name] = collect(root / "raw" / f"{name}.jsonl")
        manifest = create_manifest(root, collected_at, "https://legimonaco.mc ; labour-law queries")
        manifest["coverage"] = {"legimonaco": counts, "journal_de_monaco": "not_collected"}
        (root / "snapshot.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status = {"status": "collected", "counts": counts, "release_ready": False, "remaining": ["Journal de Monaco coverage", "effective-date review", "index build", "independent evaluation"]}
    except Exception as error:
        status_path.write_text(json.dumps({"status": "failed", "error_type": type(error).__name__, "completed_counts": counts}), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New, non-existing snapshot directory")
    args = parser.parse_args()
    print(json.dumps(collect_snapshot(args.out)))


if __name__ == "__main__":
    main()
