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


def collect_snapshot(root: Path, collectors=None, journal_scope=None) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    collected_at = datetime.now(timezone.utc).isoformat()
    counts = {}
    status_path = root / "collection-status.json"
    try:
        for name, collect in (collectors or COLLECTORS).items():
            counts[name] = collect(root / "raw" / f"{name}.jsonl")
        journal_coverage = "not_collected"
        provenance = "https://legimonaco.mc ; labour-law queries"
        if journal_scope is not None:
            from data_ingest.journal_snapshot import collect_journal

            journal_coverage = collect_journal(root, journal_scope)
            counts["journal_monaco"] = journal_coverage["documents"]
            provenance += " ; https://journaldemonaco.gouv.mc ; declared keyword/theme scope"
        manifest = create_manifest(root, collected_at, provenance)
        manifest["coverage"] = {"legimonaco": {name: count for name, count in counts.items() if name != "journal_monaco"},
                                "journal_de_monaco": journal_coverage}
        (root / "snapshot.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status = {"status": "collected", "counts": counts, "release_ready": False,
                  "remaining": ["Journal de Monaco coverage review", "effective-date review", "index build", "independent evaluation"]}
    except Exception as error:
        status_path.write_text(json.dumps({"status": "failed", "error_type": type(error).__name__, "completed_counts": counts}), encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New, non-existing snapshot directory")
    parser.add_argument("--include-journal", action="store_true", help="Also collect the declared labour circulars search scope")
    parser.add_argument("--journal-max-per-query", type=int, default=200)
    args = parser.parse_args()
    scope = None
    if args.include_journal:
        from data_ingest.journal_snapshot import JournalScope

        scope = JournalScope(max_per_query=args.journal_max_per_query)
    print(json.dumps(collect_snapshot(args.out, journal_scope=scope)))


if __name__ == "__main__":
    main()
